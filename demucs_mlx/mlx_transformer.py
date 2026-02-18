"""
MLX transformer blocks for HTDemucs (dense attention only).
"""
from __future__ import annotations

import math
import random

import mlx.core as mx
import mlx.nn as nn

from .mlx_demucs import LayerScale
from .mlx_hdemucs import ScaledEmbedding


def create_sin_embedding(length: int, dim: int, shift: int = 0, max_period: float = 10000.0):
    if dim % 2 != 0:
        raise ValueError("dim must be even")
    pos = shift + mx.arange(length).reshape(-1, 1, 1)
    half_dim = dim // 2
    adim = mx.arange(half_dim).reshape(1, 1, -1)
    phase = pos / (max_period ** (adim / (half_dim - 1)))
    return mx.concatenate([mx.cos(phase), mx.sin(phase)], axis=-1)


def create_2d_sin_embedding(d_model: int, height: int, width: int, max_period: float = 10000.0):
    """
    Vectorized creation of 2D sinusoidal embeddings.
    Avoids Python loops for interleaving sin/cos, providing a significant speedup.
    """
    if d_model % 4 != 0:
        raise ValueError("d_model must be divisible by 4")
    half = d_model // 2
    div_term = mx.exp(mx.arange(0.0, half, 2) * -(math.log(max_period) / half))
    
    pos_w = mx.arange(0.0, width).reshape(-1, 1)
    pos_h = mx.arange(0.0, height).reshape(-1, 1)

    # --- Vectorized Width Embeddings ---
    # Shape: (D/4, 1, W) -> broadcast to (D/4, H, W)
    sin_w = mx.sin(pos_w * div_term).transpose(1, 0).reshape(-1, 1, width)
    sin_w = mx.broadcast_to(sin_w, (sin_w.shape[0], height, width))
    
    cos_w = mx.cos(pos_w * div_term).transpose(1, 0).reshape(-1, 1, width)
    cos_w = mx.broadcast_to(cos_w, (cos_w.shape[0], height, width))

    # Interleave sin/cos using stack + reshape instead of loop
    # Stack: (D/4, H, W) -> (D/4, 2, H, W) -> Reshape: (D/2, H, W)
    pe_w = mx.stack([sin_w, cos_w], axis=1).reshape(-1, height, width)

    # --- Vectorized Height Embeddings ---
    # Shape: (D/4, H, 1) -> broadcast to (D/4, H, W)
    sin_h = mx.sin(pos_h * div_term).transpose(1, 0).reshape(-1, height, 1)
    sin_h = mx.broadcast_to(sin_h, (sin_h.shape[0], height, width))
    
    cos_h = mx.cos(pos_h * div_term).transpose(1, 0).reshape(-1, height, 1)
    cos_h = mx.broadcast_to(cos_h, (cos_h.shape[0], height, width))

    # Interleave sin/cos
    pe_h = mx.stack([sin_h, cos_h], axis=1).reshape(-1, height, width)

    # Concatenate width and height embeddings
    pe = mx.concatenate([pe_w, pe_h], axis=0)
    return pe[None, :]


def create_sin_embedding_cape(
    length: int,
    dim: int,
    batch_size: int,
    mean_normalize: bool,
    augment: bool,
    max_global_shift: float = 0.0,
    max_local_shift: float = 0.0,
    max_scale: float = 1.0,
    max_period: float = 10000.0,
):
    if dim % 2 != 0:
        raise ValueError("dim must be even")
    pos = mx.arange(length).reshape(-1, 1, 1).repeat(batch_size, axis=1)
    if mean_normalize:
        pos = pos - mx.mean(pos, axis=0, keepdims=True)
    if augment:
        delta = mx.random.uniform(-max_global_shift, max_global_shift, shape=(1, batch_size, 1))
        delta_local = mx.random.uniform(
            -max_local_shift, max_local_shift,
            shape=(length, batch_size, 1))
        log_max_scale = math.log(max_scale)
        log_lambdas = mx.random.uniform(-log_max_scale, log_max_scale, shape=(1, batch_size, 1))
        pos = (pos + delta + delta_local) * mx.exp(log_lambdas)
    half_dim = dim // 2
    adim = mx.arange(half_dim).reshape(1, 1, -1)
    phase = pos / (max_period ** (adim / (half_dim - 1)))
    return mx.concatenate([mx.cos(phase), mx.sin(phase)], axis=-1)


class MyGroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels, eps=eps, pytorch_compatible=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.gn(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation,
        norm_first: bool,
        group_norm: int = 0,
        norm_out: bool = False,
        layer_scale: bool = False,
        init_values: float = 1e-4,
    ):
        super().__init__()
        self.attn = nn.MultiHeadAttention(d_model, nhead, bias=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation
        self.norm_first = norm_first
        if group_norm:
            self.norm1 = MyGroupNorm(int(group_norm), d_model)
            self.norm2 = MyGroupNorm(int(group_norm), d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        self.norm_out = MyGroupNorm(int(norm_out), d_model) if (norm_first and norm_out) else None
        self.gamma_1 = LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        self.gamma_2 = LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()

    def __call__(self, x: mx.array, attn_mask=None):
        if self.norm_first:
            x_normed = self.norm1(x)
            y = self.attn(x_normed, x_normed, x_normed, mask=attn_mask)
            y = self.dropout1(y)
            x = x + self.gamma_1(y)
            y = self.linear2(self.dropout2(self.activation(self.linear1(self.norm2(x)))))
            x = x + self.gamma_2(y)
            if self.norm_out:
                x = self.norm_out(x)
        else:
            y = self.attn(x, x, x, mask=attn_mask)
            y = self.dropout1(y)
            x = self.norm1(x + self.gamma_1(y))
            y = self.linear2(self.dropout2(self.activation(self.linear1(x))))
            x = self.norm2(x + self.gamma_2(y))
        return x


class CrossTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation,
        norm_first: bool,
        group_norm: int = 0,
        norm_out: bool = False,
        layer_scale: bool = False,
        init_values: float = 1e-4,
    ):
        super().__init__()
        self.cross_attn = nn.MultiHeadAttention(d_model, nhead, bias=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation
        self.norm_first = norm_first
        if group_norm:
            self.norm1 = MyGroupNorm(int(group_norm), d_model)
            self.norm2 = MyGroupNorm(int(group_norm), d_model)
            self.norm3 = MyGroupNorm(int(group_norm), d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
        self.norm_out = MyGroupNorm(int(norm_out), d_model) if (norm_first and norm_out) else None
        self.gamma_1 = LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        self.gamma_2 = LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()

    def __call__(self, q: mx.array, k: mx.array, attn_mask=None):
        if self.norm_first:
            k_normed = self.norm2(k)
            attn_out = self.cross_attn(
                self.norm1(q), k_normed, k_normed, mask=attn_mask)
            x = q + self.gamma_1(self.dropout1(attn_out))
            ff_out = self.linear2(
                self.dropout2(self.activation(self.linear1(self.norm3(x)))))
            x = x + self.gamma_2(ff_out)
            if self.norm_out:
                x = self.norm_out(x)
        else:
            attn_out = self.cross_attn(q, k, k, mask=attn_mask)
            x = self.norm1(
                q + self.gamma_1(self.dropout1(attn_out)))
            ff_out = self.linear2(
                self.dropout2(self.activation(self.linear1(x))))
            x = self.norm2(x + self.gamma_2(ff_out))
        return x


class CrossTransformerEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        emb: str = "sin",
        hidden_scale: float = 4.0,
        num_heads: int = 8,
        num_layers: int = 6,
        cross_first: bool = False,
        dropout: float = 0.0,
        max_positions: int = 1000,
        norm_in: bool = True,
        norm_in_group: bool = False,
        group_norm: int = False,
        norm_first: bool = False,
        norm_out: bool = False,
        max_period: float = 10000.0,
        weight_pos_embed: float = 1.0,
        layer_scale: bool = False,
        gelu: bool = True,
        sin_random_shift: int = 0,
        cape_mean_normalize: bool = True,
        cape_augment: bool = True,
        cape_glob_loc_scale: list = [5000.0, 1.0, 1.4],
        sparse_self_attn: bool = False,
        sparse_cross_attn: bool = False,
        **kwargs,
    ):
        super().__init__()
        if sparse_self_attn or sparse_cross_attn:
            raise RuntimeError("Sparse attention is not supported in MLX backend.")
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        hidden_dim = int(dim * hidden_scale)
        self.num_layers = num_layers
        self.classic_parity = 1 if cross_first else 0
        self.emb = emb
        self.max_period = max_period
        self.weight_pos_embed = weight_pos_embed
        self.sin_random_shift = sin_random_shift
        
        # Cache for positional embeddings
        self._pe_cache = {}
        
        if emb == "cape":
            self.cape_mean_normalize = cape_mean_normalize
            self.cape_augment = cape_augment
            self.cape_glob_loc_scale = cape_glob_loc_scale
        if emb == "scaled":
            self.position_embeddings = ScaledEmbedding(max_positions, dim, scale=0.2)
        activation = (lambda x: nn.gelu(x)) if gelu else (lambda x: mx.maximum(x, 0))
        if norm_in:
            self.norm_in = nn.LayerNorm(dim)
            self.norm_in_t = nn.LayerNorm(dim)
        elif norm_in_group:
            self.norm_in = MyGroupNorm(int(norm_in_group), dim)
            self.norm_in_t = MyGroupNorm(int(norm_in_group), dim)
        else:
            self.norm_in = nn.Identity()
            self.norm_in_t = nn.Identity()
        self.layers = []
        self.layers_t = []
        for idx in range(num_layers):
            if idx % 2 == self.classic_parity:
                self.layers.append(
                    TransformerEncoderLayer(
                        dim,
                        num_heads,
                        hidden_dim,
                        dropout,
                        activation,
                        norm_first,
                        group_norm,
                        norm_out,
                        layer_scale,
                    )
                )
                self.layers_t.append(
                    TransformerEncoderLayer(
                        dim,
                        num_heads,
                        hidden_dim,
                        dropout,
                        activation,
                        norm_first,
                        group_norm,
                        norm_out,
                        layer_scale,
                    )
                )
            else:
                self.layers.append(
                    CrossTransformerEncoderLayer(
                        dim,
                        num_heads,
                        hidden_dim,
                        dropout,
                        activation,
                        norm_first,
                        group_norm,
                        norm_out,
                        layer_scale,
                    )
                )
                self.layers_t.append(
                    CrossTransformerEncoderLayer(
                        dim,
                        num_heads,
                        hidden_dim,
                        dropout,
                        activation,
                        norm_first,
                        group_norm,
                        norm_out,
                        layer_scale,
                    )
                )

    def _get_pos_embedding(self, T, B, C):
        if self.emb == "sin":
            shift = random.randrange(self.sin_random_shift + 1)
            return create_sin_embedding(T, C, shift=shift, max_period=self.max_period)
        if self.emb == "cape":
            # CAPE embeddings are dynamic/augmented, usually not cached during training
            if self.training:
                return create_sin_embedding_cape(
                    T,
                    C,
                    B,
                    mean_normalize=self.cape_mean_normalize,
                    augment=self.cape_augment,
                    max_global_shift=self.cape_glob_loc_scale[0],
                    max_local_shift=self.cape_glob_loc_scale[1],
                    max_scale=self.cape_glob_loc_scale[2],
                    max_period=self.max_period,
                )
            return create_sin_embedding_cape(
                T,
                C,
                B,
                mean_normalize=self.cape_mean_normalize,
                augment=False,
                max_period=self.max_period,
            )
        pos = mx.arange(T)
        return self.position_embeddings(pos)[:, None]

    def _get_2d_embedding(self, C, Fr, T1):
        """Get cached or new 2D embedding."""
        key = (C, Fr, T1, self.max_period)
        if key in self._pe_cache:
            return self._pe_cache[key]
        
        # Generate new if not in cache
        pe = create_2d_sin_embedding(C, Fr, T1, self.max_period)
        self._pe_cache[key] = pe
        return pe

    def __call__(self, x: mx.array, xt: mx.array):
        B, C, Fr, T1 = x.shape
        
        # Use cached 2D embedding
        pos_emb_2d = self._get_2d_embedding(C, Fr, T1)

        # 1. Add batch dimension: (1, C, Fr, T1)
        pos_emb_2d = pos_emb_2d.reshape(1, C, Fr, T1)

        # 2. Broadcast to actual batch size B: (B, C, Fr, T1)
        pos_emb_2d = mx.broadcast_to(pos_emb_2d, (B, C, Fr, T1))

        # 3. Transpose and flatten
        pos_emb_2d = pos_emb_2d.transpose(0, 3, 2, 1).reshape(B, T1 * Fr, C)

        x = x.transpose(0, 3, 2, 1).reshape(B, T1 * Fr, C)
        x = self.norm_in(x)
        x = x + self.weight_pos_embed * pos_emb_2d

        B, C, T2 = xt.shape
        xt = xt.transpose(0, 2, 1)
        pos_emb = self._get_pos_embedding(T2, B, C)
        pos_emb = pos_emb.transpose(1, 0, 2)
        xt = self.norm_in_t(xt)
        xt = xt + self.weight_pos_embed * pos_emb

        for idx in range(self.num_layers):
            if idx % 2 == self.classic_parity:
                x = self.layers[idx](x)
                xt = self.layers_t[idx](xt)
            else:
                old_x = x
                x = self.layers[idx](x, xt)
                xt = self.layers_t[idx](xt, old_x)

        x = x.reshape(B, T1, Fr, C).transpose(0, 3, 2, 1)
        xt = xt.transpose(0, 2, 1)
        return x, xt
