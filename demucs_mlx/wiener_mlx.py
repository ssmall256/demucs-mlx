"""
 Pure MLX implementation of Wiener filtering for multichannel audio separation.
 Optimized to use native mx.complex64 types and JIT compilation.
"""
from __future__ import annotations

import mlx.core as mx


def _to_complex(x: mx.array) -> mx.array:
    """Convert (..., 2) real/imag tensor to native complex64."""
    if x.shape[-1] != 2:
        raise ValueError("Last dimension must be 2 for conversion to complex.")
    # Efficient conversion without copying if possible
    return mx.add(x[..., 0], x[..., 1] * 1j)

def _from_complex(x: mx.array) -> mx.array:
    """Convert native complex64 tensor to (..., 2) real/imag."""
    return mx.stack([x.real, x.imag], axis=-1)

@mx.compile
def _invert_2x2_complex(M: mx.array) -> mx.array:
    """
    Invert 2x2 complex matrices using analytical formula.
    M shape: (..., 2, 2) complex64
    """
    # Determinant: ad - bc
    # M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
    det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
    
    # Inversion factor: 1 / det
    inv_det = 1.0 / det
    
    # Analytical inverse:
    # [[d, -b], [-c, a]] * inv_det
    out_00 = M[..., 1, 1] * inv_det
    out_01 = -M[..., 0, 1] * inv_det
    out_10 = -M[..., 1, 0] * inv_det
    out_11 = M[..., 0, 0] * inv_det
    
    # Stack back to (..., 2, 2)
    # Col stacking usually cleaner
    row0 = mx.stack([out_00, out_01], axis=-1)
    row1 = mx.stack([out_10, out_11], axis=-1)
    return mx.stack([row0, row1], axis=-2)

def _invert_covariance(Cxx: mx.array) -> mx.array:
    """
    Invert covariance matrices for nb_channels in {1, 2}.
    Cxx: (Batch, Bins, C, C) complex64
    Returns: same shape
    """
    nb_channels = Cxx.shape[-1]
    if nb_channels == 1:
        # Scalar inverse, preserve (..., 1, 1) shape.
        return 1.0 / Cxx
    if nb_channels == 2:
        return _invert_2x2_complex(Cxx)
    raise NotImplementedError(
        f"Wiener inversion currently supports C=1 or C=2 (got C={nb_channels})."
    )


@mx.compile
def _compute_covariance_batch(y_batch: mx.array) -> mx.array:
    """
    Compute covariance for a batch of frames.
    y_batch: (Batch, Bins, Channels, Sources) complex64
    Returns: (Sources, Bins, Channels, Channels) complex64 (Summed over batch)
    """
    # Permute to (Sources, Batch, Bins, Channels)
    # y shape: (B, F, C, S) -> (S, B, F, C)
    # Actually, let's keep it simple.
    # Outer product: y * y.H
    
    # (B, F, C, S) -> (B, F, S, C)
    y_trans = y_batch.transpose(0, 1, 3, 2)
    
    # We want R[s] = sum(y[t, f, :, s] @ y[t, f, :, s].H)
    # Expand dims for outer product:
    # y: (B, F, S, C, 1)
    # y_conj: (B, F, S, 1, C)
    y_exp = y_trans[..., None]
    y_conj = y_trans.conj()[..., None, :]
    
    # (B, F, S, C, C)
    R_batch = y_exp @ y_conj
    
    # Sum over Batch dim (0)
    # Result: (F, S, C, C) -> Transpose to (S, F, C, C) to match expected layout
    return mx.sum(R_batch, axis=0).transpose(1, 0, 2, 3)

@mx.compile
def _compute_power_batch(y_batch: mx.array) -> mx.array:
    """
    Compute PSD for a batch.
    y_batch: (Batch, Bins, Channels, Sources) complex64
    Returns: (Batch, Bins, Sources) real
    """
    # Mean of squared magnitude over channels
    return mx.mean(mx.abs(y_batch) ** 2, axis=2)

@mx.compile
def _apply_wiener_batch(
    x_batch: mx.array,
    v_batch: mx.array,
    R: mx.array,
    eps: float
) -> mx.array:
    """
    Apply Wiener filter to a batch.
    x_batch: (Batch, Bins, Channels) complex mixture
    v_batch: (Batch, Bins, Sources) power estimate
    R: (Sources, Bins, Channels, Channels) spatial covariance
    eps: regularization
    """
    batch_size, nb_bins, nb_channels = x_batch.shape
    
    # 1. Compute Mixture Covariance Matrix: Cxx
    # Cxx = sum_j (v_j * R_j) + regularization
    
    # v_batch: (Batch, Bins, Sources) -> (Batch, Bins, Sources, 1, 1)
    v_exp = v_batch[..., None, None]
    
    # R: (Sources, Bins, C, C) -> (1, Bins, Sources, C, C)
    R_exp = R[:, None, ...].transpose(1, 2, 0, 3, 4)
    
    # Weighted Covariance: (Batch, Bins, Sources, C, C)
    # We broadcast R over Batch
    weighted_covs = v_exp * R_exp
    
    # Sum over sources to get Mixture Covariance
    # Cxx: (Batch, Bins, C, C)
    Cxx = mx.sum(weighted_covs, axis=2)
    
    # Regularization
    eye = mx.eye(nb_channels, dtype=Cxx.dtype)
    Cxx = Cxx + eye * eps
    
    # 2. Invert Cxx
    # (Batch, Bins, C, C)
    inv_Cxx = _invert_covariance(Cxx)
    
    # 3. Compute Gain and Separate
    # Gain_j = R_j @ inv_Cxx * v_j
    # We can do this in parallel for all sources
    
    # R_exp: (1, Bins, Sources, C, C)
    # inv_Cxx: (Batch, Bins, C, C) -> (Batch, Bins, 1, C, C)
    inv_Cxx_exp = inv_Cxx[:, :, None, :, :]
    
    # Gain matrix: (Batch, Bins, Sources, C, C)
    # Matmul broadcasts: (1, F, S, C, C) @ (B, F, 1, C, C) -> (B, F, S, C, C)
    # Note: R is (C, C), Inv is (C, C). Order is R * Inv
    Gain = R_exp @ inv_Cxx_exp
    
    # Scale by power v: (Batch, Bins, Sources, 1, 1)
    Gain = Gain * v_exp
    
    # Apply to mixture x
    # x_batch: (Batch, Bins, C) -> (Batch, Bins, 1, C, 1)
    x_exp = x_batch[:, :, None, :, None]
    
    # Estimate: Gain @ x
    # (B, F, S, C, C) @ (B, F, 1, C, 1) -> (B, F, S, C, 1)
    y_hat = Gain @ x_exp
    
    # Remove last singleton dim -> (B, F, S, C)
    y_hat = y_hat.squeeze(-1)
    
    # Transpose to expected output: (B, F, C, S)
    return y_hat.transpose(0, 1, 3, 2)

def expectation_maximization(
    y: mx.array,
    x: mx.array,
    iterations: int = 2,
    eps: float = 1e-10,
    batch_size: int = 200,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Optimized EM algorithm using native complex numbers.
    y: (Frames, Bins, Channels, Sources) complex64
    x: (Frames, Bins, Channels) complex64
    """
    nb_frames, nb_bins, nb_channels = x.shape
    nb_sources = y.shape[-1]
    
    # Initialize Power Spectral Densities (PSD)
    # v: (Frames, Bins, Sources)
    v = mx.zeros((nb_frames, nb_bins, nb_sources), dtype=mx.float32)
    
    # R: (Sources, Bins, Channels, Channels) complex64
    R = mx.zeros((nb_sources, nb_bins, nb_channels, nb_channels), dtype=mx.complex64)

    for it in range(iterations):
        # --- Update Step (M-step) ---
        
        # 1. Update PSD (v)
        # Simply magnitude squared averaged over channels
        v = mx.mean(mx.abs(y) ** 2, axis=2)
        
        # 2. Update Spatial Covariance (R)
        # We must accumulate over frames. To save memory, we loop in batches.
        R_accum = mx.zeros_like(R)
        weight_accum = mx.zeros((nb_bins, nb_sources), dtype=mx.float32)
        
        for pos in range(0, nb_frames, batch_size):
            end_pos = min(nb_frames, pos + batch_size)
            # y_slice: (Batch, Bins, Channels, Sources)
            y_slice = y[pos:end_pos]
            v_slice = v[pos:end_pos]
            
            # Efficient compiled covariance calculation
            # Returns: (Sources, Bins, Channels, Channels)
            R_batch = _compute_covariance_batch(y_slice)
            R_accum = R_accum + R_batch
            
            # Accumulate weights (sum of PSDs over frames)
            # v_slice: (Batch, Bins, Sources) -> sum -> (Bins, Sources)
            weight_accum = weight_accum + mx.sum(v_slice, axis=0)
            
        # Normalize R
        # weight: (Bins, Sources) -> (Sources, Bins, 1, 1)
        weight_norm = weight_accum.transpose(1, 0)[..., None, None]
        R = R_accum / (weight_norm + eps)
        
        # --- Separation Step (E-step) ---
        # Apply Wiener filter with updated R and v
        y_new_list = []
        
        for pos in range(0, nb_frames, batch_size):
            end_pos = min(nb_frames, pos + batch_size)
            x_slice = x[pos:end_pos]
            v_slice = v[pos:end_pos]
            
            # Apply compiled Wiener filter
            y_batch = _apply_wiener_batch(x_slice, v_slice, R, eps)
            y_new_list.append(y_batch)
            
        y = mx.concatenate(y_new_list, axis=0)

    return y, v, R

def wiener(
    targets_spectrograms: mx.array,
    mix_stft: mx.array,
    iterations: int = 1,
    softmask: bool = False,
    residual: bool = False,
    scale_factor: float = 10.0,
    eps: float = 1e-10,
) -> mx.array:
    """
    Wiener-based separation.
    
    Args:
        targets_spectrograms: (Frames, Bins, Channels, Sources) - Magnitude
        mix_stft: (Frames, Bins, Channels, 2) - Complex as Real/Imag
    """
    # 1. Convert Input to Native Complex
    mix_complex = _to_complex(mix_stft)  # (F, B, C)

    # 2. Initialization
    if softmask:
        # Soft masking
        sum_specs = mx.sum(targets_spectrograms, axis=-1, keepdims=True)
        ratio = targets_spectrograms / (eps + sum_specs)
        # mix: (F, B, C, 1), ratio: (F, B, C, S)
        y = mix_complex[..., None] * ratio
    else:
        # Phase copying
        # mix_complex: (F, B, C)
        angle = mx.angle(mix_complex)[..., None] # (F, B, C, 1)
        # targets: (F, B, C, S)
        # Euler's formula: mag * exp(1j * angle)
        y = targets_spectrograms.astype(mx.complex64) * mx.exp(1j * angle)
        
    if residual:
        # mix: (F, B, C) -> (F, B, C, 1)
        # sum(y): (F, B, C)
        res_target = mix_complex[..., None] - mx.sum(y, axis=-1, keepdims=True)
        y = mx.concatenate([y, res_target], axis=-1)
        
    if iterations == 0:
        return _from_complex(y)

    # 3. Scaling for Stability
    max_val = mx.maximum(
        mx.array(1.0, dtype=mix_complex.dtype),
        mx.max(mx.abs(mix_complex)) / scale_factor
    )
    
    mix_scaled = mix_complex / max_val
    y_scaled = y / max_val
    
    # 4. EM Loop
    y_refined, _, _ = expectation_maximization(
        y_scaled,
        mix_scaled,
        iterations=iterations,
        eps=eps
    )
    
    # 5. Restore Scale and Output Format
    y_final = y_refined * max_val
    
    return _from_complex(y_final)
