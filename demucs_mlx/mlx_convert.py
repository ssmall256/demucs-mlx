"""
MLX Model Weight Conversion System

Converts pretrained PyTorch HTDemucs models to MLX format with proper
weight layout transformations for Conv1d/Conv2d layers.
"""
from __future__ import annotations

import inspect
import os
import pickle
import typing as tp
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import numpy as np
from packaging import version

from .mlx_backend import MIN_MLX_VERSION
from .mlx_registry import MLX_MODEL_REGISTRY


class BagOfModelsMLX:
    """
    MLX wrapper for ensemble of models with weighted averaging.

    This mirrors the PyTorch BagOfModels but operates on MLX arrays.
    Weights are per-source: weights[model_idx][source_idx]
    """
    def __init__(self, models: tp.List, weights: tp.Optional[tp.List[tp.List[float]]] = None):
        self.models = models
        self.sources = models[0].sources
        self.samplerate = models[0].samplerate
        self.audio_channels = models[0].audio_channels

        if weights is None:
            # Default: equal weights for all models and sources
            weights = [[1.0] * len(self.sources) for _ in models]

        # Store per-source weights and compute totals for normalization
        self.weights = weights
        self.totals = [0.0] * len(self.sources)
        for model_weights in weights:
            for src_idx, w in enumerate(model_weights):
                self.totals[src_idx] += w

    def __call__(self, x: mx.array) -> mx.array:
        """Apply all models and average outputs with per-source weights."""
        estimates = None

        for model, model_weights in zip(self.models, self.weights):
            out = model(x)  # Shape: [batch, sources, channels, time]

            # Apply per-source weights - reshape to broadcast correctly
            # weights shape: [sources] -> [1, sources, 1, 1]
            weight_array = mx.array(model_weights).reshape(1, len(model_weights), 1, 1)
            out = out * weight_array

            if estimates is None:
                estimates = out
            else:
                estimates = estimates + out

        # Normalize by total weights per source
        # totals shape: [sources] -> [1, sources, 1, 1]
        totals_array = mx.array(self.totals).reshape(1, len(self.totals), 1, 1)
        estimates = estimates / totals_array

        return estimates

    def state_dict(self) -> tp.Dict:
        """Return state dict for all models."""
        return {
            f'model_{i}': model.state_dict()
            for i, model in enumerate(self.models)
        }

    def load_state_dict(self, state: tp.Dict):
        """Load state dict for all models."""
        for i, model in enumerate(self.models):
            model.load_state_dict(state[f'model_{i}'])


def convert_conv_weight(
    weight: np.ndarray,
    conv_type: str,
    transpose: bool = True
) -> np.ndarray:
    """Convert convolution weight from PyTorch to MLX layout."""
    if not transpose:
        return weight

    if conv_type == 'conv1d':
        return np.transpose(weight, (0, 2, 1))
    elif conv_type == 'conv_transpose1d':
        return np.transpose(weight, (1, 2, 0))
    elif conv_type == 'conv2d':
        return np.transpose(weight, (0, 2, 3, 1))
    elif conv_type == 'conv_transpose2d':
        return np.transpose(weight, (1, 2, 3, 0))
    else:
        raise ValueError(f"Unknown conv_type: {conv_type}")


def convert_state_dict(
    torch_state: tp.Dict[str, tp.Any], # Type Hint generic to avoid Torch import
    verbose: bool = False,
    flatten: bool = False,
    torch_model: tp.Optional[tp.Any] = None,
) -> tp.Dict:
    """
    Convert PyTorch state dict to MLX format with proper layout transformations.
    """
    import torch  # LAZY IMPORT

    flat_mlx_state = {}

    module_param_types: tp.Dict[str, str] = {}
    if torch_model is not None:
        for module_name, module in torch_model.named_modules():
            if isinstance(module, torch.nn.Conv1d):
                module_param_types[f"{module_name}.weight"] = "conv1d"
            elif isinstance(module, torch.nn.ConvTranspose1d):
                module_param_types[f"{module_name}.weight"] = "conv_transpose1d"
            elif isinstance(module, torch.nn.Conv2d):
                module_param_types[f"{module_name}.weight"] = "conv2d"
            elif isinstance(module, torch.nn.ConvTranspose2d):
                module_param_types[f"{module_name}.weight"] = "conv_transpose2d"

    for name, param in torch_state.items():
        # Convert to numpy
        np_param = param.detach().cpu().numpy()

        # Determine if this is a conv weight that needs transposition
        needs_transpose = False
        conv_type = None

        is_conv_like_weight = (
            'weight' in name and
            ('conv' in name.lower() or
             'rewrite' in name.lower() or
             'upsampler' in name.lower() or
             'downsampler' in name.lower())
        )

        if name in module_param_types:
            conv_type = module_param_types[name]
            needs_transpose = True
        elif is_conv_like_weight:
            # Determine convolution type by parameter shape and name
            ndim = len(np_param.shape)
            is_transpose = 'conv_tr' in name.lower() or 'transpose' in name.lower()

            if ndim == 3:  # Conv1d or ConvTranspose1d
                conv_type = 'conv_transpose1d' if is_transpose else 'conv1d'
                needs_transpose = True

            elif ndim == 4:  # Conv2d or ConvTranspose2d
                conv_type = 'conv_transpose2d' if is_transpose else 'conv2d'
                needs_transpose = True

        # Apply transformation if needed
        if needs_transpose and conv_type:
            np_param = convert_conv_weight(np_param, conv_type)
            if verbose:
                print(f"  Transposed {name}: {param.shape} → {np_param.shape}")

        # Convert to MLX array
        flat_mlx_state[name] = mx.array(np_param)

    # Map GroupNorm wrapper names: normX.weight -> normX.gn.weight
    norm_wrapper_fixes = {}
    for name in list(flat_mlx_state.keys()):
        if ".gn." in name:
            continue
        parts = name.split(".")
        if len(parts) < 2:
            continue
        last = parts[-1]
        if last not in ("weight", "bias"):
            continue
        prev = parts[-2]
        if prev.startswith("norm"):
            new_name = ".".join(parts[:-1] + ["gn", last])
            if new_name not in flat_mlx_state:
                norm_wrapper_fixes[new_name] = flat_mlx_state[name]
    flat_mlx_state.update(norm_wrapper_fixes)

    # Map Torch BLSTM (bidirectional LSTM) params to MLX BLSTM layout.
    lstm_bias = {}
    for name in list(flat_mlx_state.keys()):
        if ".lstm." not in name:
            continue
        prefix, rest = name.split(".lstm.", 1)
        is_reverse = rest.endswith("_reverse")
        if is_reverse:
            rest = rest[:-len("_reverse")]
        if "_l" not in rest:
            continue
        base, layer_str = rest.rsplit("_l", 1)
        if not layer_str.isdigit():
            continue
        layer = int(layer_str)
        if base not in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            continue
        dir_name = "backward_lstms" if is_reverse else "forward_lstms"
        if base.startswith("weight_"):
            # MLX LSTM uses Wx/Wh parameter names
            mlx_name = "Wx" if base == "weight_ih" else "Wh"
            new_name = f"{prefix}.{dir_name}.{layer}.{mlx_name}"
            if new_name not in flat_mlx_state:
                flat_mlx_state[new_name] = flat_mlx_state[name]
        else:
            key = (prefix, dir_name, layer)
            entry = lstm_bias.setdefault(key, {})
            entry[base] = flat_mlx_state[name]

    for (prefix, dir_name, layer), entry in lstm_bias.items():
        bias_ih = entry.get("bias_ih")
        bias_hh = entry.get("bias_hh")
        if bias_ih is None and bias_hh is None:
            continue
        bias = bias_ih if bias_hh is None else (bias_hh if bias_ih is None else (bias_ih + bias_hh))
        new_name = f"{prefix}.{dir_name}.{layer}.bias"
        if new_name not in flat_mlx_state:
            flat_mlx_state[new_name] = bias

    # Post-process for transformer layers
    transformer_fixes = {}

    for name in list(flat_mlx_state.keys()):
        if 'self_attn' in name or 'cross_attn' in name:
            new_name = name.replace('self_attn', 'attn')

            if '.in_proj_weight' in name:
                weight = np.array(flat_mlx_state[name])
                embed_dim = weight.shape[0] // 3
                query_weight = weight[:embed_dim, :]
                key_weight = weight[embed_dim:2*embed_dim, :]
                value_weight = weight[2*embed_dim:, :]

                base = new_name.replace('.in_proj_weight', '')
                transformer_fixes[f"{base}.query_proj.weight"] = mx.array(query_weight)
                transformer_fixes[f"{base}.key_proj.weight"] = mx.array(key_weight)
                transformer_fixes[f"{base}.value_proj.weight"] = mx.array(value_weight)

            elif '.in_proj_bias' in name:
                bias = np.array(flat_mlx_state[name])
                embed_dim = bias.shape[0] // 3
                query_bias = bias[:embed_dim]
                key_bias = bias[embed_dim:2*embed_dim]
                value_bias = bias[2*embed_dim:]

                base = new_name.replace('.in_proj_bias', '')
                transformer_fixes[f"{base}.query_proj.bias"] = mx.array(query_bias)
                transformer_fixes[f"{base}.key_proj.bias"] = mx.array(key_bias)
                transformer_fixes[f"{base}.value_proj.bias"] = mx.array(value_bias)

            elif '.out_proj.' in name:
                transformer_fixes[new_name] = flat_mlx_state[name]

        elif '.norm_out.weight' in name or '.norm_out.bias' in name:
            new_name = name.replace('.norm_out.', '.norm_out.gn.')
            transformer_fixes[new_name] = flat_mlx_state[name]

    flat_mlx_state.update(transformer_fixes)

    if flatten:
        return flat_mlx_state
    raise NotImplementedError("Nested conversion is not supported; use flatten=True.")


def convert_single_model(
    torch_model,
    verbose: bool = False
) -> tp.Any:
    """
    Convert a single PyTorch model to MLX.
    """
    from .mlx_demucs import DemucsMLX
    from .mlx_hdemucs import HDemucsMLX
    from .mlx_htdemucs import HTDemucsMLX

    model_class = torch_model.__class__.__name__

    if verbose:
        print(f"Converting {model_class}...")

    # Extract initialization arguments
    if hasattr(torch_model, '_init_args_kwargs'):
        args, kwargs = torch_model._init_args_kwargs
    else:
        raise ValueError(f"Model {model_class} doesn't have _init_args_kwargs")

    def _filter_kwargs(target_cls, in_kwargs):
        sig = inspect.signature(target_cls)
        allowed = set(sig.parameters.keys())
        filtered = {k: v for k, v in in_kwargs.items() if k in allowed}
        if verbose:
            dropped = sorted(k for k in in_kwargs.keys() if k not in allowed)
            if dropped:
                print(f"  Dropping unsupported kwargs for {target_cls.__name__}: {dropped}")
        return filtered

    # Create MLX model
    if model_class == 'HTDemucs':
        if kwargs.get('t_sparse_self_attn'):
            raise ValueError("Sparse self-attention not supported in MLX backend")
        if kwargs.get('t_sparse_cross_attn'):
            raise ValueError("Sparse cross-attention not supported in MLX backend")
        mlx_model = HTDemucsMLX(*args, **_filter_kwargs(HTDemucsMLX, kwargs))

    elif model_class == 'HDemucs':
        mlx_model = HDemucsMLX(*args, **_filter_kwargs(HDemucsMLX, kwargs))

    elif model_class == 'Demucs':
        if "gelu" in kwargs and "gelu_act" not in kwargs:
            kwargs["gelu_act"] = kwargs["gelu"]
        if "glu" in kwargs and "glu_act" not in kwargs:
            kwargs["glu_act"] = kwargs["glu"]
        mlx_model = DemucsMLX(*args, **_filter_kwargs(DemucsMLX, kwargs))
    else:
        raise NotImplementedError(
            f"MLX conversion not implemented for {model_class}"
        )
    if hasattr(torch_model, "segment") and hasattr(mlx_model, "segment"):
        mlx_model.segment = torch_model.segment

    # Convert and load state dict
    if verbose:
        print(f"Converting {len(torch_model.state_dict())} parameters...")

    torch_state = torch_model.state_dict()
    flat_mlx_state = convert_state_dict(
        torch_state, verbose=verbose, flatten=True, torch_model=torch_model
    )

    if verbose:
        print("  Using manual weight loading...")

    _load_weights_into_model(mlx_model, flat_mlx_state)

    if verbose:
        print("  Loaded parameters manually")
    if verbose:
        print(f"✓ Converted {model_class}")

    return mlx_model


def convert_htdemucs_weights(
    model_name: str,
    output_dir: tp.Optional[str] = None,
    verify: bool = False,
    verbose: bool = True,
) -> str:
    """
    Convert Demucs/HDemucs/HTDemucs PyTorch weights to MLX format.
    """
    # Lazy imports — requires the PyTorch demucs package for conversion
    try:
        from demucs.apply import BagOfModels
        from demucs.pretrained import get_model
    except ImportError:
        raise ImportError(
            "Model conversion requires the PyTorch 'demucs' package. "
            "Install with: pip install 'demucs-mlx[convert]'"
        ) from None

    if model_name not in MLX_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(MLX_MODEL_REGISTRY.keys())}"
        )

    if output_dir is None:
        output_dir = "./mlx_checkpoints"

    os.makedirs(output_dir, exist_ok=True)

    config = MLX_MODEL_REGISTRY[model_name]

    if verbose:
        print("=" * 70)
        print(f"Converting {model_name} to MLX format")
        print(f"Description: {config['description']}")
        print("=" * 70)

    if verbose:
        print("\n1. Loading PyTorch model(s)...")

    torch_model = get_model(model_name)

    if isinstance(torch_model, BagOfModels):
        if verbose:
            print(f"   Found bag with {len(torch_model.models)} models")
        torch_models = torch_model.models
        if hasattr(torch_model, 'weights') and torch_model.weights is not None:
            weights = torch_model.weights
        else:
            num_sources = len(torch_model.sources)
            weights = [[1.0] * num_sources for _ in torch_models]
    else:
        if verbose:
            print("   Loaded single model")
        torch_models = [torch_model]
        num_sources = len(torch_model.sources)
        weights = [[1.0] * num_sources]

    if verbose:
        print(f"\n2. Converting {len(torch_models)} model(s)...")

    mlx_models = []
    for i, tm in enumerate(torch_models):
        if verbose and len(torch_models) > 1:
            print(f"\n   Model {i+1}/{len(torch_models)}:")
        mlx_model = convert_single_model(tm, verbose=verbose)
        mlx_models.append(mlx_model)

    if isinstance(torch_model, BagOfModels):
        if verbose:
            print(
                f"\n3. Creating ensemble with {len(mlx_models)} model(s) "
                f"and weights {weights}..."
            )
        final_model = BagOfModelsMLX(mlx_models, weights)
        model_class = 'BagOfModelsMLX'
        sub_model_class = type(mlx_models[0]).__name__
    else:
        final_model = mlx_models[0]
        model_class = type(final_model).__name__
        sub_model_class = None

    args, kwargs = torch_models[0]._init_args_kwargs
    kwargs = dict(kwargs)
    if hasattr(torch_models[0], "segment"):
        kwargs["segment"] = torch_models[0].segment

    per_model_args = []
    per_model_kwargs = []
    per_model_class = []
    for tm in torch_models:
        tm_args, tm_kwargs = tm._init_args_kwargs
        tm_kwargs = dict(tm_kwargs)
        if hasattr(tm, "segment"):
            tm_kwargs["segment"] = tm.segment
        per_model_args.append(list(tm_args))
        per_model_kwargs.append(tm_kwargs)
        per_model_class.append(type(tm).__name__)

    if verbose:
        print("\n4. Saving MLX checkpoint...")

    checkpoint = {
        'model_name': model_name,
        'model_class': model_class,
        'sub_model_class': sub_model_class,
        'args': args,
        'kwargs': kwargs,
        'state': final_model.state_dict(),
        'mlx_version': MIN_MLX_VERSION,
        'num_models': len(mlx_models),
        'weights': weights if isinstance(torch_model, BagOfModels) else None,
        'conversion_date': datetime.now().isoformat(),
        'torch_signatures': config['signatures'],
    }
    if isinstance(torch_model, BagOfModels):
        args_differ = any(per_model_args[0] != a for a in per_model_args[1:])
        kwargs_differ = any(per_model_kwargs[0] != k for k in per_model_kwargs[1:])
        if args_differ or kwargs_differ:
            checkpoint['per_model_args'] = per_model_args
            checkpoint['per_model_kwargs'] = per_model_kwargs
        if any(per_model_class[0] != c for c in per_model_class[1:]):
            checkpoint['per_model_class'] = per_model_class

    output_path = os.path.join(output_dir, f"{model_name}_mlx.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(checkpoint, f)

    if verbose:
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   Saved to: {output_path}")
        print(f"   File size: {file_size_mb:.1f} MB")

    if verify:
        if verbose:
            print("\n5. Running verification tests...")
        try:
            verify_conversion(torch_models[0], mlx_models[0], verbose=verbose)
            checkpoint['verification_passed'] = True
            with open(output_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            if verbose:
                print("   ✓ Verification passed")
        except Exception as e:
            if verbose:
                print(f"   ✗ Verification failed: {e}")
            checkpoint['verification_passed'] = False

    if verbose:
        print("\n" + "=" * 70)
        print(f"✓ Conversion complete: {output_path}")
        print("=" * 70)

    return output_path


def verify_conversion(
    torch_model,
    mlx_model,
    tolerance: float = 1e-4,
    verbose: bool = True
) -> bool:
    """Verify MLX conversion by comparing outputs."""
    import torch  # LAZY IMPORT
    from torch.utils.dlpack import from_dlpack, to_dlpack

    if verbose:
        print("   Testing with random input...")

    torch_input = torch.randn(1, 2, 44100 * 4)
    # Zero-copy torch -> mlx via DLPack
    mlx_input = mx.core.from_dlpack(to_dlpack(torch_input.contiguous()))

    with torch.no_grad():
        torch_model.eval()
        torch_output = torch_model(torch_input)

    if hasattr(mlx_model, "eval"):
        mlx_model.eval()
    mlx_output = mlx_model(mlx_input)

    # Zero-copy mlx -> torch via DLPack
    mx.eval(mlx_output)
    mlx_output_torch = from_dlpack(mlx_output)

    max_diff = (torch_output - mlx_output_torch).abs().max().item()
    mean_diff = (torch_output - mlx_output_torch).abs().mean().item()

    torch_max = torch_output.abs().max().item()
    rel_error = max_diff / (torch_max + 1e-8)

    if verbose:
        print(f"   Max absolute difference: {max_diff:.2e}")
        print(f"   Mean absolute difference: {mean_diff:.2e}")
        print(f"   Relative error: {rel_error:.2e}")
        print(f"   Output shape: {tuple(mlx_output_torch.shape)}")

    if rel_error > tolerance:
        raise ValueError(
            f"Verification failed: relative error {rel_error:.2e} > {tolerance:.2e}"
        )

    return True


def load_mlx_model(
    model_name: str,
    cache_dir: str = "./mlx_checkpoints",
    auto_convert: bool = True,
    verbose: bool = False,
) -> tp.Any:
    """Load MLX model from cache or convert if needed."""
    from .mlx_demucs import DemucsMLX
    from .mlx_hdemucs import HDemucsMLX
    from .mlx_htdemucs import HTDemucsMLX

    def _normalize_model_class(name: tp.Optional[str]) -> str:
        if not name:
            return "HTDemucsMLX"
        if name in {"HTDemucsMLX", "HDemucsMLX", "DemucsMLX"}:
            return name
        if "HTDemucs" in name:
            return "HTDemucsMLX"
        if "HDemucs" in name:
            return "HDemucsMLX"
        if "Demucs" in name:
            return "DemucsMLX"
        return "HTDemucsMLX"

    safetensors_path = os.path.join(cache_dir, f"{model_name}.safetensors")
    safetensors_config_path = os.path.join(cache_dir, f"{model_name}_config.json")

    if os.path.exists(safetensors_path) and os.path.exists(safetensors_config_path):
        if verbose:
            print(f"Loading from safetensors (preferred format): {safetensors_path}")
        try:
            return load_mlx_model_from_safetensors(
                model_name, cache_dir=cache_dir, verbose=verbose
            )
        except Exception as e:
            if verbose:
                print(f"Warning: Safetensors loading failed ({e}), falling back to pickle")

    cache_path = os.path.join(cache_dir, f"{model_name}_mlx.pkl")

    if os.path.exists(cache_path):
        if verbose:
            print(f"Loading cached MLX model: {cache_path}")

        with open(cache_path, 'rb') as f:
            checkpoint = pickle.load(f)

        if _version_lt(checkpoint.get('mlx_version'), MIN_MLX_VERSION):
            print("Warning: MLX version mismatch. Re-converting...")
            if auto_convert:
                return load_mlx_model(model_name, cache_dir, auto_convert=False, verbose=verbose)
            raise ValueError("MLX version mismatch")

        model_class_name = checkpoint['model_class']
        args = checkpoint['args']
        kwargs = checkpoint['kwargs']

        if model_class_name == 'BagOfModelsMLX':
            num_models = checkpoint['num_models']
            weights = checkpoint['weights']
            sub_model_class = _normalize_model_class(checkpoint.get('sub_model_class'))
            per_model_args = checkpoint.get('per_model_args')
            per_model_kwargs = checkpoint.get('per_model_kwargs')
            per_model_class = checkpoint.get('per_model_class')

            models = []
            for i in range(num_models):
                model_args = args
                model_kwargs = kwargs
                if per_model_args and i < len(per_model_args):
                    model_args = tuple(per_model_args[i])
                if per_model_kwargs and i < len(per_model_kwargs):
                    model_kwargs = per_model_kwargs[i]
                model_class = sub_model_class
                if per_model_class and i < len(per_model_class):
                    model_class = _normalize_model_class(per_model_class[i])

                if model_class == 'HTDemucsMLX':
                    model = HTDemucsMLX(*model_args, **model_kwargs)
                elif model_class == 'HDemucsMLX':
                    model = HDemucsMLX(*model_args, **model_kwargs)
                elif model_class == 'DemucsMLX':
                    model = DemucsMLX(*model_args, **model_kwargs)
                else:
                    raise ValueError(f"Unknown sub-model class: {model_class}")
                models.append(model)

            final_model = BagOfModelsMLX(models, weights)
            final_model.load_state_dict(checkpoint['state'])

        else:
            if model_class_name == 'HTDemucsMLX':
                final_model = HTDemucsMLX(*args, **kwargs)
            elif model_class_name == 'HDemucsMLX':
                final_model = HDemucsMLX(*args, **kwargs)
            elif model_class_name == 'DemucsMLX':
                final_model = DemucsMLX(*args, **kwargs)
            else:
                raise ValueError(f"Unknown model class: {model_class_name}")

            final_model.load_state_dict(checkpoint['state'])

        if verbose:
            print(f"✓ Loaded {model_class_name}")

        if model_class_name == 'BagOfModelsMLX':
            for model in final_model.models:
                model.eval()
        else:
            final_model.eval()

        return final_model

    elif auto_convert:
        if verbose:
            print(f"No cached model found. Converting {model_name}...")
        
        # Fall back to pickle conversion if safetensors script missing
        convert_htdemucs_weights(
            model_name,
            output_dir=cache_dir,
            verify=False,
            verbose=verbose
        )

        return load_mlx_model(
            model_name, cache_dir, auto_convert=False, verbose=verbose
        )

    else:
        raise FileNotFoundError(
            f"No cached MLX model found at {cache_path}. "
            f"Run convert_htdemucs_weights('{model_name}') first."
        )


def load_mlx_model_from_safetensors(
    model_name: str,
    cache_dir: str = "./mlx_checkpoints",
    verbose: bool = False,
) -> tp.Any:
    """
    Load MLX model from safetensors format (faster, safer than pickle).

    This function loads models converted with convert_to_safetensors.py.
    Benefits:
    - 10-16x faster loading via lazy loading
    - 40%+ less memory usage
    - No PyTorch dependency for inference
    - Safer format (no arbitrary code execution)

    Args:
        model_name: Model name (e.g., 'htdemucs')
        cache_dir: Directory containing .safetensors files
        verbose: Print loading information

    Returns:
        MLX model instance with loaded weights

    Raises:
        FileNotFoundError: If safetensors or config files not found
        ValueError: If model configuration is invalid
    """
    import json
    import os

    from safetensors.mlx import load_file

    from .mlx_demucs import DemucsMLX
    from .mlx_hdemucs import HDemucsMLX
    from .mlx_htdemucs import HTDemucsMLX

    def _filter_kwargs(target_cls, in_kwargs):
        import inspect
        sig = inspect.signature(target_cls)
        allowed = set(sig.parameters.keys())
        filtered = {k: v for k, v in in_kwargs.items() if k in allowed}
        if verbose:
            dropped = sorted(k for k in in_kwargs.keys() if k not in allowed)
            if dropped:
                print(f"  Dropping unsupported kwargs for {target_cls.__name__}: {dropped}")
        return filtered

    def _normalize_demucs_kwargs(in_kwargs):
        out = dict(in_kwargs)
        if "gelu" in out and "gelu_act" not in out:
            out["gelu_act"] = out["gelu"]
        if "glu" in out and "glu_act" not in out:
            out["glu_act"] = out["glu"]
        return out

    def _normalize_model_class(name: tp.Optional[str]) -> str:
        if not name:
            return "HTDemucsMLX"
        if name in {"HTDemucsMLX", "HDemucsMLX", "DemucsMLX"}:
            return name
        if "HTDemucs" in name:
            return "HTDemucsMLX"
        if "HDemucs" in name:
            return "HDemucsMLX"
        if "Demucs" in name:
            return "DemucsMLX"
        return "HTDemucsMLX"

    safetensors_path = os.path.join(cache_dir, f"{model_name}.safetensors")
    config_path = os.path.join(cache_dir, f"{model_name}_config.json")

    # Check if files exist
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(
            f"Safetensors file not found: {safetensors_path}\n"
            f"Please run: python scripts/convert_to_safetensors.py {model_name}"
        )

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Please run: python scripts/convert_to_safetensors.py {model_name}"
        )

    if verbose:
        print(f"Loading MLX model from safetensors: {safetensors_path}")

    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Version check
    if _version_lt(config.get('mlx_version'), MIN_MLX_VERSION):
        print(
            f"Warning: Checkpoint created with MLX {config.get('mlx_version')}, "
            f"but using {MIN_MLX_VERSION}. "
            f"Consider reconverting."
        )
    elif config.get('mlx_version') != MIN_MLX_VERSION:
        print(
            f"Warning: Checkpoint created with MLX {config.get('mlx_version')}, "
            f"but using {MIN_MLX_VERSION}. "
            f"Consider reconverting."
        )

    # Load weights with lazy loading (MLX feature!)
    if verbose:
        print("Loading weights...")

    weights_dict = load_file(safetensors_path)

    if verbose:
        print(f"Loaded {len(weights_dict)} weight tensors")

    # Reconstruct model(s)
    model_class_name = config['model_class']
    args = tuple(config['args']) if config['args'] else ()
    kwargs = config['kwargs']
    per_model_args = config.get('per_model_args')
    per_model_kwargs = config.get('per_model_kwargs')
    per_model_class = config.get('per_model_class')
    num_models = config['num_models']
    bag_weights = config.get('weights')
    sub_model_class = _normalize_model_class(config.get('sub_model_class'))

    if model_class_name == 'BagOfModelsMLX':
        # Reconstruct bag of models
        if verbose:
            print(f"Reconstructing bag with {num_models} models...")

        # Create individual models
        models = []
        for i in range(num_models):
            model_args = args
            model_kwargs = kwargs
            if per_model_args is not None and i < len(per_model_args):
                model_args = tuple(per_model_args[i])
            if per_model_kwargs is not None and i < len(per_model_kwargs):
                model_kwargs = per_model_kwargs[i]
            model_class = sub_model_class
            if per_model_class is not None and i < len(per_model_class):
                model_class = _normalize_model_class(per_model_class[i])

            if model_class == 'HTDemucsMLX':
                model = HTDemucsMLX(*model_args, **_filter_kwargs(HTDemucsMLX, model_kwargs))
            elif model_class == 'HDemucsMLX':
                model = HDemucsMLX(*model_args, **_filter_kwargs(HDemucsMLX, model_kwargs))
            elif model_class == 'DemucsMLX':
                demucs_kwargs = _normalize_demucs_kwargs(model_kwargs)
                model = DemucsMLX(*model_args, **_filter_kwargs(DemucsMLX, demucs_kwargs))
            else:
                raise ValueError(f"Unknown sub-model class: {model_class}")

            # Extract this model's weights (flat keys)
            model_prefix = f"model_{i}."
            flat_model_state = {}

            for key, value in weights_dict.items():
                if key.startswith(model_prefix):
                    # Remove prefix to get original key
                    original_key = key[len(model_prefix):]
                    flat_model_state[original_key] = value

            # Load weights using the same copy_weights logic as pickle conversion
            # This handles MLX's nested conv structure correctly
            _load_weights_into_model(model, flat_model_state)

            models.append(model)

        # Create bag (BagOfModelsMLX is defined in this file)
        final_model = BagOfModelsMLX(models, bag_weights)

        if verbose:
            print(f"✓ Loaded BagOfModelsMLX with {num_models} models")

    else:
        # Single model
        if verbose:
            print(f"Reconstructing {model_class_name}...")

        if model_class_name == 'HTDemucsMLX':
            final_model = HTDemucsMLX(*args, **_filter_kwargs(HTDemucsMLX, kwargs))
        elif model_class_name == 'HDemucsMLX':
            final_model = HDemucsMLX(*args, **_filter_kwargs(HDemucsMLX, kwargs))
        elif model_class_name == 'DemucsMLX':
            demucs_kwargs = _normalize_demucs_kwargs(kwargs)
            final_model = DemucsMLX(*args, **_filter_kwargs(DemucsMLX, demucs_kwargs))
        else:
            raise ValueError(f"Unknown model class: {model_class_name}")

        _load_weights_into_model(final_model, weights_dict)

        if verbose:
            print(f"✓ Loaded {model_class_name}")

    # Set to evaluation mode
    if model_class_name == 'BagOfModelsMLX':
        for model in final_model.models:
            model.eval()
    else:
        final_model.eval()

    return final_model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert HTDemucs PyTorch models to MLX format'
    )
    parser.add_argument(
        'model_name',
        choices=['htdemucs', 'htdemucs_ft', 'htdemucs_6s'],
        help='Model to convert'
    )
    parser.add_argument(
        '--output-dir',
        default='./mlx_checkpoints',
        help='Output directory for MLX checkpoints'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Run verification tests after conversion'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    convert_htdemucs_weights(
        args.model_name,
        output_dir=args.output_dir,
        verify=args.verify,
        verbose=not args.quiet
    )
    
def _version_lt(a: tp.Optional[str], b: str) -> bool:
    if not a:
        return True
    try:
        return version.parse(a) < version.parse(b)
    except Exception:
        return a != b


def _model_root_dir() -> Path:
    here = Path(__file__).resolve()
    parts = list(here.parts)
    if "variants" in parts:
        root = Path(*parts[:parts.index("variants")])
        if root.exists():
            return root
    return here.parents[4]


def _load_weights_into_model(model, flat_weights: tp.Dict[str, mx.array]):
    """Load flat weights into MLX model state (handles MLX conv wrappers)."""
    model_state = model.state_dict()

    def copy_weights_from_flat(model_dict, flat_dict, prefix="", inside_sequential=False):
        if isinstance(model_dict, dict):
            for key, value in model_dict.items():
                is_sequential_conv = (inside_sequential and
                                     key in ['conv', 'conv_tr', 'rewrite'] and
                                     isinstance(value, dict))

                if is_sequential_conv:
                    path_for_content = prefix
                else:
                    path_for_content = f"{prefix}.{key}" if prefix else key

                if isinstance(value, dict):
                    has_conv_wrapper = ('conv' in value and
                                      isinstance(value['conv'], dict) and
                                      ('weight' in value['conv'] or 'bias' in value['conv']))

                    if has_conv_wrapper:
                        copy_weights_from_flat(
                            value['conv'], flat_dict, path_for_content,
                            inside_sequential=inside_sequential)
                    else:
                        copy_weights_from_flat(
                            value, flat_dict, path_for_content,
                            inside_sequential=inside_sequential)
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        idx_path = f"{path_for_content}.{i}"
                        if isinstance(item, dict) and list(item.keys()) == ['layers']:
                            copy_weights_from_flat(
                                item['layers'], flat_dict, idx_path,
                                inside_sequential=True)
                        else:
                            copy_weights_from_flat(
                                item, flat_dict, idx_path,
                                inside_sequential=inside_sequential)
                else:
                    if path_for_content in flat_dict:
                        model_dict[key] = flat_dict[path_for_content]

        elif isinstance(model_dict, list):
            for i, item in enumerate(model_dict):
                idx_path = f"{prefix}.{i}"
                copy_weights_from_flat(
                    item, flat_dict, idx_path,
                    inside_sequential=inside_sequential)

    copy_weights_from_flat(model_state, flat_weights)
    model.update(model_state)
