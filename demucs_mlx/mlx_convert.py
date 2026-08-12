"""
MLX Model Weight Conversion System

Converts pretrained PyTorch HTDemucs models to MLX format with proper
weight layout transformations for Conv1d/Conv2d layers.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import shlex
import tempfile
import typing as tp
import warnings
from datetime import datetime, timezone
from fractions import Fraction
from importlib.metadata import version as distribution_version
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from packaging import version

from .mlx_registry import MLX_MODEL_REGISTRY

SAFE_CACHE_FORMAT_VERSION = 1
SAFE_CACHE_CONFIG_LIMIT = 1024 * 1024
SAFE_CACHE_MAX_DEPTH = 32
SAFE_CACHE_MAX_NODES = 100_000
SAFE_CACHE_MAX_COLLECTION = 10_000
SAFE_CACHE_MAX_STRING = 4_096
SAFE_CACHE_MAX_INTEGER_BITS = 63
_MLX_MODEL_CLASSES = {"HTDemucsMLX", "HDemucsMLX", "DemucsMLX"}
_SAFE_CACHE_FIELDS = {
    "format_version",
    "model_name",
    "model_class",
    "sub_model_class",
    "args",
    "kwargs",
    "per_model_args",
    "per_model_kwargs",
    "per_model_classes",
    "num_models",
    "weights",
    "source_artifacts",
    "mlx_version",
    "conversion_date",
    "verification_passed",
    "safetensors_sha256",
}


class SafeCacheError(ValueError):
    """Raised when a Demucs safetensors/config cache is incomplete or invalid."""


class _JsonBudget:
    def __init__(self) -> None:
        self.nodes = 0

    def consume(self, path: str, depth: int) -> None:
        if depth > SAFE_CACHE_MAX_DEPTH:
            raise SafeCacheError(f"JSON value at {path} is nested too deeply")
        self.nodes += 1
        if self.nodes > SAFE_CACHE_MAX_NODES:
            raise SafeCacheError("Demucs cache config contains too many values")


def _reject_json_constant(value: str) -> tp.NoReturn:
    raise SafeCacheError(f"Invalid JSON constant: {value}")


def _strict_json_object(pairs: list[tuple[str, tp.Any]]) -> dict[str, tp.Any]:
    value: dict[str, tp.Any] = {}
    for key, item in pairs:
        if key in value:
            raise SafeCacheError(f"Duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _validate_json_string(value: str, path: str) -> None:
    if len(value) > SAFE_CACHE_MAX_STRING:
        raise SafeCacheError(f"String at {path} is too long")


def _encode_json_value(
    value: tp.Any,
    path: str = "config",
    *,
    budget: tp.Optional[_JsonBudget] = None,
    depth: int = 0,
) -> tp.Any:
    if budget is None:
        budget = _JsonBudget()
    budget.consume(path, depth)

    if value is None or isinstance(value, (str, bool)):
        if isinstance(value, str):
            _validate_json_string(value, path)
        return value
    if isinstance(value, int):
        if value.bit_length() > SAFE_CACHE_MAX_INTEGER_BITS:
            raise SafeCacheError(f"Integer at {path} is out of range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SafeCacheError(f"Non-finite value at {path}")
        return value
    if isinstance(value, Fraction):
        return {
            "__type__": _encode_json_value(
                "fraction", f"{path}.__type__", budget=budget, depth=depth + 1
            ),
            "numerator": _encode_json_value(
                value.numerator,
                f"{path}.numerator",
                budget=budget,
                depth=depth + 1,
            ),
            "denominator": _encode_json_value(
                value.denominator,
                f"{path}.denominator",
                budget=budget,
                depth=depth + 1,
            ),
        }
    if isinstance(value, np.generic):
        return _encode_json_value(value.item(), path, budget=budget, depth=depth)
    if isinstance(value, (list, tuple)):
        if len(value) > SAFE_CACHE_MAX_COLLECTION:
            raise SafeCacheError(f"Sequence at {path} is too large")
        return [
            _encode_json_value(item, f"{path}[{index}]", budget=budget, depth=depth + 1)
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        if len(value) > SAFE_CACHE_MAX_COLLECTION:
            raise SafeCacheError(f"Mapping at {path} is too large")
        encoded: dict[str, tp.Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SafeCacheError(f"Mapping at {path} must use string keys")
            _validate_json_string(key, f"{path} key")
            encoded[key] = _encode_json_value(
                item,
                f"{path}.{key}",
                budget=budget,
                depth=depth + 1,
            )
        return encoded
    raise SafeCacheError(f"Unsupported value at {path}: {type(value).__name__}")


def _decode_json_value(
    value: tp.Any,
    path: str = "config",
    *,
    budget: tp.Optional[_JsonBudget] = None,
    depth: int = 0,
) -> tp.Any:
    if budget is None:
        budget = _JsonBudget()
    budget.consume(path, depth)

    if value is None or isinstance(value, (str, bool)):
        if isinstance(value, str):
            _validate_json_string(value, path)
        return value
    if isinstance(value, int):
        if value.bit_length() > SAFE_CACHE_MAX_INTEGER_BITS:
            raise SafeCacheError(f"Integer at {path} is out of range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SafeCacheError(f"Non-finite value at {path}")
        return value
    if isinstance(value, list):
        if len(value) > SAFE_CACHE_MAX_COLLECTION:
            raise SafeCacheError(f"Sequence at {path} is too large")
        return [
            _decode_json_value(item, f"{path}[{index}]", budget=budget, depth=depth + 1)
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        if len(value) > SAFE_CACHE_MAX_COLLECTION:
            raise SafeCacheError(f"Mapping at {path} is too large")
        if "__type__" in value:
            if set(value) != {"__type__", "numerator", "denominator"}:
                raise SafeCacheError(f"Invalid tagged value at {path}")
            tag = _decode_json_value(
                value["__type__"],
                f"{path}.__type__",
                budget=budget,
                depth=depth + 1,
            )
            if tag != "fraction":
                raise SafeCacheError(f"Unknown tagged value at {path}: {tag!r}")
            numerator = _decode_json_value(
                value["numerator"],
                f"{path}.numerator",
                budget=budget,
                depth=depth + 1,
            )
            denominator = _decode_json_value(
                value["denominator"],
                f"{path}.denominator",
                budget=budget,
                depth=depth + 1,
            )
            if (
                isinstance(numerator, bool)
                or not isinstance(numerator, int)
                or isinstance(denominator, bool)
                or not isinstance(denominator, int)
                or denominator == 0
            ):
                raise SafeCacheError(f"Invalid fraction at {path}")
            return Fraction(numerator, denominator)
        decoded: dict[str, tp.Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SafeCacheError(f"Mapping at {path} must use string keys")
            _validate_json_string(key, f"{path} key")
            decoded[key] = _decode_json_value(
                item,
                f"{path}.{key}",
                budget=budget,
                depth=depth + 1,
            )
        return decoded
    raise SafeCacheError(f"Unsupported JSON value at {path}: {type(value).__name__}")


def _runtime_mlx_version() -> str:
    try:
        installed = distribution_version("mlx")
        version.Version(installed)
    except Exception as exc:
        raise SafeCacheError("Could not determine the installed MLX version") from exc
    return installed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regeneration_command(model_name: str, cache_dir: tp.Union[str, Path]) -> str:
    return (
        "python -m demucs_mlx.mlx_convert "
        f"{shlex.quote(model_name)} --output-dir {shlex.quote(str(cache_dir))}"
    )


def _mlx_model_classes() -> dict[str, type]:
    from .mlx_demucs import DemucsMLX
    from .mlx_hdemucs import HDemucsMLX
    from .mlx_htdemucs import HTDemucsMLX

    return {
        "HTDemucsMLX": HTDemucsMLX,
        "HDemucsMLX": HDemucsMLX,
        "DemucsMLX": DemucsMLX,
    }


def _validate_constructor_data(
    class_name: str,
    args: tp.Any,
    kwargs: tp.Any,
    path: str,
) -> inspect.BoundArguments:
    if class_name not in _MLX_MODEL_CLASSES:
        raise SafeCacheError(f"Unknown MLX model class: {class_name!r}")
    if not isinstance(args, list) or not isinstance(kwargs, dict):
        raise SafeCacheError(f"Constructor data at {path} has invalid types")
    target_class = _mlx_model_classes()[class_name]
    try:
        return inspect.signature(target_class).bind(*args, **kwargs)
    except TypeError as exc:
        raise SafeCacheError(f"Invalid constructor data at {path}") from exc


def _validate_safe_cache_config(raw: tp.Any, model_name: str) -> dict[str, tp.Any]:
    from .secure_demucs import expected_official_sources

    if not isinstance(raw, dict):
        raise SafeCacheError("Demucs cache config must be a JSON object")
    fields = set(raw)
    missing = _SAFE_CACHE_FIELDS.difference(fields)
    unknown = fields.difference(_SAFE_CACHE_FIELDS)
    if missing:
        raise SafeCacheError(f"Demucs cache config is missing fields: {sorted(missing)}")
    if unknown:
        raise SafeCacheError(f"Demucs cache config has unknown fields: {sorted(unknown)}")
    if (
        isinstance(raw["format_version"], bool)
        or not isinstance(raw["format_version"], int)
        or raw["format_version"] != SAFE_CACHE_FORMAT_VERSION
    ):
        raise SafeCacheError(f"Unsupported Demucs cache format version: {raw['format_version']!r}")

    decoded = _decode_json_value(raw)
    if decoded["model_name"] != model_name:
        raise SafeCacheError(f"Demucs cache is for {decoded['model_name']!r}, not {model_name!r}")
    if model_name not in MLX_MODEL_REGISTRY:
        raise SafeCacheError(f"Unknown Demucs model: {model_name!r}")

    model_class = decoded["model_class"]
    sub_model_class = decoded["sub_model_class"]
    if model_class == "BagOfModelsMLX":
        if sub_model_class not in _MLX_MODEL_CLASSES:
            raise SafeCacheError(f"Invalid bag sub-model class: {sub_model_class!r}")
    elif model_class in _MLX_MODEL_CLASSES:
        if sub_model_class is not None:
            raise SafeCacheError("Single-model cache must use a null sub-model class")
    else:
        raise SafeCacheError(f"Unknown MLX model class: {model_class!r}")

    num_models = decoded["num_models"]
    if isinstance(num_models, bool) or not isinstance(num_models, int) or not 1 <= num_models <= 32:
        raise SafeCacheError(f"Invalid model count: {num_models!r}")
    if model_class != "BagOfModelsMLX" and num_models != 1:
        raise SafeCacheError("Single-model cache must contain exactly one model")

    per_model_args = decoded["per_model_args"]
    per_model_kwargs = decoded["per_model_kwargs"]
    per_model_classes = decoded["per_model_classes"]
    if not all(
        isinstance(value, list) and len(value) == num_models
        for value in (
            per_model_args,
            per_model_kwargs,
            per_model_classes,
        )
    ):
        raise SafeCacheError("Per-model metadata must contain one entry per model")

    registry_classes = MLX_MODEL_REGISTRY[model_name]["model_classes"]
    if per_model_classes != registry_classes:
        raise SafeCacheError("Demucs cache model classes do not match the official registry")
    if model_class == "BagOfModelsMLX" and sub_model_class != per_model_classes[0]:
        raise SafeCacheError("Bag sub-model class does not match its first model")
    if model_class != "BagOfModelsMLX" and model_class != per_model_classes[0]:
        raise SafeCacheError("Single-model class does not match its model metadata")
    if decoded["args"] != per_model_args[0] or decoded["kwargs"] != per_model_kwargs[0]:
        raise SafeCacheError("Top-level constructor data must match the first model")

    bound_models = []
    for index, class_name in enumerate(per_model_classes):
        if not isinstance(class_name, str):
            raise SafeCacheError("Per-model classes must be strings")
        bound_models.append(
            _validate_constructor_data(
                class_name,
                per_model_args[index],
                per_model_kwargs[index],
                f"model[{index}]",
            )
        )

    expected_sources = expected_official_sources(model_name)
    expected_records = [
        {"signature": source.signature, "checksum": source.checksum} for source in expected_sources
    ]
    if decoded["source_artifacts"] != expected_records:
        raise SafeCacheError("Demucs cache source signatures/checksums do not match the registry")
    if num_models != len(expected_records):
        raise SafeCacheError("Demucs cache model count does not match its official sources")

    bag_weights = decoded["weights"]
    if model_class == "BagOfModelsMLX":
        if not isinstance(bag_weights, list) or len(bag_weights) != num_models:
            raise SafeCacheError("Demucs bag weights must contain one row per model")
        source_names: tp.Optional[list[str]] = None
        totals: tp.Optional[list[float]] = None
        for index, (row, bound) in enumerate(zip(bag_weights, bound_models)):
            sources = bound.arguments.get("sources")
            if (
                not isinstance(sources, list)
                or not sources
                or not all(isinstance(source, str) for source in sources)
            ):
                raise SafeCacheError(f"Model {index} has invalid source metadata")
            if source_names is None:
                source_names = sources
                totals = [0.0] * len(sources)
            elif sources != source_names:
                raise SafeCacheError("All bag models must use the same sources")
            if not isinstance(row, list) or len(row) != len(sources):
                raise SafeCacheError("Demucs bag weight rows must match the model sources")
            for source_index, weight in enumerate(row):
                if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                    raise SafeCacheError("Demucs bag weights must be numeric")
                if not math.isfinite(float(weight)):
                    raise SafeCacheError("Demucs bag weights must be finite")
                if float(weight) < 0:
                    raise SafeCacheError("Demucs bag weights must be non-negative")
                tp.cast(list[float], totals)[source_index] += float(weight)
        if any(total == 0 for total in tp.cast(list[float], totals)):
            raise SafeCacheError("Demucs bag weights must have non-zero totals")
    elif bag_weights is not None:
        raise SafeCacheError("Single-model cache must not define ensemble weights")

    mlx_version = decoded["mlx_version"]
    if not isinstance(mlx_version, str):
        raise SafeCacheError("Demucs cache MLX version must be a string")
    try:
        version.Version(mlx_version)
    except version.InvalidVersion as exc:
        raise SafeCacheError("Demucs cache MLX version is invalid") from exc
    conversion_date = decoded["conversion_date"]
    if not isinstance(conversion_date, str):
        raise SafeCacheError("Demucs cache conversion date must be a string")
    try:
        parsed_date = datetime.fromisoformat(conversion_date)
    except ValueError as exc:
        raise SafeCacheError("Demucs cache conversion date is invalid") from exc
    if parsed_date.tzinfo is None:
        raise SafeCacheError("Demucs cache conversion date must include a timezone")
    if not isinstance(decoded["verification_passed"], bool):
        raise SafeCacheError("Demucs cache verification flag must be boolean")

    expected_hash = decoded["safetensors_sha256"]
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or any(char not in "0123456789abcdef" for char in expected_hash)
    ):
        raise SafeCacheError("Demucs cache safetensors SHA-256 is invalid")
    return decoded


def _load_safe_cache_config(config_path: Path, model_name: str) -> dict[str, tp.Any]:
    try:
        if config_path.stat().st_size > SAFE_CACHE_CONFIG_LIMIT:
            raise SafeCacheError(f"Demucs cache config is too large: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(
                handle,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_strict_json_object,
            )
    except SafeCacheError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SafeCacheError(f"Could not read Demucs cache config: {config_path}") from exc
    return _validate_safe_cache_config(raw, model_name)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _save_safe_cache(
    model_name: str,
    output_dir: str,
    weights: dict[str, mx.array],
    config: dict[str, tp.Any],
) -> str:
    if not weights or not all(
        isinstance(key, str) and isinstance(value, mx.array) for key, value in weights.items()
    ):
        raise SafeCacheError("Demucs safetensors state must contain named MLX arrays")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    safetensors_path = output / f"{model_name}.safetensors"
    config_path = output / f"{model_name}_config.json"
    weight_fd, temporary_weights = tempfile.mkstemp(
        prefix=f".{model_name}.", suffix=".safetensors", dir=output
    )
    config_fd, temporary_config = tempfile.mkstemp(
        prefix=f".{model_name}.", suffix=".json", dir=output
    )
    os.close(weight_fd)
    os.close(config_fd)
    temporary_weights_path = Path(temporary_weights)
    temporary_config_path = Path(temporary_config)

    try:
        mx.save_safetensors(str(temporary_weights_path), weights)
        with temporary_weights_path.open("rb") as handle:
            os.fsync(handle.fileno())

        final_config = dict(config)
        final_config["mlx_version"] = _runtime_mlx_version()
        final_config["safetensors_sha256"] = _sha256_file(temporary_weights_path)
        encoded_config = _encode_json_value(final_config)
        _validate_safe_cache_config(encoded_config, model_name)
        serialized_config = (
            json.dumps(
                encoded_config,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        if len(serialized_config.encode("utf-8")) > SAFE_CACHE_CONFIG_LIMIT:
            raise SafeCacheError("Demucs cache config is too large")
        with temporary_config_path.open("w", encoding="utf-8") as handle:
            handle.write(serialized_config)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_weights_path, safetensors_path)
        _fsync_directory(output)
        os.replace(temporary_config_path, config_path)
        _fsync_directory(output)
    finally:
        temporary_weights_path.unlink(missing_ok=True)
        temporary_config_path.unlink(missing_ok=True)
    return str(safetensors_path)


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
        return {f"model_{i}": model.state_dict() for i, model in enumerate(self.models)}

    def load_state_dict(self, state: tp.Dict):
        """Load state dict for all models."""
        for i, model in enumerate(self.models):
            model.load_state_dict(state[f"model_{i}"])


def convert_conv_weight(weight: np.ndarray, conv_type: str, transpose: bool = True) -> np.ndarray:
    """Convert convolution weight from PyTorch to MLX layout."""
    if not transpose:
        return weight

    if conv_type == "conv1d":
        return np.transpose(weight, (0, 2, 1))
    elif conv_type == "conv_transpose1d":
        return np.transpose(weight, (1, 2, 0))
    elif conv_type == "conv2d":
        return np.transpose(weight, (0, 2, 3, 1))
    elif conv_type == "conv_transpose2d":
        return np.transpose(weight, (1, 2, 3, 0))
    else:
        raise ValueError(f"Unknown conv_type: {conv_type}")


def convert_state_dict(
    torch_state: tp.Dict[str, tp.Any],  # Type Hint generic to avoid Torch import
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

        is_conv_like_weight = "weight" in name and (
            "conv" in name.lower()
            or "rewrite" in name.lower()
            or "upsampler" in name.lower()
            or "downsampler" in name.lower()
        )

        if name in module_param_types:
            conv_type = module_param_types[name]
            needs_transpose = True
        elif is_conv_like_weight:
            # Determine convolution type by parameter shape and name
            ndim = len(np_param.shape)
            is_transpose = "conv_tr" in name.lower() or "transpose" in name.lower()

            if ndim == 3:  # Conv1d or ConvTranspose1d
                conv_type = "conv_transpose1d" if is_transpose else "conv1d"
                needs_transpose = True

            elif ndim == 4:  # Conv2d or ConvTranspose2d
                conv_type = "conv_transpose2d" if is_transpose else "conv2d"
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
            rest = rest[: -len("_reverse")]
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
        if "self_attn" in name or "cross_attn" in name:
            new_name = name.replace("self_attn", "attn")

            if ".in_proj_weight" in name:
                weight = np.array(flat_mlx_state[name])
                embed_dim = weight.shape[0] // 3
                query_weight = weight[:embed_dim, :]
                key_weight = weight[embed_dim : 2 * embed_dim, :]
                value_weight = weight[2 * embed_dim :, :]

                base = new_name.replace(".in_proj_weight", "")
                transformer_fixes[f"{base}.query_proj.weight"] = mx.array(query_weight)
                transformer_fixes[f"{base}.key_proj.weight"] = mx.array(key_weight)
                transformer_fixes[f"{base}.value_proj.weight"] = mx.array(value_weight)

            elif ".in_proj_bias" in name:
                bias = np.array(flat_mlx_state[name])
                embed_dim = bias.shape[0] // 3
                query_bias = bias[:embed_dim]
                key_bias = bias[embed_dim : 2 * embed_dim]
                value_bias = bias[2 * embed_dim :]

                base = new_name.replace(".in_proj_bias", "")
                transformer_fixes[f"{base}.query_proj.bias"] = mx.array(query_bias)
                transformer_fixes[f"{base}.key_proj.bias"] = mx.array(key_bias)
                transformer_fixes[f"{base}.value_proj.bias"] = mx.array(value_bias)

            elif ".out_proj." in name:
                transformer_fixes[new_name] = flat_mlx_state[name]

        elif ".norm_out.weight" in name or ".norm_out.bias" in name:
            new_name = name.replace(".norm_out.", ".norm_out.gn.")
            transformer_fixes[new_name] = flat_mlx_state[name]

    flat_mlx_state.update(transformer_fixes)

    if flatten:
        return flat_mlx_state
    raise NotImplementedError("Nested conversion is not supported; use flatten=True.")


def _normalize_demucs_kwargs(kwargs: dict[str, tp.Any]) -> dict[str, tp.Any]:
    normalized = dict(kwargs)
    if "gelu" in normalized and "gelu_act" not in normalized:
        normalized["gelu_act"] = normalized.pop("gelu")
    if "glu" in normalized and "glu_act" not in normalized:
        normalized["glu_act"] = normalized.pop("glu")
    return normalized


def _adapt_mlx_constructor(
    torch_model: tp.Any,
    target_class: type,
    *,
    verbose: bool = False,
) -> tuple[list[tp.Any], dict[str, tp.Any]]:
    if not hasattr(torch_model, "_init_args_kwargs"):
        raise ValueError(f"Model {type(torch_model).__name__} has no constructor metadata")
    raw_args, raw_kwargs = torch_model._init_args_kwargs
    args = list(raw_args)
    kwargs = dict(raw_kwargs)
    if hasattr(torch_model, "segment"):
        kwargs["segment"] = torch_model.segment
    if type(torch_model).__name__ == "Demucs":
        kwargs = _normalize_demucs_kwargs(kwargs)

    signature = inspect.signature(target_class)
    allowed = set(signature.parameters)
    filtered = {key: value for key, value in kwargs.items() if key in allowed}
    dropped = sorted(set(kwargs).difference(filtered))
    if verbose and dropped:
        print(f"  Dropping unsupported kwargs for {target_class.__name__}: {dropped}")
    try:
        signature.bind(*args, **filtered)
    except TypeError as exc:
        raise ValueError(f"Invalid constructor metadata for {target_class.__name__}") from exc
    return args, filtered


def _mlx_target_class(torch_model: tp.Any) -> type:
    try:
        mlx_class_name = {
            "HTDemucs": "HTDemucsMLX",
            "HDemucs": "HDemucsMLX",
            "Demucs": "DemucsMLX",
        }[type(torch_model).__name__]
    except KeyError:
        raise NotImplementedError(
            f"MLX conversion not implemented for {type(torch_model).__name__}"
        ) from None
    return _mlx_model_classes()[mlx_class_name]


def convert_single_model(torch_model: tp.Any, verbose: bool = False) -> tp.Any:
    """
    Convert a single PyTorch model to MLX.
    """
    model_class = type(torch_model).__name__
    target_class = _mlx_target_class(torch_model)

    if verbose:
        print(f"Converting {model_class}...")

    args, kwargs = _adapt_mlx_constructor(torch_model, target_class, verbose=verbose)

    # Create MLX model
    if model_class == "HTDemucs":
        if kwargs.get("t_sparse_self_attn"):
            raise ValueError("Sparse self-attention not supported in MLX backend")
        if kwargs.get("t_sparse_cross_attn"):
            raise ValueError("Sparse cross-attention not supported in MLX backend")
    mlx_model = target_class(*args, **kwargs)

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
    # Lazy import — conversion extras are needed to identify upstream bags.
    try:
        from demucs.apply import BagOfModels
    except ImportError:
        raise ImportError(
            "Model conversion requires the [convert] extras. "
            "Install with: pip install 'demucs-mlx[convert]'"
        ) from None

    if model_name not in MLX_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(MLX_MODEL_REGISTRY.keys())}"
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

    from .secure_demucs import get_restricted_demucs_model

    restricted = get_restricted_demucs_model(model_name)
    torch_model = restricted.model

    torch_models: list[tp.Any]
    if isinstance(torch_model, BagOfModels):
        if verbose:
            print(f"   Found bag with {len(torch_model.models)} models")
        torch_models = list(tp.cast(tp.Any, torch_model.models))
        if hasattr(torch_model, "weights") and torch_model.weights is not None:
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
            print(f"\n   Model {i + 1}/{len(torch_models)}:")
        mlx_model = convert_single_model(tm, verbose=verbose)
        mlx_models.append(mlx_model)

    if isinstance(torch_model, BagOfModels):
        if verbose:
            print(
                f"\n3. Creating ensemble with {len(mlx_models)} model(s) and weights {weights}..."
            )
        final_model = BagOfModelsMLX(mlx_models, weights)
        model_class = "BagOfModelsMLX"
        sub_model_class = type(mlx_models[0]).__name__
    else:
        final_model = mlx_models[0]
        model_class = type(final_model).__name__
        sub_model_class = None

    per_model_args: list[list[tp.Any]] = []
    per_model_kwargs: list[dict[str, tp.Any]] = []
    per_model_classes: list[str] = []
    for torch_submodel, mlx_submodel in zip(torch_models, mlx_models):
        mlx_class = type(mlx_submodel)
        model_args, model_kwargs = _adapt_mlx_constructor(torch_submodel, mlx_class)
        per_model_args.append(model_args)
        per_model_kwargs.append(model_kwargs)
        per_model_classes.append(mlx_class.__name__)

    checkpoint_config = {
        "format_version": SAFE_CACHE_FORMAT_VERSION,
        "model_name": model_name,
        "model_class": model_class,
        "sub_model_class": sub_model_class,
        "args": per_model_args[0],
        "kwargs": per_model_kwargs[0],
        "per_model_args": per_model_args,
        "per_model_kwargs": per_model_kwargs,
        "per_model_classes": per_model_classes,
        "num_models": len(mlx_models),
        "weights": weights if isinstance(torch_model, BagOfModels) else None,
        "source_artifacts": [
            {"signature": source.signature, "checksum": source.checksum}
            for source in restricted.sources
        ],
        "mlx_version": _runtime_mlx_version(),
        "conversion_date": datetime.now(timezone.utc).isoformat(),
        "verification_passed": False,
        "safetensors_sha256": "",
    }

    if verify:
        if verbose:
            print("\n5. Running verification tests...")
        try:
            verify_conversion(torch_models[0], mlx_models[0], verbose=verbose)
            checkpoint_config["verification_passed"] = True
            if verbose:
                print("   ✓ Verification passed")
        except Exception as e:
            if verbose:
                print(f"   ✗ Verification failed: {e}")
            checkpoint_config["verification_passed"] = False

    if verbose:
        print(f"\n{5 if verify else 4}. Saving safe MLX checkpoint...")

    flat_state = dict(tree_flatten(final_model.state_dict()))
    output_path = _save_safe_cache(model_name, output_dir, flat_state, checkpoint_config)

    if verbose:
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   Saved to: {output_path}")
        print(f"   File size: {file_size_mb:.1f} MB")

    if verbose:
        print("\n" + "=" * 70)
        print(f"✓ Conversion complete: {output_path}")
        print("=" * 70)

    return output_path


def verify_conversion(
    torch_model, mlx_model, tolerance: float = 1e-4, verbose: bool = True
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
        raise ValueError(f"Verification failed: relative error {rel_error:.2e} > {tolerance:.2e}")

    return True


def load_mlx_model(
    model_name: str,
    cache_dir: str = "./mlx_checkpoints",
    auto_convert: bool = True,
    verbose: bool = False,
) -> tp.Any:
    """Load a v1 safe cache, or regenerate it from the official registry."""
    cache = Path(cache_dir)
    safetensors_path = cache / f"{model_name}.safetensors"
    config_path = cache / f"{model_name}_config.json"
    legacy_path = cache / f"{model_name}_mlx.pkl"
    regeneration = _regeneration_command(model_name, cache)

    safe_files_present = (safetensors_path.exists(), config_path.exists())
    if any(safe_files_present):
        if not all(safe_files_present):
            missing = config_path if safetensors_path.exists() else safetensors_path
            raise SafeCacheError(
                f"Incomplete Demucs safe cache: missing {missing}. "
                f"Regenerate it with: {regeneration}"
            )
        return load_mlx_model_from_safetensors(
            model_name,
            cache_dir=str(cache),
            verbose=verbose,
        )

    if legacy_path.exists():
        warnings.warn(
            f"Ignoring unsafe legacy Demucs pickle cache {legacy_path}. "
            "The file will not be opened, rewritten, or deleted.",
            FutureWarning,
            stacklevel=2,
        )

    if auto_convert:
        if verbose:
            print(f"Regenerating a safe cache for {model_name} from the official registry...")
        convert_htdemucs_weights(
            model_name,
            output_dir=str(cache),
            verify=False,
            verbose=verbose,
        )
        return load_mlx_model(
            model_name,
            cache_dir=str(cache),
            auto_convert=False,
            verbose=verbose,
        )

    legacy_note = f" Unsafe legacy cache ignored: {legacy_path}." if legacy_path.exists() else ""
    raise FileNotFoundError(
        f"No complete v1 Demucs safe cache exists for {model_name!r}.{legacy_note} "
        f"Regenerate it with: {regeneration}"
    )


def _load_exact_model_state(
    model: tp.Any,
    flat_weights: dict[str, mx.array],
    *,
    context: str,
) -> None:
    expected = dict(tree_flatten(model.state_dict()))
    expected_keys = set(expected)
    actual_keys = set(flat_weights)
    if expected_keys != actual_keys:
        missing = sorted(expected_keys - actual_keys)[:10]
        unexpected = sorted(actual_keys - expected_keys)[:10]
        raise SafeCacheError(
            f"{context} state keys do not match the constructed model "
            f"(missing={missing}, unexpected={unexpected})"
        )
    for key, value in flat_weights.items():
        if tuple(value.shape) != tuple(expected[key].shape):
            raise SafeCacheError(
                f"{context} tensor {key!r} has shape {tuple(value.shape)}, "
                f"expected {tuple(expected[key].shape)}"
            )
    model.update(tree_unflatten(list(flat_weights.items())))


def load_mlx_model_from_safetensors(
    model_name: str,
    cache_dir: str = "./mlx_checkpoints",
    verbose: bool = False,
) -> tp.Any:
    """Load a complete v1 cache after validating metadata and its digest."""
    cache = Path(cache_dir)
    safetensors_path = cache / f"{model_name}.safetensors"
    config_path = cache / f"{model_name}_config.json"
    regeneration = _regeneration_command(model_name, cache)
    if not safetensors_path.exists() or not config_path.exists():
        raise SafeCacheError(
            f"Incomplete Demucs safe cache for {model_name!r}. Regenerate it with: {regeneration}"
        )

    config = _load_safe_cache_config(config_path, model_name)
    actual_digest = _sha256_file(safetensors_path)
    if actual_digest != config["safetensors_sha256"]:
        raise SafeCacheError(
            f"Demucs safetensors digest mismatch for {safetensors_path}; "
            f"regenerate it with: {regeneration}"
        )

    if verbose:
        print(f"Loading verified MLX arrays from {safetensors_path}")
    try:
        loaded = mx.load(str(safetensors_path))
    except Exception as exc:
        raise SafeCacheError(f"Could not load Demucs safetensors: {safetensors_path}") from exc
    if (
        not isinstance(loaded, dict)
        or not loaded
        or not all(
            isinstance(key, str) and isinstance(value, mx.array) for key, value in loaded.items()
        )
    ):
        raise SafeCacheError("Demucs safetensors must contain named MLX arrays")
    weights_dict = tp.cast(dict[str, mx.array], loaded)

    classes = _mlx_model_classes()
    models = []
    consumed: set[str] = set()
    for index in range(config["num_models"]):
        class_name = config["per_model_classes"][index]
        args = config["per_model_args"][index]
        kwargs = config["per_model_kwargs"][index]
        _validate_constructor_data(class_name, args, kwargs, f"model[{index}]")
        try:
            model = classes[class_name](*args, **kwargs)
        except Exception as exc:
            raise SafeCacheError(f"Could not construct validated Demucs model {index}") from exc

        if config["model_class"] == "BagOfModelsMLX":
            prefix = f"model_{index}."
            model_weights = {
                key[len(prefix) :]: value
                for key, value in weights_dict.items()
                if key.startswith(prefix)
            }
            consumed.update(prefix + key for key in model_weights)
        else:
            model_weights = weights_dict
            consumed.update(weights_dict)
        _load_exact_model_state(model, model_weights, context=f"model {index}")
        model.eval()
        models.append(model)

    if consumed != set(weights_dict):
        unexpected = sorted(set(weights_dict) - consumed)[:10]
        raise SafeCacheError(f"Demucs safetensors contains unexpected arrays: {unexpected}")

    if config["model_class"] == "BagOfModelsMLX":
        final_model = BagOfModelsMLX(models, config["weights"])
    else:
        final_model = models[0]
    if verbose:
        print(f"✓ Loaded verified {config['model_class']}")
    return final_model


def _model_root_dir() -> Path:
    here = Path(__file__).resolve()
    parts = list(here.parts)
    if "variants" in parts:
        root = Path(*parts[: parts.index("variants")])
        if root.exists():
            return root
    return here.parents[4]


def _load_weights_into_model(model, flat_weights: tp.Dict[str, mx.array]):
    """Load flat weights into MLX model state (handles MLX conv wrappers)."""
    model_state = model.state_dict()

    def copy_weights_from_flat(model_dict, flat_dict, prefix="", inside_sequential=False):
        if isinstance(model_dict, dict):
            for key, value in model_dict.items():
                is_sequential_conv = (
                    inside_sequential
                    and key in ["conv", "conv_tr", "rewrite"]
                    and isinstance(value, dict)
                )

                if is_sequential_conv:
                    path_for_content = prefix
                else:
                    path_for_content = f"{prefix}.{key}" if prefix else key

                if isinstance(value, dict):
                    has_conv_wrapper = (
                        "conv" in value
                        and isinstance(value["conv"], dict)
                        and ("weight" in value["conv"] or "bias" in value["conv"])
                    )

                    if has_conv_wrapper:
                        copy_weights_from_flat(
                            value["conv"],
                            flat_dict,
                            path_for_content,
                            inside_sequential=inside_sequential,
                        )
                    else:
                        copy_weights_from_flat(
                            value, flat_dict, path_for_content, inside_sequential=inside_sequential
                        )
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        idx_path = f"{path_for_content}.{i}"
                        if isinstance(item, dict) and list(item.keys()) == ["layers"]:
                            copy_weights_from_flat(
                                item["layers"], flat_dict, idx_path, inside_sequential=True
                            )
                        else:
                            copy_weights_from_flat(
                                item, flat_dict, idx_path, inside_sequential=inside_sequential
                            )
                else:
                    if path_for_content in flat_dict:
                        model_dict[key] = flat_dict[path_for_content]

        elif isinstance(model_dict, list):
            for i, item in enumerate(model_dict):
                idx_path = f"{prefix}.{i}"
                copy_weights_from_flat(
                    item, flat_dict, idx_path, inside_sequential=inside_sequential
                )

    copy_weights_from_flat(model_state, flat_weights)
    model.update(model_state)


def main(argv: tp.Optional[list[str]] = None) -> None:
    """Command-line entry point for regenerating safe MLX caches."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert an official Demucs model to the v1 MLX safe cache format"
    )
    parser.add_argument(
        "model_name",
        choices=sorted(MLX_MODEL_REGISTRY),
        help="Official registered model to convert",
    )
    parser.add_argument(
        "--output-dir",
        default="./mlx_checkpoints",
        help="Output directory for MLX checkpoints",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run a numerical verification after conversion",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args(argv)
    convert_htdemucs_weights(
        args.model_name,
        output_dir=args.output_dir,
        verify=args.verify,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
