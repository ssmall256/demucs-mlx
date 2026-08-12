"""Restricted loading for the packaged official Demucs model repository."""

from __future__ import annotations

import inspect
import math
import typing as tp
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from packaging.version import InvalidVersion, Version

from .mlx_registry import MLX_MODEL_REGISTRY

MIN_SAFE_TORCH_VERSION = Version("2.6")
OFFICIAL_DEMUCS_ROOT_URL = "https://dl.fbaipublicfiles.com/demucs/"

_MAX_CONTAINER_ITEMS = 100_000
_MAX_METADATA_DEPTH = 32
_MAX_METADATA_NODES = 100_000
_MAX_STATE_KEYS = 100_000
_MAX_STRING_LENGTH = 4_096
_MAX_INTEGER_BITS = 63
_PACKAGE_REQUIRED_FIELDS = {"klass", "args", "kwargs", "state"}
_PACKAGE_OPTIONAL_FIELDS = {"training_args", "metrics"}


@dataclass(frozen=True)
class OfficialSource:
    """Verified source identity encoded in the official checkpoint filename."""

    signature: str
    checksum: str
    url: str


@dataclass(frozen=True)
class RestrictedDemucsLoad:
    """A constructed official model and the ordered sources used to build it."""

    model: tp.Any
    sources: tuple[OfficialSource, ...]


class _MetadataBudget:
    def __init__(self) -> None:
        self.nodes = 0

    def consume(self, depth: int) -> None:
        if depth > _MAX_METADATA_DEPTH:
            raise ValueError("Demucs checkpoint metadata is nested too deeply")
        self.nodes += 1
        if self.nodes > _MAX_METADATA_NODES:
            raise ValueError("Demucs checkpoint metadata contains too many values")


def _require_safe_torch(torch: tp.Any) -> None:
    raw_version = str(getattr(torch, "__version__", "")).split("+", 1)[0]
    try:
        installed = Version(raw_version)
    except InvalidVersion as exc:
        raise RuntimeError(f"Could not validate installed PyTorch version {raw_version!r}") from exc
    if installed < MIN_SAFE_TORCH_VERSION:
        raise RuntimeError("PyTorch 2.6 or newer is required to safely load Demucs checkpoints.")


def _diff_quantizer_class() -> tp.Optional[type]:
    try:
        from diffq.diffq import DiffQuantizer
    except ImportError:
        return None
    return DiffQuantizer


def _safe_globals() -> list[tp.Any]:
    """Return only the exact globals required by official Demucs packages."""
    import numpy as np

    try:
        import numpy._core.multiarray as np_core_multiarray
    except ImportError:  # NumPy < 2
        import numpy.core.multiarray as np_core_multiarray  # type: ignore[no-redef]

    from demucs.demucs import Demucs
    from demucs.hdemucs import HDemucs
    from demucs.htdemucs import HTDemucs

    allowed: list[tp.Any] = [
        Demucs,
        HDemucs,
        HTDemucs,
        Fraction,
        np.dtype,
        (np_core_multiarray.scalar, "numpy.core.multiarray.scalar"),
        (np_core_multiarray.scalar, "numpy._core.multiarray.scalar"),
    ]
    # NumPy constructs dtype classes dynamically. PyTorch documents that these
    # classes must be allowlisted separately from numpy.dtype.
    dtype_specs = (
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    )
    allowed.extend({type(np.dtype(spec)) for spec in dtype_specs})

    diff_quantizer = _diff_quantizer_class()
    if diff_quantizer is not None:
        allowed.append(diff_quantizer)
    return allowed


def _validate_string(value: str, path: str) -> None:
    if len(value) > _MAX_STRING_LENGTH:
        raise ValueError(f"String at {path} is too long")


def _validate_metadata_value(
    value: tp.Any,
    path: str,
    *,
    budget: tp.Optional[_MetadataBudget] = None,
    depth: int = 0,
) -> None:
    import numpy as np

    if budget is None:
        budget = _MetadataBudget()
    budget.consume(depth)

    if value is None or isinstance(value, bool):
        return
    if isinstance(value, (int, np.integer)):
        if int(value).bit_length() > _MAX_INTEGER_BITS:
            raise ValueError(f"Integer metadata value at {path} is out of range")
        return
    if isinstance(value, Fraction):
        if (
            value.numerator.bit_length() > _MAX_INTEGER_BITS
            or value.denominator.bit_length() > _MAX_INTEGER_BITS
        ):
            raise ValueError(f"Fraction metadata value at {path} is out of range")
        return
    if isinstance(value, str):
        _validate_string(value, path)
        return
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite metadata value at {path}")
        return
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_CONTAINER_ITEMS:
            raise ValueError(f"Sequence at {path} is too large")
        for index, item in enumerate(value):
            _validate_metadata_value(
                item,
                f"{path}[{index}]",
                budget=budget,
                depth=depth + 1,
            )
        return
    if isinstance(value, dict):
        if len(value) > _MAX_CONTAINER_ITEMS:
            raise ValueError(f"Mapping at {path} is too large")
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"Mapping at {path} must use string keys")
            _validate_string(key, f"{path} key")
            _validate_metadata_value(
                item,
                f"{path}.{key}",
                budget=budget,
                depth=depth + 1,
            )
        return
    raise ValueError(f"Unsupported metadata value at {path}: {type(value).__name__}")


def _validate_tensor_list(value: tp.Any, path: str, torch: tp.Any) -> None:
    if not isinstance(value, list) or len(value) > _MAX_CONTAINER_ITEMS:
        raise ValueError(f"{path} must be a bounded list")
    if not all(isinstance(item, torch.Tensor) for item in value):
        raise ValueError(f"{path} must contain only tensors")


def _validate_scale(value: tp.Any, path: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{path} must be a two-value scale")
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{path} must contain numeric values")
        if not math.isfinite(float(item)):
            raise ValueError(f"{path} must contain finite values")


def _validate_quantized_state(state: dict[str, tp.Any], torch: tp.Any) -> None:
    expected = {"quantized", "float16", "others", "meta", "__quantized"}
    if set(state) != expected or state["__quantized"] is not True:
        raise ValueError("Demucs quantized state has an unexpected structure")

    meta = state["meta"]
    if not isinstance(meta, dict):
        raise ValueError("Demucs quantized state metadata must be a dictionary")
    required_meta = {"init_kwargs", "klass", "packed"}
    optional_meta = {"torch_pack"}
    if not required_meta.issubset(meta) or set(meta).difference(required_meta | optional_meta):
        raise ValueError("Demucs quantized state metadata has unexpected fields")

    quantizer_class = _diff_quantizer_class()
    if quantizer_class is None or meta["klass"] is not quantizer_class:
        raise ValueError("Demucs quantized state requires the exact DiffQuantizer class")
    if not isinstance(meta["packed"], bool):
        raise ValueError("Demucs quantized packed flag must be boolean")
    if "torch_pack" in meta and not isinstance(meta["torch_pack"], bool):
        raise ValueError("Demucs quantized torch_pack flag must be boolean")

    init_kwargs = meta["init_kwargs"]
    if not isinstance(init_kwargs, dict):
        raise ValueError("Demucs quantizer constructor metadata must be a dictionary")
    _validate_metadata_value(init_kwargs, "state.meta.init_kwargs")
    try:
        inspect.signature(quantizer_class).bind(object(), **init_kwargs)
    except TypeError as exc:
        raise ValueError("Invalid DiffQuantizer constructor metadata") from exc

    quantized = state["quantized"]
    if not isinstance(quantized, list) or not quantized:
        raise ValueError("Demucs quantized parameters must be a non-empty list")
    if len(quantized) > _MAX_CONTAINER_ITEMS:
        raise ValueError("Demucs quantized parameter list is too large")

    for index, item in enumerate(quantized):
        path = f"state.quantized[{index}]"
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError(f"{path} must contain levels, scale, and bits")
        levels, scale, bits = item
        _validate_scale(scale, f"{path}.scale")
        if not isinstance(bits, torch.Tensor):
            raise ValueError(f"{path}.bits must be a tensor")
        if meta["packed"]:
            if not isinstance(levels, list) or len(levels) != 14:
                raise ValueError(f"{path}.levels must contain 14 packed bit groups")
            if not all(level is None or isinstance(level, torch.Tensor) for level in levels):
                raise ValueError(f"{path}.levels contains an invalid packed value")
        elif not isinstance(levels, torch.Tensor):
            raise ValueError(f"{path}.levels must be a tensor")

    _validate_tensor_list(state["float16"], "state.float16", torch)
    _validate_tensor_list(state["others"], "state.others", torch)


def _validate_package(package: tp.Any, torch: tp.Any) -> dict[str, tp.Any]:
    from demucs.demucs import Demucs
    from demucs.hdemucs import HDemucs
    from demucs.htdemucs import HTDemucs

    if not isinstance(package, dict) or not all(isinstance(key, str) for key in package):
        raise ValueError(f"Unexpected Demucs checkpoint type: {type(package).__name__}")
    fields = set(package)
    missing = _PACKAGE_REQUIRED_FIELDS.difference(fields)
    unknown = fields.difference(_PACKAGE_REQUIRED_FIELDS | _PACKAGE_OPTIONAL_FIELDS)
    if missing:
        raise ValueError(f"Demucs checkpoint is missing required keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"Demucs checkpoint has unexpected keys: {sorted(unknown)}")

    model_classes = (Demucs, HDemucs, HTDemucs)
    if not any(package["klass"] is model_class for model_class in model_classes):
        raise ValueError(f"Unsupported Demucs model class: {package['klass']!r}")
    args = package["args"]
    kwargs = package["kwargs"]
    if not isinstance(args, (list, tuple)) or len(args) > _MAX_CONTAINER_ITEMS:
        raise ValueError("Demucs checkpoint args must be a bounded list or tuple")
    if not isinstance(kwargs, dict) or len(kwargs) > _MAX_CONTAINER_ITEMS:
        raise ValueError("Demucs checkpoint kwargs must be a bounded dictionary")
    _validate_metadata_value(args, "args")
    _validate_metadata_value(kwargs, "kwargs")
    try:
        inspect.signature(package["klass"]).bind(*args, **kwargs)
    except TypeError as exc:
        raise ValueError("Invalid Demucs constructor args or kwargs") from exc

    for optional_field in _PACKAGE_OPTIONAL_FIELDS.intersection(package):
        _validate_metadata_value(package[optional_field], optional_field)

    state = package["state"]
    if not isinstance(state, dict) or not state or len(state) > _MAX_STATE_KEYS:
        raise ValueError("Demucs checkpoint state must be a non-empty bounded dictionary")
    if not all(isinstance(key, str) for key in state):
        raise ValueError("Demucs checkpoint state must use string keys")
    for key in state:
        _validate_string(key, "state key")
    if state.get("__quantized") is True:
        _validate_quantized_state(state, torch)
    else:
        invalid = [key for key, value in state.items() if not isinstance(value, torch.Tensor)]
        if invalid:
            raise ValueError(f"Demucs checkpoint state contains non-tensor values: {invalid[:5]}")
    return package


def _load_package_from_url(url: str, torch: tp.Any) -> dict[str, tp.Any]:
    _require_safe_torch(torch)
    try:
        with torch.serialization.safe_globals(_safe_globals()):
            package = torch.hub.load_state_dict_from_url(
                url,
                map_location="cpu",
                check_hash=True,
                weights_only=True,
            )
    except Exception as exc:
        raise RuntimeError(
            f"Restricted Demucs checkpoint load failed for {url!r}; "
            "no unrestricted fallback was attempted"
        ) from exc
    return _validate_package(package, torch)


def _construct_validated_model(package: tp.Any, torch: tp.Any) -> tp.Any:
    """Validate again immediately before trusted Demucs constructs the model."""
    from demucs.states import load_model

    validated = _validate_package(package, torch)
    model = load_model(validated, strict=True)
    model.eval()
    return model


def _remote_root() -> Path:
    return Path(__file__).resolve().parent / "remote"


def _official_source_map() -> dict[str, OfficialSource]:
    sources: dict[str, OfficialSource] = {}
    root = ""
    for raw_line in (_remote_root() / "files.txt").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("root:"):
            root = line.split(":", 1)[1].strip()
            continue
        filename = Path(line).name
        if Path(filename).suffix != ".th" or "-" not in Path(filename).stem:
            raise RuntimeError(f"Invalid packaged Demucs source filename: {filename!r}")
        signature, checksum = Path(filename).stem.split("-", 1)
        if (
            len(signature) != 8
            or len(checksum) != 8
            or any(char not in "0123456789abcdef" for char in signature + checksum)
        ):
            raise RuntimeError(f"Invalid packaged Demucs source identity: {filename!r}")
        if signature in sources:
            raise RuntimeError(f"Duplicate packaged Demucs signature: {signature}")
        sources[signature] = OfficialSource(
            signature=signature,
            checksum=checksum,
            url=OFFICIAL_DEMUCS_ROOT_URL + root + line,
        )
    return sources


def expected_official_sources(model_name: str) -> tuple[OfficialSource, ...]:
    try:
        signatures = MLX_MODEL_REGISTRY[model_name]["signatures"]
    except KeyError:
        raise ValueError(f"Unknown Demucs model: {model_name!r}") from None
    source_map = _official_source_map()
    try:
        return tuple(source_map[signature] for signature in signatures)
    except KeyError as exc:
        raise RuntimeError(
            f"Packaged Demucs registry is missing signature {exc.args[0]!r}"
        ) from None


def get_restricted_demucs_model(model_name: str) -> RestrictedDemucsLoad:
    """Load an official named model without any unrestricted pickle path."""
    try:
        import torch
        from demucs.repo import AnyModelRepo, BagOnlyRepo, ModelLoadingError, ModelOnlyRepo
    except ImportError:
        raise ImportError(
            "Model conversion requires the [convert] extras. "
            "Install with: pip install 'demucs-mlx[convert]'"
        ) from None

    _require_safe_torch(torch)
    expected_sources = expected_official_sources(model_name)
    source_map = {source.signature: source for source in expected_sources}

    class RestrictedRemoteRepo(ModelOnlyRepo):
        def __init__(self) -> None:
            self.loaded: list[OfficialSource] = []

        def has_model(self, signature: str) -> bool:
            return signature in source_map

        def get_model(self, signature: str) -> tp.Any:
            try:
                source = source_map[signature]
            except KeyError:
                raise ModelLoadingError(
                    f"Could not find an official model with signature {signature}."
                ) from None
            package = _load_package_from_url(source.url, torch)
            model = _construct_validated_model(package, torch)
            self.loaded.append(source)
            return model

    model_repo = RestrictedRemoteRepo()
    bag_repo = BagOnlyRepo(_remote_root(), model_repo)
    model = AnyModelRepo(model_repo, bag_repo).get_model(model_name)
    if tuple(model_repo.loaded) != expected_sources:
        raise RuntimeError(
            f"Packaged Demucs model {model_name!r} resolved unexpected source signatures"
        )
    model.eval()
    return RestrictedDemucsLoad(model=model, sources=expected_sources)
