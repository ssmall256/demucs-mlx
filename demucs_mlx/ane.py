"""Experimental Core ML first waveform convolution for default HTDemucs.

Conversion uses the restricted official PyTorch loader. Inference needs only
PyObjC: Core ML runs on a worker while MLX evaluates the spectral branch.
"""

from __future__ import annotations

import argparse
import json
import platform
import queue
import shutil
import sys
import threading
import time
import typing as tp
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx

from .ane_paths import ANE_MODULE, DISTRIBUTION, ane_cache_dir

MODEL_NAME = "htdemucs"
BATCH = 2
LENGTH = 343_980  # 7.8 seconds at 44.1 kHz, the official training segment.
OUTPUT_NAME = "y0"
DEFAULT_TILE_OUTPUTS = 12285
SUPPORTED_TILE_OUTPUTS = (4095, 12285)


def _tile_count(tile_outputs: int) -> int:
    if tile_outputs not in SUPPORTED_TILE_OUTPUTS:
        raise ValueError(f"Supported ANE tile output lengths: {SUPPORTED_TILE_OUTPUTS}")
    return 85_995 // tile_outputs


def _asset_name(tile_outputs: int) -> str:
    count = _tile_count(tile_outputs)
    suffix = "" if tile_outputs == 4095 else f"_t{count}"
    return f"htdemucs_time_conv_b2{suffix}"


def asset_dir() -> Path:
    return ane_cache_dir()


def compiled_path(tile_outputs: int = DEFAULT_TILE_OUTPUTS) -> Path:
    return asset_dir() / f"{_asset_name(tile_outputs)}.mlmodelc"


def manifest_path(tile_outputs: int = DEFAULT_TILE_OUTPUTS) -> Path:
    return asset_dir() / f"{_asset_name(tile_outputs)}.json"


def tail_compiled_path() -> Path:
    return asset_dir() / "htdemucs_time_tail_b2.mlmodelc"


def tail_manifest_path() -> Path:
    return asset_dir() / "htdemucs_time_tail_b2.json"


def _cache_identity() -> str:
    from .model_converter import get_mlx_cache_dir, get_mlx_model

    # The ordinary loader checks the bounded config and the safetensors digest.
    get_mlx_model(MODEL_NAME)
    config_path = get_mlx_cache_dir() / f"{MODEL_NAME}_config.json"
    with config_path.open() as file:
        config = json.load(file)
    digest = config["safetensors_sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("The validated HTDemucs cache has no weight digest")
    return digest


def _torch_conv(torch_model, tile_outputs: int = DEFAULT_TILE_OUTPUTS):
    import torch
    import torch.nn.functional as functional

    tile_count = _tile_count(tile_outputs)
    tile_step = 4 * tile_outputs
    tile_width = tile_step + 4

    class TiledFirstConv(torch.nn.Module):
        """Normalize globally, then partition one convolution exactly."""

        def __init__(self, conv):
            super().__init__()
            self.conv = conv

        def forward(self, x):
            mean = x.mean(dim=(1, 2), keepdim=True)
            std = x.std(dim=(1, 2), keepdim=True, unbiased=False)
            x = (x - mean) / (1e-5 + std)
            # (343980 + 4 - 8) / 4 + 1 = 85995. Adjacent tiles overlap
            # by four samples, preserving the original padded convolution.
            padded = functional.pad(x, (2, 2))
            return torch.cat(
                [
                    functional.conv1d(
                        padded[:, :, index * tile_step : index * tile_step + tile_width],
                        self.conv.weight,
                        self.conv.bias,
                        stride=4,
                    )
                    for index in range(tile_count)
                ],
                dim=-1,
            )

    return TiledFirstConv(torch_model.tencoder[0].conv).eval()


def _torch_tail_convs(torch_model):
    import torch
    import torch.nn.functional as functional

    class TiledConv1(torch.nn.Module):
        def __init__(self, conv):
            super().__init__()
            self.conv = conv

        def forward(self, x):
            padded = functional.pad(x, (2, 3))
            tiles = []
            for i in range(4):
                inp = padded[:, :, i * 17200 : i * 17200 + 17204]
                tiles.append(functional.conv1d(inp, self.conv.weight, self.conv.bias, stride=4))
            inp4 = padded[:, :, 4 * 17200 : 4 * 17200 + 17200]
            tiles.append(functional.conv1d(inp4, self.conv.weight, self.conv.bias, stride=4))
            return torch.cat(tiles, dim=-1)

    class ConvStage(torch.nn.Module):
        def __init__(self, conv):
            super().__init__()
            self.conv = conv

        def forward(self, x):
            return self.conv(functional.pad(x, (0, 1)))

    m1 = TiledConv1(torch_model.tencoder[1].conv).eval()
    m2 = ConvStage(torch_model.tencoder[2].conv).eval()
    m3 = ConvStage(torch_model.tencoder[3].conv).eval()
    return m1, m2, m3


def _device_placement(path: Path) -> dict:
    """Inspect Core ML's anticipated placement, including estimated cost."""
    import coremltools as ct

    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        str(path), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    program = plan.model_structure.program
    if program is None:
        raise RuntimeError("Expected an ML Program compute plan")
    counts: dict[str, int] = {}
    weights: dict[str, float] = {}

    def visit(block):
        for operation in block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
            if usage is not None:
                device = type(usage.preferred_compute_device).__name__
                counts[device] = counts.get(device, 0) + 1
                cost = plan.get_estimated_cost_for_mlprogram_operation(operation)
                if cost is not None:
                    weights[device] = weights.get(device, 0.0) + float(cost.weight)
            for nested in getattr(operation, "blocks", ()):
                visit(nested)

    visit(program.functions["main"].block)
    return {"operations": counts, "estimated_cost": weights}


def convert(tile_outputs: int = DEFAULT_TILE_OUTPUTS) -> dict:
    """Convert the first convolution and verify ANE placement."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("The Neural Engine prototype requires Apple Silicon macOS")
    try:
        import coremltools as ct
        import torch
        from coremltools.converters.mil.mil import types as mil_types
        from demucs.apply import BagOfModels
    except ImportError as exc:
        raise RuntimeError(
            "Conversion needs coremltools 9, PyTorch, and Demucs; install "
            f"{DISTRIBUTION}[ane-convert]"
        ) from exc

    tile_count = _tile_count(tile_outputs)
    digest = _cache_identity()
    from .secure_demucs import get_restricted_demucs_model

    restricted = get_restricted_demucs_model(MODEL_NAME)
    # demucs-mlx returns a wrapper with .model; mlx-audio-separator returns the model.
    source = getattr(restricted, "model", restricted)
    models = source.models if isinstance(source, BagOfModels) else [source]
    if len(models) != 1 or type(models[0]).__name__ != "HTDemucs":
        raise RuntimeError("Expected one official HTDemucs model")
    torch_model = models[0].eval()
    length = int(torch_model.segment * torch_model.samplerate)
    if length != LENGTH or torch_model.audio_channels != 2 or len(torch_model.tencoder) != 4:
        raise RuntimeError("Unexpected HTDemucs waveform encoder shape")

    wrapper = _torch_conv(torch_model, tile_outputs)
    example = torch.zeros((BATCH, 2, LENGTH), dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, check_trace=False)
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="mix", shape=tuple(example.shape), dtype=mil_types.fp32)],
        outputs=[ct.TensorType(name=OUTPUT_NAME, dtype=mil_types.fp16)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    target = compiled_path(tile_outputs)
    target.parent.mkdir(parents=True, exist_ok=True)
    package = target.with_suffix(".mlpackage")
    ml.save(str(package))
    built = Path(ct.models.utils.compile_model(str(package)))
    placement = _device_placement(built)
    manifest_path(tile_outputs).unlink(missing_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(built, target)
    ane_preferred = any("NeuralEngine" in device for device in placement["operations"])
    manifest = {
        "model": MODEL_NAME,
        "batch": BATCH,
        "length": LENGTH,
        "partition": f"normalized_tencoder_0_conv_tiled_{tile_count}",
        "tile_outputs": tile_outputs,
        "safetensors_sha256": digest,
        "placement": placement,
        "ane_preferred": ane_preferred,
    }
    temporary_manifest = manifest_path(tile_outputs).with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary_manifest.replace(manifest_path(tile_outputs))
    if not ane_preferred:
        raise RuntimeError(
            "Core ML converted the convolution, but its compute plan chose CPU for "
            f"every operation. Diagnostic assets are at {target}; placement: {placement}"
        )
    return manifest


def convert_tail() -> dict:
    """Convert the remaining waveform downsamplers and verify ANE placement."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("The Neural Engine prototype requires Apple Silicon macOS")
    try:
        import coremltools as ct
        import torch
        from coremltools.converters.mil.mil import types as mil_types
        from demucs.apply import BagOfModels
    except ImportError as exc:
        raise RuntimeError(
            "Conversion needs coremltools 9, PyTorch, and Demucs; install "
            f"{DISTRIBUTION}[ane-convert]"
        ) from exc

    digest = _cache_identity()
    from .secure_demucs import get_restricted_demucs_model

    restricted = get_restricted_demucs_model(MODEL_NAME)
    # demucs-mlx returns a wrapper with .model; mlx-audio-separator returns the model.
    source = getattr(restricted, "model", restricted)
    models = source.models if isinstance(source, BagOfModels) else [source]
    if len(models) != 1 or type(models[0]).__name__ != "HTDemucs":
        raise RuntimeError("Expected one official HTDemucs model")
    torch_model = models[0].eval()
    if (
        int(torch_model.segment * torch_model.samplerate) != LENGTH
        or torch_model.audio_channels != 2
        or len(torch_model.tencoder) != 4
    ):
        raise RuntimeError("Unexpected HTDemucs waveform encoder shape")

    m1, m2, m3 = _torch_tail_convs(torch_model)
    stages = [
        ("c1", m1, (BATCH, 48, 85_995)),
        ("c2", m2, (BATCH, 96, 21_499)),
        ("c3", m3, (BATCH, 192, 5_375)),
    ]

    target = tail_compiled_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    placements = {}
    ane_preferred = True

    for name, module, shape in stages:
        example = torch.zeros(shape, dtype=torch.float32)
        with torch.no_grad():
            traced = torch.jit.trace(module, example, check_trace=False)
        ml = ct.convert(
            traced,
            inputs=[ct.TensorType(name="x", shape=shape, dtype=mil_types.fp32)],
            outputs=[ct.TensorType(name="y", dtype=mil_types.fp16)],
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        package = target.parent / f"{target.stem}_{name}.mlpackage"
        ml.save(str(package))
        built = Path(ct.models.utils.compile_model(str(package)))
        placement = _device_placement(built)
        placements[name] = placement
        stage_target = target / f"{name}.mlmodelc"
        if stage_target.exists():
            shutil.rmtree(stage_target)
        shutil.copytree(built, stage_target)
        if not any("NeuralEngine" in d for d in placement["operations"]):
            ane_preferred = False

    manifest = {
        "model": MODEL_NAME,
        "batch": BATCH,
        "input_shape": [BATCH, 48, 85_995],
        "partition": "hybrid_tencoder_1_3",
        "safetensors_sha256": digest,
        "placement": placements,
        "ane_preferred": ane_preferred,
    }
    temporary_manifest = tail_manifest_path().with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary_manifest.replace(tail_manifest_path())
    if not ane_preferred:
        raise RuntimeError(
            f"Core ML converted the waveform tail, but placement lacked NeuralEngine: {placements}"
        )
    return manifest


def _strides(shape: tp.Sequence[int]) -> list[int]:
    """Element strides of a row-contiguous array."""
    strides, step = [], 1
    for dim in reversed(shape):
        strides.append(step)
        step *= int(dim)
    return strides[::-1]


def _ready(array: mx.array, dtype: mx.Dtype) -> mx.array:
    """An evaluated, row-contiguous copy-if-needed of ``array`` in ``dtype``.

    Evaluate on the submitting thread: MLX streams belong to the thread that
    built a graph, and the Core ML worker reads the memory directly.
    """
    ready = mx.contiguous(array.astype(dtype))
    mx.eval(ready)
    return ready


def _materialized(array: mx.array) -> mx.array:
    """Evaluate on the worker before handing a result to another thread.

    A lazy op (even a slice) is bound to the stream of the thread that built it,
    so the consumer could not evaluate it.
    """
    mx.eval(array)
    return array


def _multiarray(coreml, array: mx.array, c_dtype) -> tp.Any:
    """Wrap an evaluated MLX array's memory as an MLMultiArray (no copy).

    The returned object borrows the memory; keep ``array`` alive while it is used.
    """
    multi, error = (
        coreml.MLMultiArray.alloc()
        .initWithDataPointer_shape_dataType_strides_deallocator_error_(
            memoryview(array), list(array.shape), c_dtype, _strides(array.shape), None, None
        )
    )
    if multi is None:
        raise RuntimeError(f"Could not wrap an MLX buffer for Core ML: {error}")
    return multi


def _multiarray_to_mlx(coreml, value: tp.Any) -> mx.array:
    """Copy a Core ML output MLMultiArray into a row-contiguous MLX array."""
    dtype = {
        coreml.MLMultiArrayDataTypeFloat16: mx.float16,
        coreml.MLMultiArrayDataTypeFloat32: mx.float32,
    }[value.dataType()]
    shape = [int(s) for s in value.shape()]
    strides = [int(s) for s in value.strides()]
    nbytes = int(value.count()) * (2 if dtype == mx.float16 else 4)
    pointer = value.dataPointer()
    if hasattr(pointer, "as_buffer"):
        flat = mx.array(memoryview(pointer.as_buffer(nbytes)).cast("B"))
    else:
        held = {}

        def grab(raw_bytes, size):
            held["flat"] = mx.array(memoryview(raw_bytes).cast("B")[:size])

        value.getBytesWithHandler_(grab)
        flat = held["flat"]
    return mx.contiguous(mx.as_strided(flat.view(dtype), shape, strides))


class WaveformConv:
    """One Core ML prediction at a time on a PyObjC worker thread."""

    def __init__(self, tile_outputs: int = DEFAULT_TILE_OUTPUTS):
        if sys.platform != "darwin" or platform.machine() != "arm64":
            raise RuntimeError("The Neural Engine prototype requires Apple Silicon macOS")
        tile_count = _tile_count(tile_outputs)
        self.path = compiled_path(tile_outputs)
        if not self.path.is_dir() or not manifest_path(tile_outputs).is_file():
            raise FileNotFoundError(
                "Converted waveform convolution missing; run "
                f"`python -m {ANE_MODULE} convert` first"
            )
        manifest = json.loads(manifest_path(tile_outputs).read_text())
        if (
            manifest.get("model") != MODEL_NAME
            or manifest.get("batch") != BATCH
            or manifest.get("length") != LENGTH
            or manifest.get("partition") != f"normalized_tencoder_0_conv_tiled_{tile_count}"
            or manifest.get("tile_outputs", 4095) != tile_outputs
            or manifest.get("safetensors_sha256") != _cache_identity()
        ):
            raise RuntimeError("Core ML convolution does not match the validated MLX weights")
        if not manifest.get("ane_preferred"):
            raise RuntimeError("Core ML convolution asset has no Neural Engine placement")
        self._init_coreml(manifest["placement"], "mix", (OUTPUT_NAME,))

    def _init_coreml(self, placement: dict, input_name: str, output_names: tuple[str, ...]):
        if sys.platform != "darwin" or platform.machine() != "arm64":
            raise RuntimeError("The Neural Engine prototype requires Apple Silicon macOS")
        try:
            import CoreML
            import Foundation
        except ImportError as exc:
            raise RuntimeError(f"Install {DISTRIBUTION}[ane] for the Core ML runtime") from exc

        self.placement = placement
        self._input_name = input_name
        self._output_names = output_names
        self._coreml = CoreML
        config = CoreML.MLModelConfiguration.alloc().init()
        config.setComputeUnits_(CoreML.MLComputeUnitsCPUAndNeuralEngine)
        model, error = CoreML.MLModel.modelWithContentsOfURL_configuration_error_(
            Foundation.NSURL.fileURLWithPath_(str(self.path)), config, None
        )
        if model is None:
            raise RuntimeError(f"Core ML could not load {self.path}: {error}")
        desc = model.modelDescription()

        self._has_output_backings = hasattr(CoreML.MLPredictionOptions, "setOutputBackings_")
        self._output_specs: dict[str, tuple[tuple[int, ...], tp.Any, mx.Dtype]] = {}
        out_descs = desc.outputDescriptionsByName()
        for name in output_names:
            fdesc = out_descs.get(name)
            if fdesc and fdesc.multiArrayConstraint():
                c = fdesc.multiArrayConstraint()
                shape = tuple(int(s) for s in c.shape())
                c_dtype = c.dataType()
                is_half = c_dtype == CoreML.MLMultiArrayDataTypeFloat16
                self._output_specs[name] = (shape, c_dtype, mx.float16 if is_half else mx.float32)
            else:
                self._has_output_backings = False

        self.model = model
        self.busy_seconds = 0.0
        self.wait_seconds = 0.0
        self.transfer_seconds = 0.0
        self.predictions = 0
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="demucs-ane-sub")
        self._jobs: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True, name="demucs-ane")
        self._worker.start()

    def submit(self, mix: tp.Any) -> Future:
        """Queue (N, 2, 343980) audio; the future yields a float16 MLX array."""
        if not isinstance(mix, mx.array):
            mix = mx.asarray(mix)
        if mix.ndim != 3 or tuple(mix.shape[1:]) != (2, LENGTH) or mix.shape[0] < 1:
            raise ValueError(f"ANE convolution expects (N, 2, {LENGTH}), got {tuple(mix.shape)}")
        return self._submit(mix)

    def _submit(self, mix: mx.array) -> Future:
        data = _ready(mix, mx.float32)
        future: Future = Future()
        self._jobs.put((data, future))
        return future

    def close(self) -> None:
        if self._worker.is_alive():
            self._jobs.put(None)
            self._worker.join()
        self._executor.shutdown(wait=False)

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            data, future = job
            try:
                start = time.perf_counter()
                result = self._predict(data)
                self.busy_seconds += time.perf_counter() - start
                self.predictions += 1
                future.set_result(result)
            except BaseException as exc:
                future.set_exception(exc)

    def _output_like(self, count: int) -> mx.array:
        """A fresh output array for one prediction.

        Never reused: the caller hands it to a lazy MLX graph that may still be
        running when the next prediction starts.
        """
        shape, _, dtype = self._output_specs[self._output_names[0]]
        out = mx.zeros((count, *shape[1:]), dtype=dtype)
        mx.eval(out)
        return out

    def _predict(self, data: mx.array):
        count = int(data.shape[0])
        out_target = self._output_like(count)

        from .native_ane import predict_waveform_conv_native

        if predict_waveform_conv_native(self.path, data, out_target):
            return out_target

        if count > 2:
            futures = [
                self._executor.submit(self._predict, data[start : min(start + 2, count)])
                for start in range(0, count, 2)
            ]
            return _materialized(mx.concatenate([f.result() for f in futures], axis=0))

        coreml = self._coreml
        if count == 1:
            data = _ready(mx.concatenate([data, data], axis=0), mx.float32)
        array = _multiarray(coreml, data, coreml.MLMultiArrayDataTypeFloat32)
        features, error = coreml.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(
            {self._input_name: coreml.MLFeatureValue.featureValueWithMultiArray_(array)}, None
        )
        if features is None:
            raise RuntimeError(f"Could not create Core ML input features: {error}")

        if self._has_output_backings:
            outputs = {name: self._output_like(BATCH) for name in self._output_names}
            backings = {
                name: _multiarray(coreml, outputs[name], self._output_specs[name][1])
                for name in self._output_names
            }
            options = coreml.MLPredictionOptions.alloc().init()
            options.setOutputBackings_(backings)
            result, error = self.model.predictionFromFeatures_options_error_(
                features, options, None
            )
            if result is None:
                raise RuntimeError(f"Core ML prediction with output backings failed: {error}")
            values = [_materialized(outputs[name][:count]) for name in self._output_names]
            return values[0] if len(values) == 1 else tuple(values)

        result, error = self.model.predictionFromFeatures_error_(features, None)
        if result is None:
            raise RuntimeError(f"Core ML prediction failed: {error}")
        values = [
            _materialized(
                _multiarray_to_mlx(
                    coreml, result.featureValueForName_(name).multiArrayValue()
                )[:count]
            )
            for name in self._output_names
        ]
        return values[0] if len(values) == 1 else tuple(values)


class WaveformTail:
    """High-ROI mixed-precision waveform tail with ANE downsampling and MLX layers."""

    _SHAPES = {
        "c1": ((BATCH, 48, 85_995), (BATCH, 96, 21_499)),
        "c2": ((BATCH, 96, 21_499), (BATCH, 192, 5_375)),
        "c3": ((BATCH, 192, 5_375), (BATCH, 384, 1_344)),
    }

    def __init__(self):
        if sys.platform != "darwin" or platform.machine() != "arm64":
            raise RuntimeError("The Neural Engine prototype requires Apple Silicon macOS")
        self.path = tail_compiled_path()
        if not self.path.is_dir() or not tail_manifest_path().is_file():
            raise FileNotFoundError(
                "Converted waveform tail missing; run "
                f"`python -m {ANE_MODULE} convert-tail` first"
            )
        manifest = json.loads(tail_manifest_path().read_text())
        if (
            manifest.get("model") != MODEL_NAME
            or manifest.get("batch") != BATCH
            or manifest.get("input_shape") != [BATCH, 48, 85_995]
            or manifest.get("partition") != "hybrid_tencoder_1_3"
            or manifest.get("safetensors_sha256") != _cache_identity()
        ):
            raise RuntimeError("Core ML waveform tail does not match validated MLX weights")
        if not manifest.get("ane_preferred"):
            raise RuntimeError("Core ML waveform tail asset has no Neural Engine placement")

        try:
            import CoreML
            import Foundation
        except ImportError as exc:
            raise RuntimeError(f"Install {DISTRIBUTION}[ane] for the Core ML runtime") from exc

        self.placement = manifest["placement"]
        self._coreml = CoreML

        from .model_converter import get_mlx_model

        self._mlx_model = get_mlx_model(MODEL_NAME).models[0]
        mx.eval(self._mlx_model.parameters())

        config = CoreML.MLModelConfiguration.alloc().init()
        config.setComputeUnits_(CoreML.MLComputeUnitsCPUAndNeuralEngine)

        self._models = {}
        self._out_buffers: dict[str, mx.array] = {}
        self._out_backings = {}
        for name, (_, out_sh) in self._SHAPES.items():
            model_url = Foundation.NSURL.fileURLWithPath_(str(self.path / f"{name}.mlmodelc"))
            m, err = CoreML.MLModel.modelWithContentsOfURL_configuration_error_(
                model_url, config, None
            )
            if m is None:
                raise RuntimeError(f"Could not load Core ML model {name}: {err}")
            self._models[name] = m
            # Each stage's output is consumed and evaluated before the next
            # prediction, so one backing buffer per stage can be reused.
            out_buf = mx.zeros(out_sh, dtype=mx.float16)
            mx.eval(out_buf)
            self._out_buffers[name] = out_buf
            self._out_backings[name] = _multiarray(
                CoreML, out_buf, CoreML.MLMultiArrayDataTypeFloat16
            )

        self.busy_seconds = 0.0
        self.wait_seconds = 0.0
        self.transfer_seconds = 0.0
        self.predictions = 0
        self._jobs: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True, name="demucs-ane-tail")
        self._worker.start()

    def submit(self, encoded: tp.Any) -> Future:
        """Queue (1 or 2, 48, 85995) encoder output; yields three MLX arrays."""
        if not isinstance(encoded, mx.array):
            encoded = mx.asarray(encoded)
        if (
            encoded.ndim != 3
            or tuple(encoded.shape[1:]) != (48, 85_995)
            or encoded.shape[0] not in (1, 2)
        ):
            raise ValueError(
                f"ANE waveform tail expects (1 or 2, 48, 85995), got {tuple(encoded.shape)}"
            )
        data = _ready(encoded, mx.float32)
        future: Future = Future()
        self._jobs.put((data, future))
        return future

    def close(self) -> None:
        if self._worker.is_alive():
            self._jobs.put(None)
            self._worker.join()

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            data, future = job
            try:
                start = time.perf_counter()
                result = self._predict(data)
                self.busy_seconds += time.perf_counter() - start
                self.predictions += 1
                future.set_result(result)
            except BaseException as exc:
                future.set_exception(exc)

    def _predict_stage(self, name: str, data: mx.array) -> mx.array:
        coreml = self._coreml
        count = int(data.shape[0])
        if count == 1:
            data = mx.concatenate([data, data], axis=0)
        data = _ready(data, mx.float32)
        array = _multiarray(coreml, data, coreml.MLMultiArrayDataTypeFloat32)
        features, error = coreml.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(
            {"x": coreml.MLFeatureValue.featureValueWithMultiArray_(array)}, None
        )
        if features is None:
            raise RuntimeError(f"Could not create Core ML input features for {name}: {error}")

        options = coreml.MLPredictionOptions.alloc().init()
        options.setOutputBackings_({"y": self._out_backings[name]})
        result, error = self._models[name].predictionFromFeatures_options_error_(
            features, options, None
        )
        if result is None:
            raise RuntimeError(f"Core ML prediction failed for {name}: {error}")
        return self._out_buffers[name][:count]

    def _predict(self, data: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        outputs = []
        x = data
        for stage, name in enumerate(("c1", "c2", "c3"), start=1):
            conv = self._predict_stage(name, x)
            x = self._mlx_model.tencoder[stage](None, precomputed_conv=conv)
            # Evaluate before the next prediction rewrites this stage's buffer.
            mx.eval(x)
            outputs.append(x)
        return outputs[0], outputs[1], outputs[2]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Experimental HTDemucs Neural Engine convolution")
    parser.add_argument(
        "command", choices=["convert", "placement", "convert-tail", "placement-tail"]
    )
    parser.add_argument("--tile-outputs", type=int, default=DEFAULT_TILE_OUTPUTS)
    args = parser.parse_args(argv)
    if args.command == "convert":
        print(json.dumps(convert(args.tile_outputs), indent=2))
    elif args.command == "convert-tail":
        print(json.dumps(convert_tail(), indent=2))
    elif args.command == "placement":
        print(json.dumps(_device_placement(compiled_path(args.tile_outputs)), indent=2))
    else:
        print(json.dumps(_device_placement(tail_compiled_path()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
