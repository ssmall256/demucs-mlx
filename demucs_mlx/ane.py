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
from concurrent.futures import Future
from pathlib import Path

import numpy as np

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
    return Path.home() / ".cache" / "demucs-mlx" / "ane"


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
        from demucs.apply import BagOfModels
    except ImportError as exc:
        raise RuntimeError(
            "Conversion needs coremltools 9, PyTorch, and Demucs; install the "
            "ane-convert extra"
        ) from exc

    tile_count = _tile_count(tile_outputs)
    digest = _cache_identity()
    from .secure_demucs import get_restricted_demucs_model

    restricted = get_restricted_demucs_model(MODEL_NAME)
    source = restricted.model
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
        inputs=[ct.TensorType(name="mix", shape=tuple(example.shape), dtype=np.float32)],
        outputs=[ct.TensorType(name=OUTPUT_NAME, dtype=np.float16)],
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
        from demucs.apply import BagOfModels
    except ImportError as exc:
        raise RuntimeError(
            "Conversion needs coremltools 9, PyTorch, and Demucs; install the "
            "ane-convert extra"
        ) from exc

    digest = _cache_identity()
    from .secure_demucs import get_restricted_demucs_model

    restricted = get_restricted_demucs_model(MODEL_NAME)
    source = restricted.model
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
            inputs=[ct.TensorType(name="x", shape=shape, dtype=np.float32)],
            outputs=[ct.TensorType(name="y", dtype=np.float16)],
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
                "`python -m demucs_mlx.ane convert` first"
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
            raise RuntimeError("Install demucs-mlx[ane] for the Core ML runtime") from exc

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
        in_desc = desc.inputDescriptionsByName().get(input_name)
        if in_desc and in_desc.multiArrayConstraint():
            in_shape = tuple(int(s) for s in in_desc.multiArrayConstraint().shape())
        else:
            in_shape = (BATCH, 2, LENGTH)
        self._in_pad_buf = np.empty(in_shape, dtype=np.float32)

        self._has_output_backings = hasattr(CoreML.MLPredictionOptions, "setOutputBackings_")
        self._output_specs: dict[str, tuple[tuple[int, ...], int, type]] = {}
        out_descs = desc.outputDescriptionsByName()
        for name in output_names:
            fdesc = out_descs.get(name)
            if fdesc and fdesc.multiArrayConstraint():
                c = fdesc.multiArrayConstraint()
                shape = tuple(int(s) for s in c.shape())
                c_dtype = c.dataType()
                np_dtype = np.float16 if c_dtype == CoreML.MLMultiArrayDataTypeFloat16 else np.float32
                self._output_specs[name] = (shape, c_dtype, np_dtype)
            else:
                self._has_output_backings = False

        self.model = model
        self.busy_seconds = 0.0
        self.wait_seconds = 0.0
        self.transfer_seconds = 0.0
        self.predictions = 0
        self._jobs: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True, name="demucs-ane")
        self._worker.start()

    def submit(self, mix: np.ndarray | tp.Any) -> Future:
        if not isinstance(mix, np.ndarray):
            mix = np.array(mix, copy=False)
        if mix.ndim != 3 or mix.shape[1:] != (2, LENGTH) or mix.shape[0] not in (1, 2):
            raise ValueError(f"ANE convolution expects (1 or 2, 2, {LENGTH}), got {mix.shape}")
        return self._submit(mix)

    def _submit(self, mix: np.ndarray) -> Future:
        data = np.ascontiguousarray(mix, dtype=np.float32)
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

    def _predict(self, data: np.ndarray):
        coreml = self._coreml
        count = len(data)
        if count == 1:
            self._in_pad_buf[0] = data[0]
            self._in_pad_buf[1] = data[0]
            data = self._in_pad_buf
        elif not data.flags.c_contiguous:
            data = np.ascontiguousarray(data)

        init_array = (
            coreml.MLMultiArray.alloc()
            .initWithDataPointer_shape_dataType_strides_deallocator_error_
        )
        array, error = init_array(
            data, list(data.shape), coreml.MLMultiArrayDataTypeFloat32,
            [stride // data.itemsize for stride in data.strides], None, None,
        )
        if array is None:
            raise RuntimeError(f"Could not create Core ML input array: {error}")
        features, error = coreml.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(
            {self._input_name: coreml.MLFeatureValue.featureValueWithMultiArray_(array)}, None
        )
        if features is None:
            raise RuntimeError(f"Could not create Core ML input features: {error}")

        if self._has_output_backings:
            output_buffers = {}
            backings = {}
            for name in self._output_names:
                shape, c_dtype, np_dtype = self._output_specs[name]
                out_buf = np.empty(shape, dtype=np_dtype)
                output_buffers[name] = out_buf
                strides = [s // out_buf.itemsize for s in out_buf.strides]
                ma, err = (
                    coreml.MLMultiArray.alloc()
                    .initWithDataPointer_shape_dataType_strides_deallocator_error_(
                        out_buf, list(shape), c_dtype, strides, None, None
                    )
                )
                if ma is None:
                    raise RuntimeError(f"Could not create Core ML output backing for {name}: {err}")
                backings[name] = ma
            options = coreml.MLPredictionOptions.alloc().init()
            options.setOutputBackings_(backings)
            result, error = self.model.predictionFromFeatures_options_error_(features, options, None)
            if result is None:
                raise RuntimeError(f"Core ML prediction with output backings failed: {error}")
            outputs = [output_buffers[name][:count] for name in self._output_names]
            return outputs[0] if len(outputs) == 1 else tuple(outputs)

        result, error = self.model.predictionFromFeatures_error_(features, None)
        if result is None:
            raise RuntimeError(f"Core ML prediction failed: {error}")
        outputs = []
        for name in self._output_names:
            y = result.featureValueForName_(name).multiArrayValue()
            dtype = {
                coreml.MLMultiArrayDataTypeFloat16: np.float16,
                coreml.MLMultiArrayDataTypeFloat32: np.float32,
            }[y.dataType()]
            shape = tuple(int(s) for s in y.shape())
            strides = tuple(int(s) for s in y.strides())
            dp = y.dataPointer()
            if hasattr(dp, "as_buffer"):
                raw = dp.as_buffer(y.count() * np.dtype(dtype).itemsize)
                flat = np.frombuffer(raw, dtype=dtype)
            else:
                held = {}
                def grab(raw_bytes, size):
                    held["flat"] = np.frombuffer(
                        raw_bytes, dtype=dtype, count=size // np.dtype(dtype).itemsize
                    ).copy()
                y.getBytesWithHandler_(grab)
                flat = held["flat"]
            view = np.lib.stride_tricks.as_strided(
                flat, shape, [stride * flat.itemsize for stride in strides]
            )
            outputs.append(np.ascontiguousarray(view[:count]))
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


class WaveformTail:
    """High-ROI mixed-precision waveform tail with ANE downsampling and MLX layers."""

    def __init__(self):
        if sys.platform != "darwin" or platform.machine() != "arm64":
            raise RuntimeError("The Neural Engine prototype requires Apple Silicon macOS")
        self.path = tail_compiled_path()
        if not self.path.is_dir() or not tail_manifest_path().is_file():
            raise FileNotFoundError(
                "Converted waveform tail missing; run "
                "`python -m demucs_mlx.ane convert-tail` first"
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
            raise RuntimeError("Install demucs-mlx[ane] for the Core ML runtime") from exc

        self.placement = manifest["placement"]
        self._coreml = CoreML

        import mlx.core as mx
        import mlx.utils

        from .model_converter import get_mlx_model
        self._mlx_model = get_mlx_model(MODEL_NAME).models[0]
        for _, p in mlx.utils.tree_flatten(self._mlx_model.parameters()):
            mx.eval(p)

        config = CoreML.MLModelConfiguration.alloc().init()
        config.setComputeUnits_(CoreML.MLComputeUnitsCPUAndNeuralEngine)

        self._models = {}
        self._out_buffers = {}
        self._out_backings = {}
        self._in_pad_bufs = {}

        shapes = {
            "c1": ((BATCH, 48, 85_995), (BATCH, 96, 21_499)),
            "c2": ((BATCH, 96, 21_499), (BATCH, 192, 5_375)),
            "c3": ((BATCH, 192, 5_375), (BATCH, 384, 1_344)),
        }

        for name, (in_sh, out_sh) in shapes.items():
            model_url = Foundation.NSURL.fileURLWithPath_(str(self.path / f"{name}.mlmodelc"))
            m, err = CoreML.MLModel.modelWithContentsOfURL_configuration_error_(model_url, config, None)
            if m is None:
                raise RuntimeError(f"Could not load Core ML model {name}: {err}")
            self._models[name] = m
            self._in_pad_bufs[name] = np.empty(in_sh, dtype=np.float32)

            out_buf = np.empty(out_sh, dtype=np.float16)
            self._out_buffers[name] = out_buf
            strides = [s // out_buf.itemsize for s in out_buf.strides]
            ma, err = (
                CoreML.MLMultiArray.alloc()
                .initWithDataPointer_shape_dataType_strides_deallocator_error_(
                    out_buf, list(out_sh), CoreML.MLMultiArrayDataTypeFloat16, strides, None, None
                )
            )
            if ma is None:
                raise RuntimeError(f"Could not create output backing for {name}: {err}")
            self._out_backings[name] = ma

        self.busy_seconds = 0.0
        self.wait_seconds = 0.0
        self.transfer_seconds = 0.0
        self.predictions = 0
        self._jobs: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True, name="demucs-ane-tail")
        self._worker.start()

    def submit(self, encoded: np.ndarray | tp.Any) -> Future:
        if not isinstance(encoded, np.ndarray):
            encoded = np.array(encoded, copy=False)
        if encoded.ndim != 3 or encoded.shape[1:] != (48, 85_995) or encoded.shape[0] not in (1, 2):
            raise ValueError(
                f"ANE waveform tail expects (1 or 2, 48, 85995), got {encoded.shape}"
            )
        data = np.ascontiguousarray(encoded, dtype=np.float32)
        future: Future = Future()
        self._jobs.put((data, future))
        return future

    def close(self) -> None:
        if self._worker.is_alive():
            self._jobs.put(None)
            self._worker.join()

    def _run(self) -> None:
        import mlx.core as mx

        with mx.stream(mx.default_stream(mx.default_device())):
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

    def _predict_stage(self, name: str, data: np.ndarray) -> np.ndarray:
        coreml = self._coreml
        count = len(data)
        if count == 1:
            buf = self._in_pad_bufs[name]
            buf[0] = data[0]
            buf[1] = data[0]
            data = buf
        elif not data.flags.c_contiguous:
            data = np.ascontiguousarray(data)

        init_array = (
            coreml.MLMultiArray.alloc()
            .initWithDataPointer_shape_dataType_strides_deallocator_error_
        )
        array, error = init_array(
            data, list(data.shape), coreml.MLMultiArrayDataTypeFloat32,
            [stride // data.itemsize for stride in data.strides], None, None,
        )
        if array is None:
            raise RuntimeError(f"Could not create Core ML input array for {name}: {error}")
        features, error = coreml.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(
            {"x": coreml.MLFeatureValue.featureValueWithMultiArray_(array)}, None
        )
        if features is None:
            raise RuntimeError(f"Could not create Core ML input features for {name}: {error}")

        options = coreml.MLPredictionOptions.alloc().init()
        options.setOutputBackings_({"y": self._out_backings[name]})
        result, error = self._models[name].predictionFromFeatures_options_error_(features, options, None)
        if result is None:
            raise RuntimeError(f"Core ML prediction failed for {name}: {error}")
        return self._out_buffers[name][:count]

    def _predict(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import mlx.core as mx

        # Stage 1: ANE conv1 + MLX layer
        c1 = self._predict_stage("c1", data)
        c1_mx = mx.asarray(c1, copy=False)
        y1 = self._mlx_model.tencoder[1](None, precomputed_conv=c1_mx)
        mx.eval(y1)
        y1_np = np.asarray(y1)

        # Stage 2: ANE conv2 + MLX layer
        c2 = self._predict_stage("c2", y1_np)
        c2_mx = mx.asarray(c2, copy=False)
        y2 = self._mlx_model.tencoder[2](None, precomputed_conv=c2_mx)
        mx.eval(y2)
        y2_np = np.asarray(y2)

        # Stage 3: ANE conv3 + MLX layer
        c3 = self._predict_stage("c3", y2_np)
        c3_mx = mx.asarray(c3, copy=False)
        y3 = self._mlx_model.tencoder[3](None, precomputed_conv=c3_mx)
        mx.eval(y3)
        y3_np = np.asarray(y3)

        return y1_np, y2_np, y3_np


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
