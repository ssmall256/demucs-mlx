"""Experimental Core ML waveform encoder for the default HTDemucs model.

Conversion uses the restricted official PyTorch loader. Inference needs only
PyObjC: Core ML runs on a worker while MLX evaluates the spectral branch.
"""

from __future__ import annotations

import argparse
import json
import queue
import shutil
import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path

import numpy as np

MODEL_NAME = "htdemucs"
BATCH = 2
LENGTH = 343_980  # 7.8 seconds at 44.1 kHz, the official training segment.
OUTPUT_NAMES = ("y0", "y1", "y2", "y3")


def asset_dir() -> Path:
    return Path.home() / ".cache" / "demucs-mlx" / "ane"


def compiled_path() -> Path:
    return asset_dir() / "htdemucs_time_encoder_b2.mlmodelc"


def manifest_path() -> Path:
    return asset_dir() / "htdemucs_time_encoder_b2.json"


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


def _torch_encoder(torch_model):
    import torch

    class TorchWaveformEncoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.layers = model.tencoder

        def forward(self, mix):
            # Match mx.std's population variance, rather than torch.std's
            # default Bessel correction.
            mean = mix.mean(dim=(1, 2), keepdim=True)
            std = mix.std(dim=(1, 2), keepdim=True, unbiased=False)
            x = (mix - mean) / (1e-5 + std)
            outputs = []
            for index, layer in enumerate(self.layers):
                # The official fixed 7.8 s shapes are divisible by four only
                # at the first stage. Spell out the three known one-sample
                # pads so TorchScript does not trace dynamic shape -> int ops,
                # which Core ML Tools cannot translate for this model.
                if index:
                    x = torch.nn.functional.pad(x, (0, 1))
                x = layer(x)
                outputs.append(x)
            return tuple(outputs)

    return TorchWaveformEncoder(torch_model).eval()


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


def convert() -> dict:
    """Convert the official default model and verify ANE placement."""
    if sys.platform != "darwin":
        raise RuntimeError("The Neural Engine prototype requires macOS")
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
    length = int(torch_model.segment * torch_model.samplerate)
    if length != LENGTH or torch_model.audio_channels != 2 or len(torch_model.tencoder) != 4:
        raise RuntimeError("Unexpected HTDemucs waveform encoder shape")

    wrapper = _torch_encoder(torch_model)
    example = torch.zeros((BATCH, 2, LENGTH), dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, check_trace=False)
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="mix", shape=tuple(example.shape), dtype=np.float32)],
        outputs=[ct.TensorType(name=name, dtype=np.float16) for name in OUTPUT_NAMES],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    target = compiled_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    package = target.with_suffix(".mlpackage")
    ml.save(str(package))
    built = Path(ct.models.utils.compile_model(str(package)))
    placement = _device_placement(built)
    manifest_path().unlink(missing_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(built, target)
    ane_preferred = any("NeuralEngine" in device for device in placement["operations"])
    manifest = {
        "model": MODEL_NAME,
        "batch": BATCH,
        "length": LENGTH,
        "safetensors_sha256": digest,
        "placement": placement,
        "ane_preferred": ane_preferred,
    }
    temporary_manifest = manifest_path().with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary_manifest.replace(manifest_path())
    if not ane_preferred:
        raise RuntimeError(
            "Core ML converted the encoder, but its compute plan chose CPU for "
            f"every operation. Diagnostic assets are at {target}; placement: {placement}"
        )
    return manifest


class WaveformEncoder:
    """One Core ML prediction at a time on a PyObjC worker thread."""

    def __init__(self):
        if sys.platform != "darwin":
            raise RuntimeError("The Neural Engine prototype requires macOS")
        try:
            import CoreML
            import Foundation
        except ImportError as exc:
            raise RuntimeError("Install demucs-mlx[ane] for the Core ML runtime") from exc

        self.path = compiled_path()
        if not self.path.is_dir() or not manifest_path().is_file():
            raise FileNotFoundError(
                "Converted waveform encoder missing; run "
                "`python -m demucs_mlx.ane convert` first"
            )
        manifest = json.loads(manifest_path().read_text())
        if (
            manifest.get("model") != MODEL_NAME
            or manifest.get("batch") != BATCH
            or manifest.get("length") != LENGTH
            or manifest.get("safetensors_sha256") != _cache_identity()
        ):
            raise RuntimeError("Core ML encoder does not match the validated MLX weights")
        self.placement = manifest["placement"]
        self._coreml = CoreML
        config = CoreML.MLModelConfiguration.alloc().init()
        config.setComputeUnits_(CoreML.MLComputeUnitsCPUAndNeuralEngine)
        model, error = CoreML.MLModel.modelWithContentsOfURL_configuration_error_(
            Foundation.NSURL.fileURLWithPath_(str(self.path)), config, None
        )
        if model is None:
            raise RuntimeError(f"Core ML could not load {self.path}: {error}")
        self.model = model
        self.busy_seconds = 0.0
        self.wait_seconds = 0.0
        self.transfer_seconds = 0.0
        self.predictions = 0
        self._jobs: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True, name="demucs-ane")
        self._worker.start()

    def submit(self, mix: np.ndarray) -> Future:
        if mix.ndim != 3 or mix.shape[1:] != (2, LENGTH) or mix.shape[0] not in (1, 2):
            raise ValueError(f"ANE encoder expects (1 or 2, 2, {LENGTH}), got {mix.shape}")
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

    def _predict(self, data: np.ndarray) -> tuple[np.ndarray, ...]:
        coreml = self._coreml
        count = len(data)
        if count == 1:
            data = np.concatenate((data, data), axis=0)
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
            {"mix": coreml.MLFeatureValue.featureValueWithMultiArray_(array)}, None
        )
        if features is None:
            raise RuntimeError(f"Could not create Core ML input features: {error}")
        result, error = self.model.predictionFromFeatures_error_(features, None)
        if result is None:
            raise RuntimeError(f"Core ML prediction failed: {error}")
        outputs = []
        for name in OUTPUT_NAMES:
            y = result.featureValueForName_(name).multiArrayValue()
            dtype = {
                coreml.MLMultiArrayDataTypeFloat16: np.float16,
                coreml.MLMultiArrayDataTypeFloat32: np.float32,
            }[y.dataType()]
            shape = tuple(int(s) for s in y.shape())
            strides = tuple(int(s) for s in y.strides())
            held = {}

            def grab(raw, size):
                held["flat"] = np.frombuffer(
                    raw, dtype=dtype, count=size // np.dtype(dtype).itemsize
                ).copy()

            y.getBytesWithHandler_(grab)
            flat = held["flat"]
            view = np.lib.stride_tricks.as_strided(
                flat, shape, [stride * flat.itemsize for stride in strides]
            )
            outputs.append(np.ascontiguousarray(view[:count]))
        return tuple(outputs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Experimental HTDemucs Neural Engine encoder")
    parser.add_argument("command", choices=["convert", "placement"])
    args = parser.parse_args(argv)
    if args.command == "convert":
        print(json.dumps(convert(), indent=2))
    else:
        print(json.dumps(_device_placement(compiled_path()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
