"""Security regressions for Demucs checkpoint and MLX cache loading."""

from __future__ import annotations

import copy
import os
import pickle
import shlex
import tempfile
import types
import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from unittest import mock

import mlx.core as mx
import numpy as np

from demucs_mlx import mlx_convert
from demucs_mlx.mlx_convert import SafeCacheError
from demucs_mlx.mlx_registry import MLX_MODEL_REGISTRY
from demucs_mlx.secure_demucs import (
    _construct_validated_model,
    _load_package_from_url,
    _validate_package,
    expected_official_sources,
)

try:
    import torch
    from demucs.demucs import Demucs
except ImportError:
    torch = None
    Demucs = None


SOURCES = ["drums", "bass", "other", "vocals"]


class _ExecutablePayload:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        command = f"printf executed > {shlex.quote(str(self.marker))}"
        return os.system, (command,)


class _FakeHTDemucsMLX:
    constructions = 0

    def __init__(
        self,
        sources: list[str],
        ratio: Fraction = Fraction(1, 2),
        segment: int = 10,
    ) -> None:
        type(self).constructions += 1
        self.sources = sources
        self.ratio = ratio
        self.segment = segment
        self.samplerate = 44_100
        self.audio_channels = 2
        self.weight = mx.zeros((2,), dtype=mx.float32)
        self.training = True

    def state_dict(self):
        return {"weight": self.weight}

    def update(self, state):
        self.weight = state["weight"]

    def eval(self):
        self.training = False
        return self


@contextmanager
def _fake_mlx_class():
    _FakeHTDemucsMLX.constructions = 0
    with mock.patch(
        "demucs_mlx.mlx_htdemucs.HTDemucsMLX",
        _FakeHTDemucsMLX,
    ):
        yield


def _config(
    model_name: str = "htdemucs",
    *,
    bag: bool = False,
    ratio: Fraction = Fraction(1, 2),
) -> dict:
    classes = list(MLX_MODEL_REGISTRY[model_name]["model_classes"])
    kwargs = [{"sources": list(SOURCES), "ratio": ratio} for _ in classes]
    sources = expected_official_sources(model_name)
    return {
        "format_version": mlx_convert.SAFE_CACHE_FORMAT_VERSION,
        "model_name": model_name,
        "model_class": "BagOfModelsMLX" if bag else classes[0],
        "sub_model_class": classes[0] if bag else None,
        "args": [],
        "kwargs": copy.deepcopy(kwargs[0]),
        "per_model_args": [[] for _ in classes],
        "per_model_kwargs": kwargs,
        "per_model_classes": classes,
        "num_models": len(classes),
        "weights": [[1.0] * len(SOURCES) for _ in classes] if bag else None,
        "source_artifacts": [
            {"signature": source.signature, "checksum": source.checksum} for source in sources
        ],
        "mlx_version": mlx_convert._runtime_mlx_version(),
        "conversion_date": datetime.now(timezone.utc).isoformat(),
        "verification_passed": True,
        "safetensors_sha256": "0" * 64,
    }


def _weights(config: dict) -> dict[str, mx.array]:
    if config["model_class"] == "BagOfModelsMLX":
        return {
            f"model_{index}.weight": mx.array([index, index + 1], dtype=mx.float32)
            for index in range(config["num_models"])
        }
    return {"weight": mx.array([3, 4], dtype=mx.float32)}


def _write_safe_cache(cache: Path, config: dict) -> None:
    mlx_convert._save_safe_cache(
        config["model_name"],
        str(cache),
        _weights(config),
        config,
    )


def _demucs_package():
    assert torch is not None and Demucs is not None
    model = Demucs(
        sources=["left", "right"],
        audio_channels=1,
        channels=4,
        depth=2,
        rewrite=False,
        dconv_mode=0,
        resample=False,
        normalize=False,
    )
    args, kwargs = getattr(model, "_init_args_kwargs")
    return {
        "klass": Demucs,
        "args": args,
        "kwargs": kwargs,
        "state": model.state_dict(),
        "training_args": {"fraction": Fraction(1, 3), "epoch": np.int64(7)},
        "metrics": {"loss": np.float64(0.5)},
    }


class RestrictedCheckpointTests(unittest.TestCase):
    def test_torch_2_5_is_rejected_before_download(self) -> None:
        download = mock.Mock()
        fake_torch = types.SimpleNamespace(
            __version__="2.5.1",
            hub=types.SimpleNamespace(load_state_dict_from_url=download),
        )
        with self.assertRaisesRegex(RuntimeError, "PyTorch 2.6 or newer"):
            _load_package_from_url("https://example.invalid/model-deadbeef.th", fake_torch)
        download.assert_not_called()

    @unittest.skipUnless(torch is not None, "conversion dependencies are not installed")
    def test_checkpoint_payload_is_rejected_without_execution(self) -> None:
        assert torch is not None
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "executed"
            checkpoint = root / "payload-deadbeef.th"
            torch.save(_ExecutablePayload(marker), checkpoint)
            seen = {}

            def fake_download(url, **kwargs):
                seen.update(kwargs)
                load_kwargs = {key: value for key, value in kwargs.items() if key != "check_hash"}
                return torch.load(checkpoint, **load_kwargs)

            with mock.patch.object(
                torch.hub,
                "load_state_dict_from_url",
                side_effect=fake_download,
            ):
                with self.assertRaises(RuntimeError):
                    _load_package_from_url(str(checkpoint), torch)

            self.assertFalse(marker.exists())
            self.assertIs(seen["weights_only"], True)
            self.assertIs(seen["check_hash"], True)
            self.assertEqual(seen["map_location"], "cpu")

    @unittest.skipUnless(torch is not None, "conversion dependencies are not installed")
    def test_allowed_package_reconstructs_through_restricted_load(self) -> None:
        assert torch is not None and Demucs is not None
        package = _demucs_package()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "demucs-deadbeef.th"
            torch.save(package, checkpoint)

            def fake_download(url, **kwargs):
                load_kwargs = {key: value for key, value in kwargs.items() if key != "check_hash"}
                return torch.load(checkpoint, **load_kwargs)

            with mock.patch.object(
                torch.hub,
                "load_state_dict_from_url",
                side_effect=fake_download,
            ):
                loaded_package = _load_package_from_url(str(checkpoint), torch)
            loaded_model = _construct_validated_model(loaded_package, torch)

        self.assertIs(type(loaded_model), Demucs)
        self.assertEqual(loaded_model.sources, ["left", "right"])
        self.assertEqual(set(loaded_model.state_dict()), set(package["state"]))

    @unittest.skipUnless(torch is not None, "conversion dependencies are not installed")
    def test_unknown_class_and_malformed_packages_fail_closed(self) -> None:
        assert torch is not None and Demucs is not None
        package = _demucs_package()

        class UnknownDemucs(Demucs):
            pass

        bad_class = dict(package, klass=UnknownDemucs)
        with self.assertRaisesRegex(ValueError, "Unsupported Demucs model class"):
            _validate_package(bad_class, torch)

        unexpected = dict(package, executable_fallback=True)
        with self.assertRaisesRegex(ValueError, "unexpected keys"):
            _validate_package(unexpected, torch)

        bad_constructor = copy.copy(package)
        bad_constructor["kwargs"] = dict(package["kwargs"], unknown_argument=True)
        with self.assertRaisesRegex(ValueError, "constructor"):
            _validate_package(bad_constructor, torch)

        bad_state = copy.copy(package)
        bad_state["state"] = {"weight": _ExecutablePayload(Path("unused"))}
        with self.assertRaisesRegex(ValueError, "non-tensor"):
            _validate_package(bad_state, torch)

        bad_quantized = copy.copy(package)
        bad_quantized["state"] = {"__quantized": True, "unexpected": []}
        with self.assertRaisesRegex(ValueError, "quantized state"):
            _validate_package(bad_quantized, torch)


class SafeCacheTests(unittest.TestCase):
    def test_single_model_nullable_class_and_fraction_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, _fake_mlx_class():
            cache = Path(temporary)
            config = _config(ratio=Fraction(7, 11))
            _write_safe_cache(cache, config)
            loaded_config = mlx_convert._load_safe_cache_config(
                cache / "htdemucs_config.json", "htdemucs"
            )
            model = mlx_convert.load_mlx_model("htdemucs", cache_dir=str(cache), auto_convert=False)

        self.assertIsNone(loaded_config["sub_model_class"])
        self.assertEqual(loaded_config["kwargs"]["ratio"], Fraction(7, 11))
        self.assertEqual(model.ratio, Fraction(7, 11))
        self.assertEqual(model.weight.tolist(), [3.0, 4.0])
        self.assertEqual(
            loaded_config["mlx_version"],
            mlx_convert._runtime_mlx_version(),
        )

    def test_single_model_bag_and_ensemble_round_trip(self) -> None:
        with _fake_mlx_class():
            one_model_bag = _config(bag=True)
            validated = mlx_convert._validate_safe_cache_config(
                mlx_convert._encode_json_value(one_model_bag), "htdemucs"
            )
            self.assertEqual(validated["num_models"], 1)
            self.assertEqual(validated["weights"], [[1.0] * 4])

            ensemble = _config("htdemucs_ft", bag=True)
            decoded = mlx_convert._validate_safe_cache_config(
                mlx_convert._encode_json_value(ensemble), "htdemucs_ft"
            )
            self.assertEqual(decoded["num_models"], 4)
            self.assertEqual(len(decoded["source_artifacts"]), 4)
            self.assertEqual(decoded["per_model_classes"], ["HTDemucsMLX"] * 4)

    def test_one_model_bag_loads_from_resolved_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, _fake_mlx_class():
            cache = Path(temporary)
            config = _config(bag=True)
            _write_safe_cache(cache, config)
            model = mlx_convert.load_mlx_model("htdemucs", cache_dir=str(cache), auto_convert=False)
        self.assertIsInstance(model, mlx_convert.BagOfModelsMLX)
        self.assertEqual(len(model.models), 1)

    def test_legacy_pickle_is_never_opened_or_executed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            marker = cache / "executed"
            legacy = cache / "htdemucs_mlx.pkl"
            original = pickle.dumps(_ExecutablePayload(marker))
            legacy.write_bytes(original)
            expected_command = f"python -m demucs_mlx.mlx_convert htdemucs --output-dir {cache}"

            with self.assertWarns(FutureWarning):
                with self.assertRaisesRegex(FileNotFoundError, "legacy") as raised:
                    mlx_convert.load_mlx_model(
                        "htdemucs",
                        cache_dir=str(cache),
                        auto_convert=False,
                    )

            self.assertIn(str(legacy), str(raised.exception))
            self.assertIn(expected_command, str(raised.exception))
            self.assertFalse(marker.exists())
            self.assertEqual(legacy.read_bytes(), original)

    def test_legacy_only_cache_regenerates_safe_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, _fake_mlx_class():
            cache = Path(temporary)
            marker = cache / "executed"
            legacy = cache / "htdemucs_mlx.pkl"
            original = pickle.dumps(_ExecutablePayload(marker))
            legacy.write_bytes(original)

            def regenerate(model_name, output_dir, **kwargs):
                self.assertEqual(model_name, "htdemucs")
                _write_safe_cache(Path(output_dir), _config())
                return str(Path(output_dir) / "htdemucs.safetensors")

            with (
                mock.patch.object(
                    mlx_convert,
                    "convert_htdemucs_weights",
                    side_effect=regenerate,
                ) as converter,
                self.assertWarns(FutureWarning),
            ):
                model = mlx_convert.load_mlx_model(
                    "htdemucs",
                    cache_dir=str(cache),
                    auto_convert=True,
                )

            converter.assert_called_once()
            self.assertIsInstance(model, _FakeHTDemucsMLX)
            self.assertFalse(marker.exists())
            self.assertEqual(legacy.read_bytes(), original)
            self.assertTrue((cache / "htdemucs.safetensors").is_file())
            self.assertTrue((cache / "htdemucs_config.json").is_file())
            self.assertEqual(list(cache.glob("*.pkl")), [legacy])

    def test_incomplete_safe_cache_never_downgrades_to_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            marker = cache / "executed"
            (cache / "htdemucs.safetensors").write_bytes(b"incomplete")
            (cache / "htdemucs_mlx.pkl").write_bytes(pickle.dumps(_ExecutablePayload(marker)))
            with (
                mock.patch.object(mlx_convert, "convert_htdemucs_weights") as converter,
                mock.patch.object(mlx_convert.mx, "load") as array_loader,
                self.assertRaisesRegex(SafeCacheError, "Incomplete"),
            ):
                mlx_convert.load_mlx_model(
                    "htdemucs",
                    cache_dir=str(cache),
                    auto_convert=True,
                )
            converter.assert_not_called()
            array_loader.assert_not_called()
            self.assertFalse(marker.exists())

    def test_digest_failure_never_downgrades_to_legacy_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, _fake_mlx_class():
            cache = Path(temporary)
            marker = cache / "executed"
            _write_safe_cache(cache, _config())
            with (cache / "htdemucs.safetensors").open("ab") as handle:
                handle.write(b"tampered")
            (cache / "htdemucs_mlx.pkl").write_bytes(pickle.dumps(_ExecutablePayload(marker)))
            _FakeHTDemucsMLX.constructions = 0
            with (
                mock.patch.object(mlx_convert.mx, "load") as array_loader,
                mock.patch.object(mlx_convert, "convert_htdemucs_weights") as converter,
                self.assertRaisesRegex(SafeCacheError, "digest mismatch"),
            ):
                mlx_convert.load_mlx_model("htdemucs", cache_dir=str(cache), auto_convert=True)
            array_loader.assert_not_called()
            converter.assert_not_called()
            self.assertEqual(_FakeHTDemucsMLX.constructions, 0)
            self.assertFalse(marker.exists())

    def test_invalid_json_schema_class_source_and_weights_are_rejected(self) -> None:
        with _fake_mlx_class():
            unknown_field = _config()
            unknown_field["fallback"] = "pickle"
            with self.assertRaisesRegex(SafeCacheError, "unknown fields"):
                mlx_convert._validate_safe_cache_config(unknown_field, "htdemucs")

            unknown_class = mlx_convert._encode_json_value(_config())
            unknown_class["model_class"] = "ExecutableModel"
            with self.assertRaisesRegex(SafeCacheError, "Unknown MLX model class"):
                mlx_convert._validate_safe_cache_config(unknown_class, "htdemucs")

            wrong_source = mlx_convert._encode_json_value(_config())
            wrong_source["source_artifacts"][0]["checksum"] = "deadbeef"
            with self.assertRaisesRegex(SafeCacheError, "signatures/checksums"):
                mlx_convert._validate_safe_cache_config(wrong_source, "htdemucs")

            zero_weights = mlx_convert._encode_json_value(_config(bag=True))
            zero_weights["weights"] = [[0.0] * 4]
            with self.assertRaisesRegex(SafeCacheError, "non-zero totals"):
                mlx_convert._validate_safe_cache_config(zero_weights, "htdemucs")

    def test_invalid_json_tags_bounds_and_hashes_are_rejected(self) -> None:
        tagged = {"__type__": "callable", "numerator": 1, "denominator": 2}
        with self.assertRaisesRegex(SafeCacheError, "Unknown tagged"):
            mlx_convert._decode_json_value(tagged)

        nested = None
        for _ in range(mlx_convert.SAFE_CACHE_MAX_DEPTH + 2):
            nested = [nested]
        with self.assertRaisesRegex(SafeCacheError, "nested too deeply"):
            mlx_convert._decode_json_value(nested)

        with _fake_mlx_class():
            invalid_hash = mlx_convert._encode_json_value(_config())
            invalid_hash["safetensors_sha256"] = "not-a-digest"
            with self.assertRaisesRegex(SafeCacheError, "SHA-256"):
                mlx_convert._validate_safe_cache_config(invalid_hash, "htdemucs")

    def test_oversized_config_is_rejected_before_json_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "htdemucs_config.json"
            path.write_bytes(b" " * (mlx_convert.SAFE_CACHE_CONFIG_LIMIT + 1))
            with self.assertRaisesRegex(SafeCacheError, "too large"):
                mlx_convert._load_safe_cache_config(path, "htdemucs")


if __name__ == "__main__":
    unittest.main(verbosity=2)
