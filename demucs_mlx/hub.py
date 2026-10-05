"""Download published MLX weights, verified against digests shipped in this package.

Each model is two files: ``<model>.safetensors`` and ``<model>_config.json``.
Their sizes and SHA-256 digests are fixed below, so a download is accepted only
if it is byte-identical to the files this release was built against. Nothing is
written to the cache until both files have been verified.

Environment:

- ``DEMUCS_MLX_NO_DOWNLOAD=1`` or ``HF_HUB_OFFLINE=1`` disables downloading.
- ``HF_ENDPOINT`` selects a Hugging Face mirror.
- ``DEMUCS_MLX_HUB_URL`` replaces the whole base URL (files are fetched as
  ``<base>/<filename>``).
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import typing as tp
import urllib.error
import urllib.request
from pathlib import Path

REPO_ID = "ssmall256/demucs-mlx"
_CHUNK = 1 << 20
_TIMEOUT_SECONDS = 30


class PublishedFile(tp.NamedTuple):
    size: int
    sha256: str


class PublishedModel(tp.NamedTuple):
    weights: PublishedFile
    config: PublishedFile


class DownloadError(RuntimeError):
    """A published model could not be fetched or did not match its digest."""


# Sizes and SHA-256 digests of the files published at
# https://huggingface.co/ssmall256/demucs-mlx. The weights are what
# ``python -m demucs_mlx.mlx_convert <model>`` produces from the official
# Demucs checkpoints.
PUBLISHED: dict[str, PublishedModel] = {
    "htdemucs": PublishedModel(
        PublishedFile(
            168005865, "339d267a7a6983a11eedbdc00413c602a65e9b9103f695fb5c2b2a481cd9d297"
        ),
        PublishedFile(4215, "23657b19db14771aecf366ceedbf846c1386836c10043c97f4569acf05526b78"),
    ),
    "htdemucs_ft": PublishedModel(
        PublishedFile(
            672024519, "53f03b1ad4b4d211025a35da65460ba61a17547adf9c0544cad0ebcc8d7bbabb"
        ),
        PublishedFile(10253, "6708535a698d4d4dd285a6f0a80d9d3c228fdd6514e53bdf1ddbc2ee9b0fa872"),
    ),
    "htdemucs_6s": PublishedModel(
        PublishedFile(
            109726583, "d298f7f746bf53c21baad44fb08e88807ef47feb551dd22f1601a546c85b8e02"
        ),
        PublishedFile(4302, "d5e18c0209be583027d6eb5ec3013a02cd7ef55bc3096e644d8d3577d92e9bd0"),
    ),
    "hdemucs_mmi": PublishedModel(
        PublishedFile(
            334522864, "39f359110433930c2a589131f84c03c26bbd209e89e10e6abbcb6062c131debc"
        ),
        PublishedFile(2400, "0a3a645fc281824d8b38077bf081afdae625eab7728b68f66e334e1751bfb938"),
    ),
    "mdx": PublishedModel(
        PublishedFile(
            1381657640, "c95dab261c766fc50caadcd047aa2c125759b9b48efdac3f2aaab0fa35d8c41f"
        ),
        PublishedFile(4834, "ea7f81dce21e668e91c1ec36f8e11fa138d4828b6b622e0b4e9001afba7357a7"),
    ),
    "mdx_q": PublishedModel(
        PublishedFile(
            1381657640, "d7f31edb6b37b5ee391d104e1f88cb70c56ca82e3f6f9e8f4f3cd2df6e9bddfc"
        ),
        PublishedFile(4836, "6f976a1c96128daa7bf4652399e49cfffd7bb68ba156965533f4d880699fa7a5"),
    ),
    "mdx_extra": PublishedModel(
        PublishedFile(
            1338062104, "d1c969aa0a69417e767b23f97d10df944a4c9883febe853ab6d91ad0cb2276fe"
        ),
        PublishedFile(5602, "d925ba3dbad48ccc99d48e151e5d96a0fcc71ef57bd3862f4cb46d5a71c0db11"),
    ),
    "mdx_extra_q": PublishedModel(
        PublishedFile(
            1338062104, "82310bf4d1f32b8044cba6c192af77ab8d12a0acdedd7bf841caa78a61bd5839"
        ),
        PublishedFile(5599, "a45609a3cefd6fd69fb5058046a4ed33716f11ec313237146372fbe39a583530"),
    ),
}


def downloads_enabled() -> bool:
    """False when the environment asks for offline operation."""
    on = {"1", "true", "yes", "on"}
    return not any(
        os.getenv(name, "").strip().lower() in on
        for name in ("DEMUCS_MLX_NO_DOWNLOAD", "HF_HUB_OFFLINE")
    )


def base_url() -> str:
    override = os.getenv("DEMUCS_MLX_HUB_URL", "").strip()
    if override:
        return override.rstrip("/")
    endpoint = os.getenv("HF_ENDPOINT", "").strip().rstrip("/") or "https://huggingface.co"
    return f"{endpoint}/{REPO_ID}/resolve/main"


def _user_agent() -> str:
    from . import __version__

    return f"demucs-mlx/{__version__}"


def _fetch(url: str, expected: PublishedFile, directory: Path, label: str, progress: bool) -> Path:
    """Stream ``url`` to a temporary file in ``directory`` and verify it."""
    request = urllib.request.Request(url, headers={"User-Agent": _user_agent()})
    digest = hashlib.sha256()
    received = 0
    handle, name = tempfile.mkstemp(prefix=".download-", suffix=".part", dir=directory)
    path = Path(name)
    bar = None
    try:
        with os.fdopen(handle, "wb") as out, urllib.request.urlopen(
            request, timeout=_TIMEOUT_SECONDS
        ) as response:
            if progress:
                from tqdm import tqdm

                bar = tqdm(
                    total=expected.size, unit="B", unit_scale=True, desc=label, disable=None
                )
            while True:
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                received += len(chunk)
                if received > expected.size:
                    raise DownloadError(f"{label} is larger than the published {expected.size} B")
                digest.update(chunk)
                out.write(chunk)
                if bar is not None:
                    bar.update(len(chunk))
        if received != expected.size:
            raise DownloadError(
                f"{label} is {received} B; the published file is {expected.size} B"
            )
        if digest.hexdigest() != expected.sha256:
            raise DownloadError(f"{label} does not match its published SHA-256")
        return path
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        if bar is not None:
            bar.close()


def download_model(
    name: str, cache_dir: tp.Union[str, Path], *, progress: bool = True
) -> Path:
    """Fetch and verify one published model into ``cache_dir``.

    Returns the path of the weights. Raises :class:`DownloadError` if the model
    is not published, the network fails, or either file fails verification; in
    every failure case the cache directory is left as it was.
    """
    published = PUBLISHED.get(name)
    if published is None:
        raise DownloadError(f"No published MLX weights for {name!r}")
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    base = base_url()
    weights_name = f"{name}.safetensors"
    config_name = f"{name}_config.json"
    if progress:
        megabytes = published.weights.size / (1 << 20)
        print(
            f"Downloading {name} MLX weights ({megabytes:,.0f} MB) from {base} ...",
            file=sys.stderr,
            flush=True,
        )
    staged: list[Path] = []
    try:
        try:
            staged.append(
                _fetch(f"{base}/{config_name}", published.config, directory, config_name, False)
            )
            staged.append(
                _fetch(
                    f"{base}/{weights_name}", published.weights, directory, weights_name, progress
                )
            )
        except DownloadError:
            raise
        except (urllib.error.URLError, OSError, ValueError) as error:
            raise DownloadError(f"Could not download {name}: {error}") from error
        # Weights first: a crash between the two renames leaves an incomplete
        # pair, which the loader rejects, rather than a config without weights.
        os.replace(staged[1], directory / weights_name)
        os.replace(staged[0], directory / config_name)
        staged.clear()
    finally:
        for leftover in staged:
            leftover.unlink(missing_ok=True)
    return directory / weights_name


def verify_directory(directory: tp.Union[str, Path]) -> list[str]:
    """Compare a directory with the published digests; return the problems found."""
    root = Path(directory)
    problems: list[str] = []
    for name, published in PUBLISHED.items():
        for filename, expected in (
            (f"{name}.safetensors", published.weights),
            (f"{name}_config.json", published.config),
        ):
            path = root / filename
            if not path.is_file():
                problems.append(f"{filename}: missing")
                continue
            if path.stat().st_size != expected.size:
                problems.append(f"{filename}: size {path.stat().st_size}, expected {expected.size}")
                continue
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(_CHUNK):
                    digest.update(chunk)
            if digest.hexdigest() != expected.sha256:
                problems.append(f"{filename}: SHA-256 mismatch")
    return problems


def main(argv: tp.Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download or check the published demucs-mlx weights"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download models into a directory")
    download.add_argument("models", nargs="*", help="Model names (default: all)")
    download.add_argument("--output-dir", required=True)
    verify = commands.add_parser(
        "verify", help="Check that a directory matches the published digests"
    )
    verify.add_argument("directory")
    args = parser.parse_args(argv)
    if args.command == "verify":
        problems = verify_directory(args.directory)
        for problem in problems:
            print(problem)
        if not problems:
            print(f"All {len(PUBLISHED)} models match the published digests.")
        return 1 if problems else 0
    for name in args.models or list(PUBLISHED):
        try:
            print(download_model(name, args.output_dir))
        except DownloadError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
