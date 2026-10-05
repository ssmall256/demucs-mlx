"""Measure demucs-mlx RTFx: first call, warm calls and sustained use.

RTFx = audio seconds / wall seconds for one ``Separator.separate_tensor`` call
(CPU audio in, NumPy stems out). Definitions used in the report:

- first call: the first separation in a fresh Python process. It includes
  graph compilation, Metal kernel/pipeline setup and MLX buffer allocation.
  Model loading is timed separately ("load s") and excluded.
- warm call: any later separation in the same process on the same audio, after
  compiled graphs, pipelines and MLX's buffer cache already exist.
- sustained: total audio / total wall time over every call of a process run
  back to back (first call included), plus the median of the last five calls
  to show any drift (thermal or otherwise) over the session.

Each process is a fresh interpreter, so first calls are real cold calls. Pass
``--python`` more than once to alternate environments round by round, e.g. two
installed MLX releases. Audio is decoded once by this parent process, so child
environments need only numpy, mlx, mlx-spectro and demucs_mlx.

Examples::

    python benchmarks/bench_rtfx.py --audio "song.m4a" --processes 3 --calls 10
    python benchmarks/bench_rtfx.py --audio song.m4a --processes 3 --calls 10 \\
        --python .venv/bin/python \\
        --python /path/to/other-venv/bin/python
    python benchmarks/bench_rtfx.py --seconds 120 --calls 30 --processes 1   # sustained

Run GPU measurements one at a time (for example through a queue) and close
GPU-heavy apps: other processes on the GPU slow calls 2-10x.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THERMAL = ["nominal", "fair", "serious", "critical"]


def thermal_state() -> str:
    try:
        import Foundation  # pyobjc, present in the project environment

        state = int(Foundation.NSProcessInfo.processInfo().thermalState())
        return THERMAL[state] if state < len(THERMAL) else str(state)
    except Exception:  # noqa: BLE001 - optional diagnostic only
        return "unknown"


def wait_for_nominal(cooldown: float, timeout: float = 600) -> None:
    """Hold until nominal, then for `cooldown` seconds; never inside a timer."""
    started = time.monotonic()
    while thermal_state() not in ("nominal", "unknown"):
        if time.monotonic() - started > timeout:
            raise RuntimeError("thermal state did not return to nominal")
        time.sleep(1)
    time.sleep(cooldown)


def child(args: argparse.Namespace) -> None:
    import numpy as np

    audio = np.load(args.child)
    import mlx.core as mx

    if args.cache:
        import demucs_mlx.model_converter as converter

        cache = Path(args.cache).resolve()
        converter.get_mlx_cache_dir = lambda: cache
    from demucs_mlx import Separator

    batch = args.batch_size if args.batch_size == "auto" else int(args.batch_size)
    compile_flag = {"on": True, "off": False, "default": None}[args.compile]
    started = time.perf_counter()
    sep = Separator(
        model=args.model, seed=args.seed, batch_size=batch,
        attention_precision=args.attention, compile=compile_flag,
    )
    load = time.perf_counter() - started
    seconds = audio.shape[-1] / sep.samplerate
    calls = []
    for index in range(args.calls):
        before = thermal_state()
        started = time.perf_counter()
        _, stems = sep.separate_tensor(audio)
        elapsed = time.perf_counter() - started
        record = {"seconds": elapsed, "rtfx": seconds / elapsed,
                  "thermal": [before, thermal_state()]}
        if index in (0, args.calls - 1):
            stacked = np.stack([stems[name] for name in stems])
            record["stems_sha256"] = hashlib.sha256(stacked.tobytes()).hexdigest()[:16]
        calls.append(record)
    print(json.dumps({
        "mlx": mx.__version__, "python": sys.executable, "audio_seconds": seconds,
        "load_seconds": load, "calls": calls,
        "cache_gb_after": mx.get_cache_memory() / 1e9,
    }), flush=True)


def summarize(run: dict) -> dict:
    calls = run["calls"]
    warm = [c["rtfx"] for c in calls[1:]]
    total = sum(c["seconds"] for c in calls)
    return {
        "first_rtfx": calls[0]["rtfx"],
        "warm_median_rtfx": statistics.median(warm) if warm else None,
        "sustained_rtfx": run["audio_seconds"] * len(calls) / total,
        "late_median_rtfx": statistics.median(c["rtfx"] for c in calls[-5:]),
        "thermal_seen": sorted({s for c in calls for s in c["thermal"]}),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--audio", type=Path, help="Audio file (decoded once with mlx-audio-io)")
    src.add_argument("--fixture", type=Path, help="Safetensors file with an 'audio' tensor")
    src.add_argument("--seconds", type=int, default=120,
                     help="Synthetic stereo audio length when no file is given")
    p.add_argument("--model", default="htdemucs")
    p.add_argument("--batch-size", default="auto")
    p.add_argument("--compile", choices=["default", "on", "off"], default="default")
    p.add_argument("--attention", default="fp16")
    p.add_argument("--seed", type=int, default=481)
    p.add_argument("--calls", type=int, default=10, help="Back-to-back calls per process")
    p.add_argument("--processes", type=int, default=3, help="Fresh processes per environment")
    p.add_argument("--python", action="append", help="Interpreter(s); repeat to alternate")
    p.add_argument("--pythonpath", action="append", default=[],
                   help="Extra import path for child processes (repeatable)")
    p.add_argument("--cache", type=Path, help="Model cache directory (default: demucs-mlx's)")
    p.add_argument("--cooldown", type=float, default=0,
                   help="Seconds of nominal thermal state before each process (untimed)")
    p.add_argument("--json", type=Path, help="Write every measurement here")
    p.add_argument("--child", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.child:
        child(args)
        return
    if args.calls < 1 or args.processes < 1:
        p.error("--calls and --processes must be positive")

    import numpy as np

    if args.audio:
        from demucs_mlx.audio import load_audio

        decoded, _ = load_audio(args.audio, sr=44_100, dtype="float32")
        audio = np.asarray(decoded)
    elif args.fixture:
        import mlx.core as mx

        audio = np.asarray(mx.load(str(args.fixture))["audio"])
    else:
        rng = np.random.default_rng(481)
        t = np.arange(44_100 * args.seconds, dtype=np.float32) / 44_100
        tone = 0.05 * (np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t))
        audio = (tone[None] + 0.01 * rng.standard_normal((2, t.size))).astype(np.float32)
    pythons = args.python or [sys.executable]
    with tempfile.TemporaryDirectory() as temporary:
        audio_path = Path(temporary) / "audio.npy"
        np.save(audio_path, np.ascontiguousarray(audio, dtype=np.float32))
        env = dict(os.environ)
        paths = [str(ROOT), *map(str, args.pythonpath)]
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(paths)
        forwarded = ["--model", args.model, "--batch-size", str(args.batch_size),
                     "--compile", args.compile, "--attention", args.attention,
                     "--seed", str(args.seed), "--calls", str(args.calls)]
        if args.cache:
            forwarded += ["--cache", str(args.cache)]
        runs = []
        print(f"{audio.shape[-1] / 44_100:.1f} s of audio, model {args.model}, batch "
              f"{args.batch_size}, {args.calls} calls/process. RTFx = audio s / wall s.\n")
        print("| round | python | load s | first RTFx | warm median RTFx | sustained RTFx "
              "| last-5 median RTFx | thermal | MLX cache GB |")
        print("|---|---|---:|---:|---:|---:|---:|---|---:|")
        for round_index in range(args.processes):
            order = pythons if round_index % 2 == 0 else list(reversed(pythons))
            for python in order:
                wait_for_nominal(args.cooldown)
                result = subprocess.run(
                    [python, __file__, "--child", str(audio_path), *forwarded],
                    env=env, capture_output=True, text=True,
                )
                if result.returncode:
                    sys.exit(f"{python} failed:\n{result.stderr[-3000:]}")
                run = json.loads(result.stdout.strip().splitlines()[-1])
                run["round"] = round_index + 1
                run["arm"] = python
                run["summary"] = summarize(run)
                runs.append(run)
                s = run["summary"]
                warm = f"{s['warm_median_rtfx']:.1f}" if s["warm_median_rtfx"] else "-"
                print(f"| {round_index + 1} | {python} | {run['load_seconds']:.3f} | "
                      f"{s['first_rtfx']:.1f} | {warm} | {s['sustained_rtfx']:.1f} | "
                      f"{s['late_median_rtfx']:.1f} | {','.join(s['thermal_seen'])} | "
                      f"{run['cache_gb_after']:.1f} |", flush=True)
    print("\nMedians across processes:\n")
    print("| python | MLX | first RTFx | warm RTFx | sustained RTFx |")
    print("|---|---|---:|---:|---:|")
    for python in pythons:
        mine = [r["summary"] for r in runs if r["arm"] == python]
        warm = [m["warm_median_rtfx"] for m in mine if m["warm_median_rtfx"]]
        mlx_version = next(r["mlx"] for r in runs if r["arm"] == python)
        warm_text = f"{statistics.median(warm):.1f}" if warm else "-"
        print(f"| {python} | {mlx_version} | "
              f"{statistics.median(m['first_rtfx'] for m in mine):.1f} | {warm_text} | "
              f"{statistics.median(m['sustained_rtfx'] for m in mine):.1f} |")
    if args.json:
        args.json.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                         "runs": runs}, indent=1))


if __name__ == "__main__":
    main()
