"""Compare fresh-process first separations with and without outer compile.

Run through ``metalq submit -w``. Child Python processes execute sequentially
inside the one MetalQ job; they do not submit nested jobs.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--arm", choices=("current", "outer-only"))
parser.add_argument("--seconds", type=int)
args = parser.parse_args()

if args.arm:
    from demucs_mlx.api import Separator

    os.environ["DEMUCS_MLX_COMPILE_FORWARD"] = "0" if args.arm == "current" else "1"
    os.environ["DEMUCS_MLX_COMPILE_DCONV"] = "1" if args.arm == "current" else "0"
    rng = np.random.default_rng(982 + args.seconds)
    n = args.seconds * 44_100
    wav = (0.01 * rng.standard_normal((2, n), dtype=np.float32)).astype(np.float32)
    separator = Separator(seed=481)
    start = time.perf_counter()
    _, stems = separator.separate_tensor(wav)
    elapsed = time.perf_counter() - start
    assert len(stems) == 4
    print(
        "RESULT " + json.dumps({"seconds": args.seconds, "arm": args.arm, "elapsed": elapsed}),
        flush=True,
    )
    raise SystemExit(0)

print("## Fresh-process HTDemucs whole-forward compile", flush=True)
print("**Settings:** one shift, 25% overlap, batch two, seed 481; load excluded", flush=True)
print("| Input | Pair | Current DConv path | Opt-in outer compile | Less wall time |", flush=True)
print("|---:|---:|---:|---:|---:|", flush=True)
for seconds in (30, 60):
    for pair, order in enumerate((
        ("current", "outer-only"),
        ("outer-only", "current"),
    ), start=1):
        times = {}
        for arm in order:
            result = subprocess.run(
                [sys.executable, __file__, "--arm", arm, "--seconds", str(seconds)],
                capture_output=True, text=True, check=True,
            )
            record = next(
                line[7:] for line in result.stdout.splitlines() if line.startswith("RESULT ")
            )
            times[arm] = json.loads(record)["elapsed"]
        before, after = times["current"], times["outer-only"]
        print(f"| {seconds}s | {pair} | {before:.3f}s | {after:.3f}s | "
              f"{100 * (before - after) / before:.1f}% |", flush=True)
