"""Thermally gated, balanced alternating (ABBA) benchmark for ANE and GPU paths.

Run through metalq:
    metalq submit -w -n bench_thermally_gated -- uv run --extra ane python tests/bench_thermally_gated_ane.py
"""
import time
import Foundation
import numpy as np

from demucs_mlx.api import Separator

THERMAL_NAMES = ["nominal", "fair", "serious", "critical"]


def get_thermal_state() -> tuple[int, str]:
    state = int(Foundation.NSProcessInfo.processInfo().thermalState())
    label = THERMAL_NAMES[state] if state < len(THERMAL_NAMES) else f"unknown({state})"
    return state, label


def wait_for_cool_silicon(min_cooldown: float = 5.0) -> float:
    """Enforce return to nominal thermal state plus settle cooldown."""
    start = time.perf_counter()
    state, _ = get_thermal_state()
    while state != 0:
        time.sleep(1.0)
        state, _ = get_thermal_state()
    time.sleep(min_cooldown)
    return time.perf_counter() - start


def signal(seconds: int) -> np.ndarray:
    rng = np.random.default_rng(481 + seconds)
    n = 44_100 * seconds
    t = np.arange(n, dtype=np.float32) / 44_100
    tones = np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = rng.standard_normal((2, n), dtype=np.float32) * 0.01
    return (0.05 * tones[None, :] + noise).astype(np.float32)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Thermally gated alternating benchmark")
    parser.add_argument("--batch-size", type=int, default=4, help="Chunk batch size (default: 4)")
    parser.add_argument("--compile", action="store_true", help="Enable graph compilation")
    args = parser.parse_args()

    b_size = args.batch_size
    comp = args.compile

    print("## Thermally Gated Alternating Benchmark (ABBA)", flush=True)
    print("**Protocol:**", flush=True)
    print("- Gated to `nominal` thermal state before every single run", flush=True)
    print("- 5-second physical cooldown between runs to dissipate junction heat", flush=True)
    print("- Balanced alternating order: Round 1 (GPU->ANE), Round 2 (ANE->GPU), Round 3 (ANE->GPU), Round 4 (GPU->ANE)", flush=True)
    print(f"- Shifts=1, overlap=0.25, split=True, batch_size={b_size}, compile={comp}, seed=481\n", flush=True)

    gpu = Separator(seed=481, batch_size=b_size, compile=comp)
    with Separator(seed=481, ane_time_encoder=True, batch_size=b_size, compile=comp) as ane:
        worker = ane._ane_worker

        # Warmup
        warmup_audio = signal(10)
        gpu.separate_tensor(warmup_audio)
        ane.separate_tensor(warmup_audio)
        wait_for_cool_silicon(min_cooldown=3.0)

        for seconds in (30, 60):
            audio = signal(seconds)
            print(f"### {seconds}s Audio Benchmark", flush=True)
            print(
                "| Round | Order | Path | Thermal In->Out | Wall Time | Audio/Wall | ANE Busy | ANE Wait | ANE Xfer |",
                flush=True,
            )
            print("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|", flush=True)

            orders = [
                ("GPU", "ANE"),
                ("ANE", "GPU"),
                ("ANE", "GPU"),
                ("GPU", "ANE"),
            ]

            round_times: dict[str, list[float]] = {"GPU": [], "ANE": []}
            stem_outputs: dict[str, dict] = {}

            for r_idx, order in enumerate(orders, 1):
                for name in order:
                    wait_for_cool_silicon(min_cooldown=5.0)
                    t_in_state, t_in_label = get_thermal_state()

                    sep = gpu if name == "GPU" else ane
                    before = (worker.busy_seconds, worker.wait_seconds, worker.transfer_seconds)

                    t0 = time.perf_counter()
                    _, stems = sep.separate_tensor(audio)
                    elapsed = time.perf_counter() - t0

                    t_out_state, t_out_label = get_thermal_state()
                    after = (worker.busy_seconds, worker.wait_seconds, worker.transfer_seconds)
                    diffs = tuple(e - b for b, e in zip(before, after))

                    round_times[name].append(elapsed)
                    stem_outputs[name] = stems

                    thermal_trans = f"{t_in_label}->{t_out_label}"
                    ane_busy = f"{diffs[0]:.3f}s" if name == "ANE" else "—"
                    ane_wait = f"{diffs[1]:.3f}s" if name == "ANE" else "—"
                    ane_xfer = f"{diffs[2]:.3f}s" if name == "ANE" else "—"

                    print(
                        f"| {r_idx} | {'->'.join(order)} | {name} | {thermal_trans} | "
                        f"**{elapsed:.3f}s** | {seconds / elapsed:.1f}x | {ane_busy} | {ane_wait} | {ane_xfer} |",
                        flush=True,
                    )

            # Summaries
            gpu_med = np.median(round_times["GPU"])
            ane_med = np.median(round_times["ANE"])
            gpu_mean = np.mean(round_times["GPU"])
            ane_mean = np.mean(round_times["ANE"])

            print(f"\n**{seconds}s Summary (Median / Mean):**", flush=True)
            print(f"- GPU: {gpu_med:.3f}s median ({gpu_mean:.3f}s mean, stdev={np.std(round_times['GPU']):.3f}s)", flush=True)
            print(f"- ANE: {ane_med:.3f}s median ({ane_mean:.3f}s mean, stdev={np.std(round_times['ANE']):.3f}s)", flush=True)
            print(f"- **Thermally-gated speedup (GPU/ANE): {gpu_med / ane_med:.3f}x median ({gpu_mean / ane_mean:.3f}x mean)**\n", flush=True)

            print("| Stem | SNR (dB) | Peak Absolute Error |", flush=True)
            print("|:---|---:|---:|", flush=True)
            for stem, ref in stem_outputs["GPU"].items():
                want = ref.astype(np.float64)
                cand = stem_outputs["ANE"][stem].astype(np.float64)
                err = want - cand
                snr = 10 * np.log10(np.sum(want * want) / max(np.sum(err * err), 1e-30))
                print(f"| {stem} | {snr:.2f} dB | {np.max(np.abs(err)):.6g} |", flush=True)
            print("\n", flush=True)


if __name__ == "__main__":
    main()
