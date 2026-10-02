"""Benchmark 120s audio throughput with thermal gating and fidelity checks."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[8], help="Batch sizes to test (default: 8)")
    parser.add_argument("--seconds", type=int, default=120, help="Audio length in seconds (default: 120)")
    args = parser.parse_args()

    seconds = args.seconds
    audio = signal(seconds)
    print(f"## {seconds}s Audio Separation Benchmark", flush=True)

    for b in args.batch_sizes:
        print(f"\n### Batch Size {b} (Thermally Gated ABBA)", flush=True)
        print("| Round | Order | Path | Thermal In->Out | Wall Time | Audio/Wall (RTFx) |", flush=True)
        print("|:---:|:---:|:---:|:---:|:---:|:---:|", flush=True)

        gpu = Separator(seed=481, batch_size=b)
        with Separator(seed=481, batch_size=b, ane_time_encoder=True) as ane:
            # Warmup with full batch and audio length so all chunk shape paths are primed
            gpu.separate_tensor(signal(seconds))
            ane.separate_tensor(signal(seconds))
            import mlx.core as mx, gc
            mx.clear_cache()
            gc.collect()
            wait_for_cool_silicon(min_cooldown=3.0)

            orders = [
                ("GPU", "ANE"),
                ("ANE", "GPU"),
                ("ANE", "GPU"),
                ("GPU", "ANE"),
            ]
            round_times = {"GPU": [], "ANE": []}
            stem_outputs = {}

            for r_idx, order in enumerate(orders, 1):
                for name in order:
                    wait_for_cool_silicon(min_cooldown=5.0)
                    mx.clear_cache()
                    gc.collect()
                    t_in_state, t_in_label = get_thermal_state()

                    sep = gpu if name == "GPU" else ane
                    t0 = time.perf_counter()
                    _, stems = sep.separate_tensor(audio)
                    elapsed = time.perf_counter() - t0

                    t_out_state, t_out_label = get_thermal_state()
                    round_times[name].append(elapsed)
                    stem_outputs[name] = stems
                    thermal_trans = f"{t_in_label}->{t_out_label}"
                    rtfx = seconds / elapsed

                    print(f"| {r_idx} | {'->'.join(order)} | {name} | {thermal_trans} | **{elapsed:.3f}s** | **{rtfx:.1f}x** |", flush=True)

            gpu_med = np.median(round_times["GPU"])
            ane_med = np.median(round_times["ANE"])
            print(f"\n**{seconds}s Summary (batch={b}):**", flush=True)
            print(f"- GPU: {gpu_med:.3f}s median ({seconds/gpu_med:.1f}x Audio/Wall)", flush=True)
            print(f"- ANE: {ane_med:.3f}s median ({seconds/ane_med:.1f}x Audio/Wall)", flush=True)

            print("\n| Stem | SNR (dB) | Peak Error |", flush=True)
            print("|:---|---:|---:|", flush=True)
            for stem, ref in stem_outputs["GPU"].items():
                want = ref.astype(np.float64)
                cand = stem_outputs["ANE"][stem].astype(np.float64)
                err = want - cand
                snr = 10 * np.log10(np.sum(want**2) / max(np.sum(err**2), 1e-30))
                print(f"| {stem} | {snr:.2f} dB | {np.max(np.abs(err)):.6g} |", flush=True)

if __name__ == "__main__":
    main()
