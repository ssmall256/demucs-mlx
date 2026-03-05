"""End-to-end benchmark for demucs-mlx separation."""
import time
import mlx.core as mx
from demucs_mlx.api import Separator

def bench(duration_s=10, warmup=2, iters=5):
    sep = Separator(model="htdemucs", segment=7.8)
    sr = sep.samplerate
    wav = mx.random.normal((2, sr * duration_s))
    # warmup
    for _ in range(warmup):
        out = sep.separate_tensor(wav)
        mx.eval(out)
    mx.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = sep.separate_tensor(wav)
        mx.eval(out)
        mx.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    med = times[len(times)//2]
    print(f"{duration_s}s audio: {med:.3f}s median ({duration_s/med:.1f}x RT)")
    return med

if __name__ == "__main__":
    bench(10)
    bench(30)
    bench(60)
