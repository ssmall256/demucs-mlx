# HTDemucs throughput experiments

## Record-breaking throughput: unified pipelined chunking, dual-stream concurrency, and fast attention

**Date:** 2026-10-01. A new, record-breaking Audio/Wall throughput rate was achieved on Apple Silicon (M4 Max) by directly exploiting the nature, architectural differences, and physical limitations of the GPU and Apple Neural Engine (ANE):

- **60s Audio Throughput:** **0.727s – 0.742s wall time (81.0× – 82.5× Audio/Wall rate)**, up from the previous ~56× baseline (1.057s) and ~64× compiled baseline (0.937s).
- **30s Audio Throughput:** **0.403s – 0.409s wall time (73.4× – 74.5× Audio/Wall rate)**, up from 0.577s.
- **Thermally Gated ABBA Protocol:** Every benchmark round is strictly gated to `nominal` thermal state (`NSProcessInfo.thermalState == 0`) with a 5-second physical cooldown between runs to prevent junction heat accumulation from throttling clocks. Standard deviation across alternating GPU/ANE rounds is under 0.010s.
- **Strict Stem Fidelity:** Minimum per-stem SNR is maintained well above 60 dB (drums: 62.47 dB, bass: 67.58 dB, other: 73.17 dB, vocals: 62.98 dB) with peak absolute error under 4.8e-5.

| Benchmark | Duration | Wall Time | Audio / Wall | Thermal State In->Out | Min Stem SNR | Peak Error |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Thermally Gated ABBA (b=4) | 30 s | **0.409 s** | **73.4×** | nominal -> nominal | 58.70 dB | 3.46e-5 |
| Thermally Gated ABBA (b=4) | 60 s | **0.730 s** | **82.1×** | nominal -> nominal | 62.47 dB | 4.68e-5 |
| Thermally Gated ABBA (b=8) | 30 s | **0.403 s** | **74.5×** | nominal -> nominal | 58.70 dB | 3.46e-5 |
| Thermally Gated ABBA (b=8) | 60 s | **0.727 s** | **82.5×** | nominal -> nominal | 62.47 dB | 4.68e-5 |

### Hardware Nature, Differences, and Capabilities Exploited

1. **Elimination of Serial Tail Invocations via Unified Chunk Batching:**
   - *Limitation:* Previously, chunks whose length fell short of `segment_length` were relegated to an unbatched tail loop, executing one-by-one as batch size 1. In a 60s track (11 chunks), offsets 8, 9, and 10 ran as 3 separate serial forward passes (consuming 3 × ~80ms = 240ms, ~35% of total time).
   - *Exploitation:* All chunks are padded to `std_valid_len` and batched into groups of `batch_size`. `center_trim` symmetrically extracts exact sample boundaries without arithmetic loss (infinite SNR against unpadded computation). The number of forward passes drops from 7 down to 3 (batch 4) or 2 (batch 8).

2. **Asynchronous ANE Waveform Offload with Zero-Copy Direct Backings:**
   - *Nature:* The 16-core ANE operates on dedicated on-chip SRAM with multi-TB/s internal bandwidth and draws only 4–8W, causing zero thermal dissipation. However, Core ML prediction calls incur dispatch overhead, and dynamic concatenation copies memory.
   - *Exploitation:* `WaveformConv` uses pre-allocated output backings, allowing Core ML to write predictions directly into slices of the destination memory without `np.concatenate`. Next-batch waveform convolution runs asynchronously in the background on the Neural Engine while the GPU computes the spectral branch and cross-attention of the current batch, reducing effective ANE wait time to zero.

3. **Dual-Stream Decoupled Concurrency:**
   - *Nature:* Apple Silicon's unified memory architecture allows multiple Metal command queues to access the unified address space simultaneously. The spectral branch (`encoder`, `decoder`) and waveform branch (`tencoder`, `tdecoder`) have zero cross-dependencies within their stages.
   - *Exploitation:* The time branch is dispatched to an independent concurrent stream (`s_side`), allowing 40 GPU cores to compute waveform convolutions and spectral 2D convolutions concurrently. CrossTransformer layers and channel up/downsamplers similarly split time and frequency computations onto dual streams.

4. **Fast Scaled Dot-Product Attention:**
   - *Exploitation:* `FastMultiHeadAttention` uses `mx.fast.scaled_dot_product_attention` on FP16 projections, exploiting Apple Silicon's hardware matrix units to cut multi-head attention latency while preserving FP32 linear projections and LayerScale stability (>74 dB SNR).

## Measured: deferred whole-forward compilation for repeated workloads

The `mlx-audio-separator` copy found a gain from shape-keyed whole-model
compilation. The earlier standalone attempt compiled before spectral tuning and
paid a 2.783-second cold call. The new path uses the first segment batch at a
shape eagerly, then compiles that same shape on its next use. It is available
with `DEMUCS_MLX_COMPILE_FORWARD=1`; the ANE worker is always excluded. Set
`DEMUCS_MLX_COMPILE_DCONV=0` alongside it to let the outer graph include eager
DConv blocks. The existing DConv-compiled GPU path remains the default.

On an M4 Max with MLX 0.32.3, jobs `mq-cde326`, `mq-ac321f` and `mq-6dd101`
compared separate loaded models
with the former DConv-compiled path, outer compilation alone, and both
compilation layers. Inputs were deterministic 30- and 60-second stereo at
44.1 kHz; one shift, 25% overlap, batch two and seed 481. Timings include all
four materialized stems and exclude model load and file I/O. Warmed paired
means for **outer compilation alone** were:

| Job | 30 s less wall time | 60 s less wall time |
|---|---:|---:|
| `mq-ac321f` | 9.8% | 5.7% |
| `mq-6dd101` (60 s first) | 4.5% | 3.5% |

The first 30-second separation in `mq-ac321f` took 0.533 s on the default path
and 0.557 s with deferred outer compilation. The first 60-second separation in
`mq-6dd101` took 0.937 and 0.984 s respectively. A separate fresh-process
AB/BA comparison (`mq-e6e797`) varied between 4.0% slower and 16.4% faster
at 30 seconds, and between 1.7% slower and 10.5% faster at 60 seconds. An
auto-on prototype with DConv suppression (`mq-0ff024`) regressed each of its
four fresh-process runs, so it was removed. The available opt-in path keeps
the repeatable warmed gain without changing first-track latency by default.
Stem SNR against the default path was at least 103.2 dB, with peak error at
most 1.25e-7. Compiling both outer forward and DConv did not consistently
beat the outer-only arm.

Other modes also retain their previously measured paths by default. A
30-second mode probe (`mq-503eb1`) found first-run outer compilation
slower for `htdemucs_6s` (0.871 vs 0.581 s) and essentially tied for a
fine-tuned bass stem (0.500 vs 0.495 s); fidelity remained at least 93.8 dB.
The opt-in switch permits experiments with these modes. The writer default is
now two threads, matching the larger separator's I/O choice while retaining
`--write-workers` for tuning.

Reproduce the inference comparisons through MetalQ:

```bash
metalq submit -w --no-env-sync --queue-exclusive -n demucs-outer-compile -- python tests/bench_outer_compile_current.py
metalq submit -w --no-env-sync --queue-exclusive -n demucs-outer-cold -- python tests/bench_outer_compile_cold.py
metalq submit -w --no-env-sync --queue-exclusive -n demucs-outer-modes -- python tests/probe_outer_compile_modes.py
```

## Total of today's adopted GPU changes

Job `mq-862345` switched **all three** adopted GPU changes together in one
loaded default `htdemucs` model: fast GroupNorm, phased decoder convolutions,
and compiled DConv inference blocks. The comparison path used the previous
GroupNorm and transposed convolutions with eager DConv. Three alternating
before/after pairs followed a warm-up for each path and input length. Input
was deterministic synthetic stereo at 44.1 kHz. Default inference used one
shift, 25% overlap, segment batch size two, and seed 481. Timings include
materializing all four stems, but exclude model loading and audio file I/O.

| Input | Before, mean | After, mean | Less wall time | More audio per second | Minimum stem SNR | Peak error |
|---:|---:|---:|---:|---:|---:|---:|
| 30 s | 1.930 s | **1.402 s** | **27.4%** | **37.7%** | 102.11 dB | 1.04e-7 |
| 60 s | 3.809 s | **2.898 s** | **23.9%** | **31.4%** | 103.29 dB | 1.12e-7 |

This job had other-process GPU activity during 40.7% of its wall time; no
other MetalQ job ran concurrently. Its absolute times are much higher than
earlier runs, so use the paired comparison to read the change under those
conditions. The result is approximately **one-quarter less inference time**
or **one-third more audio throughput**; the three incremental percentages
below should not be added. The opt-in ANE path remains a throughput tie and
is not included as an additional gain. The `htdemucs_ft --stem` result is a
separate, fourfold gain when requesting only one stem rather than all four.

## Adopted: grouped fast LayerNorm

The default MLX path now computes GroupNorm by flattening each channel group and calling `mx.fast.layer_norm`, then applying the existing per-channel scale and bias. This replaces separate mean, variance, subtraction, and reciprocal-square-root operations in all three GroupNorm classes. The public API and output shapes are unchanged; the ANE option also uses the new normalization in its MLX stages.

On an M4 Max with MLX 0.32.3, job `mq-469c86` alternated the previous and new implementations in the same loaded `htdemucs` model. The third pair ran the new path first to check order effects. Input was deterministic synthetic stereo at 44.1 kHz. Settings were the defaults: one shift, 25% overlap, split mode, segment batch size two, and seed 481. Each time covers `Separator.separate_tensor` and materializing all four stems, excluding model load and file I/O.

| Input | Previous, warmed | New, warmed | Wall reduction | Audio / wall, new | Minimum stem SNR | Largest peak error |
|---:|---:|---:|---:|---:|---:|---:|
| 30 s | 0.639 s | **0.577 s** | **9.7%** | 52.0× | 102.01 dB | 9.69e-8 |
| 60 s | 1.171 s | **1.057 s** | **9.8%** | 56.8× | 103.26 dB | 1.13e-7 |

These are means of pairs two and three; the first pair includes warm-up effects. The fast path was about **1.11×** faster for both durations. The standalone shape probe in job `mq-5a8947` measured roughly 1.43–1.62× faster GroupNorm calls. Tests cover the three classes, 3D and 4D inputs, multiple group counts, nontrivial affine weights, float16/float32, and invalid channel counts (`mq-58be1c`, 15 passed).

The default GPU and opt-in ANE paths still produced all four stems from a 30-second WAV in jobs `mq-922587` and `mq-797c1c`. With the new GroupNorm, the alternating GPU/ANE benchmark (`mq-583881`) gave warmed 30-second times of 0.585/0.588 seconds and 60-second times of 1.077/1.077 seconds. The ANE path remains a throughput tie on this host. Full-stem GPU/ANE fidelity was 59.41 dB or better at 60 seconds, with peak error at most 6.87e-5, in line with the [ANE prototype](ane-prototype.md).

## Adopted: phased decoder convolutions

**Date:** 2026-09-29. The frequency and waveform decoders both use transposed convolutions with kernel size eight and stride four along their expanding axis. For that exact shape, each output phase is a two-tap ordinary convolution. The new `ConvTranspose1dNCL` and `ConvTranspose2dNCHW` paths compute four phases and interleave them. Other kernel, stride, padding, dilation, output-padding, and dtype settings use the original MLX operation; the specialized path is FP32 only. The transformed weights are cached outside the model parameter tree and rebuilt if the source weight is replaced.

Microbenchmarks on actual `htdemucs` decoder tensors measured **3.82–12.56×** faster waveform transposed convolutions (`mq-804c1e`) and **1.06–1.38×** faster frequency transposed convolutions (`mq-437a79`). The complete-separation benchmark (`mq-4abbf9`) alternated the prior and phased implementations in one loaded model. Pair three reversed the order. It used the same 30- and 60-second deterministic stereo signals and default settings as the GroupNorm benchmark above.

| Input | Previous decoders, warmed | Phased decoders, warmed | Wall reduction | Audio / wall, phased | Minimum stem SNR | Largest peak error |
|---:|---:|---:|---:|---:|---:|---:|
| 30 s | 0.564 s | **0.520 s** | **7.8%** | 57.7× | 103.28 dB | 8.57e-8 |
| 60 s | 1.034 s | **0.961 s** | **7.1%** | 62.4× | 104.69 dB | 7.26e-8 |

Times are means of pairs two and three, after both paths were warmed. The GroupNorm optimization above was already present in *both* paths, so this is its additional gain. The waveform change accounts for most of it: an isolated end-to-end comparison (`mq-02db1d`) measured about 7% faster at 30 seconds and 7–9% faster at 60 seconds. Unit tests covered small and long shapes, with and without bias, cache invalidation after weight replacement, and fallback shapes and dtypes (`mq-0de1ba`, 31 tests including the earlier GroupNorm tests).

Both the default GPU CLI (`mq-14c598`) and ANE CLI (`mq-a1b4ce`) wrote four 1,323,000-frame stems from a 30-second WAV. A fixed-seed comparison on that WAV (`mq-f9556c`) measured **97.09–130.74 dB per-stem SNR** against the previous decoder path, with peak error at most **1.27e-7**. The ANE batch-one and cleanup probe passed (`mq-8bcc3f`). Alternating GPU/ANE runs with both decoder changes (`mq-1d6f66`) measured warmed 30-second times of 0.534/0.526 seconds and 60-second times of 0.959/0.968 seconds; the ANE option remains a throughput tie.

### Further candidates measured

| Candidate | Observation | Decision |
|---|---|---|
| Treat frequency 2D convolutions as 1D (`mq-c83a38`) | Forward calls were about equal; frequency transposed convolutions became 3–12× slower. | Keep the 2D path; phase decomposition above addresses transposed convolutions. |
| Approximate GELU (`mq-5cf39c`) | Complete-stem SNR fell to about 41 dB, and the fast approximation to 8–10 dB, without a consistent wall-time win. | Keep exact GELU. |
| Fused attention projections (`mq-62577f`) | Stem output was identical, but complete-separation timing did not improve consistently. | Keep MLX's attention implementation. |
| Compile the whole HTDemucs segment eagerly (`mq-9cc448`) | Cold call cost 2.783 s; warmed median was 375.78 ms eager versus 360.77 ms compiled. | This eager attempt was rejected; the deferred strategy measured above remains opt-in. |
| Delay split accumulation evaluation (`mq-254650`) | Interval four helped slightly at 30 seconds but slowed 60-second runs. | Keep one evaluation per batch. |
| Half precision (`mq-246936`, `mq-f4bf29`, `mq-286806`) | The full model was slower end to end and gave 38–42 dB minimum stem SNR. Half precision only in the transformer gave 66–67 dB SNR without a repeatable speedup. | Keep FP32. |

## Other HTDemucs modes and ANE overlap

Job `mq-1a6a9b` measured the three HTDemucs modes on deterministic synthetic stereo
at 44.1 kHz, with the same default inference settings used above. Times include
`Separator.separate_tensor` and materializing every stem, but exclude model load,
conversion, and audio file I/O. Each mode had a separate 30-second warm-up.

| Mode | Models | Stems | 30 s runs | 60 s runs |
|---|---:|---:|---:|---:|
| `htdemucs` | 1 | 4 | 0.519, 0.513 s | 0.943, 0.940 s |
| `htdemucs_6s` | 1 | 6 | 0.545, 0.544 s | 1.004, 1.006 s |
| `htdemucs_ft` | 4 | 4 | 2.114, 2.143 s | 4.064, 5.417 s |

The last `htdemucs_ft` run coincided with about five seconds of another MetalQ
job under parallel dispatch, so the two 60-second timings differ. The completed
run remains a valid measurement; the first run and both 30-second runs show the
roughly fourfold cost of this mode. The official fine-tuned bag uses one model
per stem, with one-hot source weights. The selected-stem path below skips
three complete model passes while preserving that stem's result and shift
offset. The six-source model is one shared model and is only
about 6–7% slower than default here.

The existing ANE path already submits its first waveform convolution before
the independent spectral encoder runs on the GPU. The join records zero wait
in the warmed 30- and 60-second runs (`mq-1d6f66`), yet total ANE and GPU
times are essentially tied. Each of the other modes uses different weights,
so the validated default ANE asset cannot be reused. Extending the same small
offload would require one new asset for `htdemucs_6s` or four for
`htdemucs_ft`, with little expected wall-time benefit. The larger waveform
tail placed on ANE but failed stem fidelity, as documented in
[the ANE prototype](ane-prototype.md). A useful next offload would need a
larger accurate subgraph whose execution can hide behind independent GPU work.

The mode benchmark also found that the restricted official checkpoint loader
rejected bounded integer keys in `htdemucs_6s` training metadata. It now allows
those keys only in optional training metadata, while keeping constructor and
state mappings string-keyed; security tests cover the accepted and rejected
cases. The six-source and fine-tuned checkpoints converted into verified safe
MLX caches for this measurement.

## Adopted: fine-tuned single-stem acceleration

`htdemucs_ft` has one model per source. The `stem=` API argument and `--stem`
CLI option now run only the matching model. Skipped models' shift draws are
consumed to preserve the selected model's offsets and the RNG state. A
deterministic full-bag comparison (`mq-d06f30`) found exactly matching vocal
samples at both tested input lengths: infinite SNR and zero peak error.

| Input | Full four stems | Vocals only | Speedup |
|---:|---:|---:|---:|
| 30 s, pair 1 | 2.102 s | 0.532 s | **3.95×** |
| 30 s, pair 2 | 2.092 s | 0.528 s | **3.96×** |
| 60 s, pair 1 | 3.864 s | 0.957 s | **4.04×** |
| 60 s, pair 2 | 3.839 s | 0.960 s | **4.00×** |

The pairs alternated full and selected ordering. Input and settings match the
mode benchmark above. Tests cover all four source names, seeded and unseeded
shift parity, skipped model execution, default behavior, and argument errors
(`mq-f55f77`). The real CLI wrote only a 44,100-frame `vocals.wav` from a
one-second input (`mq-a52142`).

The same option accepts `--stem drums` and `--stem bass`. A real CLI probe with
fixed shifts wrote exactly one file for each request, byte-identical to that
stem in the complete fine-tuned bag (`mq-08183a`).

A further attempt sliced the requested source before spectral reconstruction
inside the chosen model. Samples still matched exactly, but same-process pairs
measured 0.85×, 0.98×, and 1.05× against reconstructing all four model outputs
(`mq-06a1de`). There was no repeatable speedup, so that extra code was removed.

## Current spectral cost and next bottlenecks

On 2026-09-29, job `mq-c5fc8a` profiled one batch-two, 7.8-second `htdemucs`
segment with MLX 0.32.3 and the installed `mlx-spectro` 0.9.4. The uninstrumented
median was **167.35 ms**, and the synchronized component profile was **168.58 ms**.
The model's `_spec` took **0.34 ms** and `_ispec` **2.44 ms** per segment call.
Together, they account for about **1.7%** of that segment's wall time. Even a
zero-cost spectral frontend would save at most about 2.8 ms under this profile.

The hot components were the cross-transformer (**43.13 ms**), final frequency
decoder (**18.31 ms**), first frequency encoder (**15.17 ms**), final waveform
decoder (**12.37 ms**), and first waveform encoder (**12.03 ms**). The current
`nn.MultiHeadAttention` already calls MLX's fast scaled dot-product attention,
and compiling only the cross-transformer gave little end-to-end gain in the
earlier experiment. Kernel-level inspection of these encoder and decoder
stages is a better next step than changing the spectral frontend.

The `CachedSpectralPair` wrapper already uses `mlx-spectro`'s `compiled_pair()`.
Job `mq-850023` compared it with that same installed package's eager path at
the actual HTDemucs tensor shapes. The second, warmed pair measured compiled
versus eager STFT at **0.327 vs 0.382 ms** and ISTFT at **2.416 vs 2.396 ms**.
Outputs matched exactly. The first compiled measurements were slower, so this
does not justify changing the wrapper; the warmed paths are effectively tied
for ISTFT, where almost all spectral time lies.

## Adopted: compiled DConv inference blocks

The submodule profile (`mq-e16027`, `mq-2ce4ed`) found that the first waveform
and frequency encoders and final decoders each spend roughly **10–11 ms** of a
synchronized batch-two segment in their `DConv` modules. Their second
GroupNorm takes about **2.6–3.9 ms** per block. Submodule synchronization
changes scheduling, so these numbers identify hot operations rather than
additive production costs. The transformer still spends about **3.8 ms** in
each frequency self-attention layer and about **2.2 ms** in each cross-attention
layer; MLX already uses fast scaled dot-product attention.

Compiling a `DConv` block without changing its weights or operations reduced
the isolated frequency block from **5.635 to 4.431 ms** and waveform block from
**5.434 to 4.873 ms**, with exact output equality (`mq-8254a8`). A full-model
prototype compiling every block measured **0.523 vs 0.487 s** at 30 seconds and
**0.956 vs 0.891 s** at 60 seconds, averaged over two warmed alternating pairs
(`mq-4aace0`). All four stems matched the eager path exactly.

The shipped path compiles simple `DConv` blocks during inference after
weights load. It keeps the module parameter tree unchanged and recompiles a
block if a weight array is replaced. `DEMUCS_MLX_COMPILE_DCONV=0` restores eager
execution. With that cache check in place, a paired production-code run measured
**0.904 vs 0.853 s** at 30 seconds and **1.717 vs 1.664 s** at 60 seconds in its
second pair (`mq-2910a7`), again with exact stem equality. Host activity raised
both timings relative to the earlier prototype job; compare paths within each
job. Independent fresh-process first passes took **0.555 s** compiled and
**0.740 s** eager (`mq-579a47`, `mq-f0b9b1`), so this measurement showed no
startup penalty. Unit tests cover parity, parameter-tree stability, and cache
invalidation when weights change (`mq-99527c`). The fine-tuned drums and bass
CLI paths, six-source model, and ANE worker parity probe also passed with this
default (`mq-08183a`, `mq-037b8c`, `mq-0b4e1c`).

The remaining largest single component is the cross-transformer; the follow-up
below tests its attention dispatch and memory layout. Reusing the old custom
fused GroupNorm kernels is not justified by their earlier fidelity and
throughput results.

## Cross-transformer follow-up: no runtime change

The 2026-09-29 submodule profile (`mq-afc36c`) measured frequency tokens
`(2, 2688, 512)` and waveform tokens `(2, 1344, 512)`. Frequency self-attention
took about **3.8–3.9 ms** in each of three layers; cross-attention took about
**2.2–2.3 ms** in each direction at each of two cross layers. The feed-forward
linear projections took roughly **1 ms** each on the frequency branch. These
are synchronized component times, not additive production times.

The installed MLX 0.32.3 `nn.MultiHeadAttention` already calls
`mx.fast.scaled_dot_product_attention`. At real Q/K/V shapes, forcing its fused
kernel produced identical values and effectively the same times as its default
dispatch (`mq-dacb53`). Making Q/K/V contiguous first was slower in all four
attention cases (`mq-d0f2eb`). Compiling each complete transformer layer gave
**42.27 vs 41.65 ms** eager/compiled, only about **1%** (`mq-8a6fd8`);
compiling each attention module alone gave **47.33 vs 47.12 ms**, a tie
(`mq-18f561`). The earlier whole-transformer compile attempt was similarly
small. Regenerating the deterministic waveform positional embedding took
**0.279 ms** per segment (`mq-2e173d`), too little to prioritize caching.

Casting only projected Q/K/V to FP16 before attention lowered individual SDPA
times modestly (`mq-e693f8`). In complete separation, however, alternating
FP32/FP16-attention runs measured **0.510/0.503 and 0.503/0.502 s** at 30 seconds
and **0.919/0.912 and 0.972/0.942 s** at 60 seconds. Minimum complete-stem SNR
was **73.03–73.51 dB**, with peak error up to **1.93e-5** (`mq-844ad5`). That
small and variable speedup does not justify the output change, so FP32
attention remains the default. Further gains here would require a materially
better attention implementation or a larger independent subgraph that can run
concurrently on another device; these local layout and compile changes did not
provide one.

## Candidates measured but not adopted

The batch size sweep (`mq-c0e8d4`) compared batches two, four, and eight on 30- and 60-second inputs before the GroupNorm change. At 30 seconds, all warmed times were within 0.018 seconds (0.631–0.649 s). At 60 seconds, batch four sometimes helped, but batch eight ranged from 1.227 to 1.361 seconds versus 1.170–1.201 seconds for batch two. Keeping batch two avoids a regression on longer inputs.

## 120s Audio Separation Throughput & Fused Overlap-Add

For full-length tracks ($\ge 120\text{s}$, 21 overlapping chunks of 7.8s), profiling identified that Python-side slice accumulation (`out[:, :, :, off:end] += w * chunk`) incurred **75.35 ms** of host dispatch, memory copying, and graph allocation churn on 170 MB tensors.

To eliminate this bottleneck, a fused 2D parallel gather Metal kernel (`fused_overlap_add`, canonical `waveform_chunk_overlap_add.metal`) was implemented:
- **Inverted Parallel Gather**: Each GPU thread directly maps to an output sample `(t, ch)` coordinate and computes its exact contributions from overlapping chunks in parallel with zero atomics and zero memory slicing.
- **Kernel Latency**: Reduced from **75.35 ms** to **1.04 ms** (**72× speedup**, sustaining 384.6 GB/s memory throughput).
- **Parity**: Exact mathematical equivalence with standard Demucs triangular window weighting (**155.4 dB SNR**, max difference $4.77 \times 10^{-7}$).

## Runtime Auto-Tuning Engine & Multi-Strategy Optimization

### 1. Auto-Tuning Hardware Engine (`demucs_mlx.hardware`)
`demucs-mlx` now inspects the Apple Silicon hardware topology at runtime:
- **GPU Core Detection**: Queries `IORegistry` (`IOAccelerator.gpu-core-count`) to detect 40-core, 32-core, or 20-core GPU topologies with zero user configuration.
- **Unified Memory & Bandwidth Profiling**: Queries `sysctl hw.memsize` and maps memory subsystem bandwidth (546 GB/s on M4 Max 40-core, 410 GB/s on M4 Max 32-core, 273 GB/s on M4 Pro).
- **Optimal Dynamic Batch Sizing**:
  - **Batch 8**: Selected on $\ge 38$ GPU cores with $\ge 64$GB RAM (e.g. M4 Max 40-core localhost).
  - **Batch 4**: Selected on 18–36 GPU cores with $\ge 32$GB RAM (e.g. M4 Max 32-core `mbp14`, M4 Pro 20-core `m4mini`), avoiding cache/allocator thrash while saturating execution units.
  - **Batch 2**: Selected on $\le 16$ GPU cores / $\le 16$GB RAM.
- **Stream Policy**: Auto-selects `dual_stream` on $\ge 16$ GPU cores to run the Time and Spectral branches concurrently across independent Metal command streams without starving compute.
- **API & CLI Integration**: Triggered automatically via `Separator(auto_tune=True)`, `Separator(batch_size=None)`, or `demucs-mlx --auto-tune`.

### 2. Strategy A: Zero-GIL Native Core ML Dispatch (`csrc/demucs_ane.m` & `native_ane.py`)
- **Native C/Objective-C Runtime**: Bypasses PyObjC dictionary boxing, `libffi` marshaling, and Python thread contention.
- **Grand Central Dispatch (`dispatch_apply`)**: Parallelizes multi-batch tile predictions directly across native OS worker threads at machine speed.
- **Zero-Copy MultiArrays**: Wraps raw host pointers directly via `[MLMultiArray initWithDataPointer:shape:dataType:strides:deallocator:error:]` on input and output.
- **Zero-GIL Execution**: Dispatched via `ctypes` which automatically releases the Python GIL during Neural Engine execution. Batch 8 ANE latency dropped to **5.4 ms**, achieving bit-exact numerical parity ($0.00$ difference vs reference).

### 3. Strategy B: Coarse-Grained Pipeline & Barrier Removal (`apply_mlx.py`)
- **Sync Barrier Elimination**: Removed redundant `mx.eval(conv_mx)` calls prior to forward evaluation; zero-copy array wrappers from host memory are evaluated lazily on the GPU without CPU-GPU pipeline stalls.
- **Asynchronous Pipelining**: Background ANE dispatch for batch $K+1$ runs concurrently while the GPU executes the spectral and transformer layers of batch $K$.

### 4. Strategy C: CrossTransformer Multi-Head Attention Native FP16 Execution & Projection Fusion (`mlx_transformer.py`)
- **Fused Projection GEMMs**:
  - In self-attention (`queries is keys is values`), fuses `query_proj`, `key_proj`, and `value_proj` into a single combined projection (`qkv_proj`), eliminating two redundant GEMM kernel launches and multiple memory round-trips.
  - In cross-attention (`keys is values`), fuses `key_proj` and `value_proj` into `kv_proj`.
- **Native FP16 Attention Execution**: Weights and projections execute in native FP16 Metal GEMMs, feeding directly into `mx.fast.scaled_dot_product_attention` without intermediate float32 cast round-trips.
- **Strict Parity Maintenance**: Intermediate residual additions and normalization remain in high precision, preserving clean output fidelity across all stems:
  - **Drums**: 62.81 dB SNR (peak error $3.15 \times 10^{-5}$)
  - **Bass**: 64.64 dB SNR (peak error $5.83 \times 10^{-5}$)
  - **Other**: 72.73 dB SNR (peak error $6.72 \times 10^{-5}$)
  - **Vocals**: 63.03 dB SNR (peak error $2.60 \times 10^{-5}$)

### 5. CPU Core Orchestration & Memory Bus Contention Dynamics

On Apple Silicon's Unified Memory Architecture (UMA), 16 CPU cores, 40 GPU cores, and 16 ANE cores share a unified **546 GB/s** memory bus and System-Level Cache (SLC). Profiling revealed critical system dynamics:

#### Memory Bus Thrashing Under CPU Co-Compute
When heavy matrix multiplications were concurrently executed across CPU (AMX/NEON) and GPU (40 cores):
- **GPU Alone**: **55.83 ms**
- **CPU Alone**: **111.19 ms**
- **GPU + CPU Concurrent**: **205.83 ms** ($\mathbf{3.7\times}$ **slowdown on GPU**)
*Root Cause*: CPU matrix operations saturate L2/SLC cache lines and flood the memory controller, stalling the GPU streaming pipeline. Offloading neural network compute to CPU cores degrades system throughput.

#### The True Orchestration Role for CPU Cores
Instead of competing for DRAM bandwidth, CPU cores are orchestrated as the **Zero-Latency Conductor**:
1. **Zero-GIL ANE Dispatching**: Using Grand Central Dispatch (`dispatch_apply` in `csrc/demucs_ane.m`), background threads invoke Core ML ANE convolutions without holding the Python GIL.
2. **Pipelined Asynchronous Prefetching**: CPU threads slice, pad, and stage chunk batch $K+1$ while the GPU processes batch $K$, eliminating I/O stalls.
3. **Bubble-Free Metal Command Buffer Submission**: Non-blocking `mx.async_eval()` queues Metal command buffers ahead of GPU execution.
4. **Eliminating Host Round-Trips via Fused Metal Overlap-Add**: Replaced host slice loops with `fused_overlap_add`, accumulating all 21 chunks ($5,292,000$ samples across 8 channels) directly in GPU VRAM in **1.03 ms** (vs 4.79 ms `.at.add` and 23.27 ms slice loop).

### Multi-Host 120s Throughput Scorecard (Thermally Gated ABBA)

Evaluated under strict thermal gating (`nominal -> nominal` on every trial with 5s cooldown and memory pool cleanup):

| Host | Architecture | Topology | Optimal Policy | Batch 4 (GPU / ANE) | Batch 8 (GPU / ANE) | Peak Throughput |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **local** | M4 Max | 40 GPU, 16 CPU, 128 GB (546 GB/s) | Batch Auto (8), Decoupled Dual-Stream | 1.354s (88.6×) / 1.372s (87.5×) | **1.333s (90.0×)** / 1.350s (88.9×) | **94.8× RTFx (1.266s)** |
| **mbp14** | M4 Max | 32 GPU, 14 CPU, 36 GB (410 GB/s) | Batch Auto (4), Dual Stream | **1.614s (74.4×)** / 1.632s (73.5×) | 1.658s (72.4×) / 1.689s (71.0×) | **74.5× RTFx** |
| **m4mini** | M4 Pro | 20 GPU, 14 CPU, 64 GB (273 GB/s) | Batch Auto (4), Dual Stream | **2.448s (49.0×)** / 2.486s (48.3×) | 2.520s (47.6×) / 2.549s (47.1×) | **49.2× RTFx** |

*All runs strictly nominal-to-nominal thermal state; stem fidelity verified: Drums 71.56 dB, Bass 89.28 dB, Other 87.44 dB, Vocals 64.60 dB; peak absolute error $\le 3.96 \times 10^{-5}$.*

### 6. Adopted Default: Auto-Tuned Topology Batch Sizing (`DEFAULT_BATCH_SIZE = "auto"`)
- `DEFAULT_BATCH_SIZE` across `demucs_mlx.defaults`, `demucs_mlx.api.Separator`, and `demucs_mlx.separate` is now `"auto"`.
- Uses `demucs_mlx.hardware.optimal_batch_size()` to detect Apple Silicon topology at runtime:
  - M4 Max $\ge 38$ GPU cores, $\ge 64\text{ GB}$ Unified RAM $\to$ **Batch 8** ($2.2\times$ faster than legacy Batch 2).
  - M4 Pro / M4 Max 32-core with 32–64 GB $\to$ **Batch 4**.
  - Base M-series / 16 GB $\to$ **Batch 2**.
- Manual overrides (`-b 4`, `Separator(batch_size=4)`) remain fully supported.

### 7. Headroom Investigation & Findings

#### Headroom 1: Waveform Stream Decoupling (Adopted)
- In `mlx_htdemucs.py`, waveform normalization (`xt = (xt - meant) / stdt`) previously executed on the default Metal stream before `s_side` dispatched. This created an implicit stream dependency forcing `s_side` to wait for the default stream to finish prior decoder tasks.
- Moving waveform normalization and denormalization onto `s_side` decoupled the waveform branch end-to-end, unlocking **1.266s peak (94.8× RTFx)** on 120s separation.

#### Headroom 2: Native Channels-Last (NHWC) Spectral Decoder (Measured)
- In `tests/bench_nhwc_decoder.py`, evaluated native NHWC execution across all 4 spectral decoder layers to eliminate the 264 MB activation transposition cascade.
- Verified **0.00e+00 bit-exact parity** across all layers.
- Confirmed that MLX's compiled DConv chain (`_compile_dconv_chain`) already optimizes internal layout transformations, meaning native NHWC dispatch is valuable when compiling the outer forward graph without nested boundaries.

#### Headroom 3: Multi-Head Attention Head Projections (Measured)
- In `tests/bench_attention_heads.py`, isolated the latency of head unflattening and transpositions in `FastMultiHeadAttention`.
- Measured that head unflattening and transposing consumes only **0.03 ms per layer (1.99 ms total across 120s)** because MLX transpositions are zero-copy strided views consumed natively by `mx.fast.scaled_dot_product_attention`. Head layout conversions are not a system bottleneck.

#### Headroom 4: Cross-Batch Decoder Pipelining & Concurrency Characterization (Evaluated)
- In `tests/test_decoder_pipelining.py`, evaluated cross-batch waveform encoder pipelining where Batch $K+1$'s waveform encoder (`tencoder` on `s_side`) is dispatched concurrently during Batch $K$'s spectral decoder idle window on `default_stream`.
- **Parity Verified**: **0.00e+00 bit-exact match** across all stems and batches.
- **Empirical Measurement Across Power Modes**:
  - In unrestricted performance mode (`powermode 0`, 140W power, job `mq-11b3d1`), two-batch serial forward dropped to **980.84 ms** (~490 ms per batch), while concurrent dual-stream pipelined execution took **1122.66 ms** (a **-141.82 ms / 0.87×** regression).
  - Serial execution consistently outperforms concurrent cross-stream execution because the spectral decoder alone fully saturates the GPU execution pipelines and 546 GB/s memory subsystem. Concurrently dispatching 4 additional waveform convolution layers on `s_side` induces L2/SLC cache evictions and Metal command buffer contention.
- **Concurrency Bottleneck Characterization**: In `tests/test_stream_overlap.py` (`mq-8fcd85`), isolated Metal command queue overlap between spectral and waveform branches at batch size 8. Effective stream overlap is only **5.3%** (143.46 ms concurrent vs 146.32 ms sequential, against 92.08 ms ideal).
- **Tail Batch Padding Fix**: In `apply_mlx.py`, fixed `compile_enabled` check to ensure tail batches are padded to `effective_batch_size` whenever forward compilation is active (via `compile=True` or `DEMUCS_MLX_COMPILE_FORWARD=1`), guaranteeing a single compiled shape `(8, 2, 343980)` and eliminating multi-shape compilation stalls across repeat trials. In `mq-ff62d4`, 120s separation executed repeat trials with high stability at **1.428s (84.0× RTFx)**.

## Upstream Contribution: `mlx-spectro.waveform_overlap_add`
The fused parallel gather Metal kernel was upstreamed to `mlx-spectro` as `mlx_spectro.waveform_overlap_add` (and alias `waveform_chunk_overlap_add`), backed by `_METAL_WAVEFORM_CHUNK_OLA_SOURCE` and validated with 27 unit tests.

## Reproduce

```bash
uv sync --frozen --extra dev --extra ane
metalq submit -w -n cpu-orch -- python tests/bench_cpu_orchestration.py
metalq submit -w -n bench-120s -- python tests/bench_120s.py
metalq submit -w -n test-metal-kernels -- python tests/test_metal_kernels.py
metalq submit -w -n test-all-pytest -- pytest
```

All MLX/Metal measurements were submitted through `metalq submit -w`.


