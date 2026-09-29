# HTDemucs throughput experiments

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
| Compile the whole HTDemucs segment (`mq-9cc448`) | Cold call cost 2.783 s; warmed median was 375.78 ms eager versus 360.77 ms compiled. | The startup cost outweighs this gain for ordinary tracks. |
| Delay split accumulation evaluation (`mq-254650`) | Interval four helped slightly at 30 seconds but slowed 60-second runs. | Keep one evaluation per batch. |
| Half precision (`mq-246936`, `mq-f4bf29`, `mq-286806`) | The full model was slower end to end and gave 38–42 dB minimum stem SNR. Half precision only in the transformer gave 66–67 dB SNR without a repeatable speedup. | Keep FP32. |

## Candidates measured but not adopted

The batch size sweep (`mq-c0e8d4`) compared batches two, four, and eight on 30- and 60-second inputs before the GroupNorm change. At 30 seconds, all warmed times were within 0.018 seconds (0.631–0.649 s). At 60 seconds, batch four sometimes helped, but batch eight ranged from 1.227 to 1.361 seconds versus 1.170–1.201 seconds for batch two. Keeping batch two avoids a regression on longer inputs.

The synchronized component profile (`mq-6bb707`) found a 207.56 ms uninstrumented median for one batch-two segment. The cross-transformer took about 44 ms, followed by `decoder.3` at 23 ms and `tdecoder.3` at 21 ms. Component synchronization changes scheduling, so these are bottleneck hints. Compiling only the cross-transformer (`mq-eefbc3`) reduced its median from 44.49 to 43.67 ms, too little to justify changing the default. An identity-model overlap-add probe (`mq-095bc7`) measured only about 2–7 ms, so overlap-add is not the main opportunity.

## Reproduce

```bash
uv sync --frozen --extra dev --extra ane
metalq submit -w --no-env-sync -n fast-groupnorm-parity -- python -m pytest -q tests/test_fast_groupnorm.py
metalq submit -w --no-env-sync -n fast-groupnorm-abba-benchmark -- python tests/bench_fast_groupnorm_e2e.py
metalq submit -w --no-env-sync -n fast-groupnorm-ane-benchmark -- python tests/bench_ane_waveform.py
metalq submit -w --no-env-sync -n groupnorm-shape-probe -- python tests/bench_fast_groupnorm.py
metalq submit -w --no-env-sync -n batch-size-sweep -- python tests/bench_inference_batch.py
metalq submit -w --no-env-sync -n component-profile -- python tests/profile_htdemucs_components.py
metalq submit -w --no-env-sync -n transformer-compile -- python tests/bench_transformer_compile.py
metalq submit -w --no-env-sync -n phased-waveform-parity -- python -m pytest -q tests/test_phased_waveform_deconv.py tests/test_phased_frequency_deconv.py
metalq submit -w --no-env-sync -n phased-decoder-benchmark -- python tests/bench_phased_decoders_e2e.py
metalq submit -w --no-env-sync -n phased-waveform-benchmark -- python tests/bench_phased_waveform_e2e.py
metalq submit -w --no-env-sync -n phased-frequency-benchmark -- python tests/bench_phased_deconv_e2e.py
metalq submit -w --no-env-sync -n waveform-deconv-shapes -- python tests/bench_phased_waveform_deconv.py
metalq submit -w --no-env-sync -n frequency-deconv-shapes -- python tests/bench_phased_frequency_deconv.py
metalq submit -w --no-env-sync -n phased-wav-parity -- python tests/probe_phased_wav_parity.py --input stereo-44100-pcm16.wav
```

All MLX/Metal measurements were submitted through `metalq submit -w`.
