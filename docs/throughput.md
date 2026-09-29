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
metalq submit -w --no-env-sync -n htdemucs-model-modes -- python tests/bench_model_modes.py
metalq submit -w --no-env-sync -n ft-single-stem-parity -- python -m pytest -q tests/test_ft_single_stem.py
metalq submit -w --no-env-sync -n ft-single-stem-benchmark -- python tests/bench_ft_single_stem.py
metalq submit -w --no-env-sync -n ft-single-stem-cli -- python tests/probe_ft_single_stem_cli.py
metalq submit -w --no-env-sync --queue-exclusive -n htdemucs-current-component-profile -- python tests/profile_htdemucs_components.py
metalq submit -w --no-env-sync --queue-exclusive -n htdemucs-spectro-compiled-vs-eager -- python tests/bench_htdemucs_spectral_paths.py
```

All MLX/Metal measurements were submitted through `metalq submit -w`.
