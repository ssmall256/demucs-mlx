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
```

All MLX/Metal measurements were submitted through `metalq submit -w`.
