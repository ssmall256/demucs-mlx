# Changelog

All notable changes to this project are documented in this file. This is the file
the GitHub release workflow reads, so every release needs an entry here before it
can be published.

Entries before 1.4.7 were reconstructed from the commit history, `docs/release.md`
and the README after the fact.

## Unreleased

Every model in the registry now matches upstream PyTorch Demucs. Measured end to
end against `demucs.api.Separator` (shifts 0, same input) and written as 16-bit
WAV, per stem: htdemucs/htdemucs_ft/htdemucs_6s 75–82 dB, hdemucs_mmi 76–82 dB,
and the mdx, mdx_extra, mdx_q and mdx_extra_q bags 76–83 dB, with the residual
at the 16-bit quantization floor on every stem. A new suite
(`tests/test_upstream_parity.py`) compares each architecture with upstream.

### Fixed

- **HTDemucs output was wrong since 1.5.0.** The cross-transformer fed the
  frequency branch the time branch's *updated* output instead of its input to the
  layer, as upstream does. Stems matched upstream at only 18–24 dB; they now
  match at 81–87 dB (75–82 dB written as 16-bit WAV).
- **HDemucs and the time-domain Demucs were wrong in every release.**
  `LocalState` attention contracted the softmax weights over the wrong axis
  (hdemucs_mmi matched upstream at 10–19 dB); the x2 resampler was an
  approximate 63-tap filter instead of julius' (the mdx bags' Demucs members
  matched at 29–40 dB; they are now exact to 117–122 dB); `MultiWrap` padded the
  time axis instead of frequency and dropped the transposed-convolution bias
  correction; Wiener-filtered models swapped the sources and real/imaginary
  axes; and `mx.angle`, which MLX does not have, crashed the magnitude path.
- **The mdx bags could not be converted:** the registry listed every member as
  `DemucsMLX`, so the safe cache refused the real mix of Demucs and HDemucs.
- Chunks run at the same length as upstream: HTDemucs pads each chunk to the
  segment and zero-pads to its training length internally (a custom `--segment`
  matched at 18–25 dB; now 75–82 dB), and other models run each chunk, including
  a short tail, at its own valid length.
- Chunk outputs longer than the segment (any model whose valid length exceeds
  it) are trimmed before overlap-add; the fused kernel read its window out of
  bounds. The kernel now rejects a window/frame length mismatch and indexes
  with 64-bit integers.
- A model, transform or `Separator` built on one thread now works on another:
  side streams are per thread, and cached arrays and weights are materialized
  when they are created.
- `save_audio(bits_per_sample=24)` wrote float32; it now writes 24-bit PCM.
- The wheel did not ship `csrc/demucs_ane.m`, so the native Neural Engine
  bridge was never available to installed users.
- `verify_conversion` called the nonexistent `mx.core.from_dlpack` and compared
  tensors on different devices.

### Changed

- Transformer attention runs in **float32 by default**. float16, the 1.5.x
  behavior, is ~4% faster end to end and matches upstream at 72–79 dB rather
  than 81–87 dB; opt in with `--attention fp16`,
  `Separator(attention_precision="fp16")` or `DEMUCS_MLX_ATTENTION_FP16=1`.
- Overlap-add is streamed: each output span is finalized as soon as no later
  chunk can reach it, with the same fused kernel and summation order
  (bit-identical output). Peak memory on a 21.7-minute track fell from 9.75 GB to
  6.2 GB; time is within 0.4%.
- Separation no longer imports NumPy. The converter moves weights from PyTorch
  with zero-copy CPU DLPack, the Neural Engine bridge passes MLX buffers to
  Core ML by address, and `Separator.separate_tensor` returns NumPy only when
  `return_mx=False` (its default, for API compatibility).
- The Neural Engine path allocates a fresh output per prediction. It reused two
  buffers that MLX could still be reading under `async_eval`.
- `batch_size="auto"` is sized from measurements (interleaved sweeps of batch
  1-8, htdemucs, 216 s): 3 on M4 Pro and 32-core M4 Max, where 3 was 1.5-1.9%
  faster than 2 and larger batches were not faster; 8 on 40-core M4 Max with at
  least 64 GB (about 6% faster than 2, measured under desktop load); 2 elsewhere.
  It was 4 on the former, and `fit_batch_size` then dropped any target to an
  exact divisor even if that added batches, so on many tracks `auto` silently ran
  at 2. Chunks are now spread evenly over the batches the target implies.
- Chunks that run at different input lengths (a short tail on non-HTDemucs
  models) are batched separately instead of padded to the segment.
- `ruff` and `pyright` are clean again; CI runs the parity suite.

## 1.5.1 - 2026-10-01

### Added

- Updated all-time throughput record to **94.8× RTFx** (1.266s peak for 120s separation)
  on Apple M4 Max with MLX 0.32.3 and mlx-spectro 0.9.9.

### Changed

- Upgraded dependency constraints to `mlx>=0.32.3,<0.33`, `mlx-audio-io>=1.3.21,<1.4`, and `mlx-spectro>=0.9.9`.
- Modernized sliding-window overlap-add CPU fallback (`_overlap_add_fallback`) to native
  `out.at[:, off:end].add(...)` on MLX 0.32.3.

## 1.5.0 - 2026-10-01

### Added

- All-time throughput record of **93.9× RTFx** (1.278s peak, 1.285s median for 120s
  separation) on Apple M4 Max with bit-exact reconstruction fidelity (>319 dB SNR).
- Apple Silicon runtime topology auto-tuner (`demucs_mlx/hardware.py`): dynamically
  inspects memory bandwidth, GPU core counts, and system cache capacity to select
  optimal batch sizing (`--batch-size auto`, defaulting to 8 on Max chips and 2 on base).
- Fused 128-bit vectorized 2D-coalesced zero-transpose GLU Metal kernel, eliminating
  transposition round-trips and memory stalls.
- Decoupled waveform branch execution streams (`s_side`), maximizing hardware utilization
  across independent compute paths.
- Native FP16 fused projection GEMMs for MultiHeadAttention (`qkv_proj`, `kv_proj`).
- Whole-forward compilation flag `--compile` / `Separator(compile=True)` for fixed-batch
  graph acceleration.
- Pre-fused attention projections across all 4 sub-models in `BagOfModelsMLX` (`htdemucs_ft`),
  guaranteeing zero repeated projection GEMMs during ensemble inference.
- High-speed single-stem fine-tuned inference (`--stem vocals`), running at **81.3× RTFx**
  (1.476s for 120s audio, 3.95× faster than full ensemble separation).
- Apple Neural Engine (ANE) waveform encoder offload via PyObjC and Grand Central Dispatch
  (`--ane-time-encoder`), bypassing Python GIL overhead.
- Streamlined audio I/O utilizing native `channels_first` decoding and `layout="auto"`
  WAV encoding from `mlx-audio-io` 1.3.20.

### Changed

- Default `--batch-size` is now `'auto'`, configuring inference batches dynamically
  to fit Apple Silicon hardware characteristics.
- Standalone CLI defaults to two concurrent stem writers.

## 1.4.14 - 2026-09-23

### Changed

- The release smoke gate installs the published artifact by URL resolved from the
  release API, instead of resolving the version through the package index. The two
  content-negotiated renderings of `/simple/<project>/` can serve different
  snapshots, so an index resolve could fail for a version that was published and
  intact.
- The publish step refuses to run when the version is already on the index unless
  `allow_existing` is set, so a re-dispatch cannot skip the upload and still report
  success.
- Publishing to PyPI now requires a matching TestPyPI release, with `skip_rc_check`
  to override.

### Fixed

- The GitHub release workflow compared `info.version` from the PyPI project API,
  a cached "latest" view that lags a just-published release; it now relies on the
  per-version endpoint. Its changelog heading matcher also accepts the bracketed
  `## [1.2.3]` style.
- The comment on the overlap-add accumulation guard now records the accurate MLX
  range for the strided scatter-add corruption (below 0.32.0, ml-explore/mlx#3676)
  and notes that the package floor keeps the safe path unconditional.

## 1.4.13 - 2026-09-23

### Removed

- A `.gitignore` entry naming assistant tooling. It lives in a global ignore
  file instead, so the repository does not carry it.

## 1.4.12 - 2026-09-23

### Fixed

- README stated that mlx-audio-io did not yet support MLX 0.32 and that the
  runtime pair was MLX 0.31.2 with mlx-audio-io 1.3.11. Both have been untrue
  since 1.4.7; the package requires `mlx>=0.31.2,<0.33`. Corrected here and in
  `docs/platform.md` and `docs/release.md`.
- The `DEMUCS_MLX_USE_FUSED_GN_GLU` row quoted parity figures from before the
  kernels were fixed in 1.4.9. It now states the measured behaviour: 118 dB SNR
  against the unfused path, deterministic run to run, and a wash on speed.
- Per-version "What changed in" sections replaced with a pointer to this file,
  which the release workflow already reads. They had stopped at 1.4.7 and
  disagreed with the changelog.

## 1.4.11 - 2026-09-23

### Fixed

- `uv.lock` pinned mlx 0.31.2 while mlx-audio-io's sdist, built in an isolated
  environment that resolves `mlx` on its own, compiled against 0.32.2. `uv sync`
  therefore produced a binary the runtime could not load, and CI failed with
  `MLX version mismatch`. The lock now resolves to a consistent pair; the loader
  error that caught it is working as intended.

## 1.4.10 - 2026-09-23

### Changed

- `mlx-spectro` floor raised to 0.9.3. 0.9.2 fixed a compiled `stft`/`istft`
  raising `no usable threadgroup size` on a machine with no tuning cache yet --
  `CachedSpectralPair` was never affected, because `compiled_pair` calls the
  transform eagerly before compiling, but anyone wrapping this package's
  transform in their own `mx.compile` was.

## 1.4.9 - 2026-09-23

### Fixed

- **Fused GroupNorm Metal kernels: added a missing threadgroup barrier.** The
  three-pass reduction shares one `shared_sums` array, and pass 2 could overwrite
  the group mean before every simdgroup had read it — most likely at small
  `elems_per_group`, i.e. the frequency-branch DConv shapes. With the barrier,
  relative error at those shapes drops from 1.6e-02 to **2.0e-07**, output becomes
  run-to-run deterministic, and end-to-end SNR against the unfused path is
  **118.2 dB**. New tests cover parity and determinism at the real shapes.
  `DEMUCS_MLX_USE_FUSED_GN_GLU=1` enables the kernels; they remain off by default
  because the unfused path is the same speed.
- Release workflows retry the install step — a freshly published dependency can
  still be missing from whichever CDN mirror pip resolves against.
- Import ordering in `mlx_demucs.py`, `mlx_hdemucs.py` and `mlx_layers.py`.

## 1.4.8 - 2026-09-23

### Fixed

- **Upgrading past 1.4.6 made every existing cache a hard failure.** 1.4.6
  hardened the safetensors cache and its loader now requires fields no earlier
  cache contains, but `get_mlx_model` only caught `FileNotFoundError` when
  deciding to convert. A cache that existed and could not be validated raised
  `SafeCacheError` straight out to the caller, so the one code path that would
  have recovered never ran — every user with a cache written before 1.4.6 hit an
  unrecoverable error on their first run after upgrading. An unusable cache now
  regenerates from the official registry, which is also the right response to a
  digest mismatch: discard the suspect file and refetch something verified.

## 1.4.7 - 2026-09-22

### Changed

- **Fused GroupNorm+GELU/GLU Metal kernels are off by default.** Measured on a 45 s
  clip through `htdemucs`, they cost about 20 dB SNR against the unfused path
  (19.7 dB on drums, 23.7 dB on other) — audible, not float noise — because the
  kernel uses an erf-approximation GELU and threadgroup reductions whose width
  varies with tensor shape. They are not faster either: 0.783 s fused against
  0.776 s unfused, median of five timed runs, because at real Demucs shapes the
  group size mostly exceeds the hybrid threshold and falls back anyway. Strictly
  worse on both axes. Set `DEMUCS_MLX_USE_FUSED_GN_GLU=1` to re-enable them for
  benchmarking.
- MLX 0.32.x is now allowed (`mlx>=0.31.2,<0.33`). Verified against 0.32.2
  alongside the rest of the stack: the suite passes and the separation path is
  unchanged.
- Requires `mlx-audio-io>=1.3.12,<1.4` and `mlx-spectro>=0.9.0`. The latter carries
  the fix for wrong `differentiable_istft` gradients at batch sizes above 1.
- GitHub Actions updated to Node 24 runtimes.

### Added

- `DEMUCS_MLX_USE_FUSED_GN_GLU` as a runtime switch. The fused kernels were wired
  in unconditionally, so there was no way to rule them out when output looked wrong
  without editing the package. They use simdgroup reductions and threadgroup
  barriers — the kind of code an OS or driver update can perturb — and were a
  leading suspect while investigating
  [mlx-audio-separator#4](https://github.com/ssmall256/mlx-audio-separator/issues/4).
  They turned out to be innocent there; the cause was a cache key mismatch in that
  project. Fused and unfused layers expose identical parameter names, verified by
  test, so an already-converted cache loads either way and the strict key check in
  `_load_exact_model_state` is unaffected; the two paths agree to within 1e-6 on
  the same weights.
- A `## Tuning` section in the README. `DEMUCS_MLX_USE_FUSED_GN_GLU` was
  discoverable only by reading `mlx_layers.py`, and the batch size, shifts and
  overlap defaults were listed in CLI help without saying why they are what they
  are.
- This changelog. Release notes previously lived in `docs/release.md` (covering
  1.0.0 and 1.4.4–1.4.6) and in README "What changed in X" sections (1.4.0,
  1.4.2–1.4.7), which disagreed and were each incomplete.

## 1.4.6 - 2026-08-12

### Security

- Demucs checkpoint loading requires PyTorch 2.6+ with `weights_only=True`, scoped
  safe globals, official filename-hash verification and strict package validation.
- Executable pickle caches replaced with safetensors plus a versioned JSON sidecar,
  verified by SHA-256 before model construction. Legacy pickle caches are never
  opened. Fail-closed regression and CI coverage added.

## 1.4.5 - 2026-08-12

### Fixed

- MLX 0.31.2 thread-local stream failures (issues #5 and #7): decoded arrays are
  now evaluated on their producer thread before handoff to the CLI queue.
- The soxr regression check is capability-aware rather than assuming soxr is
  present.
- Release smoke tests hardened against PyPI index propagation delay.

### Changed

- Default batch size 8 → 2, to avoid memory thrashing on 16–36 GB Macs.
- MLX 0.31.2 and mlx-audio-io 1.3.11 pinned as a compatible native pair.

## 1.4.4 - 2026-06-13

### Fixed

- Split-mode overlap-add corruption on MLX 0.31.2 (issue #1). Long multi-segment
  inputs reconstructed with amplitude spikes of 100x and more.

### Added

- An identity-model overlap-add regression test across durations, overlaps and
  batch sizes, needing no weights; an optional `htdemucs` reproduction test; and an
  overlap-add benchmark script.

## 1.4.3 - 2026-03-06

### Changed

- `resample_mx()` calls `mac.resample()` directly instead of writing and reading a
  temporary file, keeping data on the MLX device.
- Loading simplified to `mac.load(sr=)` now that mlx-audio-io selects `soxr_vhq`
  automatically. Requires `mlx-audio-io>=1.3.9`.

## 1.4.2 - 2026-03-06

### Changed

- Native MLX arrays end to end in `api.py` and `separate.py`, removing the numpy
  round-trip.
- Automatic resampling through `mac.resample()`, using `soxr_vhq` where available.
- Requires `mlx>=0.31.0`, `mlx-audio-io>=1.3.8`, `mlx-spectro>=0.2.4`.

## 1.4.1 - 2026-03-06

Version bump only; no source changes. Its notes were folded into 1.4.2.

## 1.4.0 - 2026-03-02

### Changed

- GroupNorm uses `mx.var`, and `GELUNCL` is replaced with a direct `nn.gelu` call.
- GroupNorm Metal kernel fixed for GPU underutilization at `groups=1`.

### Fixed

- `apply_model` chunk handling, with optional seeded shifts for reproducibility.

### Added

- A `compiled_pair` benchmark script.

## 1.0.0

Initial release: an MLX-native port of Demucs. See `docs/release.md` for the
differences from upstream Demucs.
