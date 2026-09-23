# Changelog

All notable changes to this project are documented in this file. This is the file
the GitHub release workflow reads, so every release needs an entry here before it
can be published.

Entries before 1.4.7 were reconstructed from the commit history, `docs/release.md`
and the README after the fact.

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
