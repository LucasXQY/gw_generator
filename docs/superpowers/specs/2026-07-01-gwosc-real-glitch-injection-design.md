# GWOSC Real-Glitch Injection — Design

**Date:** 2026-07-01
**Status:** Approved (design), pending implementation
**Scope:** `data_generator/`

## Goal

Replace the synthetic single sine-Gaussian glitch with **real detector glitches**
fetched from GWOSC, so the dataset's glitches have realistic (and generally
larger, more varied) time-frequency morphology. The change must preserve the
A1 cross-detector premise (glitches are **independent** across H1/L1) and record
every injected glitch's details to CSV.

## Background / motivation

- Original glitch generator (`noise_generator.sample_glitch`) always emits one
  sine-Gaussian regardless of `glitch_type`; median footprint ≈ 42 ms × 135 Hz
  (~1% × 14% of the image). Real Gravity Spy glitches (Scattered_Light arches,
  Whistles, Koi Fish, …) are much larger and varied. This is why synthetic
  glitches "look smaller" than real GW-YOLO training glitches.
- GWOSC serves **strain by GPS time**; it cannot "randomly generate" a glitch.
  The random-generation analog is **random sampling from a pool of real glitch
  GPS times** (a Gravity Spy catalog), done independently per detector.

## Original random-generation logic (reference)

- Event level: `global_class = rng.choice(GLOBAL_CLASSES)` over 6 classes →
  `has_glitch` true for `bbh_glitch`/`bns_glitch`/`glitch_only` (~50%).
- Detector level (H1, L1 each): independent background + independent glitch →
  distinct `glitch_id` / params. Chirps are shared (coherent); glitches are not.
- Synthetic glitch params: `center_time=U(0.15,0.85·dur)`,
  `center_freq=U(flo·1.5,fhi·0.8)`, `q=U(5,30)`, `amp=U(3,8)`; box = ±3σ.

## Decisions (approved)

1. **Catalog source:** user supplies a Gravity Spy pool CSV via
   `glitch_metadata_csv`. Code randomly samples rows (per detector).
2. **Label box:** derived from the **rendered Q-transform energy ridge** (reuse
   the chirp ridge logic, class = glitch), not from analytic/catalog params.
3. **Amplitude:** whiten + normalize to unit std, then scale by a configurable
   amplitude range (`glitch_amplitude_range`, default (3.0, 8.0)).
4. **Failure mode:** default **hard error** on missing/failed GWOSC/gwpy;
   optional `glitch_allow_synthetic_fallback` to degrade to synthetic.
5. **Synthetic path is unchanged** (backward compatible).

### Additional constraints (approved)

1. **Cache raw/resampled strain only** — never the scaled glitch. Cache the
   GWOSC-fetched, resampled segment plus a sidecar metadata file. whiten / crop
   / unit-std / amp-scale are done at generation time, so one real glitch can be
   reused with different amplitude/placement.
2. **Crop-window fallback** — snippet length = CSV `duration` if present, else
   `glitch_default_duration` (default 0.5 s). The final bbox still comes from the
   ridge; CSV `duration` only controls how much strain is cut for injection.
3. **Placement clamp** — keep `center_time ∈ [0.15, 0.85]·dur`, then additionally
   clamp to `[half_len + margin, dur - half_len - margin]` so long glitches are
   not truncated.
4. **No silent empty box** — for a real glitch, if `glitch_box_from_ridge` finds
   no hot region, resample a new glitch up to `max_glitch_attempts` times; if all
   fail, **hard error** (unless `glitch_allow_synthetic_fallback` is set, in which
   case that sample uses a synthetic glitch with its analytic box).

## Components

### `real_glitch.py` (new) — `RealGlitchProvider`

- `__init__(config)`: load & validate the pool CSV; index rows per detector
  (filter by `ifo` and `label ∈ glitch_types`). Empty/missing pool → hard error
  when `glitch_source == "gwosc"`.
- `sample_glitch(rng, detector) -> RealGlitch`:
  1. Randomly pick a row from `pool[detector]`.
  2. Fetch strain (cache-first): `real_glitch_cache_dir/<ifo>_<gps>_<sr>_<halfwin>.npy`
     + `.json` sidecar; miss → `TimeSeries.fetch_open_data(ifo, gps-halfwin,
     gps+halfwin, sample_rate=sr)`, resample to `sr` if needed, save cache.
     Fetch/import failure → hard error (unless fallback enabled).
  3. Reconstruct `TimeSeries` from cache → `highpass(~15 Hz)` → `whiten()`.
  4. Crop `crop_len = (row.duration or glitch_default_duration)` seconds centered
     on the GPS peak; apply a Tukey taper.
  5. Normalize to unit std; scale by `amp = U(*glitch_amplitude_range)`.
  6. Choose `center_time` per placement clamp; embed into `zeros(n_samples)`.
  7. Return `RealGlitch(series, glitch_id="gspy_<ifo>_<gps>", glitch_type=label,
     start_time, end_time, center_freq, low_freq, high_freq, amplitude=amp,
     gps, snr_catalog, source="gwosc")`. Freq fields are provisional (metadata);
     the label box comes from the ridge.

`RealGlitch` is a dataclass exposing the same attributes `build_dataset` already
reads from `GlitchRealization`, plus `gps`, `snr_catalog`, `source`.

gwpy is imported lazily inside these methods, so synthetic/offline runs are
unaffected and import errors surface only on the gwosc path.

### `label_generator.py` — add `glitch_box_from_ridge`

Mirror `chirp_box_from_ridge` but emit `CLASS_GLITCH` (clip via `glitch_box`).
Signature: `(energy, freqs, times, time_window, ridge_threshold)`.

### `build_dataset.py` — integration

- `__init__`: `self.real_glitch_provider = RealGlitchProvider(cfg) if
  cfg.glitch_source == "gwosc" else None`.
- New helper `_acquire_real_glitch(rng, detector, background)`:
  loop up to `max_glitch_attempts`: sample real glitch → trial render
  `energy_stats(preprocess(background + glitch.series))` → `glitch_box_from_ridge`
  in the placement window. Return `(glitch, trial_box)` on first success.
  Exhausted → synthetic fallback (if enabled) else hard error.
- In the `if has_glitch:` block:
  - gwosc: `glitch, trial_box = self._acquire_real_glitch(...)`; add to series;
    defer box to after the main render.
  - synthetic (or fallback): unchanged analytic `glitch_box` appended pre-render.
- After the main combined render, for the gwosc path compute the final glitch box
  from `norm_stats` energy ridge (fallback to `trial_box`), append it, and update
  the metadata glitch freq/time fields from the final box.
- Trial render uses `energy_stats` (no PNG); extra cost only on gwosc glitch
  samples.

### `config.py` — new fields + metadata columns

New `DatasetConfig` fields:
- `glitch_source: str = "synthetic"`  (`"synthetic"` | `"gwosc"`)
- `real_glitch_cache_dir: Optional[Path] = None`  (default `<output_dir>/glitch_cache`)
- `glitch_fetch_halfwin: float = 4.0`  (seconds each side of GPS; ≥ whiten need)
- `glitch_default_duration: float = 0.5`  (crop fallback)
- `glitch_whiten: bool = True`
- `glitch_amplitude_range: Tuple[float, float] = (3.0, 8.0)`
- `glitch_placement_margin: float = 0.1`  (seconds)
- `max_glitch_attempts: int = 5`
- `glitch_allow_synthetic_fallback: bool = False`

Add to `METADATA_FIELDS`: `glitch_source`, `glitch_gps`, `glitch_snr_catalog`.

### CSV output

- `metadata.csv`: existing glitch fields populated from the final box; new columns
  above.
- `glitch_catalog_used.csv` (new, written in finalize from `self._meta_rows`):
  one row per injected glitch — `sample_id, split, detector, glitch_source,
  glitch_id, glitch_type, glitch_gps, glitch_snr_catalog, glitch_amplitude,
  glitch_start_time, glitch_end_time, glitch_low_freq, glitch_high_freq`.

## Affected files

- new `real_glitch.py`
- `config.py` (fields + `METADATA_FIELDS`)
- `label_generator.py` (+`glitch_box_from_ridge`)
- `build_dataset.py` (provider wiring, acquire-with-retry, ridge box, used-catalog CSV)
- `noise_generator.py` — **unchanged**

## Testing

- Unit: `glitch_box_from_ridge` on a synthetic hot-region energy map (no network).
- Unit: `RealGlitchProvider` cache hit path with a monkeypatched fetch (no network),
  verifying whiten/crop/scale/placement clamp and retry-on-no-ridge.
- Config default (`glitch_source="synthetic"`) reproduces current behavior.
- gwpy/GWOSC network calls are not exercised in CI (local env has gwpy).

## Non-goals

- Downloading/curating the Gravity Spy catalog (user supplies it).
- Changing the synthetic generator or existing datasets.
- Real *noise* backgrounds (only glitches).
