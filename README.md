# gw_generator — two-detector synthetic gravitational-wave dataset

A **dual-detector (H1 + L1), event-level** synthetic gravitational-wave dataset
generator for GW-YOLO-style machine-learning benchmarks. Each astrophysical
*event* shares one physical chirp across detectors, while every detector-level
*sample* gets its own noise realization, detector effects, strain files,
Q-transform images, YOLO labels, and pairing metadata.

> This is a *GW-YOLO-style* approximation for research/education, **not** an
> official reproduction of any published GW-YOLO dataset.

## Benchmark tasks

| # | Task | Model type | Labels |
|---|------|------------|--------|
| 1 | Single-detector chirp/glitch detection | YOLO object detection | `labels_yolo/` |
| 2 | Cross-detector same-chirp matching | Siamese / pair classifier | `match_pairs.csv`, `negative_pairs.csv`, `pair_metadata.csv` |
| 3 | Coherent event-level detection | Event-level fusion | `event_metadata.csv` |
| 4 | Low-SNR robustness & glitch rejection | Any of the above, stratified | `metadata.csv` (SNR bins, glitch flags) |

## Features

- **Shared chirp, independent detectors** — one waveform + intrinsic parameters
  per event; per-detector noise, optional glitch, arrival-time delay (±10 ms),
  amplitude scaling, optional sign flip, and per-detector SNR.
- **Physical waveforms** via PyCBC (BBH/BNS), with an analytic inspiral-chirp
  fallback (frequency sweep set by chirp mass + `f_lower`) so the pipeline runs
  with no scientific stack.
- **LIGO Q-transform** via GWpy (the Omega Q-scan) by default, with a built-in
  constant-Q (Gaussian filterbank) fallback that needs only numpy/scipy.
- **aLIGO-colored noise** (seismic wall, ~215 Hz bucket, shot-noise rise) coloring
  the strain; GWpy's `q_transform` returns *normalized energy* (per-frequency
  median normalization), so the Q-transform images are flattened by GWpy itself
  rather than by a separate strain-whitening step.
- **Leakage-safe** event-level train/val/test split with post-build validators.
- **Rich metadata** — detector-sample, event, positive/negative/combined pair
  tables, plus a full `dataset_config.json` for reproducibility.

## Install

```bash
pip install -r requirements.txt
```

Core deps (`numpy`, `scipy`, `matplotlib`, `pillow`) run everything. `pycbc` and
`gwpy` are optional but recommended for physical waveforms and the real LIGO
Q-transform; install them in a conda env (e.g. `conda activate gw-yolo`).

## Quickstart

```bash
# Default: GWpy Q-transform + PyCBC waveforms
python build_dataset.py --num-events 5 --detectors H1 L1 --duration 4.0 \
  --output-dir out_smoke --seed 42

# No scientific stack required (built-in constant-Q + analytic chirp)
python build_dataset.py --num-events 5 --detectors H1 L1 --duration 4.0 \
  --output-dir out_smoke --qtransform-backend scipy --no-pycbc --seed 42
```

Run the tests (no PyCBC/GWpy needed — they use the offline fallbacks):

```bash
python -m unittest discover -s tests -v
```

### Scaling up

Each detector-level sample writes 1 normalized-train image (the YOLO input);
each event has H1+L1. So **N events ⇒ 2N training images**. For ~1000 YOLO
training images use `--num-events 500`. Full outputs (raw + display) add 3 more
PNGs per sample. Speed tips: `--no-raw-outputs` / `--no-display-images` to write
less, `--no-pycbc` to skip long BNS inspiral generation, or
`--qtransform-backend scipy` for a faster transform. `metadata.csv` is streamed
per event, so progress is visible (`wc -l <out>/metadata.csv`) and interrupted
runs still leave a coherent dataset.

## Frequency coordinate system (fixed)

The Q-transform window, display axis, and YOLO y-normalization all use a
**hardcoded log 20–1000 Hz** system (`coords.py`), so images and labels always
share one coordinate frame. A constant-Q transform is log-spaced in frequency by
construction (and GWpy returns log-spaced tiles), so a log axis matches both the
transform's native resolution and the GW-YOLO / LIGO Omega-scan convention; 20 Hz
is the low edge (a log axis excludes 0 Hz, and 20 Hz is the common CBC cutoff).
`--frange-*` / `--frequency-axis-scale` are accepted but ignored. Chirp
*insertion* is band-limited to each source type's web-verified LIGO-band range:

| Type | Insertion band | Basis |
|------|----------------|-------|
| BBH  | 20–350 Hz  | GW150914 swept 35→250 Hz, FLSO ~220 Hz |
| BNS  | 20–1000 Hz | inspiral sweeps from ~10 Hz to ~1 kHz coalescence |
| NSBH | 20–400 Hz  | typical FLSO ~400 Hz |

## Output layout

```
<output-dir>/
  metadata.csv            one row per detector-level sample (rich schema)
  event_metadata.csv      one row per astrophysical event
  match_pairs.csv         positive (same-chirp, cross-detector) pairs
  negative_pairs.csv      negatives (7 types incl. hard negatives)
  pair_metadata.csv       combined positive + negative table (pair_label 0/1)
  task_protocols.yaml     the four benchmark task definitions + rules
  gw_data.yaml            YOLO data config (points at qtransform_normalized)
  dataset_config.json     full DatasetConfig dump (provenance)
  raw_series/{train,val,test}/{H1,L1}/{sample_id}.npy
  normalized_series/{...}/{sample_id}.npy
  qtransform_raw/{...}/{sample_id}.png               pure spectrogram (analysis)
  qtransform_normalized/{...}/{sample_id}.png        default YOLO input
  qtransform_display_raw/{...}/{sample_id}.png       axes + 0–25 colorbar
  qtransform_display_normalized/{...}/{sample_id}.png
  labels_yolo/{train,val,test}/{H1,L1}/{sample_id}.txt
```

## Metadata schema (highlights)

`metadata.csv` (one row per sample) records IDs and relationships
(`sample_id`/`event_id`/`chirp_id`/`counterpart_sample_ids`), class & SNR
(`global_class`/`snr_bin`/`detector_snr`/`network_snr`), detector effects
(`detector_time_delay`/`amplitude_scale`/`phase_or_sign_flip`), the frequency
coordinate provenance, all file paths, energy stats, noise/glitch IDs, and —
denormalized for single-table ML use — physical parameters (`mass1/2`,
`spin1z/2z`, `chirp_mass`, `total_mass`, `mass_ratio`, `chi_eff`, `distance`,
`f_lower`, `waveform_approximant`, `waveform_source`), the YOLO chirp box
(`chirp_yolo_cx/cy/w/h`, `has_label`, `num_boxes`), glitch time-frequency detail,
and acquisition fields (`sample_rate`, `duration`, `n_samples`).

## Module map

| File | Responsibility |
|------|----------------|
| `config.py` | `DatasetConfig` + all CSV schemas (single source of truth) |
| `waveform_generator.py` | PyCBC BBH/BNS + analytic-chirp fallback |
| `noise_generator.py` | aLIGO-colored Gaussian noise + synthetic glitches |
| `injection.py` | target-SNR injection, merger-anchored, detector effects |
| `preprocessing.py` | band-limit the strain to the Q-transform window (energy normalization is GWpy's job) |
| `coords.py` | the shared frequency ↔ image-coordinate mapping |
| `qtransform.py` | GWpy / constant-Q transforms, energy norm, train+display images |
| `label_generator.py` | instantaneous-frequency labels + Q-ridge fallback |
| `pairs.py` | positive/negative/combined pair generation |
| `validation.py` | leakage & pair-consistency validators |
| `protocols.py` | `task_protocols.yaml` + `gw_data.yaml` writers |
| `build_dataset.py` | `DatasetBuilder` + CLI |

## Notes & limitations

- The analytic waveform is a leading-order (Newtonian/quadrupole) inspiral; use
  PyCBC for production fidelity.
- The constant-Q `scipy` backend approximates a Q-transform; GWpy is the real
  Omega Q-scan.
- Per-detector delay/amplitude are sampled, not derived from sky position +
  antenna response (so no `ra/dec/inclination` fields are stored).

## Acknowledgements

Builds on the open LIGO/Virgo software ecosystem — [GWpy](https://gwpy.github.io/),
[PyCBC](https://pycbc.org/) — and is inspired by the GW-YOLO approach to
treating gravitational-wave detection as time-frequency object detection.
