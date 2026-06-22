# GW-YOLO-style two-detector synthetic dataset generator

Generates a **dual-detector (H1 + L1), event-level** synthetic gravitational-wave
dataset for four benchmark tasks:

1. Single-detector chirp/glitch detection (YOLO)
2. Cross-detector same-chirp matching (Siamese / pair classification)
3. Coherent event-level detection
4. Low-SNR robustness & glitch rejection

It is a *GW-YOLO-style* approximation, not an official reproduction.

## Install

```bash
pip install -r requirements.txt
```

Core deps (numpy, scipy, matplotlib, pillow) are enough to run everything.
PyCBC / GWpy are optional: when present they provide physical waveforms and a
true Q-transform; when absent the builder uses an analytic inspiral chirp
(frequency sweep determined by masses + `f_lower`) and a scipy spectrogram.

## Frequency coordinate system (fixed)

The Q-transform window, the display axis, and the YOLO y-normalization all use a
**hardcoded linear 0-1000 Hz** coordinate system. It is intentionally not
configurable — `--frange-low/--frange-high/--frequency-axis-scale` are accepted
for backward compatibility but **ignored**.

Chirp *insertion* is constrained to each source type's web-verified common
LIGO-band range, so a signal's visible energy lands where that type radiates:

| Type | Insertion band | Basis |
|---|---|---|
| BBH  | 20-350 Hz  | GW150914 swept 35->250 Hz, FLSO ~220 Hz |
| BNS  | 20-1000 Hz | inspiral sweeps from ~10 Hz to ~1 kHz coalescence |
| NSBH | 20-400 Hz  | typical FLSO ~400 Hz |

## Run

```bash
python build_dataset.py --num-events 5 --detectors H1 L1 --duration 4.0 \
  --output-dir out_smoke --qtransform-backend scipy --seed 42

python build_dataset.py --num-events 5 --detectors H1 L1 --duration 4.0 \
  --output-dir out_smoke_2 --qtransform-backend scipy --seed 43
```

Add `--no-pycbc` to force the analytic fallback even if PyCBC is installed.
(The original `--frange-high 512 / 2048` commands still run, but the window
stays fixed at 0-1000 Hz and a notice is printed.)

## Tests

```bash
python -m unittest discover -s tests -v
```

## Output layout (flat, event-level 3-way split)

```
<output-dir>/
  metadata.csv  event_metadata.csv
  match_pairs.csv  negative_pairs.csv  pair_metadata.csv
  task_protocols.yaml  gw_data.yaml
  raw_series/{train,val,test}/{H1,L1}/{sample_id}.npy
  normalized_series/{train,val,test}/{H1,L1}/{sample_id}.npy
  qtransform_raw/...                  (pure spectrogram, train image)
  qtransform_normalized/...           (default YOLO input)
  qtransform_display_raw/...          (axes + colorbar)
  qtransform_display_normalized/...
  labels_yolo/{train,val,test}/{H1,L1}/{sample_id}.txt
```

## Key design points

- **Shared chirp, independent detectors.** One event = one waveform + intrinsic
  params, shared by H1/L1. Each detector gets its own noise realization
  (`noise_id`), optional glitch, arrival-time delay (±10 ms), amplitude scale,
  optional sign flip, and its own SNR.
- **Frequency coordinate system.** A fixed linear 0-1000 Hz window defines the
  Q-transform image *and* the YOLO y-axis via the single mapping in `coords.py`,
  so image and labels always share one coordinate system. Chirp insertion is
  band-limited to each source type's realistic LIGO-band range.
- **Labels** come from the clean waveform's instantaneous frequency (Hilbert),
  with a Q-transform energy-ridge fallback, clipped to the configured window.
- **Leakage-safe.** Splitting is at the event level first; chirps and pairs
  never cross train/val/test. Validators run before success is reported.
```
