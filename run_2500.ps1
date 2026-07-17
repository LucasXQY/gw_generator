# Generate a 2500-sample dataset (1250 events x 2 detectors = 2500 detector samples).
# Same settings as the original 1000-sample run (gw_dataset_1000): GWpy Q-transform,
# PyCBC waveforms, all 8 glitch types, BBH + BNS chirps. Run inside the conda env
# that has pycbc + gwpy installed (e.g. `conda activate gw-yolo`).

python build_dataset.py `
  --num-events 1250 `
  --detectors H1 L1 `
  --duration 4.0 `
  --sample-rate 4096 `
  --output-dir gw_dataset_2500 `
  --seed 123
