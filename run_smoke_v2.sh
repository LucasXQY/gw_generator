#!/bin/bash
# G1-5: 50-event smoke build of the v2 grouped real-background dataset.
cd "$(dirname "$0")"
PY=/home/bensonxqy/miniforge3/envs/gw-yolo/bin/python
exec $PY build_dataset.py \
  --num-events 50 \
  --seed 2026 \
  --glitch-source gwosc \
  --noise-source gwosc \
  --glitch-metadata-csv gravityspy_pool_3000.csv \
  --real-glitch-cache-dir glitch_cache \
  --output-dir datasets/gw_dataset_v2_realbg_grouped_50_seed2026
