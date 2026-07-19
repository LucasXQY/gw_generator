#!/bin/bash
# G1-6 main dataset: 1500 events, real glitches + real off-source backgrounds,
# grouped 4096s source isolation (D1/D2/D3).
cd "$(dirname "$0")"
PY=/home/bensonxqy/miniforge3/envs/gw-yolo/bin/python
exec $PY -u build_dataset.py \
  --num-events 1500 \
  --seed 2026 \
  --glitch-source gwosc \
  --noise-source gwosc \
  --glitch-metadata-csv gravityspy_pool_3000.csv \
  --real-glitch-cache-dir glitch_cache \
  --output-dir datasets/gw_dataset_v2_realbg_grouped_1500_seed2026
