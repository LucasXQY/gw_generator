#!/bin/bash
# G1-6 secondary dataset: 1500 events, fully synthetic (glitches + noise) --
# the D2 ablation counterpart with the same event-level split protocol.
cd "$(dirname "$0")"
PY=/home/bensonxqy/miniforge3/envs/gw-yolo/bin/python
exec $PY -u build_dataset.py \
  --num-events 1500 \
  --seed 2026 \
  --glitch-source synthetic \
  --noise-source synthetic \
  --output-dir datasets/gw_dataset_v2_synthbg_grouped_1500_seed2026
