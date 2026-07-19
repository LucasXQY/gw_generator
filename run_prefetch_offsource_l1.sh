#!/bin/bash
# L1-only companion to run_prefetch_offsource.sh (halves wall-clock time).
cd "$(dirname "$0")"
PY=/home/bensonxqy/miniforge3/envs/gw-yolo/bin/python
exec $PY -u prefetch_offsource_cache.py \
  --pool gravityspy_pool_3000.csv \
  --cache-dir glitch_cache \
  --detectors L1 \
  --sleep 1.0
