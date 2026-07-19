#!/bin/bash
# Bulk off-source prefetch: one whole-file download per (detector, 4096 s file).
cd "$(dirname "$0")"
PY=/home/bensonxqy/miniforge3/envs/gw-yolo/bin/python
exec $PY -u prefetch_offsource_cache.py \
  --pool gravityspy_pool_3000.csv \
  --cache-dir glitch_cache \
  --sleep 1.0
