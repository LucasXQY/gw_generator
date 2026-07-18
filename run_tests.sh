#!/bin/bash
# Run the data_generator test suite in the WSL gw-yolo env and print a
# buffering-safe summary (stdout prints otherwise scramble the tail).
cd "$(dirname "$0")"
PY=/home/bensonxqy/miniforge3/envs/gw-yolo/bin/python
$PY -m unittest discover -s tests "$@" > /tmp/suite.log 2>&1
status=$?
grep -E 'Ran [0-9]+ tests|^OK|^FAILED' /tmp/suite.log
grep -E '^(FAIL|ERROR):' /tmp/suite.log
exit $status
