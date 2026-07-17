"""Tests for label_viz.draw_boxes: metadata/label-file box extraction and
mismatch detection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from label_viz.draw_boxes import boxes_from_metadata_row, boxes_mismatch


def test_boxes_from_metadata_row_chirp_present():
    row = {"has_chirp": "1", "has_label": "1",
           "chirp_yolo_cx": "0.5", "chirp_yolo_cy": "0.67",
           "chirp_yolo_w": "0.99", "chirp_yolo_h": "0.65"}
    boxes = boxes_from_metadata_row(row)
    assert boxes == [(0, 0.5, 0.67, 0.99, 0.65)]


def test_boxes_from_metadata_row_no_chirp():
    row = {"has_chirp": "0", "has_label": "0",
           "chirp_yolo_cx": "", "chirp_yolo_cy": "",
           "chirp_yolo_w": "", "chirp_yolo_h": ""}
    assert boxes_from_metadata_row(row) == []


def test_boxes_mismatch_detects_offset_and_missing():
    meta = [(0, 0.5, 0.67, 0.99, 0.65)]
    same = [(0, 0.5005, 0.6702, 0.99, 0.65)]
    off = [(0, 0.42, 0.67, 0.99, 0.65)]
    assert not boxes_mismatch(meta, same, tol=0.01)
    assert boxes_mismatch(meta, off, tol=0.01)
    assert boxes_mismatch(meta, [], tol=0.01)   # label file lost the chirp
    assert boxes_mismatch([], same, tol=0.01)   # metadata lost the chirp
    assert not boxes_mismatch([], [], tol=0.01)
