"""Writers for ``task_protocols.yaml`` and ``gw_data.yaml``.

The protocol file is emitted as literal, valid YAML (no PyYAML dependency at
build time). It documents the four benchmark tasks, the frequency-coordinate
rule, and the data-leakage rule at the dataset root.
"""

from __future__ import annotations

from pathlib import Path

from config import DatasetConfig

DATASET_NAME = "gw_synthetic_h1_l1_coherent"


def write_task_protocols(config: DatasetConfig, root: Path) -> Path:
    path = Path(root) / "task_protocols.yaml"
    content = f"""dataset_name: {DATASET_NAME}
default_training_images: qtransform_normalized
default_label_dir: labels_yolo
split_key: event_id

tasks:
  single_detector_detection:
    description: Detect chirp and glitch candidates from a single detector Q-transform image.
    inputs:
      - qtransform_normalized/{{split}}/{{detector}}/{{sample_id}}.png
    labels:
      - labels_yolo/{{split}}/{{detector}}/{{sample_id}}.txt
    metadata:
      - metadata.csv
    metrics:
      - mAP50
      - mAP50-95
      - precision
      - recall

  cross_detector_matching:
    description: Decide whether two detector-level candidates correspond to the same astrophysical chirp.
    inputs:
      - anchor_sample_id
      - candidate_sample_id
      - qtransform_normalized images
      - optional detector metadata
    labels:
      - pair_metadata.csv
    positive_pairs:
      - match_pairs.csv
    negative_pairs:
      - negative_pairs.csv
    metrics:
      - pair_matching_accuracy
      - pair_matching_AUC
      - pair_matching_F1
      - false_positive_rate
      - false_negative_rate

  coherent_event_detection:
    description: Combine H1 and L1 detector-level evidence to decide whether a coherent astrophysical event exists.
    inputs:
      - all detector samples sharing an event_id
      - detector-level YOLO predictions
      - pairwise matching scores
      - event_metadata.csv
    labels:
      - event_metadata.csv
    metrics:
      - event_level_precision
      - event_level_recall
      - event_level_F1
      - event_false_positive_rate

  low_snr_glitch_rejection:
    description: Evaluate whether cross-detector coherence improves robustness under low SNR and glitch contamination.
    inputs:
      - qtransform_normalized images
      - metadata.csv
      - pair_metadata.csv
    evaluation_groups:
      - very_low_snr
      - low_snr
      - medium_snr
      - high_snr
      - glitch_contaminated
      - pure_noise
    metrics:
      - low_SNR_recall
      - glitch_rejection_rate
      - event_level_AUC
      - calibration_error

frequency_coordinate_rule:
  qtransform_window: fixed_linear_0_1000_hz
  frange_low: {config.frange_low:g}
  frange_high: {config.frange_high:g}
  frequency_axis_scale: {config.frequency_axis_scale}
  label_normalization_source: same_as_qtransform
  window_configurable: false
  per_type_insertion_bands:
    BBH: [{config.signal_freq_bands['BBH'][0]:g}, {config.signal_freq_bands['BBH'][1]:g}]
    BNS: [{config.signal_freq_bands['BNS'][0]:g}, {config.signal_freq_bands['BNS'][1]:g}]
    NSBH: [{config.signal_freq_bands['NSBH'][0]:g}, {config.signal_freq_bands['NSBH'][1]:g}]

data_leakage_rule:
  split_unit: event_id
  forbid_same_event_across_splits: true
  forbid_same_chirp_across_splits: true
  forbid_positive_pair_across_splits: true
  forbid_negative_pair_across_splits: true
"""
    path.write_text(content, encoding="utf-8")
    return path


def write_gw_data_yaml(config: DatasetConfig, root: Path) -> Path:
    """YOLO data config: training points at the normalized Q-transform images."""
    path = Path(root) / "gw_data.yaml"
    content = (
        f"path: {Path(root).as_posix()}\n"
        "train: qtransform_normalized/train\n"
        "val: qtransform_normalized/val\n"
        "test: qtransform_normalized/test\n"
        "nc: 2\n"
        "names: ['chirp', 'glitch']\n"
    )
    path.write_text(content, encoding="utf-8")
    return path
