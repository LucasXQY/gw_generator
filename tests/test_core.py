"""Unit tests for the two-detector synthetic GW dataset builder.

Runs on a plain numpy/scipy/matplotlib/pillow install: PyCBC is bypassed via
the analytic waveform fallback and the scipy Q-transform backend is used.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Make the project root importable regardless of unittest's start directory.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from config import DatasetConfig, NEGATIVE_PAIR_TYPES  # noqa: E402
from build_dataset import DatasetBuilder  # noqa: E402
from label_generator import LabelGenerator, TimeFrequencyBox, CLASS_CHIRP  # noqa: E402
from qtransform import normalize_energy_map  # noqa: E402
from validation import (  # noqa: E402
    validate_no_event_leakage,
    validate_no_chirp_leakage,
    validate_positive_pairs,
    validate_pair_split_consistency,
)


def _read_csv(path: Path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _build(tmp: Path, **overrides) -> DatasetConfig:
    # The frequency window is hardcoded to linear 0-1000 Hz inside DatasetConfig;
    # nothing passed here can change it.
    kwargs = dict(
        num_events=12,
        detectors=("H1", "L1"),
        duration=4.0,
        output_dir=tmp,
        qtransform_backend="scipy",
        qtransform_image_width=128,
        qtransform_image_height=128,
        seed=7,
    )
    kwargs.update(overrides)
    config = DatasetConfig(**kwargs)
    DatasetBuilder(config, use_pycbc=False).build()
    return config


class SharedDatasetTests(unittest.TestCase):
    """Most checks share one generated dataset for speed."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name) / "ds"
        cls.config = _build(cls.root)
        cls.meta = _read_csv(cls.root / "metadata.csv")
        cls.events = _read_csv(cls.root / "event_metadata.csv")
        cls.match = _read_csv(cls.root / "match_pairs.csv")
        cls.negatives = _read_csv(cls.root / "negative_pairs.csv")
        cls.pairs = _read_csv(cls.root / "pair_metadata.csv")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # 1 — every event has at least one H1 and one L1 sample
    def test_01_each_event_has_both_detectors(self):
        by_event = {}
        for r in self.meta:
            by_event.setdefault(r["event_id"], set()).add(r["detector"])
        self.assertTrue(by_event)
        for event, dets in by_event.items():
            self.assertIn("H1", dets, event)
            self.assertIn("L1", dets, event)

    # 2 — H1/L1 of the same event share chirp_id
    def test_02_same_event_shares_chirp_id(self):
        by_event = {}
        for r in self.meta:
            by_event.setdefault(r["event_id"], []).append(r)
        for rows in by_event.values():
            chirps = {r["chirp_id"] for r in rows}
            self.assertEqual(len(chirps), 1)

    # 3 — H1/L1 of the same event have different noise_id
    def test_03_same_event_different_noise(self):
        by_event = {}
        for r in self.meta:
            by_event.setdefault(r["event_id"], []).append(r)
        for rows in by_event.values():
            ids = [r["noise_id"] for r in rows]
            self.assertEqual(len(ids), len(set(ids)))

    # 4 — raw strain files exist
    def test_04_raw_strain_files_exist(self):
        found = False
        for r in self.meta:
            if r["raw_strain_path"]:
                found = True
                self.assertTrue((self.root / r["raw_strain_path"]).exists())
        self.assertTrue(found)

    # 5 — normalized strain files exist
    def test_05_normalized_strain_files_exist(self):
        for r in self.meta:
            self.assertTrue((self.root / r["normalized_strain_path"]).exists())

    # 6 — raw Q-transform train images exist
    def test_06_raw_qtransform_train_images_exist(self):
        for r in self.meta:
            if r["qtransform_raw_path"]:
                self.assertTrue((self.root / r["qtransform_raw_path"]).exists())

    # 7 — normalized Q-transform train images exist
    def test_07_normalized_qtransform_train_images_exist(self):
        for r in self.meta:
            self.assertTrue((self.root / r["qtransform_normalized_path"]).exists())

    # 8 — display Q-transform images exist and include a colorbar (wider than train)
    def test_08_display_images_have_colorbar(self):
        from PIL import Image

        checked = False
        for r in self.meta:
            disp = r["qtransform_display_normalized_path"]
            if not disp:
                continue
            train = r["qtransform_normalized_path"]
            with Image.open(self.root / disp) as d, Image.open(self.root / train) as t:
                # The colorbar + axis labels make the display image wider.
                self.assertGreater(d.size[0], t.size[0])
            checked = True
        self.assertTrue(checked)

    # 9 — Q-transform train image size is stable / exact
    def test_09_train_image_size_stable(self):
        from PIL import Image

        for r in self.meta:
            with Image.open(self.root / r["qtransform_normalized_path"]) as img:
                self.assertEqual(img.size, (128, 128))

    # 10 — normalized energy maximum is clipped at/below 25
    def test_10_energy_clipped_at_25(self):
        energy = np.linspace(0, 1000, 4096).reshape(64, 64)
        norm, meta = normalize_energy_map(energy, method="percentile", vmax=25.0, percentile=99.5)
        self.assertLessEqual(float(np.max(norm)), 25.0 + 1e-9)
        self.assertEqual(meta["energy_vmax"], 25.0)
        # And in the actual dataset metadata, vmax is 25.
        for r in self.meta:
            self.assertEqual(float(r["energy_vmax"]), 25.0)

    # 14 — metadata / label / image frequency ranges are the fixed 0-1000 window
    def test_14_frequency_window_fixed_0_1000(self):
        for r in self.meta:
            self.assertEqual(float(r["qtransform_frange_low"]), 0.0)
            self.assertEqual(float(r["qtransform_frange_high"]), 1000.0)
            self.assertEqual(r["frequency_axis_scale"], "linear")
            if r["label_frange_low"]:
                self.assertGreaterEqual(float(r["label_frange_low"]), 0.0 - 1e-6)
                self.assertLessEqual(float(r["label_frange_high"]), 1000.0 + 1e-6)

    # per-type insertion band: chirp energy stays in the source type's band
    def test_14b_per_type_insertion_bands(self):
        bands = self.config.signal_freq_bands
        checked = False
        for r in self.meta:
            if r["has_chirp"] != "1" or not r["chirp_freq_high"]:
                continue
            band = bands[r["signal_type"]]
            self.assertLessEqual(float(r["chirp_freq_high"]), band[1] + 30.0)
            self.assertGreaterEqual(float(r["chirp_freq_low"]), band[0] - 30.0)
            checked = True
        self.assertTrue(checked)

    # 15 — SNR bins all valid; metadata snr_bin values are recognised
    def test_15_snr_bins_valid(self):
        valid = set(self.config.bbh_snr_bins) | set(self.config.bns_snr_bins)
        for r in self.meta:
            if r["snr_bin"]:
                self.assertIn(r["snr_bin"], valid)
        for lo, hi in list(self.config.bbh_snr_bins.values()) + list(self.config.bns_snr_bins.values()):
            self.assertLess(lo, hi)

    # 16 — match_pairs expresses same-chirp cross-detector matches
    def test_16_match_pairs_same_chirp_cross_detector(self):
        by_id = {r["sample_id"]: r for r in self.meta}
        for m in self.match:
            a, p = by_id[m["anchor_sample_id"]], by_id[m["positive_sample_id"]]
            self.assertEqual(a["chirp_id"], p["chirp_id"])
            self.assertEqual(a["event_id"], p["event_id"])
            self.assertNotEqual(m["anchor_detector"], m["positive_detector"])
            self.assertEqual(int(m["same_chirp"]), 1)

    # 17 — every event_id appears in exactly one split
    def test_17_no_event_leakage(self):
        self.assertTrue(validate_no_event_leakage(self.meta))

    # 18 — every chirp_id appears in exactly one split
    def test_18_no_chirp_leakage(self):
        self.assertTrue(validate_no_chirp_leakage(self.meta))

    # 19 — no positive pair crosses splits
    def test_19_positive_pairs_within_split(self):
        self.assertTrue(validate_pair_split_consistency(self.match, self.meta))
        self.assertTrue(validate_positive_pairs(self.match, self.meta))

    # 20 — no negative pair crosses splits (default: not allowed)
    def test_20_negative_pairs_within_split(self):
        self.assertTrue(validate_pair_split_consistency(self.negatives, self.meta))

    # 21 — pair_metadata combines positives and negatives
    def test_21_pair_metadata_combines(self):
        labels = {r["pair_label"] for r in self.pairs}
        self.assertIn("1", labels)
        self.assertIn("0", labels)
        n_pos = sum(1 for r in self.pairs if r["pair_label"] == "1")
        n_neg = sum(1 for r in self.pairs if r["pair_label"] == "0")
        self.assertEqual(n_pos, len(self.match))
        self.assertEqual(n_neg, len(self.negatives))

    # 23 — invalid_delay_same_chirp negatives are pair_label 0 even if same chirp
    def test_23_invalid_delay_is_negative(self):
        rows = [r for r in self.pairs if r["pair_type"] == "invalid_delay_same_chirp"]
        for r in rows:
            self.assertEqual(r["pair_label"], "0")
            self.assertEqual(r["anchor_chirp_id"], r["candidate_chirp_id"])
            self.assertEqual(int(r["same_chirp"]), 1)

    # 24/25/26 — task_protocols.yaml content
    def test_24_task_protocols_has_four_tasks(self):
        text = (self.root / "task_protocols.yaml").read_text(encoding="utf-8")
        for task in (
            "single_detector_detection",
            "cross_detector_matching",
            "coherent_event_detection",
            "low_snr_glitch_rejection",
        ):
            self.assertIn(task, text)

    def test_25_task_protocols_frequency_rule(self):
        text = (self.root / "task_protocols.yaml").read_text(encoding="utf-8")
        self.assertIn("frequency_coordinate_rule", text)
        self.assertIn("frange_high: 1000", text)
        self.assertIn("frequency_axis_scale: linear", text)
        self.assertIn("label_normalization_source: same_as_qtransform", text)
        self.assertIn("per_type_insertion_bands", text)

    def test_26_task_protocols_leakage_rule(self):
        text = (self.root / "task_protocols.yaml").read_text(encoding="utf-8")
        self.assertIn("data_leakage_rule", text)
        self.assertIn("split_unit: event_id", text)

    # 27 — gw_data.yaml points training at normalized Q-transform images
    def test_27_gw_data_yaml_normalized(self):
        text = (self.root / "gw_data.yaml").read_text(encoding="utf-8")
        self.assertIn("train: qtransform_normalized/train", text)
        self.assertIn("val: qtransform_normalized/val", text)
        self.assertIn("test: qtransform_normalized/test", text)


class LabelNormalizationTests(unittest.TestCase):
    """Labels use the fixed linear 0-1000 Hz window as the (explicit) denominator."""

    def test_labels_normalized_against_fixed_0_1000(self):
        # Window is fixed 0-1000 linear; a [0, 500] box -> height 0.5, cy 0.75.
        gen = LabelGenerator(4.0, (0.0, 1000.0), "linear")
        box = TimeFrequencyBox(1.0, 2.0, 0.0, 500.0, CLASS_CHIRP, "test")
        parts = gen.to_yolo(box).split()
        cy, h = float(parts[2]), float(parts[4])
        self.assertAlmostEqual(h, 0.5, places=5)      # 500/1000
        self.assertAlmostEqual(cy, 0.75, places=5)    # top = high freq

    def test_y_mapping_is_window_driven(self):
        # The mapping is driven by the window it is given (not a hidden constant):
        # the same box normalizes differently under a different window.
        box = TimeFrequencyBox(1.0, 2.0, 0.0, 250.0, CLASS_CHIRP, "test")
        h_1000 = float(LabelGenerator(4.0, (0.0, 1000.0), "linear").to_yolo(box).split()[4])
        h_500 = float(LabelGenerator(4.0, (0.0, 500.0), "linear").to_yolo(box).split()[4])
        self.assertAlmostEqual(h_1000, 0.25, places=5)
        self.assertAlmostEqual(h_500, 0.5, places=5)
        self.assertNotAlmostEqual(h_1000, h_500, places=3)


class NegativeTypeCoverageTests(unittest.TestCase):
    """22 — negative_pairs.csv contains all enabled negative pair types."""

    def test_22_all_enabled_negative_types_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ds"
            _build(
                root,
                num_events=60,
                seed=11,
                save_display_images=False,
                save_raw_outputs=False,
            )
            rows = _read_csv(root / "negative_pairs.csv")
            present = {r["negative_type"] for r in rows}
            for ntype in NEGATIVE_PAIR_TYPES:
                self.assertIn(ntype, present, f"missing negative type {ntype}")


class ConfigTests(unittest.TestCase):
    def test_three_way_ratio_must_sum_to_one(self):
        with self.assertRaises(ValueError):
            DatasetConfig(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)

    def test_fixed_window_above_nyquist_rejected(self):
        # The fixed 1000 Hz upper edge requires a sample rate of at least 2 kHz.
        with self.assertRaises(ValueError):
            DatasetConfig(sample_rate=1024)

    def test_window_is_hardcoded_0_1000_linear(self):
        cfg = DatasetConfig(num_events=1)
        self.assertEqual(cfg.frange_low, 0.0)
        self.assertEqual(cfg.frange_high, 1000.0)
        self.assertEqual(cfg.frequency_axis_scale, "linear")


if __name__ == "__main__":
    unittest.main(verbosity=2)
