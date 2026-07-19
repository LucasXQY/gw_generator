"""G1 leakage tests: GWOSC 4096 s source-group isolation across splits.

Covers three layers:

1. ``split_source_groups`` primitives: the group-id formula (must match
   ``select_pool_subset.FILE_SECONDS`` semantics) and the deterministic
   group-to-split assignment.
2. ``validation`` audit functions: cross-split ``glitch_id`` leakage,
   cross-split source-group leakage (derived from detector+gps when the
   explicit column is absent, so pre-G1 datasets can be audited), and
   background-domain/label collinearity.
3. On-disk audits: ``gw_dataset_3000_real`` is known-leaky and the audit
   must detect that; the primary dataset (newest ``gw_dataset_v2_*`` if
   built, else ``gw_dataset_3000_real``) must pass — this test is RED by
   design until the G1 rebuild lands.
"""

from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

# Make the project root importable regardless of unittest's start directory.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from split_source_groups import (  # noqa: E402
    FILE_SECONDS,
    assign_groups_to_splits,
    offsource_candidate_gps,
    source_group,
)
from validation import (  # noqa: E402
    run_all_validations,
    validate_background_domain_decoupled,
    validate_no_background_group_leakage,
    validate_no_glitch_leakage,
    validate_no_source_group_leakage,
)

DATASETS = ROOT / "datasets"
LEGACY_REAL = DATASETS / "gw_dataset_3000_real"


def _row(split, detector="H1", glitch_id="", glitch_gps="", has_glitch="0",
         noise_type="gaussian_aligo_colored", **extra):
    r = {
        "sample_id": f"{split}_{len(extra)}_{glitch_id or 'x'}",
        "event_id": f"ev_{split}_{glitch_id or 'x'}_{glitch_gps}",
        "chirp_id": "",
        "split": split,
        "detector": detector,
        "glitch_id": glitch_id,
        "glitch_gps": glitch_gps,
        "has_glitch": has_glitch,
        "noise_type": noise_type,
    }
    r.update(extra)
    return r


def _glitch_row(split, gps, detector="H1", glitch_id=None, **extra):
    gid = glitch_id if glitch_id is not None else f"gspy_{detector}_{gps:.4f}"
    return _row(split, detector=detector, glitch_id=gid,
                glitch_gps=f"{gps:.6f}", has_glitch="1",
                noise_type="real_gwosc", **extra)


class SourceGroupFormulaTests(unittest.TestCase):
    def test_group_is_detector_prefixed_4096s_file_id(self):
        # 1126259462 // 4096 == 274965 (the GW150914 GWOSC file).
        self.assertEqual(source_group("H1", 1126259462.0), "H1:274965")
        self.assertEqual(source_group("L1", 1126259462.0), "L1:274965")

    def test_group_accepts_string_gps_from_csv(self):
        self.assertEqual(source_group("H1", "1126259462.000000"), "H1:274965")

    def test_same_file_same_group_different_file_different_group(self):
        base = 274965 * 4096.0
        self.assertEqual(source_group("H1", base + 1.0), source_group("H1", base + 4095.0))
        self.assertNotEqual(source_group("H1", base + 1.0), source_group("H1", base + 4096.0))

    def test_file_seconds_matches_select_pool_subset(self):
        import select_pool_subset
        self.assertEqual(FILE_SECONDS, select_pool_subset.FILE_SECONDS)


class AssignGroupsToSplitsTests(unittest.TestCase):
    RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

    def _pool(self, groups_per_det=10, rows_per_group=5):
        rows = []
        for ifo in ("H1", "L1"):
            for g in range(groups_per_det):
                base = (500000 + g) * FILE_SECONDS
                for k in range(rows_per_group):
                    rows.append({"ifo": ifo, "gps": base + 10.0 + 300.0 * k})
        return rows

    def test_every_group_assigned_to_exactly_one_split(self):
        rows = self._pool()
        assignment = assign_groups_to_splits(rows, self.RATIOS, seed=2026)
        expected_groups = {source_group(r["ifo"], r["gps"]) for r in rows}
        self.assertEqual(set(assignment), expected_groups)
        self.assertTrue(set(assignment.values()) <= set(self.RATIOS))

    def test_deterministic_for_same_seed_different_for_other_seed(self):
        rows = self._pool()
        a = assign_groups_to_splits(rows, self.RATIOS, seed=2026)
        b = assign_groups_to_splits(rows, self.RATIOS, seed=2026)
        c = assign_groups_to_splits(rows, self.RATIOS, seed=7)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_each_split_gets_groups_for_each_detector(self):
        rows = self._pool(groups_per_det=12)
        assignment = assign_groups_to_splits(rows, self.RATIOS, seed=2026)
        for split in self.RATIOS:
            for ifo in ("H1", "L1"):
                groups = [g for g, s in assignment.items()
                          if s == split and g.startswith(f"{ifo}:")]
                self.assertTrue(
                    groups,
                    f"split {split} has no {ifo} source groups: {assignment}",
                )

    def test_row_weighted_shares_approximate_ratios(self):
        rows = self._pool(groups_per_det=40, rows_per_group=5)
        assignment = assign_groups_to_splits(rows, self.RATIOS, seed=2026)
        counts = {s: 0 for s in self.RATIOS}
        for r in rows:
            counts[assignment[source_group(r["ifo"], r["gps"])]] += 1
        total = sum(counts.values())
        for split, ratio in self.RATIOS.items():
            self.assertAlmostEqual(counts[split] / total, ratio, delta=0.10)


class OffsourceGridTests(unittest.TestCase):
    """Off-source candidate GPS come from a deterministic per-file grid so a
    bulk prefetch and the build-time sampler always agree on cache keys."""

    def test_grid_deterministic_interior_and_increasing(self):
        pts = offsource_candidate_gps("H1:274965", halfwin=4.0, step=32.0)
        self.assertEqual(
            list(pts), list(offsource_candidate_gps("H1:274965", halfwin=4.0, step=32.0))
        )
        base = 274965 * FILE_SECONDS
        self.assertTrue(all(base + 4.0 <= p <= base + FILE_SECONDS - 4.0 for p in pts))
        self.assertTrue(all(b > a for a, b in zip(pts, pts[1:])))
        self.assertGreater(len(pts), 100)

    def test_grid_depends_only_on_file_id(self):
        h1 = offsource_candidate_gps("H1:500000", halfwin=4.0, step=32.0)
        l1 = offsource_candidate_gps("L1:500000", halfwin=4.0, step=32.0)
        self.assertEqual(list(h1), list(l1))


class GlitchLeakageValidatorTests(unittest.TestCase):
    def test_same_glitch_id_across_splits_raises(self):
        gps = 274965 * 4096.0 + 100.0
        rows = [_glitch_row("train", gps), _glitch_row("test", gps)]
        with self.assertRaisesRegex(AssertionError, "glitch_id"):
            validate_no_glitch_leakage(rows)

    def test_duplicates_within_one_split_are_allowed(self):
        gps = 274965 * 4096.0 + 100.0
        rows = [_glitch_row("train", gps), _glitch_row("train", gps)]
        self.assertTrue(validate_no_glitch_leakage(rows))

    def test_empty_glitch_id_rows_are_skipped(self):
        rows = [_row("train"), _row("val"), _row("test")]
        self.assertTrue(validate_no_glitch_leakage(rows))


class SourceGroupLeakageValidatorTests(unittest.TestCase):
    def test_distinct_ids_same_4096s_file_across_splits_raises(self):
        base = 274965 * 4096.0
        rows = [
            _glitch_row("train", base + 100.0),
            _glitch_row("test", base + 900.0),  # different glitch, same file
        ]
        with self.assertRaisesRegex(AssertionError, "source_group|source group"):
            validate_no_source_group_leakage(rows)

    def test_disjoint_files_across_splits_pass(self):
        rows = [
            _glitch_row("train", 274965 * 4096.0 + 100.0),
            _glitch_row("val", 274966 * 4096.0 + 100.0),
            _glitch_row("test", 274967 * 4096.0 + 100.0),
        ]
        self.assertTrue(validate_no_source_group_leakage(rows))

    def test_same_file_id_different_detectors_is_not_leakage(self):
        base = 274965 * 4096.0
        rows = [
            _glitch_row("train", base + 100.0, detector="H1"),
            _glitch_row("test", base + 900.0, detector="L1"),
        ]
        self.assertTrue(validate_no_source_group_leakage(rows))

    def test_explicit_glitch_source_group_column_is_preferred(self):
        # Contradictory on purpose: gps values collide but the explicit
        # column says the groups differ -> the column must win.
        base = 274965 * 4096.0
        rows = [
            _glitch_row("train", base + 100.0, glitch_source_group="H1:1"),
            _glitch_row("test", base + 900.0, glitch_source_group="H1:2"),
        ]
        self.assertTrue(validate_no_source_group_leakage(rows))
        rows[1]["glitch_source_group"] = "H1:1"
        with self.assertRaises(AssertionError):
            validate_no_source_group_leakage(rows)


class BackgroundDomainValidatorTests(unittest.TestCase):
    def test_domain_collinear_with_has_glitch_raises(self):
        # The gw_dataset_3000_real pattern: real noise iff has_glitch.
        rows = [
            _glitch_row("train", 274965 * 4096.0 + 1.0),
            _row("train", has_glitch="0", noise_type="gaussian_aligo_colored"),
        ]
        with self.assertRaisesRegex(AssertionError, "background"):
            validate_background_domain_decoupled(rows)

    def test_uniform_real_domain_passes(self):
        rows = [
            _glitch_row("train", 274965 * 4096.0 + 1.0),
            _row("train", has_glitch="0", noise_type="real_gwosc_offsource"),
        ]
        self.assertTrue(validate_background_domain_decoupled(rows))

    def test_uniform_synthetic_domain_passes(self):
        rows = [
            _row("train", has_glitch="1", noise_type="gaussian_aligo_colored",
                 glitch_id="synth_001"),
            _row("train", has_glitch="0", noise_type="gaussian_aligo_colored"),
        ]
        self.assertTrue(validate_background_domain_decoupled(rows))

    def test_explicit_background_source_column_is_preferred(self):
        rows = [
            _row("train", has_glitch="1", glitch_id="g1",
                 noise_type="real_gwosc", background_source="gwosc"),
            _row("train", has_glitch="0",
                 noise_type="gaussian_aligo_colored", background_source="gwosc"),
        ]
        self.assertTrue(validate_background_domain_decoupled(rows))


class BackgroundGroupLeakageValidatorTests(unittest.TestCase):
    def test_background_group_across_splits_raises(self):
        rows = [
            _row("train", background_source="gwosc",
                 background_source_group="H1:274965", background_gps="1126256641.0"),
            _row("test", background_source="gwosc",
                 background_source_group="H1:274965", background_gps="1126257000.0"),
        ]
        with self.assertRaises(AssertionError):
            validate_no_background_group_leakage(rows)

    def test_disjoint_background_groups_pass(self):
        rows = [
            _row("train", background_source="gwosc",
                 background_source_group="H1:274965"),
            _row("test", background_source="gwosc",
                 background_source_group="H1:274966"),
        ]
        self.assertTrue(validate_no_background_group_leakage(rows))

    def test_rows_without_background_group_are_skipped(self):
        rows = [_row("train"), _row("test")]
        self.assertTrue(validate_no_background_group_leakage(rows))


class RunAllValidationsWiringTests(unittest.TestCase):
    """The builder-side aggregator must enforce the glitch/source-group
    checks unconditionally -- allow_cross_split only relaxes pair checks."""

    def test_glitch_leakage_fails_run_all_validations(self):
        gps = 274965 * 4096.0 + 100.0
        rows = [_glitch_row("train", gps), _glitch_row("test", gps)]
        with self.assertRaisesRegex(AssertionError, "glitch"):
            run_all_validations(rows, [], [], [], allow_cross_split=True)

    def test_source_group_leakage_fails_run_all_validations(self):
        base = 274965 * 4096.0
        rows = [_glitch_row("train", base + 100.0),
                _glitch_row("test", base + 900.0)]
        with self.assertRaises(AssertionError):
            run_all_validations(rows, [], [], [], allow_cross_split=True)

    def test_background_collinearity_fails_run_all_validations(self):
        rows = [
            _glitch_row("train", 274965 * 4096.0 + 100.0),
            _row("train", has_glitch="0", noise_type="gaussian_aligo_colored"),
        ]
        with self.assertRaisesRegex(AssertionError, "background"):
            run_all_validations(rows, [], [], [], allow_cross_split=True)

    def test_background_group_leakage_fails_run_all_validations(self):
        rows = [
            _row("train", background_source="gwosc",
                 background_source_group="H1:274965"),
            _row("test", background_source="gwosc",
                 background_source_group="H1:274965"),
        ]
        with self.assertRaises(AssertionError):
            run_all_validations(rows, [], [], [], allow_cross_split=True)


class AuditCliTests(unittest.TestCase):
    """validate_real_glitch.py --audit runs the leakage/domain audits on a
    dataset dir and exits nonzero on failure (G1-8 gate tool)."""

    FIELDS = [
        "sample_id", "event_id", "chirp_id", "split", "detector", "has_glitch",
        "glitch_id", "glitch_gps", "noise_type",
    ]

    def _write_dataset(self, tmp: Path, leaky: bool) -> Path:
        gps_a = 274965 * 4096.0 + 100.0
        gps_b = (274966 * 4096.0 + 100.0) if not leaky else gps_a
        rows = [
            _glitch_row("train", gps_a),
            _glitch_row("test", gps_b),
            _row("train", has_glitch="0", noise_type="real_gwosc_offsource"),
            _row("test", has_glitch="0", noise_type="real_gwosc_offsource"),
        ]
        with (tmp / "metadata.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=self.FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        return tmp

    def test_audit_exits_nonzero_on_leaky_dataset(self):
        import tempfile

        import validate_real_glitch as vrg

        with tempfile.TemporaryDirectory() as td:
            d = self._write_dataset(Path(td), leaky=True)
            with self.assertRaises(SystemExit) as cm:
                vrg.main(["--audit", "--output-dir", str(d)])
            # 1 = audit failure (argparse usage errors exit 2 and must not
            # be mistaken for a detected leak).
            self.assertEqual(cm.exception.code, 1)

    def test_audit_passes_on_clean_dataset(self):
        import tempfile

        import validate_real_glitch as vrg

        with tempfile.TemporaryDirectory() as td:
            d = self._write_dataset(Path(td), leaky=False)
            vrg.main(["--audit", "--output-dir", str(d)])  # must not raise


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _v2_dataset_dirs():
    """All COMPLETED v2 datasets. Crashed builds leave partial directories
    (streaming CSVs) without the BUILD_COMPLETE marker; they must never be
    audited as datasets."""
    if not DATASETS.is_dir():
        return []
    return sorted(d for d in DATASETS.iterdir()
                  if d.is_dir() and d.name.startswith("gw_dataset_v2_")
                  and (d / "BUILD_COMPLETE").exists())


def _primary_dataset_dir():
    """Any completed v2 dataset wins; else the legacy real dataset."""
    v2 = _v2_dataset_dirs()
    if v2:
        return v2[-1]
    if LEGACY_REAL.is_dir():
        return LEGACY_REAL
    return None


@unittest.skipUnless(LEGACY_REAL.is_dir(), "gw_dataset_3000_real not on disk")
class KnownLeakyDatasetAuditTests(unittest.TestCase):
    """The pre-G1 dataset is known-leaky; the audit must detect it forever."""

    @classmethod
    def setUpClass(cls):
        cls.metadata = _read_csv(LEGACY_REAL / "metadata.csv")

    def test_glitch_id_leakage_is_detected(self):
        with self.assertRaises(AssertionError):
            validate_no_glitch_leakage(self.metadata)

    def test_source_group_leakage_is_detected(self):
        with self.assertRaises(AssertionError):
            validate_no_source_group_leakage(self.metadata)

    def test_background_collinearity_is_detected(self):
        with self.assertRaises(AssertionError):
            validate_background_domain_decoupled(self.metadata)


@unittest.skipUnless(_primary_dataset_dir() is not None, "no dataset on disk")
class CurrentPrimaryDatasetAuditTests(unittest.TestCase):
    """RED until the G1 rebuild: with no completed v2 dataset this resolves
    to gw_dataset_3000_real and fails on its source leakage; afterwards
    EVERY completed v2 dataset must pass all three audits."""

    @classmethod
    def setUpClass(cls):
        dirs = _v2_dataset_dirs() or [_primary_dataset_dir()]
        cls.datasets = [(d.name, _read_csv(d / "metadata.csv")) for d in dirs]

    def _each(self, check):
        for name, metadata in self.datasets:
            with self.subTest(dataset=name):
                self.assertTrue(check(metadata))

    def test_primary_dataset_has_no_glitch_id_leakage(self):
        self._each(validate_no_glitch_leakage)

    def test_primary_dataset_has_no_source_group_leakage(self):
        self._each(validate_no_source_group_leakage)

    def test_primary_dataset_background_domain_is_decoupled(self):
        self._each(validate_background_domain_decoupled)


if __name__ == "__main__":
    unittest.main()
