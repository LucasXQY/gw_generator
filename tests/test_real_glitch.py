"""Unit tests for the GWOSC real-glitch injection path.

All tests run offline: GWOSC fetches are monkeypatched and whitening is
disabled (``glitch_whiten=False``), so gwpy is never imported.
"""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

# Make the project root importable regardless of unittest's start directory.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from config import DatasetConfig  # noqa: E402
from injection import inject_chirp  # noqa: E402
from label_generator import CLASS_GLITCH, LabelGenerator, TimeFrequencyBox  # noqa: E402
from real_glitch import (  # noqa: E402
    GlitchFetchError,
    GlitchUnavailableError,
    RealGlitchProvider,
)
from split_source_groups import FILE_SECONDS, source_group  # noqa: E402
from validation import (  # noqa: E402
    validate_background_domain_decoupled,
    validate_no_background_group_leakage,
    validate_no_glitch_leakage,
    validate_no_source_group_leakage,
)


def _write_pool(path: Path, gps_h1=(1126259462.0, 1126259500.0),
                gps_l1=(1126259470.0, 1126259510.0)) -> Path:
    rows = [
        {"gps": g, "ifo": "H1", "label": "Blip", "snr": 12.0, "duration": 0.5}
        for g in gps_h1
    ] + [
        {"gps": g, "ifo": "L1", "label": "Blip", "snr": 12.0, "duration": 0.5}
        for g in gps_l1
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["gps", "ifo", "label", "snr", "duration"])
        w.writeheader()
        w.writerows(rows)
    return path


def _make_config(tmp: Path, **overrides) -> DatasetConfig:
    kwargs = dict(
        num_events=1,
        output_dir=tmp / "out",
        glitch_source="gwosc",
        glitch_whiten=False,
        glitch_amplitude_range=(5.0, 5.0),
    )
    kwargs.update(overrides)
    if "glitch_metadata_csv" not in overrides:
        # Written lazily so an override's pool file is never clobbered by
        # the default pool sharing the same path.
        kwargs["glitch_metadata_csv"] = _write_pool(tmp / "pool.csv")
    return DatasetConfig(**kwargs)


def _segment_with_spike(config: DatasetConfig, peak: float, seed: int = 42) -> np.ndarray:
    """Fake fetched strain: unit-std noise + a compact spike at the GPS center."""
    n = int(2 * config.glitch_fetch_halfwin * config.sample_rate)
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 1.0, n)
    if peak > 0:
        center = n // 2
        t = np.arange(-64, 65) / config.sample_rate
        wavelet = np.cos(2 * np.pi * 80.0 * t) * np.exp(-0.5 * (t / 0.005) ** 2)
        raw[center - 64 : center + 65] += peak * wavelet / np.max(np.abs(wavelet))
    return raw


def _robust_floor(series: np.ndarray) -> float:
    """1.4826*MAD of the placed (non-zero) crop region."""
    nz = np.flatnonzero(series)
    seg = series[nz[0] : nz[-1] + 1]
    med = np.median(seg)
    return float(1.4826 * np.median(np.abs(seg - med)))


class RealSegmentTests(unittest.TestCase):
    """gwosc glitch samples use the real whitened segment AS the whole sample
    (noise included) instead of transplanting a noisy crop into synthetic
    noise -- transplanting inevitably adds a broadband power pedestal across
    the glitch window."""

    def test_series_is_full_length_segment_with_unit_floor_and_natural_peak(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_config(tmp)
            provider = RealGlitchProvider(cfg)
            provider._gwosc_fetch = (
                lambda det, gps, hw: _segment_with_spike(cfg, peak=50.0)
            )
            glitch = provider.sample_glitch(np.random.default_rng(0), "H1")
            series = glitch.series
            self.assertEqual(series.size, cfg.n_samples)
            # The whole series is real noise: unit robust floor everywhere,
            # no zero padding.
            self.assertGreater(float(np.min(np.abs(series[::100])).max()), 0.0)
            floor = _robust_floor(series)
            self.assertGreater(floor, 0.8)
            self.assertLess(floor, 1.2)
            # The glitch keeps its NATURAL contrast (no amplitude rescaling).
            self.assertGreater(float(np.max(np.abs(series))), 30.0)

    def test_glitch_lands_inside_placement_window(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_config(tmp)
            provider = RealGlitchProvider(cfg)
            provider._gwosc_fetch = (
                lambda det, gps, hw: _segment_with_spike(cfg, peak=50.0)
            )
            for seed in range(5):
                glitch = provider.sample_glitch(np.random.default_rng(seed), "H1")
                pos = int(np.argmax(np.abs(glitch.series))) / cfg.sample_rate
                frac = pos / cfg.duration
                self.assertGreaterEqual(frac, 0.2)
                self.assertLessEqual(frac, 0.8)
                # The box window brackets the glitch position.
                self.assertLessEqual(glitch.start_time, pos + 1e-6)
                self.assertGreaterEqual(glitch.end_time, pos - 1e-6)


class UnavailableTrackingTests(unittest.TestCase):
    def test_unavailable_gps_is_blacklisted_and_persisted(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_config(tmp)
            provider = RealGlitchProvider(cfg)

            def unavailable_fetch(det, gps, hw):
                raise GlitchUnavailableError(f"no GWOSC data for {det} @ {gps}")

            provider._gwosc_fetch = unavailable_fetch
            with self.assertRaises(GlitchFetchError):
                provider.sample_glitch(np.random.default_rng(0), "H1")
            self.assertEqual(len(provider._unavailable), 1)

            # A fresh provider over the same cache dir must reload the blacklist.
            reloaded = RealGlitchProvider(cfg)
            self.assertEqual(reloaded._unavailable, provider._unavailable)

    def test_transient_fetch_error_is_not_blacklisted(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_config(tmp)
            provider = RealGlitchProvider(cfg)

            def flaky_fetch(det, gps, hw):
                raise GlitchFetchError("connection timed out")

            provider._gwosc_fetch = flaky_fetch
            with self.assertRaises(GlitchFetchError):
                provider.sample_glitch(np.random.default_rng(0), "H1")
            self.assertEqual(len(provider._unavailable), 0)


class GlitchBoxFromRidgeTests(unittest.TestCase):
    def setUp(self):
        self.gen = LabelGenerator(4.0, (20.0, 1000.0), "log")
        self.freqs = np.geomspace(20.0, 1000.0, 64)
        self.times = np.linspace(0.0, 4.0, 128, endpoint=False)

    def test_hot_block_yields_matching_glitch_box(self):
        energy = np.full((64, 128), 0.5)
        energy[20:30, 40:50] = 10.0
        box = self.gen.glitch_box_from_ridge(
            energy, self.freqs, self.times, (1.0, 2.0), 0.30
        )
        self.assertIsNotNone(box)
        self.assertEqual(box.class_id, CLASS_GLITCH)
        self.assertAlmostEqual(box.freq_low, self.freqs[20], places=6)
        self.assertAlmostEqual(box.freq_high, self.freqs[29], places=6)
        self.assertAlmostEqual(box.time_start, self.times[40], places=6)
        self.assertAlmostEqual(box.time_end, self.times[49], places=6)

    def test_no_energy_returns_none(self):
        energy = np.zeros((64, 128))
        box = self.gen.glitch_box_from_ridge(
            energy, self.freqs, self.times, (1.0, 2.0), 0.30
        )
        self.assertIsNone(box)

    def test_noise_outliers_do_not_inflate_a_weak_glitch_box(self):
        # Realistic weak case: noise floor 1.0, weak glitch block at 6x floor,
        # scattered noise outliers at 3-4x floor OUTSIDE the glitch band. With a
        # peak of 6 the naive 0.3*peak threshold (1.8) lets the outliers in and
        # the union box spans the full band.
        energy = np.full((64, 128), 1.0)
        energy[20:26, 40:60] = 6.0                      # the glitch
        energy[2, 45] = energy[60, 50] = energy[55, 42] = 4.0   # noise outliers
        box = self.gen.glitch_box_from_ridge(
            energy, self.freqs, self.times, (1.0, 2.0), 0.30
        )
        self.assertIsNotNone(box)
        self.assertAlmostEqual(box.freq_low, self.freqs[20], places=6)
        self.assertAlmostEqual(box.freq_high, self.freqs[25], places=6)

    def test_single_bright_outlier_is_rejected_by_mass_quantile(self):
        # One isolated pixel brighter than the floor gate but carrying a
        # negligible fraction of the hot energy mass must not stretch the box.
        energy = np.full((64, 128), 1.0)
        energy[20:26, 40:60] = 8.0
        energy[60, 50] = 7.0  # isolated bright outlier far below the band
        box = self.gen.glitch_box_from_ridge(
            energy, self.freqs, self.times, (1.0, 2.0), 0.30
        )
        self.assertIsNotNone(box)
        self.assertAlmostEqual(box.freq_low, self.freqs[20], places=6)
        self.assertAlmostEqual(box.freq_high, self.freqs[25], places=6)


class BoxAreaFractionTests(unittest.TestCase):
    def setUp(self):
        self.gen = LabelGenerator(4.0, (20.0, 1000.0), "log")

    def test_full_image_box_has_area_one(self):
        box = TimeFrequencyBox(0.0, 4.0, 20.0, 1000.0, CLASS_GLITCH)
        self.assertAlmostEqual(self.gen.box_area_fraction(box), 1.0, places=6)

    def test_small_box_area_uses_log_frequency_axis(self):
        box = TimeFrequencyBox(1.0, 1.4, 100.0, 200.0, CLASS_GLITCH)
        expected = (0.4 / 4.0) * (np.log(2.0) / np.log(1000.0 / 20.0))
        self.assertAlmostEqual(self.gen.box_area_fraction(box), expected, places=6)


class InjectChirpNoiseReferenceTests(unittest.TestCase):
    def test_glitch_in_background_does_not_inflate_chirp_amplitude(self):
        sr = 4096
        rng = np.random.default_rng(7)
        noise = rng.normal(0.0, 1.0, 4 * sr)
        contaminated = noise.copy()
        contaminated[sr : 3 * sr] += rng.normal(0.0, 5.0, 2 * sr)
        t = np.arange(sr) / sr
        waveform = np.sin(2 * np.pi * 100.0 * t)

        clean_ref = inject_chirp(noise, waveform, 2.0, 10.0, sr)
        contaminated_inj = inject_chirp(
            contaminated, waveform, 2.0, 10.0, sr, noise_reference=noise
        )
        self.assertAlmostEqual(
            float(np.max(np.abs(contaminated_inj.clean_signal))),
            float(np.max(np.abs(clean_ref.clean_signal))),
            places=9,
        )


class PrefetchPoolTests(unittest.TestCase):
    def test_prefetch_caches_available_and_blacklists_unavailable(self):
        from prefetch_glitch_cache import prefetch_pool

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_config(tmp)
            provider = RealGlitchProvider(cfg)
            bad_gps = 1126259462.0  # first H1 pool row

            def fetch(det, gps, hw):
                if gps == bad_gps:
                    raise GlitchUnavailableError("no data")
                return _segment_with_spike(cfg, peak=10.0)

            provider._gwosc_fetch = fetch
            counts = prefetch_pool(provider, log=lambda *a, **k: None)
            self.assertEqual(counts["fetched"], 3)
            self.assertEqual(counts["unavailable"], 1)
            self.assertEqual(counts["failed"], 0)
            self.assertIn(provider._key("H1", bad_gps), provider._unavailable)
            npys = list(Path(cfg.real_glitch_cache_dir).glob("*.npy"))
            self.assertEqual(len(npys), 3)

            # Second pass: everything already cached or blacklisted.
            counts = prefetch_pool(provider, log=lambda *a, **k: None)
            self.assertEqual(counts["fetched"], 0)
            self.assertEqual(counts["cached"], 3)
            self.assertEqual(counts["skipped_unavailable"], 1)


def _spread_pool_gps(n_files=6, per_file=2, first_file=274960):
    """GPS times spanning ``n_files`` distinct 4096 s GWOSC files."""
    return tuple(
        (first_file + i) * 4096.0 + off
        for i in range(n_files)
        for off in (100.0, 900.0)[:per_file]
    )


_POOL_GPS = set(_spread_pool_gps()) | set(_spread_pool_gps(first_file=275100))


def _fetch_spike_only_at_pool_gps(self, det, gps, hw):
    """Fake GWOSC fetch: a loud transient at pool glitch GPS times, pure
    noise elsewhere (so off-source background candidates come out clean)."""
    peak = 50.0 if gps in _POOL_GPS else 0.0
    return _segment_with_spike(self.config, peak=peak, seed=int(gps) % 2**16)


class GwoscBuildIntegrationTests(unittest.TestCase):
    """Full offline build with a monkeypatched GWOSC fetch: real-glitch boxes
    must hug the glitch, not the whole image; glitch source groups must be
    split-isolated and recorded; clean/pure_noise samples must sit on real
    off-source backgrounds (D2 domain decoupling)."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.cfg = _make_config(
            tmp,
            num_events=10,
            seed=11,
            qtransform_backend="scipy",
            noise_source="gwosc",
            glitch_amplitude_range=(8.0, 8.0),
            glitch_metadata_csv=_write_pool(
                tmp / "pool.csv",
                gps_h1=_spread_pool_gps(),
                gps_l1=_spread_pool_gps(first_file=275100),
            ),
        )

        from build_dataset import DatasetBuilder

        original = RealGlitchProvider._gwosc_fetch
        RealGlitchProvider._gwosc_fetch = _fetch_spike_only_at_pool_gps
        try:
            builder = DatasetBuilder(cls.cfg, use_pycbc=False)
            builder.build()
        finally:
            RealGlitchProvider._gwosc_fetch = original

        with (cls.cfg.output_dir / "metadata.csv").open(newline="", encoding="utf-8") as fh:
            cls.meta = list(csv.DictReader(fh))
        cls.glitch_rows = [r for r in cls.meta if r["has_glitch"] == "1"]

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_build_produced_gwosc_glitches(self):
        self.assertGreater(len(self.glitch_rows), 0)
        for r in self.glitch_rows:
            self.assertEqual(r["glitch_source"], "gwosc")

    def test_glitch_boxes_are_not_full_band(self):
        gen = LabelGenerator(self.cfg.duration, (20.0, 1000.0), "log")
        for r in self.glitch_rows:
            lf, hf = float(r["glitch_low_freq"]), float(r["glitch_high_freq"])
            t0, t1 = float(r["glitch_start_time"]), float(r["glitch_end_time"])
            box = TimeFrequencyBox(t0, t1, lf, hf, CLASS_GLITCH)
            frac = gen.box_area_fraction(box)
            self.assertLessEqual(
                frac, self.cfg.glitch_max_box_frac,
                f"{r['sample_id']}: glitch box covers {frac:.0%} of the image "
                f"(t=[{t0:.2f},{t1:.2f}]s f=[{lf:.0f},{hf:.0f}]Hz)",
            )
            self.assertFalse(
                lf <= 21.0 and hf >= 990.0,
                f"{r['sample_id']}: full-band glitch box f=[{lf:.0f},{hf:.0f}]Hz",
            )

    def test_glitch_center_freq_consistent_with_box(self):
        for r in self.glitch_rows:
            lf, hf = float(r["glitch_low_freq"]), float(r["glitch_high_freq"])
            cf = float(r["glitch_center_freq"])
            self.assertTrue(lf <= cf <= hf,
                            f"{r['sample_id']}: cf={cf} outside [{lf},{hf}]")

    # ---- G1: source-group isolation of the built dataset

    def test_glitch_source_group_column_matches_gps(self):
        for r in self.glitch_rows:
            expected = source_group(r["detector"], r["glitch_gps"])
            self.assertEqual(
                r["glitch_source_group"], expected,
                f"{r['sample_id']}: glitch_source_group={r['glitch_source_group']!r}",
            )

    def test_built_dataset_has_no_glitch_or_group_leakage(self):
        self.assertTrue(validate_no_glitch_leakage(self.meta))
        self.assertTrue(validate_no_source_group_leakage(self.meta))

    def test_background_provenance_fields(self):
        # Real-glitch samples: the glitch's own segment IS the background.
        # (Clean rows are covered by test_clean_rows_have_real_offsource_background.)
        for r in self.glitch_rows:
            self.assertEqual(r["background_source"], "gwosc", r["sample_id"])
            self.assertEqual(
                r["background_source_group"], r["glitch_source_group"], r["sample_id"]
            )
            self.assertEqual(r["background_gps"], r["glitch_gps"], r["sample_id"])

    def test_source_groups_manifest_written_and_disjoint(self):
        import json

        manifest_path = self.cfg.output_dir / "source_groups.json"
        self.assertTrue(manifest_path.exists(), "source_groups.json missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assignment = manifest["assignment"]
        # Every used group is listed and assigned to the split it served.
        for r in self.glitch_rows:
            grp = r["glitch_source_group"]
            self.assertEqual(assignment[grp], r["split"], grp)
        # Per-split bookkeeping: sample counts and glitch_id reuse.
        per_split = manifest["per_split"]
        n_glitch_rows = sum(
            info["glitch_samples"] for info in per_split.values()
        )
        self.assertEqual(n_glitch_rows, len(self.glitch_rows))
        for info in per_split.values():
            self.assertEqual(
                sum(info["glitch_id_reuse"].values()), info["glitch_samples"]
            )

    # ---- G1/D2: real off-source backgrounds for non-glitch samples

    def test_clean_rows_have_real_offsource_background(self):
        clean_rows = [r for r in self.meta if r["has_glitch"] != "1"]
        self.assertTrue(clean_rows)
        pool_gps = sorted(_POOL_GPS)
        for r in clean_rows:
            self.assertEqual(r["background_source"], "gwosc", r["sample_id"])
            self.assertEqual(r["noise_type"], "real_gwosc_offsource", r["sample_id"])
            gps = float(r["background_gps"])
            group = r["background_source_group"]
            self.assertEqual(group, source_group(r["detector"], gps))
            # Same split's source-group pool (D2): group must be assigned to
            # this row's split.
            import json

            manifest = json.loads(
                (self.cfg.output_dir / "source_groups.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["assignment"][group], r["split"], group)
            # Known glitches excluded by at least 8 s.
            nearest = min(abs(gps - g) for g in pool_gps)
            self.assertGreaterEqual(
                nearest, 8.0, f"{r['sample_id']}: off-source gps {gps} within "
                f"8 s of a pool glitch",
            )
            # Fetch window stays inside the group's 4096 s file.
            offset = gps % FILE_SECONDS
            hw = float(self.cfg.glitch_fetch_halfwin)
            self.assertGreaterEqual(offset, hw, r["sample_id"])
            self.assertLessEqual(offset, FILE_SECONDS - hw, r["sample_id"])

    def test_background_domain_decoupled_and_groups_isolated(self):
        self.assertTrue(validate_background_domain_decoupled(self.meta))
        self.assertTrue(validate_no_background_group_leakage(self.meta))

    def test_dataset_config_records_pool_hash_and_code_commit(self):
        """D3: dataset_config.json must pin the pool contents and code commit."""
        import hashlib
        import json

        cfg_json = json.loads(
            (self.cfg.output_dir / "dataset_config.json").read_text(encoding="utf-8")
        )
        expected = hashlib.sha256(
            Path(self.cfg.glitch_metadata_csv).read_bytes()
        ).hexdigest()
        self.assertEqual(cfg_json["pool_sha256"], expected)
        self.assertTrue(cfg_json.get("code_commit"))


class SplitAwareSamplingTests(unittest.TestCase):
    """sample_glitch(split=...) must only draw glitches whose 4096 s source
    group was assigned to that split."""

    RATIOS = {"train": 0.5, "val": 0.25, "test": 0.25}

    def _provider(self, tmp: Path, n_files: int = 4) -> RealGlitchProvider:
        cfg = _make_config(
            tmp,
            glitch_metadata_csv=_write_pool(
                tmp / "pool.csv",
                gps_h1=_spread_pool_gps(n_files=n_files),
                gps_l1=_spread_pool_gps(n_files=n_files, first_file=275100),
            ),
        )
        provider = RealGlitchProvider(cfg)
        provider._gwosc_fetch = (
            lambda det, gps, hw: _segment_with_spike(cfg, peak=50.0)
        )
        return provider

    def test_assign_split_groups_partitions_every_pool_group(self):
        with tempfile.TemporaryDirectory() as td:
            provider = self._provider(Path(td))
            assignment = provider.assign_split_groups(self.RATIOS, seed=3)
            pool_groups = {
                source_group(det, r["gps"])
                for det, rows in provider.pool.items()
                for r in rows
            }
            self.assertEqual(set(assignment), pool_groups)
            self.assertTrue(set(assignment.values()) <= set(self.RATIOS))

    def test_sample_glitch_draws_only_from_the_splits_groups(self):
        with tempfile.TemporaryDirectory() as td:
            provider = self._provider(Path(td))
            assignment = provider.assign_split_groups(self.RATIOS, seed=3)
            rng = np.random.default_rng(0)
            for split in self.RATIOS:
                if not any(
                    s == split and g.startswith("H1:") for g, s in assignment.items()
                ):
                    continue
                for _ in range(4):
                    glitch = provider.sample_glitch(rng, "H1", split=split)
                    self.assertEqual(
                        assignment[source_group("H1", glitch.gps)], split
                    )

    def test_sample_glitch_without_split_is_unrestricted(self):
        with tempfile.TemporaryDirectory() as td:
            provider = self._provider(Path(td))
            glitch = provider.sample_glitch(np.random.default_rng(0), "H1")
            self.assertTrue(glitch.glitch_id.startswith("gspy_H1_"))

    def test_split_without_groups_raises_fetch_error(self):
        with tempfile.TemporaryDirectory() as td:
            # 2 files per detector cannot cover 3 splits: some split ends up
            # with an empty H1 pool and sampling from it must fail loudly.
            provider = self._provider(Path(td), n_files=2)
            assignment = provider.assign_split_groups(self.RATIOS, seed=3)
            empty = [
                split for split in self.RATIOS
                if not any(
                    s == split and g.startswith("H1:")
                    for g, s in assignment.items()
                )
            ]
            self.assertTrue(empty, "expected an uncovered split with 2 files")
            with self.assertRaisesRegex(GlitchFetchError, empty[0]):
                provider.sample_glitch(np.random.default_rng(0), "H1", split=empty[0])

    def test_sample_with_split_before_assignment_raises(self):
        with tempfile.TemporaryDirectory() as td:
            provider = self._provider(Path(td))
            with self.assertRaisesRegex(GlitchFetchError, "assign_split_groups"):
                provider.sample_glitch(np.random.default_rng(0), "H1", split="train")


class OffSourceBackgroundTests(unittest.TestCase):
    """sample_background draws real off-source segments from the split's own
    4096 s source groups, away from known glitches and file edges."""

    RATIOS = {"train": 0.5, "val": 0.25, "test": 0.25}

    def _provider(self, tmp: Path, n_files: int = 4) -> RealGlitchProvider:
        cfg = _make_config(
            tmp,
            glitch_metadata_csv=_write_pool(
                tmp / "pool.csv",
                gps_h1=_spread_pool_gps(n_files=n_files),
                gps_l1=_spread_pool_gps(n_files=n_files, first_file=275100),
            ),
        )
        provider = RealGlitchProvider(cfg)
        provider._gwosc_fetch = (
            lambda det, gps, hw: _segment_with_spike(cfg, peak=0.0, seed=int(gps) % 2**16)
        )
        return provider

    def _split_with_h1_groups(self, assignment):
        for split in self.RATIOS:
            if any(s == split and g.startswith("H1:") for g, s in assignment.items()):
                return split
        self.fail("no split has H1 groups")

    def test_background_is_full_length_with_unit_robust_floor(self):
        with tempfile.TemporaryDirectory() as td:
            provider = self._provider(Path(td))
            assignment = provider.assign_split_groups(self.RATIOS, seed=5)
            split = self._split_with_h1_groups(assignment)
            bg = provider.sample_background(np.random.default_rng(0), "H1", split=split)
            self.assertEqual(bg.series.size, provider.n_samples)
            self.assertEqual(bg.noise_type, "real_gwosc_offsource")
            med = np.median(bg.series)
            floor = 1.4826 * np.median(np.abs(bg.series - med))
            self.assertAlmostEqual(floor, 1.0, delta=0.15)

    def test_background_gps_in_split_groups_and_file_interior(self):
        with tempfile.TemporaryDirectory() as td:
            provider = self._provider(Path(td))
            assignment = provider.assign_split_groups(self.RATIOS, seed=5)
            split = self._split_with_h1_groups(assignment)
            rng = np.random.default_rng(1)
            hw = float(provider.config.glitch_fetch_halfwin)
            for _ in range(8):
                bg = provider.sample_background(rng, "H1", split=split)
                self.assertEqual(assignment[source_group("H1", bg.gps)], split)
                offset = bg.gps % FILE_SECONDS
                self.assertGreaterEqual(offset, hw)
                self.assertLessEqual(offset, FILE_SECONDS - hw)

    def test_background_avoids_known_glitches_by_8_seconds(self):
        with tempfile.TemporaryDirectory() as td:
            provider = self._provider(Path(td))
            assignment = provider.assign_split_groups(self.RATIOS, seed=5)
            split = self._split_with_h1_groups(assignment)
            pool_gps = [float(r["gps"]) for r in provider.pool["H1"]]
            rng = np.random.default_rng(2)
            for _ in range(12):
                bg = provider.sample_background(rng, "H1", split=split)
                nearest = min(abs(bg.gps - g) for g in pool_gps)
                self.assertGreaterEqual(nearest, 8.0)

    def test_background_requires_assignment_for_split(self):
        with tempfile.TemporaryDirectory() as td:
            provider = self._provider(Path(td))
            with self.assertRaisesRegex(GlitchFetchError, "assign_split_groups"):
                provider.sample_background(np.random.default_rng(0), "H1", split="train")

    def test_cached_candidates_win_over_uncached_groups_split_wide(self):
        """If ANY of the split's groups has cached candidates, sampling must
        never touch the network -- even when the randomly drawn group has no
        cache (e.g. its whole-file prefetch failed)."""
        from split_source_groups import offsource_candidate_gps

        with tempfile.TemporaryDirectory() as td:
            provider = self._provider(Path(td), n_files=8)
            assignment = provider.assign_split_groups(self.RATIOS, seed=5)

            def h1_groups(s):
                return [g for g, sp in assignment.items()
                        if sp == s and g.startswith("H1:")]

            split = max(self.RATIOS, key=lambda s: len(h1_groups(s)))
            groups = h1_groups(split)
            self.assertGreaterEqual(len(groups), 2, assignment)
            cfg = provider.config
            # Cache 3 candidates of ONE group only.
            seeded = offsource_candidate_gps(
                groups[0], cfg.glitch_fetch_halfwin, cfg.offsource_grid_step
            )[:3]
            n = int(2 * cfg.glitch_fetch_halfwin * provider.sr)
            for i, gps in enumerate(seeded):
                provider.cache_store(
                    "H1", gps, np.random.default_rng(i).normal(0.0, 1.0, n)
                )

            def no_network(det, gps, hw):
                raise AssertionError("network hit despite cached candidates")

            provider._gwosc_fetch = no_network
            rng = np.random.default_rng(0)
            for _ in range(10):
                bg = provider.sample_background(rng, "H1", split=split)
                self.assertIn(bg.gps, set(seeded))

    def test_background_gps_comes_from_deterministic_grid(self):
        from split_source_groups import offsource_candidate_gps

        with tempfile.TemporaryDirectory() as td:
            provider = self._provider(Path(td))
            assignment = provider.assign_split_groups(self.RATIOS, seed=5)
            split = self._split_with_h1_groups(assignment)
            cfg = provider.config
            grid = set()
            for group, s in assignment.items():
                if s == split and group.startswith("H1:"):
                    grid.update(offsource_candidate_gps(
                        group, cfg.glitch_fetch_halfwin, cfg.offsource_grid_step
                    ))
            rng = np.random.default_rng(3)
            for _ in range(6):
                bg = provider.sample_background(rng, "H1", split=split)
                self.assertIn(bg.gps, grid)


class PrefetchOffsourceTests(unittest.TestCase):
    """Bulk off-source prefetch: ONE whole-file download per (detector, file),
    all grid candidates cut locally; the build then samples fully offline."""

    RATIOS = {"train": 0.5, "val": 0.25, "test": 0.25}

    def test_prefetch_downloads_each_file_once_and_enables_offline_sampling(self):
        from prefetch_offsource_cache import prefetch_offsource

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_config(
                tmp,
                glitch_metadata_csv=_write_pool(
                    tmp / "pool.csv",
                    gps_h1=_spread_pool_gps(n_files=2),
                    gps_l1=_spread_pool_gps(n_files=2, first_file=275100),
                ),
            )
            provider = RealGlitchProvider(cfg)
            calls = []

            def fake_fetch(det, gps, hw):
                calls.append((det, gps, hw))
                n = int(2 * hw * provider.sr)
                return np.random.default_rng(len(calls)).normal(0.0, 1.0, n)

            provider._gwosc_fetch = fake_fetch
            stats = prefetch_offsource(provider, per_file=3, log=lambda *a: None)

            # One whole-file fetch per (detector, 4096 s file): 2 det x 2 files.
            self.assertEqual(len(calls), 4)
            for _det, _gps, hw in calls:
                self.assertEqual(hw, FILE_SECONDS / 2.0)
            self.assertEqual(stats["segments_cached"], 4 * 3)

            # Re-running is a no-op (resumable).
            stats2 = prefetch_offsource(provider, per_file=3, log=lambda *a: None)
            self.assertEqual(len(calls), 4)
            self.assertEqual(stats2["segments_cached"], 0)

            # Sampling now works with the network dead.
            def no_network(det, gps, hw):
                raise AssertionError("network hit during offline sampling")

            provider._gwosc_fetch = no_network
            assignment = provider.assign_split_groups(self.RATIOS, seed=5)
            split = next(
                s for g, s in assignment.items() if g.startswith("H1:")
            )
            bg = provider.sample_background(np.random.default_rng(0), "H1", split=split)
            self.assertEqual(bg.noise_type, "real_gwosc_offsource")
            self.assertEqual(bg.series.size, cfg.n_samples)


class OffSourceBuildConfigTests(unittest.TestCase):
    def test_noise_source_is_validated(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                _make_config(Path(td), noise_source="bogus")

    def test_gwosc_noise_forbids_synthetic_glitch_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                _make_config(
                    Path(td),
                    noise_source="gwosc",
                    glitch_allow_synthetic_fallback=True,
                )

    def test_transient_candidates_are_rejected_until_explicit_failure(self):
        """Every off-source candidate renders with a transient: the builder
        must retry up to max_background_attempts then raise -- never fall
        back to synthetic noise."""
        from build_dataset import DatasetBuilder

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_config(
                tmp,
                noise_source="gwosc",
                max_background_attempts=2,
                qtransform_backend="scipy",
                glitch_metadata_csv=_write_pool(
                    tmp / "pool.csv",
                    gps_h1=_spread_pool_gps(),
                    gps_l1=_spread_pool_gps(first_file=275100),
                ),
            )
            builder = DatasetBuilder(cfg, use_pycbc=False)
            provider = builder.real_glitch_provider
            provider._gwosc_fetch = (
                lambda det, gps, hw: _segment_with_spike(cfg, peak=50.0)
            )
            provider.assign_split_groups(cfg.split_ratios(), cfg.seed)
            with self.assertRaisesRegex(GlitchFetchError, "off-source"):
                builder._acquire_real_background(
                    np.random.default_rng(0), "H1", "train"
                )


class SplitPoolCoverageTests(unittest.TestCase):
    def test_build_fails_fast_when_a_split_has_no_groups(self):
        """A pool with one 4096 s file per detector cannot serve 3 splits;
        the builder must refuse before fetching anything."""
        from build_dataset import DatasetBuilder

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_config(tmp, num_events=6, qtransform_backend="scipy")
            builder = DatasetBuilder(cfg, use_pycbc=False)
            with self.assertRaisesRegex(GlitchFetchError, "source group"):
                builder.build()


if __name__ == "__main__":
    unittest.main()
