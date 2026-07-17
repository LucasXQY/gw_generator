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
        glitch_metadata_csv=_write_pool(tmp / "pool.csv"),
        glitch_whiten=False,
        glitch_amplitude_range=(5.0, 5.0),
    )
    kwargs.update(overrides)
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


class GwoscBuildIntegrationTests(unittest.TestCase):
    """Full offline build with a monkeypatched GWOSC fetch: real-glitch boxes
    must hug the glitch, not the whole image."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.cfg = _make_config(
            tmp,
            num_events=10,
            seed=11,
            qtransform_backend="scipy",
            glitch_amplitude_range=(8.0, 8.0),
        )

        from build_dataset import DatasetBuilder

        original = RealGlitchProvider._gwosc_fetch
        RealGlitchProvider._gwosc_fetch = (
            lambda self, det, gps, hw: _segment_with_spike(self.config, peak=50.0,
                                                           seed=int(gps) % 2**16)
        )
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


if __name__ == "__main__":
    unittest.main()
