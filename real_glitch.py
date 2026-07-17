"""Real detector glitches from GWOSC strain.

Randomly samples glitch entries from a Gravity Spy pool CSV (one per detector,
independently — preserving the A1 cross-detector *independence* of glitches),
fetches the corresponding real strain from GWOSC (cache-first), whitens it, and
returns the real segment — glitch in its own real noise — as the **entire
detector-sample series** (duration-long, robust-unit noise floor, glitch placed
at a random position via sub-window choice).

The segment REPLACES the synthetic background for that sample rather than being
added to it. Transplanting a cropped noisy snippet into synthetic noise was
tried first and is fundamentally flawed: the crop's own noise stacks on the
background and renders as a broadband ~2x-power pedestal across the glitch
window (near-full-band label boxes), and no time- or spectral-domain gate can
remove it without destroying weak glitch morphology. Using the real segment as
the sample has no pedestal by construction and keeps morphology exact; after
the robust (MAD) floor normalization its noise floor matches the unit-std
synthetic backgrounds, so chirp SNR definitions are unchanged.

Caching stores the raw (pre-whiten) resampled segment plus a JSON sidecar;
whitening, sub-window choice, and normalization happen at generation time.

``gwpy`` is imported lazily so synthetic and offline runs are unaffected. On the
``gwosc`` path a missing ``gwpy`` or a failed fetch raises :class:`GlitchFetchError`;
the caller decides whether to hard-fail or fall back to synthetic.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config import DatasetConfig


class GlitchFetchError(RuntimeError):
    """Raised when a real glitch cannot be fetched or prepared from GWOSC.

    Treated as *retryable*: the specific GPS/segment failed (e.g. not in GWOSC
    open data), so sampling a different glitch may succeed.
    """


class GlitchDependencyError(GlitchFetchError):
    """A non-retryable environment error (e.g. gwpy not installed).

    Retrying will not help; the caller should fall back or hard-fail.
    """


class GlitchUnavailableError(GlitchFetchError):
    """This GPS has no usable GWOSC open data (missing segment or degenerate
    strain). Permanent for the pool entry: it is blacklisted and persisted so
    later runs skip it. Transient failures (timeouts, network errors) stay
    plain :class:`GlitchFetchError` and are NOT blacklisted.
    """


@dataclass
class RealGlitch:
    """A real glitch in its own real noise. ``series`` is the FULL sample
    (length ``n_samples``, unit robust noise floor) that replaces the synthetic
    background. Mirrors ``GlitchRealization`` plus provenance fields. Frequency
    fields are provisional (catalog-derived); the label box is measured from
    the rendered energy ridge downstream, windowed to [start_time, end_time]."""

    series: np.ndarray
    glitch_id: str
    glitch_type: str
    start_time: float
    end_time: float
    center_freq: float
    low_freq: float
    high_freq: float
    amplitude: float
    gps: float
    snr_catalog: float
    source: str = "gwosc"


def _to_float(value) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class RealGlitchProvider:
    """Samples + fetches real glitches from a Gravity Spy pool CSV via GWOSC."""

    _REQUIRED = ("gps", "ifo", "label")

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.cache_dir = Path(config.real_glitch_cache_dir)
        self.sr = int(config.sample_rate)
        self.n_samples = int(config.n_samples)
        self.duration = float(config.duration)
        self.pool: Dict[str, List[dict]] = self._load_pool()
        # GPS keys known to be unavailable in GWOSC open data (skip on resample).
        # Persisted next to the cache so restarted runs do not re-fetch them.
        self._unavailable: set = self._load_unavailable()

    @staticmethod
    def _key(detector: str, gps: float) -> str:
        return f"{detector}:{gps:.4f}"

    @property
    def _unavailable_path(self) -> Path:
        return self.cache_dir / "unavailable.json"

    def _load_unavailable(self) -> set:
        try:
            return set(json.loads(self._unavailable_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return set()

    def _mark_unavailable(self, detector: str, gps: float) -> None:
        self._unavailable.add(self._key(detector, gps))
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._unavailable_path.write_text(
                json.dumps(sorted(self._unavailable)), encoding="utf-8"
            )
        except OSError:
            pass  # blacklist still applies in-memory for this run

    # -------------------------------------------------------------- pool CSV
    def _load_pool(self) -> Dict[str, List[dict]]:
        csv_path = self.config.glitch_metadata_csv
        if csv_path is None or not Path(csv_path).exists():
            raise GlitchFetchError(
                "glitch_source='gwosc' requires glitch_metadata_csv to point to a "
                f"Gravity Spy pool CSV; got {csv_path!r}."
            )
        with Path(csv_path).open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise GlitchFetchError(f"glitch pool CSV {csv_path} is empty.")
        missing = [c for c in self._REQUIRED if c not in rows[0]]
        if missing:
            raise GlitchFetchError(
                f"glitch pool CSV missing required columns {missing}; "
                f"has {list(rows[0].keys())}."
            )
        allowed = set(self.config.glitch_types)
        pool: Dict[str, List[dict]] = {d: [] for d in self.config.detectors}
        for r in rows:
            ifo = str(r.get("ifo", "")).strip()
            label = str(r.get("label", "")).strip()
            if ifo not in pool:
                continue
            if allowed and label not in allowed:
                continue
            if _to_float(r.get("gps")) is None:
                continue
            pool[ifo].append(r)
        for d in self.config.detectors:
            if not pool[d]:
                raise GlitchFetchError(
                    f"glitch pool CSV has no usable rows for detector {d} "
                    f"(need ifo={d} and label in {sorted(allowed)})."
                )
        return pool

    # --------------------------------------------------------------- sampling
    def sample_glitch(self, rng, detector: str) -> RealGlitch:
        """Randomly pick a pool row for ``detector`` and return a placed glitch.

        Rows whose GPS is known-unavailable in GWOSC open data are skipped. A
        fetch failure marks that GPS unavailable and re-raises (retryable), so a
        subsequent call samples a different, hopefully-available glitch.
        """
        rows = [
            r for r in self.pool[detector]
            if self._key(detector, float(r["gps"])) not in self._unavailable
        ]
        if not rows:
            raise GlitchFetchError(
                f"all pool glitches for {detector} are unavailable in GWOSC open "
                "data; supply a pool with GPS times inside published open-data "
                "segments (e.g. from O1-O3 observing runs)."
            )
        row = rows[int(rng.integers(len(rows)))]
        gps = float(row["gps"])
        label = str(row["label"]).strip()
        snr_cat = _to_float(row.get("snr")) or 0.0
        peak_freq = (
            _to_float(row.get("peak_frequency"))
            or _to_float(row.get("central_freq"))
            or 0.0
        )
        bandwidth = _to_float(row.get("bandwidth")) or 0.0

        margin = self.config.glitch_placement_margin
        max_len = self.duration - 2.0 * margin
        # Time support used as the ridge-box search window (not a crop length).
        crop_len = _to_float(row.get("duration")) or self.config.glitch_default_duration
        crop_len = float(min(max(crop_len, 16.0 / self.sr), max_len))

        try:
            raw = self._fetch_segment(detector, gps)
            series, pos = self._extract_segment(detector, gps, raw, rng)
        except GlitchDependencyError:
            raise  # non-retryable (e.g. gwpy missing)
        except GlitchUnavailableError:
            # No usable open data at this GPS: blacklist it (persisted).
            self._mark_unavailable(detector, gps)
            raise
        # Transient GlitchFetchError (timeout, network): propagate WITHOUT
        # blacklisting, so the same GPS can succeed on a later attempt.

        t0 = float(max(0.0, pos - crop_len / 2.0))
        t1 = float(min(self.duration, pos + crop_len / 2.0))
        window = series[int(t0 * self.sr) : max(int(t1 * self.sr), int(t0 * self.sr) + 1)]
        amp = float(np.max(np.abs(window)))  # natural peak, in noise-floor sigmas

        lo = self.config.frange_low
        hi = self.config.frange_high
        if bandwidth and peak_freq:
            lo = max(self.config.frange_low, peak_freq - bandwidth)
            hi = min(self.config.frange_high, peak_freq + bandwidth)
        return RealGlitch(
            series=series,
            glitch_id=f"gspy_{detector}_{gps:.4f}",
            glitch_type=label,
            start_time=t0,
            end_time=t1,
            center_freq=peak_freq,
            low_freq=lo,
            high_freq=hi,
            amplitude=amp,
            gps=gps,
            snr_catalog=snr_cat,
            source="gwosc",
        )

    # ---------------------------------------------------------------- fetch
    def cache_npy_path(self, detector: str, gps: float) -> Path:
        """Cache file for this (detector, gps) under the current settings."""
        halfwin = float(self.config.glitch_fetch_halfwin)
        return self.cache_dir / f"{detector}_{gps:.4f}_{self.sr}_{halfwin:g}.npy"

    def _fetch_segment(self, detector: str, gps: float) -> np.ndarray:
        """Fetch (or load from cache) the resampled raw strain around ``gps``."""
        halfwin = float(self.config.glitch_fetch_halfwin)
        npy = self.cache_npy_path(detector, gps)
        meta = npy.with_suffix(".json")
        if npy.exists():
            return np.load(npy)

        data = self._gwosc_fetch(detector, gps, halfwin)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(npy, data.astype(np.float32))
        meta.write_text(
            json.dumps(
                {
                    "detector": detector,
                    "gps": gps,
                    "sample_rate": self.sr,
                    "halfwin": halfwin,
                    "start": gps - halfwin,
                    "end": gps + halfwin,
                    "n": int(data.size),
                }
            ),
            encoding="utf-8",
        )
        return data

    def _gwosc_fetch(self, detector: str, gps: float, halfwin: float) -> np.ndarray:
        try:
            from gwpy.timeseries import TimeSeries
        except Exception as exc:  # pragma: no cover - env-dependent
            raise GlitchDependencyError(
                "gwpy is required for glitch_source='gwosc' (`pip install gwpy`)."
            ) from exc

        start, end = gps - halfwin, gps + halfwin
        try:
            ts = TimeSeries.fetch_open_data(
                detector, start, end, sample_rate=self.sr, cache=True
            )
        except TypeError:
            # Older/newer gwpy without a sample_rate kwarg: fetch then resample.
            ts = TimeSeries.fetch_open_data(detector, start, end, cache=True)
        except Exception as exc:
            # Distinguish "no data at this GPS" (permanent -> blacklist) from
            # transient network trouble (retryable, do NOT blacklist).
            msg = str(exc).lower()
            permanent = any(
                marker in msg
                for marker in ("cannot find", "not found", "404", "no files",
                               "unknown dataset", "missing")
            )
            err_cls = GlitchUnavailableError if permanent else GlitchFetchError
            raise err_cls(
                f"GWOSC fetch failed for {detector} @ {gps}: {exc}"
            ) from exc

        try:
            if int(round(float(ts.sample_rate.value))) != self.sr:
                ts = ts.resample(self.sr)
        except Exception:
            pass
        return np.asarray(ts.value, dtype=float)

    # ------------------------------------------------------------- extract
    def _extract_segment(self, detector: str, gps: float, raw: np.ndarray, rng):
        """Whiten the fetched span and cut a duration-long sub-window that
        contains the glitch at a random position.

        Returns ``(series, pos)`` where ``series`` (length ``n_samples``) has a
        unit robust noise floor and ``pos`` is the glitch time (s) within it.
        """
        halfwin = float(self.config.glitch_fetch_halfwin)
        # GWOSC segments can carry NaN/Inf where data is missing; whitening would
        # spread them across the whole segment. Zero them before/after whitening.
        raw = np.nan_to_num(np.asarray(raw, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        proc = self._whiten(raw, gps, halfwin) if self.config.glitch_whiten else raw
        proc = np.nan_to_num(np.asarray(proc, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

        # The glitch sits at ``halfwin`` seconds into ``proc``. Choose where it
        # should land inside the sample (pos), constrained so the sub-window
        # avoids the whiten-corrupted edges (~fduration/2 = 1 s per side).
        edge = 1.0 if self.config.glitch_whiten else 0.0
        avail = len(proc) / self.sr
        lo_pos = max(0.25 * self.duration, self.duration - (avail - edge - halfwin))
        hi_pos = min(0.75 * self.duration, halfwin - edge)
        if hi_pos <= lo_pos:
            raise GlitchDependencyError(
                f"glitch_fetch_halfwin={halfwin} too small for duration="
                f"{self.duration} (need halfwin >= duration/2 + {edge})."
            )
        pos = float(rng.uniform(lo_pos, hi_pos))
        i0 = int(round((halfwin - pos) * self.sr))
        seg = np.asarray(proc[i0 : i0 + self.n_samples], dtype=float)
        if seg.size < self.n_samples:
            raise GlitchUnavailableError(
                f"segment too short for {detector} @ {gps} "
                f"({seg.size} < {self.n_samples} samples)"
            )
        # Robust (MAD) floor normalization: the glitch barely moves the MAD, so
        # the segment's noise floor lands at 1 -- the same level as the unit-std
        # synthetic backgrounds -- while the glitch keeps its natural contrast.
        med = float(np.median(seg))
        mad = float(np.median(np.abs(seg - med)))
        sigma = 1.4826 * mad if mad > 0 else float(np.std(seg))
        if not np.isfinite(sigma) or sigma <= 0:
            raise GlitchUnavailableError(
                f"degenerate (zero-variance/non-finite) glitch for {detector} @ {gps}"
            )
        return (seg - med) / sigma, pos

    def _whiten(self, raw: np.ndarray, gps: float, halfwin: float) -> np.ndarray:
        try:
            from gwpy.timeseries import TimeSeries
        except Exception as exc:  # pragma: no cover - env-dependent
            raise GlitchDependencyError(
                "gwpy is required to whiten real glitches (`pip install gwpy`)."
            ) from exc
        ts = TimeSeries(np.asarray(raw, dtype=float), sample_rate=self.sr, t0=gps - halfwin)
        flo = max(self.config.frange_low, 15.0)
        try:
            ts = ts.highpass(flo)
        except Exception:
            pass
        fftlength = min(2.0, (2.0 * halfwin) / 4.0)
        try:
            # Median PSD is robust to the glitch itself; the default (mean/
            # Welch) PSD includes the loud glitch and partially whitens it away.
            try:
                w = ts.whiten(fftlength=fftlength, method="median")
            except (TypeError, ValueError):
                w = ts.whiten(fftlength=fftlength) if fftlength > 0 else ts.whiten()
        except Exception as exc:
            raise GlitchFetchError(f"whiten failed @ {gps}: {exc}") from exc
        return np.asarray(w.value, dtype=float)

