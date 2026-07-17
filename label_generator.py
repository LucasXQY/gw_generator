"""YOLO label generation from the *actual* chirp time-frequency response.

Primary path: instantaneous frequency of the clean injected waveform (Hilbert
phase derivative) over its visible amplitude support.
Fallback path: the Q-transform energy ridge inside the injection window.

Frequency boundaries are clipped to the sample's ``frange_low/frange_high``;
there is no hardcoded 1000/512 Hz denominator. The y-axis mapping is the shared
:mod:`coords` mapping, so labels align with the rendered image.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from coords import freq_to_unit

CLASS_CHIRP = 0
CLASS_GLITCH = 1


@dataclass
class TimeFrequencyBox:
    time_start: float
    time_end: float
    freq_low: float
    freq_high: float
    class_id: int
    source: str = ""


class LabelGenerator:
    def __init__(
        self,
        duration: float,
        frange: Sequence[float],
        frequency_axis_scale: str,
        drop_if_outside_window: bool = True,
    ):
        self.duration = float(duration)
        self.frange_low = float(frange[0])
        self.frange_high = float(frange[1])
        self.scale = frequency_axis_scale
        self.drop_if_outside_window = drop_if_outside_window

    # ------------------------------------------------------ chirp from waveform
    def measure_chirp_track(
        self,
        clean_signal: np.ndarray,
        sample_rate: int,
        envelope_threshold: float,
    ) -> Optional[dict]:
        """Raw (unclipped) time/frequency support of the clean chirp.

        Returns ``{t_start, t_end, f_low, f_high, source}`` in physical units,
        or ``None`` if instantaneous-frequency extraction fails.
        """
        sig = np.asarray(clean_signal, dtype=float)
        if not np.any(sig):
            return None

        analytic = self._analytic(sig)
        envelope = np.abs(analytic)
        peak = float(np.max(envelope))
        if peak <= 0:
            return None

        support = envelope >= envelope_threshold * peak
        idx = np.flatnonzero(support)
        if idx.size < 2:
            return None
        i0, i1 = int(idx[0]), int(idx[-1])

        phase = np.unwrap(np.angle(analytic))
        f_inst = np.gradient(phase, 1.0 / sample_rate) / (2.0 * np.pi)
        f_support = f_inst[i0 : i1 + 1]
        f_support = f_support[np.isfinite(f_support)]
        f_support = f_support[f_support > 0]
        if f_support.size == 0:
            return None

        return {
            "t_start": i0 / sample_rate,
            "t_end": i1 / sample_rate,
            # Robust percentiles avoid Hilbert edge artifacts.
            "f_low": float(np.percentile(f_support, 2)),
            "f_high": float(np.percentile(f_support, 98)),
            "source": "waveform_instfreq",
        }

    def chirp_box_from_waveform(
        self,
        clean_signal: np.ndarray,
        sample_rate: int,
        envelope_threshold: float,
    ) -> Optional[TimeFrequencyBox]:
        track = self.measure_chirp_track(clean_signal, sample_rate, envelope_threshold)
        if track is None:
            return None
        return self._finalize_chirp_box(
            track["t_start"], track["t_end"], track["f_low"], track["f_high"],
            source=track["source"],
        )

    # -------------------------------------------------------- chirp from ridge
    def chirp_box_from_ridge(
        self,
        energy: np.ndarray,
        freqs: np.ndarray,
        times: np.ndarray,
        time_window: Sequence[float],
        ridge_threshold: float,
    ) -> Optional[TimeFrequencyBox]:
        energy = np.asarray(energy, dtype=float)
        freqs = np.asarray(freqs, dtype=float)
        times = np.asarray(times, dtype=float)
        if energy.size == 0:
            return None

        t_lo, t_hi = float(time_window[0]), float(time_window[1])
        col_mask = (times >= t_lo) & (times <= t_hi)
        if not np.any(col_mask):
            col_mask = np.ones_like(times, dtype=bool)

        sub = energy[:, col_mask]
        peak = float(np.max(sub)) if sub.size else 0.0
        if peak <= 0:
            return None
        hot = sub >= ridge_threshold * peak
        if not np.any(hot):
            return None

        rows = np.any(hot, axis=1)
        cols_local = np.any(hot, axis=0)
        hot_freqs = freqs[rows]
        col_times = times[col_mask][cols_local]
        if hot_freqs.size == 0 or col_times.size == 0:
            return None

        t_start = float(np.min(col_times))
        t_end = float(np.max(col_times))
        f_low = float(np.min(hot_freqs))
        f_high = float(np.max(hot_freqs))
        return self._finalize_chirp_box(
            t_start, t_end, f_low, f_high, source="qtransform_ridge"
        )

    # ----------------------------------------------------------------- glitch
    def glitch_box_from_ridge(
        self,
        energy: np.ndarray,
        freqs: np.ndarray,
        times: np.ndarray,
        time_window: Sequence[float],
        ridge_threshold: float,
        floor_gate: float = 5.0,
        mass_quantile: float = 0.01,
    ) -> Optional[TimeFrequencyBox]:
        """Glitch box from the rendered Q-transform energy ridge, robust to
        real-noise outliers.

        The naive union of pixels above ``ridge_threshold * peak`` breaks on
        real noise: normalized energy has an exponential tail, so for a weak
        glitch the threshold drops into the tail and scattered noise pixels
        stretch the box to the full band. Two defenses:

        * a pixel is hot only if it also clears ``floor_gate`` times the
          window's median energy (the noise floor);
        * the box bounds are the [q, 1-q] quantiles of the hot pixels' energy
          mass in time and frequency, not the min/max union, so an isolated
          bright outlier carrying negligible mass cannot stretch the box.

        Returns ``None`` if nothing clears both gates (caller resamples).
        """
        energy = np.asarray(energy, dtype=float)
        freqs = np.asarray(freqs, dtype=float)
        times = np.asarray(times, dtype=float)
        if energy.size == 0:
            return None

        t_lo, t_hi = float(time_window[0]), float(time_window[1])
        col_mask = (times >= t_lo) & (times <= t_hi)
        if not np.any(col_mask):
            col_mask = np.ones_like(times, dtype=bool)

        sub = energy[:, col_mask]
        peak = float(np.max(sub)) if sub.size else 0.0
        if peak <= 0:
            return None
        floor = float(np.median(sub))
        threshold = max(ridge_threshold * peak, floor_gate * floor)
        hot = sub >= threshold
        if not np.any(hot):
            return None

        mass = np.where(hot, sub, 0.0)
        col_times = times[col_mask]

        def _quantile_bounds(profile: np.ndarray, coords: np.ndarray):
            cum = np.cumsum(profile)
            total = cum[-1]
            if total <= 0:
                return None
            lo_i = int(np.searchsorted(cum, mass_quantile * total, side="left"))
            hi_i = int(np.searchsorted(cum, (1.0 - mass_quantile) * total, side="left"))
            hi_i = min(hi_i, coords.size - 1)
            a, b = float(coords[lo_i]), float(coords[hi_i])
            # The rendered grid's frequency axis is DESCENDING (image row
            # order); return direction-agnostic (low, high) bounds.
            return (a, b) if a <= b else (b, a)

        t_bounds = _quantile_bounds(mass.sum(axis=0), col_times)
        f_bounds = _quantile_bounds(mass.sum(axis=1), freqs)
        if t_bounds is None or f_bounds is None:
            return None

        # A very compact glitch can collapse the quantile bounds onto a single
        # grid cell; expand degenerate extents by one cell so the box survives
        # the >0-extent clip in glitch_box.
        t0, t1 = t_bounds
        if t1 <= t0:
            dt = float(times[1] - times[0]) if times.size > 1 else 0.01
            t0, t1 = t0 - dt / 2.0, t1 + dt / 2.0
        f_lo, f_hi = f_bounds
        if f_hi <= f_lo:
            ratio = float(freqs[1] / freqs[0]) if freqs.size > 1 else 1.1
            ratio = max(ratio, 1.0 / ratio) if ratio > 0 else 1.1
            f_lo, f_hi = f_lo / np.sqrt(ratio), f_hi * np.sqrt(ratio)

        return self.glitch_box(t0, t1, f_lo, f_hi)

    def box_area_fraction(self, box: TimeFrequencyBox) -> float:
        """Fraction of the image area the box covers (log-frequency aware)."""
        t_frac = max(0.0, box.time_end - box.time_start) / self.duration
        u_low = float(freq_to_unit(box.freq_low, self.frange_low, self.frange_high, self.scale))
        u_high = float(freq_to_unit(box.freq_high, self.frange_low, self.frange_high, self.scale))
        return t_frac * max(0.0, u_high - u_low)

    def glitch_box(
        self, start_time: float, end_time: float, low_freq: float, high_freq: float
    ) -> Optional[TimeFrequencyBox]:
        t0 = max(0.0, min(start_time, self.duration))
        t1 = max(0.0, min(end_time, self.duration))
        f_low = max(self.frange_low, low_freq)
        f_high = min(self.frange_high, high_freq)
        if t1 <= t0 or f_high <= f_low:
            return None
        return TimeFrequencyBox(t0, t1, f_low, f_high, CLASS_GLITCH, "glitch_synthetic")

    # --------------------------------------------------------------- helpers
    def _finalize_chirp_box(
        self, t_start: float, t_end: float, f_low: float, f_high: float, source: str
    ) -> Optional[TimeFrequencyBox]:
        t0 = max(0.0, min(t_start, self.duration))
        t1 = max(0.0, min(t_end, self.duration))
        # Visible-window check before clipping.
        if self.drop_if_outside_window and (
            f_high <= self.frange_low or f_low >= self.frange_high
        ):
            return None
        f_low_c = max(self.frange_low, min(f_low, self.frange_high))
        f_high_c = max(self.frange_low, min(f_high, self.frange_high))
        if t1 <= t0 or f_high_c <= f_low_c:
            return None
        return TimeFrequencyBox(t0, t1, f_low_c, f_high_c, CLASS_CHIRP, source)

    @staticmethod
    def _analytic(sig: np.ndarray) -> np.ndarray:
        try:
            from scipy.signal import hilbert

            return hilbert(sig)
        except Exception:
            n = sig.size
            spec = np.fft.fft(sig)
            h = np.zeros(n)
            if n % 2 == 0:
                h[0] = h[n // 2] = 1
                h[1 : n // 2] = 2
            else:
                h[0] = 1
                h[1 : (n + 1) // 2] = 2
            return np.fft.ifft(spec * h)

    # ----------------------------------------------------------------- YOLO
    def to_yolo(self, box: TimeFrequencyBox) -> str:
        cx = ((box.time_start + box.time_end) / 2.0) / self.duration
        w = (box.time_end - box.time_start) / self.duration
        u_low = float(freq_to_unit(box.freq_low, self.frange_low, self.frange_high, self.scale))
        u_high = float(freq_to_unit(box.freq_high, self.frange_low, self.frange_high, self.scale))
        cy = 1.0 - (u_low + u_high) / 2.0  # image top = high frequency
        h = abs(u_high - u_low)
        cx = float(np.clip(cx, 0.0, 1.0))
        cy = float(np.clip(cy, 0.0, 1.0))
        w = float(np.clip(w, 0.0, 1.0))
        h = float(np.clip(h, 0.0, 1.0))
        return f"{box.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    def write_label_file(
        self, path: Path, boxes: List[TimeFrequencyBox], write_empty: bool = True
    ) -> Optional[Path]:
        path = Path(path)
        valid = [b for b in boxes if b is not None]
        if not valid and not write_empty:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [self.to_yolo(b) for b in valid]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return path
