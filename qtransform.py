"""Q-transform energy maps and image rendering.

Produces, for a strain series:

* a **train** image: pure spectrogram pixels, no axes/ticks/labels/colorbar,
  saved at exactly ``(width, height)`` so YOLO labels align pixel-for-pixel;
* a **display** image: viridis spectrogram with ``Time [secs]`` /
  ``Frequency [Hz]`` axes and a ``Normalized energy`` colorbar (0-25), matching
  the reference visual style.

The frequency axis (log or linear) is defined once in :mod:`coords` and shared
with the YOLO label coordinate system. Energy is normalized with a robust
statistic so strong chirps peak near 20-25 and values are clipped to [0, 25].
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import DatasetConfig, MissingOptionalDependency  # noqa: E402
from coords import image_row_frequencies  # noqa: E402


def _get_cmap(name: str):
    """Version-robust colormap lookup (matplotlib >= and < 3.9)."""
    try:
        return matplotlib.colormaps[name]
    except (AttributeError, KeyError):
        from matplotlib import cm

        return cm.get_cmap(name)


def normalize_energy_map(
    energy: np.ndarray,
    method: str = "percentile",
    vmax: float = 25.0,
    vmin: float = 0.0,
    percentile: float = 99.5,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Robustly normalize a raw energy map into the ``[vmin, vmax]`` display scale.

    Returns ``(normalized_energy, energy_metadata)``.
    """
    energy = np.asarray(energy, dtype=float)
    percentile_used: object = ""

    if method == "percentile":
        ref = float(np.percentile(energy, percentile))
        if ref <= 0:
            ref = float(np.max(energy)) or 1.0
        scaled = energy / ref * vmax
        percentile_used = percentile
    elif method == "mad":
        med = float(np.median(energy))
        mad = float(np.median(np.abs(energy - med)))
        madn = 1.4826 * mad if mad > 0 else (float(np.std(energy)) + 1e-12)
        z = np.clip((energy - med) / madn, 0.0, None)
        # Treat a 6-sigma feature as a "strong" peak mapped near vmax.
        scaled = z / 6.0 * vmax
    else:
        raise ValueError(f"unknown energy_norm_method {method!r}")

    peak = float(np.max(scaled)) if scaled.size else 0.0
    normalized = np.clip(scaled, vmin, vmax)
    meta = {
        "energy_norm_method": method,
        "energy_vmin": float(vmin),
        "energy_vmax": float(vmax),
        "energy_peak": peak,
        "energy_percentile_used": percentile_used,
    }
    return normalized, meta


@dataclass
class EnergyStats:
    method: str
    vmin: float
    vmax: float
    peak: float
    percentile_used: object
    energy: np.ndarray            # normalized (height, width) grid
    freqs: np.ndarray             # frequency (Hz) per image row (top = high)
    times: np.ndarray             # time (s) per image column


class QTransformRenderer:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.width = config.qtransform_image_width
        self.height = config.qtransform_image_height
        self.cmap = _get_cmap("viridis")

    # ----------------------------------------------------------- energy grid
    def _raw_energy(self, strain: np.ndarray) -> np.ndarray:
        """Energy on the (height, width) image grid for the configured axis."""
        cfg = self.config
        strain = np.asarray(strain, dtype=float)

        if cfg.qtransform_backend == "gwpy":
            # Explicitly selected -> use GWpy's Q-transform, surface errors.
            return self._gwpy_energy(strain)

        try:
            return self._cqt_energy(strain)
        except Exception:
            return self._scipy_energy(strain)

    def _cqt_energy(self, strain: np.ndarray) -> np.ndarray:
        """Constant-Q (Morlet/Gaussian filterbank) energy on the image grid.

        Each image frequency row is a Gaussian band-pass of bandwidth f/Q,
        normalized so white noise gives a flat response (so background colour
        comes from the noise PSD, not the filter bandwidth). This yields the
        clean, adaptive-resolution tracks of a real Q-transform instead of the
        blocky fixed-resolution STFT.
        """
        cfg = self.config
        sr = cfg.sample_rate
        strain = np.asarray(strain, dtype=float)
        n = strain.shape[0]

        target_f = image_row_frequencies(
            self.height, cfg.frange_low, cfg.frange_high, cfg.frequency_axis_scale
        )
        centers = np.clip(target_f, 8.0, None)  # wavelet centre needs f > 0
        q = float(np.sqrt(cfg.qrange_low * cfg.qrange_high))

        full_freqs = np.fft.fftfreq(n, d=1.0 / sr)
        strain_fft = np.fft.fft(strain)
        t_full = np.arange(n) / sr
        target_t = np.linspace(0.0, cfg.duration, self.width)

        energy = np.empty((self.height, self.width))
        block = 64  # bound peak memory
        for s in range(0, self.height, block):
            rows = centers[s : s + block]
            sigma_f = rows / q
            win = np.exp(
                -0.5 * ((full_freqs[None, :] - rows[:, None]) / sigma_f[:, None]) ** 2
            )
            # Normalize each filter so filtered white noise has flat variance.
            win /= np.sqrt(np.sum(win ** 2, axis=1, keepdims=True)) + 1e-12
            analytic = np.fft.ifft(strain_fft[None, :] * win, axis=1)
            power = np.abs(analytic) ** 2
            for j in range(power.shape[0]):
                energy[s + j, :] = np.interp(target_t, t_full, power[j])
        return np.clip(energy, 0.0, None)

    def _scipy_energy(self, strain: np.ndarray) -> np.ndarray:
        cfg = self.config
        sr = cfg.sample_rate
        try:
            from scipy.signal import spectrogram

            nperseg = min(512, len(strain))
            noverlap = int(nperseg * 0.85)
            f, t, sxx = spectrogram(
                strain, fs=sr, nperseg=nperseg, noverlap=noverlap, scaling="spectrum"
            )
        except Exception:
            f, t, sxx = self._numpy_spectrogram(strain, sr)

        return self._resample_to_grid(f, t, sxx)

    @staticmethod
    def _numpy_spectrogram(strain: np.ndarray, sr: int):
        nperseg = min(512, len(strain))
        step = max(int(nperseg * 0.15), 1)
        win = np.hanning(nperseg)
        starts = range(0, max(len(strain) - nperseg, 0) + 1, step)
        cols = []
        for s in starts:
            seg = strain[s : s + nperseg] * win
            cols.append(np.abs(np.fft.rfft(seg)) ** 2)
        if not cols:
            cols = [np.abs(np.fft.rfft(strain * np.hanning(len(strain)))) ** 2]
        sxx = np.stack(cols, axis=1)
        f = np.fft.rfftfreq(nperseg, d=1.0 / sr)
        t = (np.arange(sxx.shape[1]) * step + nperseg / 2.0) / sr
        return f, t, sxx

    def _resample_to_grid(self, f: np.ndarray, t: np.ndarray, sxx: np.ndarray) -> np.ndarray:
        cfg = self.config
        target_f = image_row_frequencies(
            self.height, cfg.frange_low, cfg.frange_high, cfg.frequency_axis_scale
        )
        target_t = np.linspace(0.0, cfg.duration, self.width)

        # Interpolate in frequency for each existing time column.
        freq_interp = np.empty((self.height, sxx.shape[1]))
        for j in range(sxx.shape[1]):
            freq_interp[:, j] = np.interp(target_f, f, sxx[:, j])
        # Then interpolate in time for each target frequency row.
        grid = np.empty((self.height, self.width))
        for i in range(self.height):
            grid[i, :] = np.interp(target_t, t, freq_interp[i, :])
        return np.clip(grid, 0.0, None)

    def _gwpy_energy(self, strain: np.ndarray) -> np.ndarray:
        """GWpy's tiled constant-Q transform (the LIGO/Virgo Omega Q-scan).

        Returns gwpy's *normalized energy* mapped onto the fixed linear 0-1000 Hz
        image grid. Raises a clear error if gwpy is unavailable.
        """
        cfg = self.config
        try:
            from gwpy.timeseries import TimeSeries
        except Exception as exc:
            raise MissingOptionalDependency(
                "gwpy is required for --qtransform-backend gwpy. Install it "
                "(`pip install gwpy` or `conda install -c conda-forge gwpy`) or "
                "use --qtransform-backend scipy."
            ) from exc

        ts = TimeSeries(strain, sample_rate=cfg.sample_rate)
        # gwpy uses log-spaced constant-Q tiles, so the low edge must be > 0.
        # We map the result onto the fixed linear 0-1000 image grid afterwards.
        gw_low = max(cfg.frange_low, 10.0)
        qgram = ts.q_transform(
            qrange=cfg.qrange,
            frange=(gw_low, cfg.frange_high),
            whiten=False,
        )
        f = np.asarray(qgram.frequencies.value)
        t = np.asarray(qgram.times.value) - float(qgram.times.value[0])
        sxx = np.asarray(qgram.value).T  # gwpy is (time, freq)
        return self._resample_to_grid(f, t, sxx)

    # --------------------------------------------------------------- render
    def energy_stats(self, strain: np.ndarray) -> EnergyStats:
        cfg = self.config
        raw = self._raw_energy(strain)
        if cfg.qtransform_backend == "gwpy":
            # gwpy already returns *normalized energy*; just clip to the fixed
            # 0-25 display scale (the canonical Omega Q-scan colorbar).
            peak = float(np.max(raw)) if raw.size else 0.0
            normalized = np.clip(raw, cfg.energy_vmin, cfg.energy_vmax)
            meta = {
                "energy_norm_method": "gwpy_normalized_energy",
                "energy_vmin": float(cfg.energy_vmin),
                "energy_vmax": float(cfg.energy_vmax),
                "energy_peak": peak,
                "energy_percentile_used": "",
            }
        else:
            normalized, meta = normalize_energy_map(
                raw,
                method=cfg.energy_norm_method,
                vmax=cfg.energy_vmax,
                vmin=cfg.energy_vmin,
                percentile=cfg.energy_percentile,
            )
        target_f = image_row_frequencies(
            self.height, cfg.frange_low, cfg.frange_high, cfg.frequency_axis_scale
        )
        target_t = np.linspace(0.0, cfg.duration, self.width)
        return EnergyStats(
            method=str(meta["energy_norm_method"]),
            vmin=float(meta["energy_vmin"]),
            vmax=float(meta["energy_vmax"]),
            peak=float(meta["energy_peak"]),
            percentile_used=meta["energy_percentile_used"],
            energy=normalized,
            freqs=target_f,
            times=target_t,
        )

    def render(
        self,
        strain: np.ndarray,
        train_path: Path,
        display_path: Optional[Path] = None,
    ) -> EnergyStats:
        cfg = self.config
        stats = self.energy_stats(strain)
        self._save_train(stats.energy, Path(train_path))
        if display_path is not None:
            self._save_display(stats, Path(display_path))
        return stats

    def _save_train(self, normalized: np.ndarray, path: Path) -> None:
        """Pure spectrogram pixels at exactly (width, height)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        vmax = self.config.energy_vmax
        scaled = np.clip(normalized / vmax, 0.0, 1.0)
        rgba = self.cmap(scaled)  # (H, W, 4) in [0, 1]
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        try:
            from PIL import Image

            Image.fromarray(rgb, mode="RGB").save(path)
        except Exception:
            # Matplotlib fallback: figure-filling axis, no decorations.
            fig = plt.figure(
                figsize=(self.width / 100.0, self.height / 100.0), dpi=100
            )
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(scaled, cmap=self.cmap, vmin=0, vmax=1, aspect="auto")
            ax.axis("off")
            fig.savefig(path, dpi=100)
            plt.close(fig)

    def _save_display(self, stats: EnergyStats, path: Path) -> None:
        """Annotated viridis spectrogram with axes and a colorbar."""
        cfg = self.config
        path.parent.mkdir(parents=True, exist_ok=True)

        dpi = cfg.qtransform_display_dpi
        fig = plt.figure(figsize=(self.width / dpi * 1.3, self.height / dpi), dpi=dpi)
        ax = fig.add_subplot(1, 1, 1)

        # Frequency / time edges for pcolormesh (one more than cell count).
        freqs = stats.freqs[::-1]  # ascending for plotting (bottom = low)
        energy = stats.energy[::-1, :]
        f_edges = self._edges(freqs)
        t_edges = self._edges(stats.times)

        mesh = ax.pcolormesh(
            t_edges,
            f_edges,
            energy,
            cmap="viridis",
            vmin=cfg.energy_vmin,
            vmax=cfg.energy_vmax,
            shading="auto",
        )
        ax.set_yscale(cfg.frequency_axis_scale if cfg.frequency_axis_scale == "log" else "linear")
        ax.set_ylim(cfg.frange_low, cfg.frange_high)
        ax.set_xlim(0.0, cfg.duration)
        ax.set_xlabel("Time [secs]")
        ax.set_ylabel("Frequency [Hz]")
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("Normalized energy")
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
        plt.close(fig)

    @staticmethod
    def _edges(centers: np.ndarray) -> np.ndarray:
        centers = np.asarray(centers, dtype=float)
        if centers.size == 1:
            return np.array([centers[0] - 0.5, centers[0] + 0.5])
        mid = 0.5 * (centers[1:] + centers[:-1])
        first = centers[0] - (mid[0] - centers[0])
        last = centers[-1] + (centers[-1] - mid[-1])
        return np.concatenate([[first], mid, [last]])
