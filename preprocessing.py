"""Strain preprocessing: bandpass -> whiten -> robust normalize.

Turns the raw injected detector strain into the *normalized* strain used for
the default YOLO Q-transform. Uses scipy when available and degrades to a
numpy FFT implementation otherwise.
"""

from __future__ import annotations

import numpy as np

from config import DatasetConfig


class Preprocessor:
    def __init__(self, config: DatasetConfig):
        self.config = config

    def preprocess(self, raw: np.ndarray, noise_reference: np.ndarray | None = None) -> np.ndarray:
        """Bandpass -> whiten -> normalize.

        ``noise_reference`` is the signal-free background used to estimate the
        whitening ASD. Estimating it from the noise (not from ``raw``, which
        contains the loud chirp/glitch) prevents the whitener from carving a
        notch at the signal frequency and self-cancelling the chirp.
        """
        x = np.asarray(raw, dtype=float)
        ref = np.asarray(noise_reference, dtype=float) if noise_reference is not None else x
        if self.config.filter_mode == "bandpass":
            x = self._bandpass(x)
            ref = self._bandpass(ref)
        x = self._whiten(x, ref)
        x = self._normalize(x)
        return x.astype(float)

    # ------------------------------------------------------------------ steps
    def _bandpass(self, x: np.ndarray) -> np.ndarray:
        flo, fhi = self.config.frange
        sr = self.config.sample_rate
        try:
            from scipy.signal import butter, sosfiltfilt

            nyq = sr / 2.0
            high = min(fhi / nyq, 0.999)
            if flo <= 0:
                # 0-1000 Hz window -> low-pass (no positive low edge for a bandpass).
                sos = butter(4, high, btype="low", output="sos")
            else:
                low = max(flo / nyq, 1e-4)
                sos = butter(4, [low, high], btype="band", output="sos")
            return sosfiltfilt(sos, x)
        except Exception:
            return self._fft_bandpass(x, flo, fhi, sr)

    @staticmethod
    def _fft_bandpass(x: np.ndarray, flo: float, fhi: float, sr: int) -> np.ndarray:
        n = x.shape[0]
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        spec = np.fft.rfft(x)
        mask = (freqs >= flo) & (freqs <= fhi)
        spec = spec * mask
        return np.fft.irfft(spec, n=n)

    def _whiten(self, x: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Flatten ``x``'s spectrum by the ASD estimated from ``reference``.

        The ASD comes from the signal-free background, so loud chirps/glitches
        in ``x`` are not divided out (no self-notching).
        """
        n = x.shape[0]
        spec = np.fft.rfft(x)
        ref_mag = np.abs(np.fft.rfft(reference, n=n))
        asd = self._smooth_spectrum(ref_mag)
        asd = np.where(asd > 0, asd, np.median(asd) + 1e-12)
        whitened = spec / asd
        return np.fft.irfft(whitened, n=n)

    @staticmethod
    def _smooth_spectrum(mag: np.ndarray) -> np.ndarray:
        """Smooth a magnitude spectrum to a clean ASD estimate (robust median)."""
        width = max(int(mag.size * 0.01), 9)
        if width % 2 == 0:
            width += 1
        try:
            from scipy.ndimage import median_filter

            return median_filter(mag, size=width, mode="nearest")
        except Exception:
            kernel = np.ones(width) / width
            return np.convolve(mag, kernel, mode="same")

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.config.normalization == "robust":
            med = np.median(x)
            mad = np.median(np.abs(x - med))
            scale = 1.4826 * mad if mad > 0 else (np.std(x) + 1e-12)
            return (x - med) / scale
        std = np.std(x)
        return (x - np.mean(x)) / (std + 1e-12)
