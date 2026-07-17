"""Strain preprocessing: bandpass only.

Turns the raw injected detector strain into the *normalized* strain used for
the default YOLO Q-transform. The actual energy normalization is done by the
Q-transform itself: GWpy's ``q_transform`` returns *normalized energy* (each
frequency row divided by its median), which already removes the PSD colouring.
A separate strain-level whiten + robust-normalize was therefore redundant with
GWpy and is intentionally omitted -- it changed the strain but had almost no
effect on the resulting normalized-energy image. Only the band-limiting filter
remains here. Uses scipy when available and degrades to a numpy FFT otherwise.
"""

from __future__ import annotations

import numpy as np

from config import DatasetConfig


class Preprocessor:
    def __init__(self, config: DatasetConfig):
        self.config = config

    def preprocess(self, raw: np.ndarray) -> np.ndarray:
        """Band-limit the strain to the Q-transform window.

        Energy normalization is left to GWpy's ``q_transform`` (normalized
        energy), so no strain-level whitening or scaling is applied here.
        """
        x = np.asarray(raw, dtype=float)
        if self.config.filter_mode == "bandpass":
            x = self._bandpass(x)
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
