"""Detector noise and glitch generation.

Each detector-level sample gets an *independent* noise realization (its own
``noise_id``) so H1 and L1 of the same event never share noise. Glitches are
synthetic time-frequency bursts by default; a real-glitch directory can be
plugged in later via the config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import DatasetConfig


def _aligo_asd(freqs: np.ndarray) -> np.ndarray:
    """Analytic Advanced-LIGO amplitude spectral density (arbitrary units).

    Uses the standard dimensionless fit (Ajith 2011) with f0 = 215 Hz, floored
    below 20 Hz (the seismic wall) to avoid a divergent DC term.
    """
    f = np.asarray(freqs, dtype=float).copy()
    f = np.where(f < 20.0, 20.0, f)  # floor the seismic wall
    x = f / 215.0
    psd = x ** (-4.14) - 5.0 * x ** (-2.0) + 111.0 * (
        1.0 - x ** 2 + 0.5 * x ** 4
    ) / (1.0 + 0.5 * x ** 2)
    psd = np.clip(psd, 1e-3, None)
    asd = np.sqrt(psd)
    return asd / np.median(asd)  # normalize scale


@dataclass
class NoiseRealization:
    series: np.ndarray
    noise_id: str
    noise_type: str  # "gaussian" or "real"


@dataclass
class GlitchRealization:
    series: np.ndarray
    glitch_id: str
    glitch_type: str
    start_time: float
    end_time: float
    center_freq: float
    low_freq: float
    high_freq: float
    amplitude: float


class NoiseGenerator:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self._noise_counter = 0
        self._glitch_counter = 0

    def background_noise(self, rng: np.random.Generator) -> NoiseRealization:
        """An independent Gaussian background coloured by an analytic aLIGO PSD.

        Colouring (seismic wall at low f, sensitivity bucket near ~200-300 Hz,
        shot noise rising at high f) makes the *raw* Q-transform look like real
        detector data, while whitening flattens it for the *normalized* image.
        """
        n = self.config.n_samples
        sr = self.config.sample_rate

        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        asd = _aligo_asd(freqs)

        # White Gaussian -> colour in the frequency domain -> back to time.
        white = rng.standard_normal(n)
        spec = np.fft.rfft(white) * asd
        colored = np.fft.irfft(spec, n=n)
        colored = colored / (np.std(colored) + 1e-12)

        self._noise_counter += 1
        return NoiseRealization(
            series=colored.astype(float),
            noise_id=f"noise_{self._noise_counter:08d}",
            noise_type="gaussian_aligo_colored",
        )

    def sample_glitch(self, rng: np.random.Generator) -> GlitchRealization:
        """A synthetic sine-Gaussian glitch placed at a random time/frequency."""
        sr = self.config.sample_rate
        n = self.config.n_samples
        t = np.arange(n) / sr
        duration = self.config.duration

        glitch_type = str(rng.choice(self.config.glitch_types))
        center_time = float(rng.uniform(0.15 * duration, 0.85 * duration))
        # Glitch centre frequency drawn inside the Q-transform window.
        flo, fhi = self.config.frange
        center_freq = float(rng.uniform(flo * 1.5, fhi * 0.8))
        # Q controls the time-frequency extent.
        q = float(rng.uniform(5.0, 30.0))
        sigma_t = q / (2.0 * np.pi * center_freq)
        sigma_t = float(np.clip(sigma_t, 0.5 / sr * 50, duration / 6.0))

        envelope = np.exp(-0.5 * ((t - center_time) / sigma_t) ** 2)
        series = envelope * np.sin(2.0 * np.pi * center_freq * (t - center_time))
        series = series / (np.std(series) + 1e-12)
        amp = float(rng.uniform(3.0, 8.0))
        series = series * amp

        # Visible support (~ +/- 3 sigma) and bandwidth from the Q.
        start_time = max(0.0, center_time - 3.0 * sigma_t)
        end_time = min(duration, center_time + 3.0 * sigma_t)
        bw = center_freq / q
        low_freq = max(flo, center_freq - 3.0 * bw)
        high_freq = min(fhi, center_freq + 3.0 * bw)

        self._glitch_counter += 1
        return GlitchRealization(
            series=series.astype(float),
            glitch_id=f"glitch_{self._glitch_counter:08d}",
            glitch_type=glitch_type,
            start_time=start_time,
            end_time=end_time,
            center_freq=center_freq,
            low_freq=low_freq,
            high_freq=high_freq,
            amplitude=amp,
        )
