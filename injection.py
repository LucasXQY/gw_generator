"""Target-SNR chirp injection with per-detector effects.

The same shared waveform is injected into each detector's background with its
own arrival-time delay, amplitude scale, and optional sign/phase flip. The
clean (noise-free) placed signal is returned so labels can be derived from the
true instantaneous frequency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Injection:
    combined: np.ndarray          # noise + scaled signal (raw detector strain)
    clean_signal: np.ndarray      # scaled signal placed in the full segment, no noise
    start_index: int              # sample index where the signal starts
    snr: float                    # achieved (whitened-norm) SNR
    injection_time: float         # merger/coalescence time within the segment (s)
    time_delay: float             # applied arrival-time delay (s)
    amp_scale: float              # applied amplitude scale
    sign_flip: bool


def _whitened_norm_snr(signal: np.ndarray, noise_sigma: float) -> float:
    """SNR of a signal in white Gaussian noise of std ``noise_sigma``."""
    if noise_sigma <= 0:
        noise_sigma = 1.0
    return float(np.sqrt(np.sum((signal / noise_sigma) ** 2)))


def inject_chirp(
    background: np.ndarray,
    waveform: np.ndarray,
    injection_time: float,
    target_snr: float,
    sample_rate: int,
    time_delay: float = 0.0,
    amp_scale: float = 1.0,
    sign_flip: bool = False,
    noise_reference: np.ndarray | None = None,
) -> Injection:
    """Inject ``waveform`` into ``background`` scaled to ``target_snr``.

    ``injection_time`` is the **merger/coalescence** time: the waveform is placed
    so its final sample (merger + ringdown) lands at that time, which keeps the
    loud late-inspiral/merger inside the segment even for long inspirals.
    ``time_delay`` shifts the arrival time (s); ``amp_scale`` rescales the
    detector amplitude; ``sign_flip`` inverts the polarisation.
    ``noise_reference`` is the glitch-free noise used to estimate the noise std
    for SNR scaling; without it a loud glitch in ``background`` inflates the
    estimate and the chirp is injected louder than ``target_snr``.
    """
    background = np.asarray(background, dtype=float)
    waveform = np.asarray(waveform, dtype=float)
    n = background.shape[0]

    sigma_source = background if noise_reference is None else np.asarray(noise_reference, dtype=float)
    noise_sigma = float(np.std(sigma_source)) or 1.0

    # Unit-SNR template, then scale to the requested target.
    base_snr = _whitened_norm_snr(waveform, noise_sigma)
    if base_snr <= 0:
        base_snr = 1.0
    scale = (float(target_snr) / base_snr) * float(amp_scale)
    signal = waveform * scale
    if sign_flip:
        signal = -signal

    seg_len = signal.shape[0]
    merger_time = injection_time + time_delay
    merger_index = int(round(merger_time * sample_rate))
    # Anchor the waveform END (merger) at merger_index.
    start_index = merger_index - seg_len

    clean = np.zeros(n, dtype=float)
    dst0 = max(0, start_index)
    dst1 = min(n, start_index + seg_len)
    src0 = dst0 - start_index
    src1 = src0 + (dst1 - dst0)
    if dst1 > dst0:
        clean[dst0:dst1] = signal[src0:src1]

    combined = background + clean
    achieved_snr = _whitened_norm_snr(clean, noise_sigma)

    return Injection(
        combined=combined,
        clean_signal=clean,
        start_index=start_index,
        snr=achieved_snr,
        injection_time=float(merger_time),
        time_delay=float(time_delay),
        amp_scale=float(amp_scale),
        sign_flip=bool(sign_flip),
    )
