"""Shared frequency <-> normalized-coordinate mapping.

This is the single definition of the Q-transform frequency axis. Both the
rendered image rows and the YOLO label y-coordinates use these functions, so
the image and the labels are guaranteed to share one coordinate system.

``unit`` is in [0, 1]: 0 at ``frange_low`` (bottom of the image), 1 at
``frange_high`` (top of the image). Supports linear and log frequency axes.
"""

from __future__ import annotations

import numpy as np


def freq_to_unit(freq, frange_low: float, frange_high: float, scale: str):
    """Map frequency (Hz) to a [0, 1] axis coordinate."""
    freq = np.asarray(freq, dtype=float)
    if scale == "log":
        lo, hi = np.log(frange_low), np.log(frange_high)
        u = (np.log(np.clip(freq, 1e-9, None)) - lo) / (hi - lo)
    else:
        u = (freq - frange_low) / (frange_high - frange_low)
    return np.clip(u, 0.0, 1.0)


def unit_to_freq(unit, frange_low: float, frange_high: float, scale: str):
    """Inverse of :func:`freq_to_unit`."""
    unit = np.asarray(unit, dtype=float)
    if scale == "log":
        lo, hi = np.log(frange_low), np.log(frange_high)
        return np.exp(lo + unit * (hi - lo))
    return frange_low + unit * (frange_high - frange_low)


def image_row_frequencies(height: int, frange_low: float, frange_high: float, scale: str):
    """Frequency (Hz) for each image row, top row = high frequency.

    Row 0 is the top of the image (``frange_high``); row ``height-1`` is the
    bottom (``frange_low``).
    """
    rows = np.arange(height)
    unit = 1.0 - rows / max(height - 1, 1)
    return unit_to_freq(unit, frange_low, frange_high, scale)
