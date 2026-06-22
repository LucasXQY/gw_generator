"""Waveform generation for BBH / BNS chirps.

The chirp time-frequency evolution is determined by *physical waveform
parameters* (masses, spins, ``f_lower``, sample rate, merger/ringdown), never
by an arbitrary fixed frequency band.

Primary path: PyCBC ``get_td_waveform`` (when installed).
Fallback path: a leading-order (Newtonian/quadrupole) analytic inspiral chirp
whose frequency sweep is fully determined by the same physical parameters. The
fallback keeps the whole pipeline runnable on a plain numpy/scipy install and
keeps instantaneous-frequency label extraction exact.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from config import DatasetConfig, MissingOptionalDependency

# Physical constants (SI).
_G = 6.67430e-11
_C = 299792458.0
_MSUN = 1.98892e30


@dataclass
class Waveform:
    """A generated, polarisation-resolved chirp at the configured sample rate."""

    hp: np.ndarray  # plus polarisation
    hc: np.ndarray  # cross polarisation
    sample_rate: int
    params: Dict[str, object]

    @property
    def series(self) -> np.ndarray:
        """Default detector-frame strain proxy (plus polarisation)."""
        return self.hp

    @property
    def analytic(self) -> np.ndarray:
        """Complex analytic signal hp + i*hc (exact instantaneous phase)."""
        return self.hp + 1j * self.hc


# Injectable PyCBC loader (overridable in tests). Returns the
# ``get_td_waveform`` callable or raises MissingOptionalDependency.
def _default_pycbc_loader() -> Callable:
    try:
        from pycbc.waveform import get_td_waveform  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only with pycbc absent
        raise MissingOptionalDependency(
            "PyCBC is not installed. Install pycbc for physical waveforms, or "
            "the analytic fallback will be used automatically."
        ) from exc
    return get_td_waveform


class WaveformGenerator:
    def __init__(
        self,
        config: DatasetConfig,
        pycbc_loader: Optional[Callable[[], Callable]] = None,
        use_pycbc: bool = True,
    ):
        self.config = config
        self._pycbc_loader = pycbc_loader or _default_pycbc_loader
        self._use_pycbc = use_pycbc
        self._get_td_waveform: Optional[Callable] = None

    # ----------------------------------------------------------- parameters
    def sample_intrinsic(self, signal_type: str, rng: np.random.Generator) -> Dict[str, object]:
        """Sample physical waveform parameters for a BBH or BNS event."""
        if signal_type == "BBH":
            mass1 = float(rng.uniform(15.0, 60.0))
            mass2 = float(rng.uniform(15.0, mass1))
            spin1z = float(rng.uniform(-0.8, 0.8))
            spin2z = float(rng.uniform(-0.8, 0.8))
            f_lower = float(rng.choice([20.0, 30.0, 35.0]))
            approximant = "SEOBNRv4_opt"
        elif signal_type == "BNS":
            mass1 = float(rng.uniform(1.2, 2.2))
            mass2 = float(rng.uniform(1.0, mass1))
            # TaylorT4 is a non-spinning approximant; NS spins are tiny anyway.
            spin1z = 0.0
            spin2z = 0.0
            f_lower = float(rng.choice([20.0, 23.0, 30.0]))
            approximant = "TaylorT4"
        else:
            raise ValueError(f"signal_type must be BBH or BNS, got {signal_type!r}")

        return {
            "signal_type": signal_type,
            "mass1": mass1,
            "mass2": mass2,
            "spin1z": spin1z,
            "spin2z": spin2z,
            "f_lower": f_lower,
            "approximant": approximant,
            "distance": float(rng.uniform(100.0, 1000.0)),
        }

    def sample_snr_bin(self, signal_type: str, rng: np.random.Generator):
        """Pick an SNR bin (weighted) and a target SNR within it."""
        bins = self.config.bbh_snr_bins if signal_type == "BBH" else self.config.bns_snr_bins
        names = list(self.config.snr_bin_weights.keys())
        weights = np.array([self.config.snr_bin_weights[n] for n in names], dtype=float)
        weights = weights / weights.sum()
        bin_name = str(rng.choice(names, p=weights))
        lo, hi = bins[bin_name]
        return bin_name, float(rng.uniform(lo, hi))

    # ----------------------------------------------------------- generation
    def generate(self, signal_type: str, params: Dict[str, object]) -> Waveform:
        if self._use_pycbc:
            try:
                return self._generate_pycbc(params)
            except MissingOptionalDependency:
                pass  # PyCBC not installed -> analytic fallback
            except Exception as exc:  # any LAL/approximant error -> analytic fallback
                warnings.warn(
                    f"PyCBC waveform generation failed for {params.get('approximant')} "
                    f"(m1={params.get('mass1')}, m2={params.get('mass2')}): {exc}. "
                    "Using the analytic chirp fallback for this event.",
                    RuntimeWarning,
                )
        return self._generate_analytic(params)

    def _generate_pycbc(self, params: Dict[str, object]) -> Waveform:
        if self._get_td_waveform is None:
            self._get_td_waveform = self._pycbc_loader()
        hp, hc = self._get_td_waveform(
            approximant=str(params["approximant"]),
            mass1=float(params["mass1"]),
            mass2=float(params["mass2"]),
            spin1z=float(params["spin1z"]),
            spin2z=float(params["spin2z"]),
            f_lower=float(params["f_lower"]),
            delta_t=1.0 / self.config.sample_rate,
            distance=float(params.get("distance", 400.0)),
        )
        hp_arr = np.asarray(hp, dtype=float)
        hc_arr = np.asarray(hc, dtype=float)
        # A low-mass inspiral from f_lower can be far longer than the analysis
        # segment; keep the last `n_samples` (late inspiral + merger + ringdown),
        # which is the visible chirp. The merger sits at the end of the array.
        max_len = self.config.n_samples
        if hp_arr.size > max_len:
            hp_arr = hp_arr[-max_len:]
            hc_arr = hc_arr[-max_len:]
        return Waveform(hp_arr, hc_arr, self.config.sample_rate, dict(params))

    def _generate_analytic(self, params: Dict[str, object]) -> Waveform:
        """Leading-order analytic inspiral chirp.

        Frequency sweep f(t) and amplitude A(t) are determined by the chirp mass
        and ``f_lower`` (so masses/f_lower genuinely drive the visible track).
        We build hp = A cos(phi), hc = A sin(phi) so the analytic signal is exact
        and instantaneous-frequency extraction is well defined.
        """
        m1 = float(params["mass1"])
        m2 = float(params["mass2"])
        f_lower = float(params["f_lower"])
        sr = self.config.sample_rate

        m_total = m1 + m2
        mu = m1 * m2 / m_total
        mchirp = mu ** 0.6 * m_total ** 0.4  # solar masses
        m_chirp_s = _G * mchirp * _MSUN / _C ** 3  # chirp mass in seconds

        # ISCO frequency of the total mass caps the inspiral.
        f_isco = _C ** 3 / (6.0 ** 1.5 * np.pi * _G * m_total * _MSUN)
        # Constrain the sweep to this source type's common LIGO-band range.
        band = self.config.signal_freq_bands.get(
            str(params.get("signal_type")), (f_lower, sr / 2.0 * 0.9)
        )
        band_low, band_high = float(band[0]), float(band[1])
        f_lower = max(f_lower, band_low)
        f_high = min(f_isco, sr / 2.0 * 0.9, band_high)
        f_high = max(f_high, f_lower * 1.5)

        # Time-to-coalescence for a given GW frequency (Newtonian, leading order):
        #   tau(f) = 5/256 * m_chirp_s * (pi m_chirp_s f)^(-8/3)
        def tau_of_f(f: float) -> float:
            return (5.0 / 256.0) * m_chirp_s * (np.pi * m_chirp_s * f) ** (-8.0 / 3.0)

        tau_start = tau_of_f(f_lower)
        tau_end = tau_of_f(f_high)
        chirp_len = max(tau_start - tau_end, 1.0 / sr)
        # Keep the inspiral comfortably inside the analysis duration.
        chirp_len = min(chirp_len, self.config.duration * 0.8)

        n = max(int(round(chirp_len * sr)), 8)
        t = np.arange(n) / sr  # time since start of segment
        tau = tau_start - t  # time to coalescence, decreasing
        tau = np.clip(tau, tau_end, None)

        # Instantaneous GW frequency from tau.
        f_inst = (1.0 / np.pi) * (5.0 / (256.0 * tau)) ** (3.0 / 8.0) * m_chirp_s ** (-5.0 / 8.0)
        f_inst = np.clip(f_inst, f_lower, f_high)

        # Phase = 2*pi * integral f dt (cumulative trapezoid).
        phase = np.concatenate([[0.0], np.cumsum(0.5 * (f_inst[1:] + f_inst[:-1]) / sr)])
        phase = 2.0 * np.pi * phase[:n]

        # Newtonian amplitude grows as f^(2/3); apply a soft early-time taper.
        amp = (np.pi * f_inst) ** (2.0 / 3.0)
        amp = amp / np.max(amp)
        taper_len = max(int(0.05 * n), 1)
        window = np.ones(n)
        window[:taper_len] = 0.5 * (1 - np.cos(np.pi * np.arange(taper_len) / taper_len))
        amp = amp * window

        hp = amp * np.cos(phase)
        hc = amp * np.sin(phase)
        out_params = dict(params)
        out_params.setdefault("f_isco", float(f_isco))
        out_params["waveform_source"] = "analytic_fallback"
        return Waveform(hp.astype(float), hc.astype(float), sr, out_params)
