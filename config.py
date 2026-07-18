"""Configuration for the GW-YOLO-style two-detector synthetic dataset builder.

This project builds a *GW-YOLO-style* synthetic gravitational-wave dataset for
four benchmark tasks (single-detector detection, cross-detector matching,
coherent event detection, low-SNR / glitch rejection). It is **not** an exact
reproduction of any official GW-YOLO dataset.

``DatasetConfig`` is the single source of generation parameters. Module-level
tuples define the CSV schemas so the builder and the readers agree on columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose
from pathlib import Path
from typing import Dict, Optional, Tuple


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
SIGNAL_TYPES = ("BBH", "BNS", "None")

# Mutually-exclusive per-event categories. Signal type and glitch presence are
# derived from this label.
GLOBAL_CLASSES = (
    "bbh_clean",
    "bns_clean",
    "bbh_glitch",
    "bns_glitch",
    "glitch_only",
    "pure_noise",
)

DEFAULT_GLITCH_TYPES = (
    "Blip",
    "Scattered_Light",
    "Tomte",
    "Koi_Fish",
    "Whistle",
    "Low_Frequency_Burst",
    "Power_Line",
    "Violin_Mode",
)

SUPPORTED_DETECTORS = frozenset({"H1", "L1"})

SPLITS = ("train", "val", "test")

# --------------------------------------------------------------------------- #
# Hardcoded frequency coordinate system.
#
# The Q-transform window, the display axis, and the YOLO y-normalization all use
# this single fixed log 20-1000 Hz window. A constant-Q transform is intrinsically
# log-spaced in frequency (tile bandwidth scales with frequency), and gwpy's
# q_transform returns log-spaced tiles, so a log axis matches both the transform's
# native resolution and the GW-YOLO / LIGO Omega-scan display convention. The low
# edge is 20 Hz (a log axis cannot include 0 Hz, and 20 Hz is the common CBC lower
# cutoff). It is intentionally NOT configurable: --frange-* / --frequency-axis-scale
# CLI flags are accepted but ignored.
# --------------------------------------------------------------------------- #
FIXED_FRANGE_LOW = 20.0
FIXED_FRANGE_HIGH = 1000.0
FIXED_FREQUENCY_AXIS_SCALE = "log"

# Web-verified common GW frequency ranges per source type, in the LIGO/Virgo
# band (~10-1000 Hz). Chirp insertion is constrained to the matching band so a
# signal's visible energy lands where that source type actually radiates.
#   BBH : GW150914 swept 35->250 Hz, FLSO ~220 Hz; lighter systems reach higher.
#   BNS : sweeps from ~10 Hz up to coalescence near ~1 kHz (FLSO ~2 kHz).
#   NSBH: FLSO ~400 Hz for typical systems.
# Sources: arXiv:gr-qc/0205122, arXiv:gr-qc/9902019, arXiv:1307.1757,
#          PRL 116, 061102 (GW150914).
def _default_signal_freq_bands() -> Dict[str, Tuple[float, float]]:
    return {
        "BBH": (20.0, 350.0),
        "BNS": (20.0, 1000.0),
        "NSBH": (20.0, 400.0),
    }


# --------------------------------------------------------------------------- #
# CSV / output schemas (single source of truth, shared with readers)
# --------------------------------------------------------------------------- #
# One row per detector-level sample.
METADATA_FIELDS = (
    "sample_id",
    "event_id",
    "chirp_id",
    "split",
    "detector",
    "counterpart_sample_ids",
    "global_class",
    "signal_type",
    "has_chirp",
    "has_glitch",
    "target_snr",
    "actual_snr",
    "snr_bin",
    "detector_snr",
    "network_snr",
    "injection_time",
    "detector_time_delay",
    "amplitude_scale",
    "phase_or_sign_flip",
    "chirp_start_time",
    "chirp_end_time",
    "chirp_freq_low",
    "chirp_freq_high",
    "qtransform_frange_low",
    "qtransform_frange_high",
    "label_frange_low",
    "label_frange_high",
    "frequency_axis_scale",
    "raw_strain_path",
    "normalized_strain_path",
    "qtransform_raw_path",
    "qtransform_normalized_path",
    "qtransform_display_raw_path",
    "qtransform_display_normalized_path",
    "yolo_label_path",
    "label_source",
    "energy_norm_method",
    "energy_vmin",
    "energy_vmax",
    "energy_peak",
    "energy_percentile_used",
    "noise_id",
    "noise_type",
    "glitch_id",
    "glitch_type",
    # --- physical parameters (denormalized per sample for single-table ML use)
    "waveform_source",
    "waveform_approximant",
    "mass1",
    "mass2",
    "spin1z",
    "spin2z",
    "chirp_mass",
    "total_mass",
    "mass_ratio",
    "chi_eff",
    "distance",
    "f_lower",
    "geocent_injection_time",
    # --- YOLO chirp box in normalized image coordinates
    "chirp_yolo_cx",
    "chirp_yolo_cy",
    "chirp_yolo_w",
    "chirp_yolo_h",
    "has_label",
    "num_boxes",
    # --- glitch time-frequency detail
    "glitch_start_time",
    "glitch_end_time",
    "glitch_center_freq",
    "glitch_low_freq",
    "glitch_high_freq",
    "glitch_amplitude",
    "glitch_source",
    "glitch_gps",
    "glitch_snr_catalog",
    # --- source-group isolation + background provenance (G1)
    "glitch_source_group",
    "background_source",
    "background_source_group",
    "background_gps",
    # --- acquisition (so a .npy can be loaded without joining other tables)
    "sample_rate",
    "duration",
    "n_samples",
)

# One row per astrophysical event.
EVENT_FIELDS = (
    "event_id",
    "chirp_id",
    "split",
    "signal_type",
    "has_chirp",
    "waveform_approximant",
    "mass1",
    "mass2",
    "spin1z",
    "spin2z",
    "f_lower",
    "sample_rate",
    "duration",
    "injection_time",
    "network_snr",
    "detectors",
    "num_detector_samples",
    "qtransform_frange_low",
    "qtransform_frange_high",
    "frequency_axis_scale",
    "distance",
    "chirp_mass",
    "total_mass",
    "mass_ratio",
    "chi_eff",
    "waveform_source",
    "n_samples",
)

# One row per positive (same-chirp cross-detector) pair.
MATCH_FIELDS = (
    "match_id",
    "event_id",
    "chirp_id",
    "split",
    "anchor_sample_id",
    "positive_sample_id",
    "anchor_detector",
    "positive_detector",
    "same_chirp",
    "same_noise",
    "notes",
)

# One row per negative pair.
NEGATIVE_FIELDS = (
    "negative_pair_id",
    "split",
    "anchor_sample_id",
    "candidate_sample_id",
    "anchor_event_id",
    "candidate_event_id",
    "anchor_chirp_id",
    "candidate_chirp_id",
    "anchor_detector",
    "candidate_detector",
    "same_chirp",
    "same_event",
    "valid_time_delay",
    "time_delay_difference",
    "frequency_overlap_score",
    "snr_bin_anchor",
    "snr_bin_candidate",
    "negative_type",
    "notes",
)

# Combined positive + negative pair table.
PAIR_FIELDS = (
    "pair_id",
    "pair_label",
    "pair_type",
    "split",
    "anchor_sample_id",
    "candidate_sample_id",
    "anchor_event_id",
    "candidate_event_id",
    "anchor_chirp_id",
    "candidate_chirp_id",
    "anchor_detector",
    "candidate_detector",
    "same_chirp",
    "same_event",
    "valid_time_delay",
    "time_delay_difference",
    "frequency_overlap_score",
    "snr_bin_anchor",
    "snr_bin_candidate",
    "notes",
)

# All negative-pair categories the generator knows how to produce.
NEGATIVE_PAIR_TYPES = (
    "different_chirp_cross_detector",
    "chirp_vs_glitch",
    "chirp_vs_noise",
    "invalid_delay_same_chirp",
    "same_detector_different_event",
    "similar_snr_different_chirp",
    "similar_frequency_different_chirp",
)

# Output subdirectory trees (each holds {split}/{detector}/ leaves).
OUTPUT_SUBDIRS: Tuple[str, ...] = (
    "raw_series",
    "normalized_series",
    "qtransform_raw",
    "qtransform_normalized",
    "qtransform_display_raw",
    "qtransform_display_normalized",
    "labels_yolo",
)


class MissingOptionalDependency(ImportError):
    """Raised when a runtime-only scientific dependency is unavailable."""


# --------------------------------------------------------------------------- #
# SNR bins
# --------------------------------------------------------------------------- #
def _default_bbh_snr_bins() -> Dict[str, Tuple[float, float]]:
    return {
        "very_low": (4.0, 6.0),
        "low": (6.0, 8.0),
        "medium": (8.0, 12.0),
        "high": (12.0, 20.0),
        "very_high": (20.0, 35.0),
    }


def _default_bns_snr_bins() -> Dict[str, Tuple[float, float]]:
    return {
        "very_low": (4.0, 6.0),
        "low": (6.0, 9.0),
        "medium": (9.0, 14.0),
        "high": (14.0, 22.0),
        "very_high": (22.0, 35.0),
    }


def _default_snr_bin_weights() -> Dict[str, float]:
    return {
        "very_low": 0.15,
        "low": 0.25,
        "medium": 0.30,
        "high": 0.20,
        "very_high": 0.10,
    }


@dataclass
class DatasetConfig:
    # --------------------------------------------------------------- core
    num_events: int = 50
    duration: float = 4.0
    sample_rate: int = 4096
    output_dir: Path | str = Path("gw_synthetic_dataset")
    seed: int = 42

    # --------------------------------------------------------------- split (event-level, 3-way)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # --------------------------------------------------------------- detectors
    detectors: Tuple[str, ...] = field(default_factory=lambda: ("H1", "L1"))
    # Physical H1/L1 light-travel delay is ~10 ms; we add a small extra jitter.
    detector_time_delay_max: float = 0.010
    detector_amp_scale_jitter: float = 0.1
    enable_sign_flip: bool = False
    # Per-detector SNR jitter (fraction); H1 and L1 SNR may differ within a bin.
    detector_snr_jitter: float = 0.15

    # --------------------------------------------------------------- noise / glitch
    use_real_noise: bool = False
    real_noise_dir: Optional[Path | str] = None
    glitch_metadata_csv: Optional[Path | str] = None
    glitch_types: Tuple[str, ...] = field(default_factory=lambda: DEFAULT_GLITCH_TYPES)

    # --- real (GWOSC) glitch injection ------------------------------------- #
    # "synthetic" (default sine-Gaussian) or "gwosc" (real strain fetched by GPS
    # from a Gravity Spy pool CSV supplied via glitch_metadata_csv).
    glitch_source: str = "synthetic"
    # Where fetched/resampled GWOSC strain segments are cached (raw, pre-whiten).
    # Default: <output_dir>/glitch_cache (set in __post_init__).
    real_glitch_cache_dir: Optional[Path | str] = None
    # Seconds fetched on each side of the glitch GPS time (>= whiten fftlength need).
    glitch_fetch_halfwin: float = 4.0
    # Crop length used when a pool row lacks a `duration` (seconds).
    glitch_default_duration: float = 0.5
    glitch_whiten: bool = True
    # Synthetic glitches only: the sine-Gaussian amplitude range. Real (gwosc)
    # glitches keep their NATURAL amplitude -- the whitened segment is used as
    # the whole sample with a unit robust noise floor, so rescaling the glitch
    # independently of its own noise is neither possible nor desirable.
    glitch_amplitude_range: Tuple[float, float] = (3.0, 8.0)
    # Keep the injected glitch this far (s) from the segment edges.
    glitch_placement_margin: float = 0.1
    # Resample a new real glitch this many times if a fetch fails (GPS not in
    # GWOSC open data) or no energy ridge is found. Unavailable GPS are cached
    # and skipped, so later samples rarely need many tries.
    max_glitch_attempts: int = 12
    # If True, fall back to a synthetic glitch when GWOSC/gwpy is unavailable or
    # no ridge is found after max_glitch_attempts; if False, raise (hard error).
    glitch_allow_synthetic_fallback: bool = False
    # Reject a real-glitch ridge box covering more than this fraction of the
    # image (log-frequency aware) and resample: an oversized box means the
    # injection lit up the background, not a localized glitch.
    glitch_max_box_frac: float = 0.60

    # --- real (GWOSC) off-source backgrounds (G1/D2) ----------------------- #
    # "synthetic" (colored Gaussian) or "gwosc": non-glitch samples draw real
    # off-source segments from the SAME split source-group pool as the
    # glitches, so background domain never correlates with has_glitch.
    noise_source: str = "synthetic"
    # Off-source candidates whose render shows a significant ridge box are
    # rejected and resampled up to this many times, then the build FAILS
    # explicitly -- never a silent synthetic fallback.
    max_background_attempts: int = 20
    # Exclude off-source GPS within this many seconds of any known pool glitch.
    background_glitch_exclusion: float = 8.0
    # Significance gate for the off-source transient veto: a candidate is
    # rejected only when its render has pixels above this multiple of the
    # median energy. Calibrated so clean-noise renders (peak/median <= ~12)
    # pass while >=12 sigma transients (ratio >= ~100) are rejected; the
    # label-time gate stays at glitch_box_from_ridge's default 5x.
    background_veto_floor_gate: float = 15.0
    # Spacing (s) of the deterministic off-source candidate grid inside each
    # 4096 s file (shared by prefetch_offsource_cache and the sampler).
    offsource_grid_step: float = 32.0

    # --------------------------------------------------------------- image / label output
    save_raw_outputs: bool = True
    save_display_images: bool = True
    generate_yolo_labels: bool = True
    write_empty_labels: bool = True
    # If the visible chirp lies fully outside the Q-transform window, drop the
    # label (True) or keep a clipped/invalid one (False).
    drop_label_if_outside_window: bool = True

    # --------------------------------------------------------------- Q-transform
    qtransform_backend: str = "gwpy"  # "gwpy" (LIGO Omega Q-scan) or "scipy" (CQT)
    qtransform_image_width: int = 640
    qtransform_image_height: int = 640
    qtransform_display_dpi: int = 100
    # NOTE: frange_low/frange_high/frequency_axis_scale are hardcoded in
    # __post_init__ to the FIXED_* values above; any value passed here is
    # overwritten (the window is deliberately not configurable).
    frange_low: float = FIXED_FRANGE_LOW
    frange_high: float = FIXED_FRANGE_HIGH
    qrange_low: float = 4.0
    qrange_high: float = 64.0
    frequency_axis_scale: str = FIXED_FREQUENCY_AXIS_SCALE

    # Per-type chirp insertion frequency bands (web-verified, see module docstring).
    signal_freq_bands: Dict[str, Tuple[float, float]] = field(
        default_factory=_default_signal_freq_bands
    )

    # --------------------------------------------------------------- energy normalization
    energy_norm_method: str = "percentile"  # "percentile" or "mad"
    energy_vmin: float = 0.0
    energy_vmax: float = 25.0
    energy_percentile: float = 99.5

    # --------------------------------------------------------------- label extraction
    label_envelope_threshold: float = 0.05  # fraction of waveform peak envelope
    label_ridge_threshold: float = 0.30  # fraction of peak energy for ridge fallback

    # --------------------------------------------------------------- SNR
    bbh_snr_bins: Dict[str, Tuple[float, float]] = field(
        default_factory=_default_bbh_snr_bins
    )
    bns_snr_bins: Dict[str, Tuple[float, float]] = field(
        default_factory=_default_bns_snr_bins
    )
    snr_bin_weights: Dict[str, float] = field(
        default_factory=_default_snr_bin_weights
    )

    # --------------------------------------------------------------- injection timing
    injection_time_min: float = 1.0
    injection_time_max: float = 3.0

    # --------------------------------------------------------------- preprocessing
    normalization: str = "robust"
    filter_mode: str = "bandpass"

    # --------------------------------------------------------------- pair generation
    negative_pairs_per_positive: int = 3
    hard_negative_fraction: float = 0.5
    enable_invalid_delay_negatives: bool = True
    enable_chirp_vs_glitch_negatives: bool = True
    enable_chirp_vs_noise_negatives: bool = True
    allow_cross_split_negative_pairs: bool = False
    # A cross-detector arrival-time difference is physically valid only up to
    # the light-travel time plus a tolerance. Used to flag invalid_delay pairs.
    max_physical_time_delay: float = 0.012

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.real_noise_dir is not None:
            self.real_noise_dir = Path(self.real_noise_dir)
        if self.glitch_metadata_csv is not None:
            self.glitch_metadata_csv = Path(self.glitch_metadata_csv)
        if isinstance(self.detectors, list):
            self.detectors = tuple(self.detectors)

        # ---- real-glitch (GWOSC) settings
        if self.glitch_source not in ("synthetic", "gwosc"):
            raise ValueError(
                f"glitch_source must be 'synthetic' or 'gwosc', got {self.glitch_source!r}"
            )
        if self.real_glitch_cache_dir is None:
            self.real_glitch_cache_dir = self.output_dir / "glitch_cache"
        else:
            self.real_glitch_cache_dir = Path(self.real_glitch_cache_dir)
        self.glitch_amplitude_range = tuple(self.glitch_amplitude_range)

        # ---- real off-source background settings (G1/D2)
        if self.noise_source not in ("synthetic", "gwosc"):
            raise ValueError(
                f"noise_source must be 'synthetic' or 'gwosc', got {self.noise_source!r}"
            )
        if self.noise_source == "gwosc":
            if self.glitch_source != "gwosc":
                raise ValueError(
                    "noise_source='gwosc' requires glitch_source='gwosc' (the "
                    "off-source sampler draws from the glitch pool's source groups)"
                )
            if self.glitch_allow_synthetic_fallback:
                raise ValueError(
                    "noise_source='gwosc' forbids glitch_allow_synthetic_fallback: "
                    "a silent synthetic fallback would re-couple background "
                    "domain with has_glitch"
                )

        # ---- hardcode the frequency coordinate system (window is not configurable)
        self.frange_low = FIXED_FRANGE_LOW
        self.frange_high = FIXED_FRANGE_HIGH
        self.frequency_axis_scale = FIXED_FREQUENCY_AXIS_SCALE

        # ---- numeric checks
        if self.num_events < 1:
            raise ValueError("num_events must be >= 1")
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.qtransform_image_width <= 0 or self.qtransform_image_height <= 0:
            raise ValueError("qtransform image dimensions must be positive")

        # ---- 3-way split ratios
        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if not isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "train_ratio + val_ratio + test_ratio must equal 1.0 "
                f"(got {ratio_sum:.8f})"
            )
        for name, r in (
            ("train_ratio", self.train_ratio),
            ("val_ratio", self.val_ratio),
            ("test_ratio", self.test_ratio),
        ):
            if r < 0:
                raise ValueError(f"{name} must be >= 0")

        # ---- frequency / Q ranges (frange is the fixed 0-1000 Hz window)
        if self.frange_low < 0 or self.frange_low >= self.frange_high:
            raise ValueError("frange_low must be >= 0 and lower than frange_high")
        if self.frange_high > self.sample_rate / 2.0:
            raise ValueError(
                "frange_high must not exceed the Nyquist frequency "
                f"(sample_rate/2 = {self.sample_rate / 2.0:g} Hz)"
            )
        if self.qrange_low <= 0 or self.qrange_low >= self.qrange_high:
            raise ValueError("qrange_low must be positive and lower than qrange_high")

        # ---- injection timing within duration
        if self.injection_time_min < 0 or self.injection_time_max > self.duration:
            raise ValueError("injection_time range must stay within the sample duration")
        if self.injection_time_min >= self.injection_time_max:
            raise ValueError("injection_time_min must be lower than injection_time_max")

        # ---- backend
        if self.qtransform_backend not in {"gwpy", "scipy"}:
            raise ValueError("qtransform_backend must be 'gwpy' or 'scipy'")

        # ---- detectors
        if not self.detectors:
            raise ValueError("detectors must be non-empty")
        if len(set(self.detectors)) != len(self.detectors):
            raise ValueError("detectors must contain unique detector names")
        for det in self.detectors:
            if det not in SUPPORTED_DETECTORS:
                raise ValueError(
                    f"detectors contains unsupported detector '{det}'; "
                    f"allowed values are {sorted(SUPPORTED_DETECTORS)}"
                )
        if self.detector_time_delay_max < 0:
            raise ValueError("detector_time_delay_max must be >= 0")

        # ---- energy normalization
        if self.energy_norm_method not in {"percentile", "mad"}:
            raise ValueError(
                "energy_norm_method must be 'percentile' or 'mad' "
                f"(got '{self.energy_norm_method}')"
            )
        if self.energy_vmin >= self.energy_vmax:
            raise ValueError("energy_vmin must be lower than energy_vmax")

        # ---- frequency axis scale
        if self.frequency_axis_scale not in {"log", "linear"}:
            raise ValueError(
                "frequency_axis_scale must be 'log' or 'linear' "
                f"(got '{self.frequency_axis_scale}')"
            )

        # ---- SNR bins consistency
        for label, bins in (("bbh", self.bbh_snr_bins), ("bns", self.bns_snr_bins)):
            for bin_name, (lo, hi) in bins.items():
                if lo >= hi:
                    raise ValueError(
                        f"{label}_snr_bins['{bin_name}'] has min ({lo}) >= max ({hi})"
                    )
        weight_sum = sum(self.snr_bin_weights.values())
        if not isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"snr_bin_weights must sum to 1.0 (got {weight_sum:.8f})")
        wkeys = set(self.snr_bin_weights)
        if wkeys != set(self.bbh_snr_bins) or wkeys != set(self.bns_snr_bins):
            raise ValueError(
                "snr_bin_weights keys must match bbh_snr_bins and bns_snr_bins keys"
            )

        # ---- pair generation
        if self.negative_pairs_per_positive < 0:
            raise ValueError("negative_pairs_per_positive must be >= 0")
        if not 0.0 <= self.hard_negative_fraction <= 1.0:
            raise ValueError("hard_negative_fraction must be within [0, 1]")

    # ------------------------------------------------------------ properties
    @property
    def n_samples(self) -> int:
        return int(round(self.duration * self.sample_rate))

    @property
    def frange(self) -> Tuple[float, float]:
        return (self.frange_low, self.frange_high)

    @property
    def qrange(self) -> Tuple[float, float]:
        return (self.qrange_low, self.qrange_high)

    @property
    def image_size(self) -> Tuple[int, int]:
        return (self.qtransform_image_width, self.qtransform_image_height)

    def split_ratios(self) -> Dict[str, float]:
        return {
            "train": self.train_ratio,
            "val": self.val_ratio,
            "test": self.test_ratio,
        }
