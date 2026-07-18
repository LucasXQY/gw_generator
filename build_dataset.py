"""Two-detector (H1 + L1) event-level synthetic GW dataset builder + CLI.

Pipeline per event (shared chirp) -> per detector (independent realization):

    WaveformGenerator  sample_intrinsic() + generate() once per chirp event
    NoiseGenerator     independent background (+ optional glitch) per detector
    injection          per-detector arrival delay / amp scale / sign flip / SNR
    Preprocessor       raw strain -> normalized strain
    QTransformRenderer raw + normalized, train (pure) + display (axes+colorbar)
    LabelGenerator     instantaneous-frequency label (ridge fallback), config frange

Then positive/negative/combined pairs, task_protocols.yaml, gw_data.yaml, and
leakage validation. Splitting is performed at the event level first, so H1/L1
of one event never cross train/val/test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    EVENT_FIELDS,
    GLOBAL_CLASSES,
    MATCH_FIELDS,
    METADATA_FIELDS,
    NEGATIVE_FIELDS,
    OUTPUT_SUBDIRS,
    PAIR_FIELDS,
    DatasetConfig,
)
from injection import inject_chirp
from label_generator import LabelGenerator, TimeFrequencyBox
from noise_generator import NoiseGenerator
from real_glitch import GlitchDependencyError, GlitchFetchError, RealGlitchProvider
from split_source_groups import FILE_SECONDS, source_group
from pairs import PairBuilder
from preprocessing import Preprocessor
from protocols import write_gw_data_yaml, write_task_protocols
from qtransform import QTransformRenderer
from validation import run_all_validations
from waveform_generator import WaveformGenerator


def _code_commit() -> str:
    """Best-effort git commit of the generator code (D3 provenance)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        sha = out.stdout.strip()
        if out.returncode == 0 and sha:
            dirty = subprocess.run(
                ["git", "-C", str(Path(__file__).resolve().parent),
                 "status", "--porcelain"],
                capture_output=True, text=True, timeout=10,
            )
            if dirty.returncode == 0 and dirty.stdout.strip():
                return f"{sha}-dirty"
            return sha
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def class_to_signal_type(global_class: str) -> str:
    if global_class.startswith("bbh"):
        return "BBH"
    if global_class.startswith("bns"):
        return "BNS"
    return "None"


def _rel(path: Optional[Path], base: Path) -> str:
    if path is None:
        return ""
    try:
        return Path(path).relative_to(base).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _fmt(value) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _chirp_derived(params: Dict[str, object]) -> Tuple[float, float, float, float]:
    """Return (chirp_mass, total_mass, mass_ratio, chi_eff) from intrinsic params."""
    m1 = float(params["mass1"])
    m2 = float(params["mass2"])
    s1 = float(params.get("spin1z", 0.0) or 0.0)
    s2 = float(params.get("spin2z", 0.0) or 0.0)
    total = m1 + m2
    chirp_mass = (m1 * m2) ** 0.6 / total ** 0.2
    mass_ratio = min(m1, m2) / max(m1, m2)
    chi_eff = (m1 * s1 + m2 * s2) / total
    return chirp_mass, total, mass_ratio, chi_eff


def _bandlimit(signal: np.ndarray, sample_rate: int, flo: float, fhi: float) -> np.ndarray:
    """FFT band-limit a waveform to [flo, fhi] (low-pass when flo <= 0).

    Ensures the inserted chirp's energy stays inside the source type's common
    LIGO-band frequency range.
    """
    signal = np.asarray(signal, dtype=float)
    n = signal.shape[0]
    if n == 0:
        return signal
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    spec = np.fft.rfft(signal)
    mask = freqs <= fhi
    if flo > 0:
        mask &= freqs >= flo
    return np.fft.irfft(spec * mask, n=n)


class DatasetBuilder:
    def __init__(self, config: DatasetConfig, use_pycbc: bool = True):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.noise_generator = NoiseGenerator(config)
        self.real_glitch_provider = (
            RealGlitchProvider(config) if config.glitch_source == "gwosc" else None
        )
        self.waveform_generator = WaveformGenerator(config, use_pycbc=use_pycbc)
        self.preprocessor = Preprocessor(config)
        self.qtransform_renderer = QTransformRenderer(config)
        self.label_generator = LabelGenerator(
            config.duration,
            config.frange,
            config.frequency_axis_scale,
            drop_if_outside_window=config.drop_label_if_outside_window,
        )
        self.root = Path(config.output_dir)
        self._meta_rows: List[dict] = []
        self._sample_info: List[dict] = []
        self._match_rows: List[dict] = []
        self._negative_rows: List[dict] = []
        self._pair_rows: List[dict] = []

    # ------------------------------------------------------------------ build
    def build(self) -> Path:
        self._prepare_directories()
        self._write_dataset_config()
        split_assignment = self._event_split_assignment()
        if self.real_glitch_provider is not None:
            # Partition GWOSC 4096 s source files across splits BEFORE any
            # event is built; a source file never serves more than one split.
            self.real_glitch_provider.assign_split_groups(
                self.config.split_ratios(), self.config.seed
            )
            self._check_split_pool_coverage(split_assignment)

        meta_path = self.root / "metadata.csv"
        event_path = self.root / "event_metadata.csv"
        completed = False

        # Stream metadata + event rows per event (flushed), so an interrupted or
        # failed run still leaves usable CSVs for the events already generated.
        with meta_path.open("w", newline="", encoding="utf-8") as mh, \
                event_path.open("w", newline="", encoding="utf-8") as eh:
            meta_writer = csv.DictWriter(mh, fieldnames=METADATA_FIELDS)
            event_writer = csv.DictWriter(eh, fieldnames=EVENT_FIELDS)
            meta_writer.writeheader()
            event_writer.writeheader()
            try:
                for index, split in enumerate(split_assignment):
                    self._build_event(index, split, meta_writer, event_writer)
                    mh.flush()
                    eh.flush()
                completed = True
            finally:
                # Always write pairs + configs from whatever completed, so a
                # partial run is still a coherent (smaller) dataset.
                self._write_pairs_and_config()

        # Validate only a fully completed run.
        if completed:
            run_all_validations(
                self._meta_rows,
                self._match_rows,
                self._negative_rows,
                self._pair_rows,
                allow_cross_split=self.config.allow_cross_split_negative_pairs,
            )
        return meta_path

    def _write_pairs_and_config(self) -> None:
        # Pairs are built within each split (leakage-safe).
        pair_builder = PairBuilder(self.config, self.rng)
        match_rows, negative_rows, pair_rows = pair_builder.generate(self._sample_info)
        self._match_rows, self._negative_rows, self._pair_rows = (
            match_rows, negative_rows, pair_rows,
        )
        self._write_csv("match_pairs.csv", MATCH_FIELDS, match_rows)
        self._write_csv("negative_pairs.csv", NEGATIVE_FIELDS, negative_rows)
        self._write_csv("pair_metadata.csv", PAIR_FIELDS, pair_rows)
        write_task_protocols(self.config, self.root)
        write_gw_data_yaml(self.config, self.root)
        self._write_glitch_catalog_used()
        self._write_source_groups_manifest()

    def _check_split_pool_coverage(self, split_assignment: List[str]) -> None:
        """Fail fast when a populated split has no source groups to draw from."""
        provider = self.real_glitch_provider
        for split in sorted(set(split_assignment)):
            for detector in self.config.detectors:
                if not provider.pool_rows(detector, split):
                    raise GlitchFetchError(
                        f"split '{split}' has no 4096 s source group for "
                        f"{detector}: the glitch pool spans too few GWOSC "
                        "source files to isolate every split; enlarge the pool "
                        "(select_pool_subset) or adjust split ratios."
                    )

    def _write_source_groups_manifest(self) -> None:
        """D1 bookkeeping: group->split assignment plus per-split usage/reuse."""
        provider = self.real_glitch_provider
        if provider is None or not provider.split_groups:
            return
        per_split: Dict[str, dict] = {}
        for r in self._meta_rows:
            group = str(r.get("glitch_source_group", "") or "")
            if not group:
                continue
            info = per_split.setdefault(
                str(r["split"]),
                {"groups_used": set(), "glitch_samples": 0, "glitch_id_reuse": {}},
            )
            info["groups_used"].add(group)
            info["glitch_samples"] += 1
            gid = str(r["glitch_id"])
            info["glitch_id_reuse"][gid] = info["glitch_id_reuse"].get(gid, 0) + 1
        manifest = {
            "file_seconds": FILE_SECONDS,
            "seed": self.config.seed,
            "assignment": provider.split_groups,
            "per_split": {
                split: {
                    "groups_assigned": sorted(
                        g for g, s in provider.split_groups.items() if s == split
                    ),
                    "groups_used": sorted(info["groups_used"]),
                    "glitch_samples": info["glitch_samples"],
                    "unique_glitch_ids": len(info["glitch_id_reuse"]),
                    "glitch_id_reuse": info["glitch_id_reuse"],
                }
                for split, info in sorted(per_split.items())
            },
        }
        path = self.root / "source_groups.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _write_glitch_catalog_used(self) -> None:
        """One row per injected glitch (details of every glitch actually used)."""
        fields = [
            "sample_id", "split", "detector", "glitch_source", "glitch_id",
            "glitch_type", "glitch_gps", "glitch_snr_catalog", "glitch_amplitude",
            "glitch_start_time", "glitch_end_time", "glitch_low_freq", "glitch_high_freq",
            "glitch_source_group",
        ]
        rows = [
            {k: r.get(k, "") for k in fields}
            for r in self._meta_rows
            if str(r.get("has_glitch", "")) in ("1", "True", "true")
        ]
        self._write_csv("glitch_catalog_used.csv", fields, rows)

    # --------------------------------------------------------- real glitch
    def _acquire_real_glitch(self, rng, detector: str, split: str):
        """Sample a real glitch segment that shows a localized energy ridge.

        The box is measured on the glitch-only render (the real segment, no
        chirp), so a chirp sharing the sample can never leak into the glitch
        label. Returns ``(glitch, trial_box, used_synthetic_fallback)``. Retries
        up to ``cfg.max_glitch_attempts`` when the fetch fails, no ridge is
        found, or the box covers more than ``cfg.glitch_max_box_frac`` of the
        image; on exhaustion (or a gwpy error) falls back to a synthetic glitch
        if ``glitch_allow_synthetic_fallback`` is set, otherwise raises
        :class:`GlitchFetchError`.
        """
        cfg = self.config
        last_exc: Optional[Exception] = None
        for _ in range(max(1, cfg.max_glitch_attempts)):
            try:
                glitch = self.real_glitch_provider.sample_glitch(
                    rng, detector, split=split
                )
            except GlitchDependencyError as exc:
                last_exc = exc
                break  # environment error (e.g. gwpy missing) won't fix on retry
            except GlitchFetchError as exc:
                # Unavailable GPS or transient fetch trouble; try another glitch.
                last_exc = exc
                continue
            trial = self.preprocessor.preprocess(glitch.series)
            stats = self.qtransform_renderer.energy_stats(trial)
            box = self.label_generator.glitch_box_from_ridge(
                stats.energy, stats.freqs, stats.times,
                (glitch.start_time, glitch.end_time), cfg.label_ridge_threshold,
            )
            if box is not None and (
                self.label_generator.box_area_fraction(box) <= cfg.glitch_max_box_frac
            ):
                return glitch, box, False
        if cfg.glitch_allow_synthetic_fallback:
            return self.noise_generator.sample_glitch(rng), None, True
        msg = (
            f"could not obtain a real glitch with a detectable ridge for {detector} "
            f"after {cfg.max_glitch_attempts} attempts"
        )
        if last_exc is not None:
            raise GlitchFetchError(f"{msg}: {last_exc}") from last_exc
        raise GlitchFetchError(msg)

    # ----------------------------------------------- real off-source background
    def _acquire_real_background(self, rng, detector: str, split: str):
        """Draw a real off-source background whose render shows NO significant
        transient (inverted ridge veto). Rejection-samples up to
        ``max_background_attempts``; on exhaustion raises -- never a silent
        synthetic fallback (D2)."""
        cfg = self.config
        last_exc: Optional[Exception] = None
        for _ in range(max(1, cfg.max_background_attempts)):
            try:
                bg = self.real_glitch_provider.sample_background(
                    rng, detector, split=split
                )
            except GlitchDependencyError:
                raise
            except GlitchFetchError as exc:
                last_exc = exc
                continue
            trial = self.preprocessor.preprocess(bg.series)
            stats = self.qtransform_renderer.energy_stats(trial)
            box = self.label_generator.glitch_box_from_ridge(
                stats.energy, stats.freqs, stats.times,
                (0.0, cfg.duration), cfg.label_ridge_threshold,
                floor_gate=cfg.background_veto_floor_gate,
            )
            if box is None:
                return bg
        msg = (
            f"could not obtain a clean off-source background for {detector} in "
            f"split '{split}' after {cfg.max_background_attempts} attempts"
        )
        if last_exc is not None:
            raise GlitchFetchError(f"{msg}: {last_exc}") from last_exc
        raise GlitchFetchError(msg)

    # ----------------------------------------------------------- event split
    def _event_split_assignment(self) -> List[str]:
        ratios = self.config.split_ratios()
        raw = {name: self.config.num_events * r for name, r in ratios.items()}
        counts = {name: math.floor(v) for name, v in raw.items()}
        remainder = self.config.num_events - sum(counts.values())
        order = sorted(raw, key=lambda n: (raw[n] - counts[n], ratios[n]), reverse=True)
        for name in order[:remainder]:
            counts[name] += 1
        seq = (
            ["train"] * counts["train"]
            + ["val"] * counts["val"]
            + ["test"] * counts["test"]
        )
        self.rng.shuffle(seq)
        return seq

    # ------------------------------------------------------------ one event
    def _build_event(self, index: int, split: str, meta_writer, event_writer) -> None:
        cfg = self.config
        rng = self.rng

        event_id = f"evt_{index:06d}"
        global_class = str(rng.choice(GLOBAL_CLASSES))
        signal_type = class_to_signal_type(global_class)
        has_chirp = signal_type in {"BBH", "BNS"}
        has_glitch = "glitch" in global_class
        chirp_id = f"chirp_{index:06d}" if has_chirp else ""

        params: Dict[str, object] = {}
        inject_waveform = None
        snr_bin = ""
        base_target_snr = None
        injection_time_value = None
        sig_duration = 0.0
        waveform_source = ""
        chirp_mass = total_mass = mass_ratio = chi_eff = distance = None
        if has_chirp:
            params = self.waveform_generator.sample_intrinsic(signal_type, rng=rng)
            waveform = self.waveform_generator.generate(signal_type, params)
            waveform_source = str(waveform.params.get("waveform_source", ""))
            chirp_mass, total_mass, mass_ratio, chi_eff = _chirp_derived(params)
            distance = params.get("distance")
            # Constrain the inserted chirp to this source type's common band.
            band = cfg.signal_freq_bands.get(signal_type, (cfg.frange_low, cfg.frange_high))
            inject_waveform = _bandlimit(
                waveform.series, cfg.sample_rate, float(band[0]), float(band[1])
            )
            snr_bin, base_target_snr = self.waveform_generator.sample_snr_bin(signal_type, rng=rng)
            injection_time_value = float(
                rng.uniform(cfg.injection_time_min, cfg.injection_time_max)
            )
            sig_duration = len(inject_waveform) / cfg.sample_rate

        rows: List[dict] = []
        detector_snrs: List[float] = []

        for detector in cfg.detectors:
            sample_id = f"{split}_{index:06d}_{detector}"
            bg = self.noise_generator.background_noise(rng)
            series = bg.series.copy()
            noise_id, noise_type = bg.noise_id, bg.noise_type
            background_source = "synthetic"
            background_group = background_gps = ""
            noise_reference = bg.series
            if cfg.noise_source == "gwosc" and not (
                has_glitch and self.real_glitch_provider is not None
            ):
                # D2: non-glitch samples sit on real off-source backgrounds
                # from the SAME split source-group pool as the glitches.
                # (Real-glitch samples get their background from the glitch
                # segment itself below.)
                real_bg = self._acquire_real_background(rng, detector, split)
                series = real_bg.series.copy()
                noise_id, noise_type = real_bg.noise_id, real_bg.noise_type
                background_source = "gwosc"
                background_group = real_bg.group
                background_gps = real_bg.gps
                noise_reference = real_bg.series

            boxes: List[TimeFrequencyBox] = []
            glitch_id = glitch_type = ""
            glitch_start = glitch_end = glitch_cf = glitch_lf = glitch_hf = glitch_amp = ""
            glitch_source = glitch_gps = glitch_snr_cat = ""
            glitch_group = ""
            if has_glitch:
                glitch_trial_box = None
                if self.real_glitch_provider is not None:
                    glitch, glitch_trial_box, _used_synth = self._acquire_real_glitch(
                        rng, detector, split
                    )
                else:
                    glitch = self.noise_generator.sample_glitch(rng)
                if glitch_trial_box is not None:
                    # Real segment: glitch in its own real noise REPLACES the
                    # synthetic background (adding it would stack two noise
                    # floors into a broadband pedestal across the window).
                    series = glitch.series.copy()
                    noise_id, noise_type = glitch.glitch_id, "real_gwosc"
                    glitch_group = source_group(detector, glitch.gps)
                    background_source = "gwosc"
                    background_group = glitch_group
                    background_gps = glitch.gps
                else:
                    series = series + glitch.series
                glitch_id, glitch_type = glitch.glitch_id, glitch.glitch_type
                glitch_start, glitch_end = glitch.start_time, glitch.end_time
                glitch_cf, glitch_lf, glitch_hf = (
                    glitch.center_freq, glitch.low_freq, glitch.high_freq,
                )
                glitch_amp = glitch.amplitude
                glitch_source = getattr(glitch, "source", "synthetic")
                glitch_gps = getattr(glitch, "gps", "")
                glitch_snr_cat = getattr(glitch, "snr_catalog", "")
                if glitch_trial_box is not None:
                    # Real glitch: the box measured on the glitch-only render is
                    # authoritative (a box from the final combined render could
                    # absorb a chirp crossing the glitch window).
                    boxes.append(glitch_trial_box)
                    glitch_start, glitch_end = (
                        glitch_trial_box.time_start, glitch_trial_box.time_end,
                    )
                    glitch_lf, glitch_hf = (
                        glitch_trial_box.freq_low, glitch_trial_box.freq_high,
                    )
                    # Geometric mean = the box center on the log-frequency axis.
                    glitch_cf = float(np.sqrt(glitch_lf * glitch_hf))
                else:
                    # Synthetic (or synthetic fallback): analytic box.
                    gbox = self.label_generator.glitch_box(
                        glitch.start_time, glitch.end_time, glitch.low_freq, glitch.high_freq
                    )
                    if gbox is not None:
                        boxes.append(gbox)

            # ---- per-detector chirp injection
            chirp_box = None
            track = None
            target_snr = detector_snr = ""
            time_delay = amp_scale = ""
            sign_flip = False
            injection_time_out = ""
            chirp_low = chirp_high = ""
            if has_chirp:
                td = float(rng.uniform(-cfg.detector_time_delay_max, cfg.detector_time_delay_max))
                amp = 1.0 + float(rng.uniform(-cfg.detector_amp_scale_jitter, cfg.detector_amp_scale_jitter))
                sign_flip = bool(rng.integers(0, 2)) if cfg.enable_sign_flip else False
                det_target = float(base_target_snr) * (
                    1.0 + float(rng.uniform(-cfg.detector_snr_jitter, cfg.detector_snr_jitter))
                )
                injected = inject_chirp(
                    series,
                    inject_waveform,
                    injection_time_value,
                    det_target,
                    cfg.sample_rate,
                    time_delay=td,
                    amp_scale=amp,
                    sign_flip=sign_flip,
                    # SNR scaling must see the glitch-free noise that actually
                    # underlies (or stands in for) this sample's background: a
                    # loud glitch in ``series`` inflates the std and the chirp
                    # comes out louder than the requested target SNR.
                    noise_reference=noise_reference,
                )
                raw = injected.combined
                target_snr = det_target
                detector_snr = float(injected.snr)
                detector_snrs.append(detector_snr)
                time_delay = float(injected.time_delay)
                amp_scale = float(injected.amp_scale)
                injection_time_out = float(injected.injection_time)

                track = self.label_generator.measure_chirp_track(
                    injected.clean_signal, cfg.sample_rate, cfg.label_envelope_threshold
                )
                if track is not None:
                    chirp_low, chirp_high = track["f_low"], track["f_high"]
                    chirp_box = self.label_generator._finalize_chirp_box(
                        track["t_start"], track["t_end"], track["f_low"], track["f_high"],
                        source=track["source"],
                    )
            else:
                raw = series

            # ---- output paths
            def out(sub: str, ext: str) -> Path:
                return self.root / sub / split / detector / f"{sample_id}.{ext}"

            raw_npy = out("raw_series", "npy")
            norm_npy = out("normalized_series", "npy")
            qt_raw = out("qtransform_raw", "png")
            qt_norm = out("qtransform_normalized", "png")
            qt_raw_disp = out("qtransform_display_raw", "png")
            qt_norm_disp = out("qtransform_display_normalized", "png")
            label_path = out("labels_yolo", "txt")

            raw_rel = ""
            if cfg.save_raw_outputs:
                raw_npy.parent.mkdir(parents=True, exist_ok=True)
                np.save(raw_npy, raw.astype(np.float32))
                raw_rel = _rel(raw_npy, self.root)

            normalized = self.preprocessor.preprocess(raw)
            norm_npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(norm_npy, normalized.astype(np.float32))

            norm_stats = self.qtransform_renderer.render(
                normalized, qt_norm, qt_norm_disp if cfg.save_display_images else None
            )
            qt_raw_rel = qt_raw_disp_rel = ""
            if cfg.save_raw_outputs:
                self.qtransform_renderer.render(
                    raw, qt_raw, qt_raw_disp if cfg.save_display_images else None
                )
                qt_raw_rel = _rel(qt_raw, self.root)
                if cfg.save_display_images:
                    qt_raw_disp_rel = _rel(qt_raw_disp, self.root)

            # ---- ridge fallback if the waveform path produced no usable box
            label_source = ""
            if has_chirp and chirp_box is None:
                merger = injection_time_out or 0.0
                window = (
                    max(0.0, merger - sig_duration - 0.2),
                    min(cfg.duration, merger + 0.2),
                )
                chirp_box = self.label_generator.chirp_box_from_ridge(
                    norm_stats.energy, norm_stats.freqs, norm_stats.times,
                    window, cfg.label_ridge_threshold,
                )
            if chirp_box is not None:
                boxes.append(chirp_box)
                label_source = chirp_box.source
            elif track is not None:
                label_source = "outside_window"

            written_label = None
            if cfg.generate_yolo_labels:
                written_label = self.label_generator.write_label_file(
                    label_path, boxes, write_empty=cfg.write_empty_labels
                )

            label_low = chirp_box.freq_low if chirp_box is not None else ""
            label_high = chirp_box.freq_high if chirp_box is not None else ""

            # YOLO chirp box in normalized image coordinates.
            yolo_cx = yolo_cy = yolo_w = yolo_h = ""
            if chirp_box is not None:
                parts = self.label_generator.to_yolo(chirp_box).split()
                yolo_cx, yolo_cy, yolo_w, yolo_h = parts[1], parts[2], parts[3], parts[4]
            num_boxes = len(boxes)
            has_label = int(num_boxes > 0)

            row = {
                "sample_id": sample_id,
                "event_id": event_id,
                "chirp_id": chirp_id,
                "split": split,
                "detector": detector,
                "counterpart_sample_ids": "",  # filled after all detectors
                "global_class": global_class,
                "signal_type": signal_type,
                "has_chirp": int(has_chirp),
                "has_glitch": int(has_glitch),
                "target_snr": _fmt(target_snr) if has_chirp else "",
                "actual_snr": _fmt(detector_snr) if has_chirp else "",
                "snr_bin": snr_bin if has_chirp else "",
                "detector_snr": _fmt(detector_snr) if has_chirp else "",
                "network_snr": "",  # filled after all detectors
                "injection_time": _fmt(injection_time_out) if has_chirp else "",
                "detector_time_delay": _fmt(time_delay) if has_chirp else "",
                "amplitude_scale": _fmt(amp_scale) if has_chirp else "",
                "phase_or_sign_flip": int(sign_flip) if has_chirp else "",
                "chirp_start_time": _fmt(track["t_start"]) if track is not None else "",
                "chirp_end_time": _fmt(track["t_end"]) if track is not None else "",
                "chirp_freq_low": _fmt(chirp_low) if has_chirp else "",
                "chirp_freq_high": _fmt(chirp_high) if has_chirp else "",
                "qtransform_frange_low": _fmt(cfg.frange_low),
                "qtransform_frange_high": _fmt(cfg.frange_high),
                "label_frange_low": _fmt(label_low) if chirp_box is not None else "",
                "label_frange_high": _fmt(label_high) if chirp_box is not None else "",
                "frequency_axis_scale": cfg.frequency_axis_scale,
                "raw_strain_path": raw_rel,
                "normalized_strain_path": _rel(norm_npy, self.root),
                "qtransform_raw_path": qt_raw_rel,
                "qtransform_normalized_path": _rel(qt_norm, self.root),
                "qtransform_display_raw_path": qt_raw_disp_rel,
                "qtransform_display_normalized_path": (
                    _rel(qt_norm_disp, self.root) if cfg.save_display_images else ""
                ),
                "yolo_label_path": _rel(written_label, self.root) if written_label else "",
                "label_source": label_source,
                "energy_norm_method": norm_stats.method,
                "energy_vmin": _fmt(norm_stats.vmin),
                "energy_vmax": _fmt(norm_stats.vmax),
                "energy_peak": _fmt(norm_stats.peak),
                "energy_percentile_used": _fmt(norm_stats.percentile_used)
                if norm_stats.percentile_used != "" else "",
                "noise_id": noise_id,
                "noise_type": noise_type,
                "glitch_id": glitch_id,
                "glitch_type": glitch_type,
                "waveform_source": waveform_source if has_chirp else "",
                "waveform_approximant": params.get("approximant", "") if has_chirp else "",
                "mass1": _fmt(params.get("mass1")) if has_chirp else "",
                "mass2": _fmt(params.get("mass2")) if has_chirp else "",
                "spin1z": _fmt(params.get("spin1z")) if has_chirp else "",
                "spin2z": _fmt(params.get("spin2z")) if has_chirp else "",
                "chirp_mass": _fmt(chirp_mass) if has_chirp else "",
                "total_mass": _fmt(total_mass) if has_chirp else "",
                "mass_ratio": _fmt(mass_ratio) if has_chirp else "",
                "chi_eff": _fmt(chi_eff) if has_chirp else "",
                "distance": _fmt(distance) if has_chirp else "",
                "f_lower": _fmt(params.get("f_lower")) if has_chirp else "",
                "geocent_injection_time": _fmt(injection_time_value) if has_chirp else "",
                "chirp_yolo_cx": yolo_cx,
                "chirp_yolo_cy": yolo_cy,
                "chirp_yolo_w": yolo_w,
                "chirp_yolo_h": yolo_h,
                "has_label": has_label,
                "num_boxes": num_boxes,
                "glitch_start_time": _fmt(glitch_start) if has_glitch else "",
                "glitch_end_time": _fmt(glitch_end) if has_glitch else "",
                "glitch_center_freq": _fmt(glitch_cf) if has_glitch else "",
                "glitch_low_freq": _fmt(glitch_lf) if has_glitch else "",
                "glitch_high_freq": _fmt(glitch_hf) if has_glitch else "",
                "glitch_amplitude": _fmt(glitch_amp) if has_glitch else "",
                "glitch_source": glitch_source if has_glitch else "",
                "glitch_gps": _fmt(glitch_gps) if (has_glitch and glitch_gps != "") else "",
                "glitch_snr_catalog": (
                    _fmt(glitch_snr_cat) if (has_glitch and glitch_snr_cat != "") else ""
                ),
                "glitch_source_group": glitch_group,
                "background_source": background_source,
                "background_source_group": background_group,
                "background_gps": _fmt(background_gps) if background_gps != "" else "",
                "sample_rate": cfg.sample_rate,
                "duration": _fmt(cfg.duration),
                "n_samples": cfg.n_samples,
            }
            rows.append(row)
            self._sample_info.append(
                {
                    "sample_id": sample_id,
                    "event_id": event_id,
                    "chirp_id": chirp_id,
                    "split": split,
                    "detector": detector,
                    "has_chirp": has_chirp,
                    "has_glitch": has_glitch,
                    "global_class": global_class,
                    "snr_bin": snr_bin if has_chirp else "",
                    "injection_time": injection_time_out if has_chirp else None,
                    "label_low": float(label_low) if label_low != "" else None,
                    "label_high": float(label_high) if label_high != "" else None,
                }
            )

        # ---- cross-detector fields, then stream rows to disk
        network_snr = math.sqrt(sum(s * s for s in detector_snrs)) if detector_snrs else ""
        for row in rows:
            others = [r["sample_id"] for r in rows if r["sample_id"] != row["sample_id"]]
            row["counterpart_sample_ids"] = ";".join(others)
            row["network_snr"] = _fmt(network_snr) if network_snr != "" else ""
            self._meta_rows.append(row)
            meta_writer.writerow({k: row.get(k, "") for k in METADATA_FIELDS})

        event_row = {
            "event_id": event_id,
            "chirp_id": chirp_id,
            "split": split,
            "signal_type": signal_type,
            "has_chirp": int(has_chirp),
            "waveform_approximant": params.get("approximant", "") if has_chirp else "",
            "mass1": _fmt(params.get("mass1")) if has_chirp else "",
            "mass2": _fmt(params.get("mass2")) if has_chirp else "",
            "spin1z": _fmt(params.get("spin1z")) if has_chirp else "",
            "spin2z": _fmt(params.get("spin2z")) if has_chirp else "",
            "f_lower": _fmt(params.get("f_lower")) if has_chirp else "",
            "sample_rate": cfg.sample_rate,
            "duration": _fmt(cfg.duration),
            "injection_time": _fmt(injection_time_value) if has_chirp else "",
            "network_snr": _fmt(network_snr) if network_snr != "" else "",
            "detectors": ";".join(cfg.detectors),
            "num_detector_samples": len(rows),
            "qtransform_frange_low": _fmt(cfg.frange_low),
            "qtransform_frange_high": _fmt(cfg.frange_high),
            "frequency_axis_scale": cfg.frequency_axis_scale,
            "distance": _fmt(distance) if has_chirp else "",
            "chirp_mass": _fmt(chirp_mass) if has_chirp else "",
            "total_mass": _fmt(total_mass) if has_chirp else "",
            "mass_ratio": _fmt(mass_ratio) if has_chirp else "",
            "chi_eff": _fmt(chi_eff) if has_chirp else "",
            "waveform_source": waveform_source if has_chirp else "",
            "n_samples": cfg.n_samples,
        }
        event_writer.writerow({k: event_row.get(k, "") for k in EVENT_FIELDS})

    # --------------------------------------------------------------- helpers
    def _prepare_directories(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in OUTPUT_SUBDIRS:
            for split in ("train", "val", "test"):
                for det in self.config.detectors:
                    (self.root / sub / split / det).mkdir(parents=True, exist_ok=True)

    def _write_dataset_config(self) -> None:
        """Dump the full DatasetConfig for reproducibility/provenance.

        D3: also pins the glitch-pool contents (SHA256) and the generator
        code commit, so a dataset can always be matched to exactly what
        built it.
        """
        data = asdict(self.config)
        data["output_dir"] = str(self.config.output_dir)
        if self.config.glitch_metadata_csv is not None:
            pool = Path(self.config.glitch_metadata_csv)
            if pool.exists():
                data["pool_sha256"] = hashlib.sha256(pool.read_bytes()).hexdigest()
        data["code_commit"] = _code_commit()
        (self.root / "dataset_config.json").write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )

    def _write_csv(self, name: str, fields, rows) -> None:
        path = self.root / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fields})


# --------------------------------------------------------------------------- CLI
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a two-detector synthetic GW dataset.")
    p.add_argument("--num-events", type=int, default=50)
    p.add_argument("--detectors", nargs="+", default=["H1", "L1"])
    p.add_argument("--duration", type=float, default=4.0)
    p.add_argument("--sample-rate", type=int, default=4096)
    # The frequency window is hardcoded to a fixed log 20-1000 Hz coordinate
    # system. These flags are accepted for backward compatibility but IGNORED.
    p.add_argument("--frange-low", type=float, default=None,
                   help="(ignored) frequency window is fixed at 20-1000 Hz")
    p.add_argument("--frange-high", type=float, default=None,
                   help="(ignored) frequency window is fixed at 20-1000 Hz")
    p.add_argument("--frequency-axis-scale", choices=["log", "linear"], default=None,
                   help="(ignored) frequency axis is fixed to log")
    p.add_argument("--output-dir", default="gw_synthetic_dataset")
    p.add_argument("--qtransform-backend", choices=["gwpy", "scipy"], default="gwpy",
                   help="gwpy = LIGO Omega Q-scan (default); scipy = built-in constant-Q")
    p.add_argument("--image-width", type=int, default=640)
    p.add_argument("--image-height", type=int, default=640)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--injection-time-min", type=float, default=None)
    p.add_argument("--injection-time-max", type=float, default=None)
    p.add_argument("--negative-pairs-per-positive", type=int, default=3)
    p.add_argument("--hard-negative-fraction", type=float, default=0.5)
    p.add_argument("--enable-invalid-delay-negatives", dest="enable_invalid_delay",
                   action="store_true", default=True)
    p.add_argument("--disable-invalid-delay-negatives", dest="enable_invalid_delay",
                   action="store_false")
    p.add_argument("--enable-chirp-vs-glitch-negatives", dest="enable_chirp_vs_glitch",
                   action="store_true", default=True)
    p.add_argument("--disable-chirp-vs-glitch-negatives", dest="enable_chirp_vs_glitch",
                   action="store_false")
    p.add_argument("--enable-chirp-vs-noise-negatives", dest="enable_chirp_vs_noise",
                   action="store_true", default=True)
    p.add_argument("--disable-chirp-vs-noise-negatives", dest="enable_chirp_vs_noise",
                   action="store_false")
    p.add_argument("--allow-cross-split-negative-pairs", action="store_true", default=False)
    p.add_argument("--no-raw-outputs", dest="save_raw_outputs", action="store_false", default=True)
    p.add_argument("--no-display-images", dest="save_display_images", action="store_false", default=True)
    p.add_argument("--no-pycbc", dest="use_pycbc", action="store_false", default=True,
                   help="Force the analytic waveform fallback (skip PyCBC).")
    # --- real (GWOSC) glitch injection
    p.add_argument("--glitch-source", choices=["synthetic", "gwosc"], default="synthetic",
                   help="synthetic sine-Gaussian (default) or real GWOSC glitches.")
    p.add_argument("--glitch-metadata-csv", default=None,
                   help="Gravity Spy pool CSV (gps,ifo,label,...); required for gwosc.")
    p.add_argument("--real-glitch-cache-dir", default=None,
                   help="Cache dir for fetched strain (default: <output-dir>/glitch_cache).")
    p.add_argument("--glitch-fetch-halfwin", type=float, default=None,
                   help="Seconds of strain fetched on each side of the glitch GPS.")
    p.add_argument("--glitch-default-duration", type=float, default=None,
                   help="Crop length (s) when a pool row lacks a duration.")
    p.add_argument("--no-glitch-whiten", dest="glitch_whiten", action="store_false", default=True,
                   help="Skip whitening the fetched real glitch.")
    p.add_argument("--glitch-amp-min", type=float, default=None,
                   help="Min amplitude scale for the unit-std glitch.")
    p.add_argument("--glitch-amp-max", type=float, default=None,
                   help="Max amplitude scale for the unit-std glitch.")
    p.add_argument("--max-glitch-attempts", type=int, default=None,
                   help="Resample a real glitch up to N times if no ridge is found.")
    p.add_argument("--glitch-max-box-frac", type=float, default=None,
                   help="Reject real-glitch boxes covering more than this image fraction.")
    # --- real (GWOSC) off-source backgrounds (G1/D2)
    p.add_argument("--noise-source", choices=["synthetic", "gwosc"], default="synthetic",
                   help="synthetic colored Gaussian (default) or real GWOSC "
                        "off-source segments for non-glitch samples.")
    p.add_argument("--max-background-attempts", type=int, default=None,
                   help="Reject/resample off-source candidates up to N times, "
                        "then fail explicitly (no synthetic fallback).")
    p.add_argument("--background-glitch-exclusion", type=float, default=None,
                   help="Exclude off-source GPS within this many seconds of a "
                        "known pool glitch.")
    p.add_argument("--background-veto-floor-gate", type=float, default=None,
                   help="Reject off-source candidates whose render has pixels "
                        "above this multiple of the median energy.")
    p.add_argument("--glitch-allow-synthetic-fallback", action="store_true", default=False,
                   help="Fall back to synthetic when GWOSC/gwpy fails or no ridge is found.")
    return p.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> DatasetConfig:
    kwargs = dict(
        num_events=args.num_events,
        detectors=tuple(args.detectors),
        duration=args.duration,
        sample_rate=args.sample_rate,
        # frange_* / frequency_axis_scale intentionally NOT passed: the window
        # is hardcoded to linear 0-1000 Hz inside DatasetConfig.
        output_dir=args.output_dir,
        qtransform_backend=args.qtransform_backend,
        qtransform_image_width=args.image_width,
        qtransform_image_height=args.image_height,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        negative_pairs_per_positive=args.negative_pairs_per_positive,
        hard_negative_fraction=args.hard_negative_fraction,
        enable_invalid_delay_negatives=args.enable_invalid_delay,
        enable_chirp_vs_glitch_negatives=args.enable_chirp_vs_glitch,
        enable_chirp_vs_noise_negatives=args.enable_chirp_vs_noise,
        allow_cross_split_negative_pairs=args.allow_cross_split_negative_pairs,
        save_raw_outputs=args.save_raw_outputs,
        save_display_images=args.save_display_images,
        glitch_source=args.glitch_source,
        glitch_whiten=args.glitch_whiten,
        glitch_allow_synthetic_fallback=args.glitch_allow_synthetic_fallback,
        noise_source=args.noise_source,
    )
    if args.max_background_attempts is not None:
        kwargs["max_background_attempts"] = args.max_background_attempts
    if args.background_glitch_exclusion is not None:
        kwargs["background_glitch_exclusion"] = args.background_glitch_exclusion
    if args.background_veto_floor_gate is not None:
        kwargs["background_veto_floor_gate"] = args.background_veto_floor_gate
    if args.injection_time_min is not None:
        kwargs["injection_time_min"] = args.injection_time_min
    if args.injection_time_max is not None:
        kwargs["injection_time_max"] = args.injection_time_max
    if args.glitch_metadata_csv is not None:
        kwargs["glitch_metadata_csv"] = args.glitch_metadata_csv
    if args.real_glitch_cache_dir is not None:
        kwargs["real_glitch_cache_dir"] = args.real_glitch_cache_dir
    if args.glitch_fetch_halfwin is not None:
        kwargs["glitch_fetch_halfwin"] = args.glitch_fetch_halfwin
    if args.glitch_default_duration is not None:
        kwargs["glitch_default_duration"] = args.glitch_default_duration
    if args.max_glitch_attempts is not None:
        kwargs["max_glitch_attempts"] = args.max_glitch_attempts
    if args.glitch_max_box_frac is not None:
        kwargs["glitch_max_box_frac"] = args.glitch_max_box_frac
    if args.glitch_amp_min is not None or args.glitch_amp_max is not None:
        lo, hi = DatasetConfig().glitch_amplitude_range
        kwargs["glitch_amplitude_range"] = (
            args.glitch_amp_min if args.glitch_amp_min is not None else lo,
            args.glitch_amp_max if args.glitch_amp_max is not None else hi,
        )
    return DatasetConfig(**kwargs)


def main(argv=None) -> None:
    args = parse_args(argv)
    if any(v is not None for v in (args.frange_low, args.frange_high, args.frequency_axis_scale)):
        print("[notice] --frange-low/--frange-high/--frequency-axis-scale are ignored; "
              "the frequency window is fixed at log 20-1000 Hz.")
    config = config_from_args(args)
    builder = DatasetBuilder(config, use_pycbc=args.use_pycbc)
    metadata_path = builder.build()
    print(f"Dataset written to: {config.output_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"Events: {config.num_events}  Detectors: {', '.join(config.detectors)}")
    print(f"Frequency window: {config.frange_low:g}-{config.frange_high:g} Hz "
          f"({config.frequency_axis_scale}, fixed)")
    print("Per-type insertion bands (Hz): "
          + ", ".join(f"{k}={v[0]:g}-{v[1]:g}" for k, v in config.signal_freq_bands.items()))


if __name__ == "__main__":
    main()
