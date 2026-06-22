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
import math
from pathlib import Path
from typing import Dict, List, Optional

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
from pairs import PairBuilder
from preprocessing import Preprocessor
from protocols import write_gw_data_yaml, write_task_protocols
from qtransform import QTransformRenderer
from validation import run_all_validations
from waveform_generator import WaveformGenerator


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
        self._event_rows: List[dict] = []
        self._sample_info: List[dict] = []

    # ------------------------------------------------------------------ build
    def build(self) -> Path:
        self._prepare_directories()
        split_assignment = self._event_split_assignment()

        for index, split in enumerate(split_assignment):
            self._build_event(index, split)

        # Pairs are built within each split (leakage-safe).
        pair_builder = PairBuilder(self.config, self.rng)
        match_rows, negative_rows, pair_rows = pair_builder.generate(self._sample_info)

        self._write_csv("metadata.csv", METADATA_FIELDS, self._meta_rows)
        self._write_csv("event_metadata.csv", EVENT_FIELDS, self._event_rows)
        self._write_csv("match_pairs.csv", MATCH_FIELDS, match_rows)
        self._write_csv("negative_pairs.csv", NEGATIVE_FIELDS, negative_rows)
        self._write_csv("pair_metadata.csv", PAIR_FIELDS, pair_rows)

        write_task_protocols(self.config, self.root)
        write_gw_data_yaml(self.config, self.root)

        # Validate before reporting success.
        run_all_validations(
            self._meta_rows,
            match_rows,
            negative_rows,
            pair_rows,
            allow_cross_split=self.config.allow_cross_split_negative_pairs,
        )
        return self.root / "metadata.csv"

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
    def _build_event(self, index: int, split: str) -> None:
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
        if has_chirp:
            params = self.waveform_generator.sample_intrinsic(signal_type, rng=rng)
            waveform = self.waveform_generator.generate(signal_type, params)
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
            noise_reference = bg.series.copy()  # signal-free ASD estimate
            noise_id, noise_type = bg.noise_id, bg.noise_type

            boxes: List[TimeFrequencyBox] = []
            glitch_id = glitch_type = ""
            if has_glitch:
                glitch = self.noise_generator.sample_glitch(rng)
                series = series + glitch.series
                glitch_id, glitch_type = glitch.glitch_id, glitch.glitch_type
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

            normalized = self.preprocessor.preprocess(raw, noise_reference=noise_reference)
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

        # ---- cross-detector fields
        network_snr = math.sqrt(sum(s * s for s in detector_snrs)) if detector_snrs else ""
        for row in rows:
            others = [r["sample_id"] for r in rows if r["sample_id"] != row["sample_id"]]
            row["counterpart_sample_ids"] = ";".join(others)
            row["network_snr"] = _fmt(network_snr) if network_snr != "" else ""
            self._meta_rows.append(row)

        self._event_rows.append(
            {
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
            }
        )

    # --------------------------------------------------------------- helpers
    def _prepare_directories(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in OUTPUT_SUBDIRS:
            for split in ("train", "val", "test"):
                for det in self.config.detectors:
                    (self.root / sub / split / det).mkdir(parents=True, exist_ok=True)

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
    # The frequency window is hardcoded to a fixed linear 0-1000 Hz coordinate
    # system. These flags are accepted for backward compatibility but IGNORED.
    p.add_argument("--frange-low", type=float, default=None,
                   help="(ignored) frequency window is fixed at 0-1000 Hz")
    p.add_argument("--frange-high", type=float, default=None,
                   help="(ignored) frequency window is fixed at 0-1000 Hz")
    p.add_argument("--frequency-axis-scale", choices=["log", "linear"], default=None,
                   help="(ignored) frequency axis is fixed to linear")
    p.add_argument("--output-dir", default="gw_synthetic_dataset")
    p.add_argument("--qtransform-backend", choices=["scipy", "gwpy"], default="scipy")
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
    )
    if args.injection_time_min is not None:
        kwargs["injection_time_min"] = args.injection_time_min
    if args.injection_time_max is not None:
        kwargs["injection_time_max"] = args.injection_time_max
    return DatasetConfig(**kwargs)


def main(argv=None) -> None:
    args = parse_args(argv)
    if any(v is not None for v in (args.frange_low, args.frange_high, args.frequency_axis_scale)):
        print("[notice] --frange-low/--frange-high/--frequency-axis-scale are ignored; "
              "the frequency window is fixed at linear 0-1000 Hz.")
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
