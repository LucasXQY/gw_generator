"""QA: sample real cached glitches through the full injection + render path
and report box tightness and the broadband noise pedestal.

Run inside the env that has gwpy (the build env). Uses only the local cache --
GPS times not in the cache raise, so run prefetch first.

    python validate_real_glitch.py --pool gravityspy_pool_3000.csv \
        --output-dir datasets/gw_dataset_3000_real --n 12

Audit mode (G1): run the cross-split leakage + background-domain audits on a
built dataset's metadata.csv; exits 1 when any audit fails.

    python validate_real_glitch.py --audit --output-dir datasets/<name>
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from config import DatasetConfig
from label_generator import LabelGenerator
from preprocessing import Preprocessor
from qtransform import QTransformRenderer
from real_glitch import GlitchFetchError, RealGlitchProvider
from validation import (
    validate_background_domain_decoupled,
    validate_no_background_group_leakage,
    validate_no_chirp_leakage,
    validate_no_event_leakage,
    validate_no_glitch_leakage,
    validate_no_source_group_leakage,
)

_AUDITS = (
    ("event_id leakage", validate_no_event_leakage),
    ("chirp_id leakage", validate_no_chirp_leakage),
    ("glitch_id leakage", validate_no_glitch_leakage),
    ("glitch source_group leakage", validate_no_source_group_leakage),
    ("background domain collinearity", validate_background_domain_decoupled),
    ("background source_group leakage", validate_no_background_group_leakage),
)


def _run_audit(output_dir: Path) -> None:
    meta_path = Path(output_dir) / "metadata.csv"
    if not meta_path.exists():
        print(f"AUDIT ERROR: {meta_path} not found")
        sys.exit(1)
    with meta_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"auditing {meta_path} ({len(rows)} rows)")
    failed = False
    for name, check in _AUDITS:
        try:
            check(rows)
        except AssertionError as exc:
            print(f"FAIL {name}: {exc}")
            failed = True
        else:
            print(f"PASS {name}")
    if failed:
        sys.exit(1)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default=None)
    p.add_argument("--output-dir", required=True,
                   help="Build output dir (its glitch_cache/ is used).")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--audit", action="store_true",
                   help="Audit <output-dir>/metadata.csv for cross-split "
                        "leakage and background-domain collinearity; exit 1 "
                        "on failure.")
    args = p.parse_args(argv)

    if args.audit:
        _run_audit(Path(args.output_dir))
        return
    if args.pool is None:
        p.error("--pool is required unless --audit is given")

    cfg = DatasetConfig(
        num_events=1,
        output_dir=args.output_dir,
        glitch_source="gwosc",
        glitch_metadata_csv=args.pool,
    )
    provider = RealGlitchProvider(cfg)
    prep = Preprocessor(cfg)
    renderer = QTransformRenderer(cfg)
    labeler = LabelGenerator(cfg.duration, (cfg.frange_low, cfg.frange_high),
                             cfg.frequency_axis_scale)

    rng = np.random.default_rng(args.seed)
    done = 0
    while done < args.n:
        det = "H1" if done % 2 == 0 else "L1"
        try:
            glitch = provider.sample_glitch(rng, det)
        except GlitchFetchError as exc:
            print(f"skip ({exc})")
            continue
        stats = renderer.energy_stats(prep.preprocess(glitch.series))
        box = labeler.glitch_box_from_ridge(
            stats.energy, stats.freqs, stats.times,
            (glitch.start_time, glitch.end_time), cfg.label_ridge_threshold,
        )
        # Pedestal: mean energy inside vs outside the glitch window, high band
        # (400-1000 Hz) where long low-frequency glitches should have nothing.
        tmask = (stats.times >= glitch.start_time) & (stats.times <= glitch.end_time)
        fmask = stats.freqs >= 400.0
        inside = float(np.mean(stats.energy[np.ix_(fmask, tmask)]))
        outside = float(np.mean(stats.energy[np.ix_(fmask, ~tmask)]))
        ped = inside / outside if outside > 0 else float("nan")
        if box is None:
            print(f"{det} {glitch.glitch_type:<22} NO BOX (would be resampled)   "
                  f"pedestal(>400Hz) x{ped:.2f}")
        else:
            frac = labeler.box_area_fraction(box)
            print(f"{det} {glitch.glitch_type:<22} "
                  f"t=[{box.time_start:.2f},{box.time_end:.2f}]s "
                  f"f=[{box.freq_low:.0f},{box.freq_high:.0f}]Hz "
                  f"area={frac:.0%} pedestal(>400Hz) x{ped:.2f}")
        done += 1


if __name__ == "__main__":
    main()
