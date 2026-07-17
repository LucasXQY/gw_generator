"""QA: sample real cached glitches through the full injection + render path
and report box tightness and the broadband noise pedestal.

Run inside the env that has gwpy (the build env). Uses only the local cache --
GPS times not in the cache raise, so run prefetch first.

    python validate_real_glitch.py --pool gravityspy_pool_3000.csv \
        --output-dir datasets/gw_dataset_3000_real --n 12
"""

from __future__ import annotations

import argparse

import numpy as np

from config import DatasetConfig
from label_generator import LabelGenerator
from preprocessing import Preprocessor
from qtransform import QTransformRenderer
from real_glitch import GlitchFetchError, RealGlitchProvider


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True)
    p.add_argument("--output-dir", required=True,
                   help="Build output dir (its glitch_cache/ is used).")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)

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
