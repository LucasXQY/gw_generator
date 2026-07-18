"""Bulk-prefetch real off-source background segments into the glitch cache.

The lesson from the B1 real-glitch build applies unchanged: fetching GWOSC
per sample at build time stalls the pipeline for days. This tool downloads
each 4096 s GWOSC source file ONCE (per detector), cuts every valid
off-source grid candidate locally, and stores them in the same npy cache the
builder reads -- after which a ``noise_source='gwosc'`` build runs fully
offline.

    python prefetch_offsource_cache.py --pool gravityspy_pool_3000.csv \
        --cache-dir glitch_cache

Resumable: existing cache entries and blacklisted GPS are skipped; segments
containing non-finite samples (data gaps) are blacklisted instead of cached.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from config import DatasetConfig
from real_glitch import (
    GlitchDependencyError,
    GlitchFetchError,
    GlitchUnavailableError,
    RealGlitchProvider,
)
from split_source_groups import FILE_SECONDS, source_group


def prefetch_offsource(
    provider: RealGlitchProvider,
    detectors=None,
    per_file: int | None = None,
    retries: int = 2,
    sleep: float = 0.0,
    log=print,
) -> dict:
    """Fill the cache with off-source grid segments for every pool file.

    Returns stats: files_fetched, segments_cached, files_skipped (already
    complete), failed_files (transient errors after retries).
    """
    cfg = provider.config
    halfwin = float(cfg.glitch_fetch_halfwin)
    seg_len = int(2 * halfwin * provider.sr)
    stats = {
        "files_fetched": 0,
        "segments_cached": 0,
        "files_skipped": 0,
        "failed_files": [],
    }
    for detector in detectors or cfg.detectors:
        groups = sorted({
            source_group(detector, r["gps"]) for r in provider.pool[detector]
        })
        for group in groups:
            candidates = list(provider.offsource_candidates(detector, group))
            if per_file is not None:
                candidates = candidates[:per_file]
            missing = [
                g for g in candidates
                if not provider.cache_npy_path(detector, g).exists()
                and provider._key(detector, g) not in provider._unavailable
            ]
            if not missing:
                stats["files_skipped"] += 1
                continue

            fid = int(group.split(":", 1)[1])
            file_start = fid * FILE_SECONDS
            center = file_start + FILE_SECONDS / 2.0
            raw = None
            for attempt in range(1 + retries):
                try:
                    raw = provider._gwosc_fetch(detector, center, FILE_SECONDS / 2.0)
                    break
                except GlitchDependencyError:
                    raise
                except GlitchUnavailableError as exc:
                    # Whole file absent from open data: blacklist its grid.
                    log(f"{group}: unavailable ({exc}); blacklisting grid")
                    for g in missing:
                        provider._mark_unavailable(detector, g)
                    raw = None
                    break
                except GlitchFetchError as exc:
                    log(f"{group}: transient fetch error "
                        f"(attempt {attempt + 1}/{1 + retries}): {exc}")
                    if sleep:
                        time.sleep(sleep)
            else:
                stats["failed_files"].append(group)
                continue
            if raw is None:
                continue

            cached_here = 0
            for gps in missing:
                i0 = int(round((gps - halfwin - file_start) * provider.sr))
                seg = raw[i0 : i0 + seg_len]
                if seg.size < seg_len or not np.all(np.isfinite(seg)) or not np.any(seg):
                    # Data gap under this candidate: never cache, never retry.
                    provider._mark_unavailable(detector, gps)
                    continue
                provider.cache_store(detector, gps, seg)
                cached_here += 1
            stats["files_fetched"] += 1
            stats["segments_cached"] += cached_here
            log(f"{group}: cached {cached_here}/{len(missing)} off-source segments")
    return stats


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--detectors", nargs="*", default=None)
    p.add_argument("--per-file", type=int, default=None,
                   help="Cache at most N candidates per 4096 s file (default all).")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--sleep", type=float, default=1.0)
    args = p.parse_args(argv)

    cfg = DatasetConfig(
        num_events=1,
        output_dir=".",
        glitch_source="gwosc",
        glitch_metadata_csv=args.pool,
        real_glitch_cache_dir=args.cache_dir,
    )
    provider = RealGlitchProvider(cfg)
    stats = prefetch_offsource(
        provider,
        detectors=args.detectors,
        per_file=args.per_file,
        retries=args.retries,
        sleep=args.sleep,
    )
    print(
        f"done: files_fetched={stats['files_fetched']} "
        f"segments_cached={stats['segments_cached']} "
        f"files_skipped={stats['files_skipped']} "
        f"failed={len(stats['failed_files'])}"
    )
    if stats["failed_files"]:
        print("failed files (rerun to retry):", ", ".join(stats["failed_files"]))


if __name__ == "__main__":
    main()
