"""Prefetch GWOSC strain for every glitch in a Gravity Spy pool CSV.

Fills ``real_glitch_cache_dir`` up front so a subsequent
``build_dataset.py --glitch-source gwosc`` run is cache-only (no network in the
build loop -- the reason the first 1250-event real-glitch build stalled).

Resumable: already-cached segments are skipped, permanently unavailable GPS
times are blacklisted to ``<cache-dir>/unavailable.json`` (shared with the
builder), and transient network failures are retried, then reported without
being blacklisted -- just re-run the script.

Example
-------
    python prefetch_glitch_cache.py --pool gravityspy_pool.csv \
        --cache-dir datasets/gw_dataset_real/glitch_cache
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Sequence

from config import DatasetConfig
from real_glitch import (
    GlitchDependencyError,
    GlitchFetchError,
    GlitchUnavailableError,
    RealGlitchProvider,
)


def prefetch_pool(
    provider: RealGlitchProvider,
    detectors: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    retries: int = 2,
    sleep: float = 0.0,
    log=print,
) -> dict:
    """Fetch every not-yet-cached pool segment. Returns outcome counts."""
    counts = {
        "fetched": 0, "cached": 0, "skipped_unavailable": 0,
        "unavailable": 0, "failed": 0,
    }
    seen: set = set()
    for det in detectors or provider.config.detectors:
        for row in provider.pool.get(det, []):
            gps = float(row["gps"])
            key = provider._key(det, gps)
            if key in seen:
                continue
            seen.add(key)
            if key in provider._unavailable:
                counts["skipped_unavailable"] += 1
                continue
            if provider.cache_npy_path(det, gps).exists():
                counts["cached"] += 1
                continue
            if limit is not None and counts["fetched"] >= limit:
                continue
            for attempt in range(retries + 1):
                try:
                    provider._fetch_segment(det, gps)
                    counts["fetched"] += 1
                    log(f"[{sum(counts.values())}] fetched {det} @ {gps:.4f}")
                    break
                except GlitchDependencyError:
                    raise  # gwpy missing: nothing else will succeed either
                except GlitchUnavailableError as exc:
                    provider._mark_unavailable(det, gps)
                    counts["unavailable"] += 1
                    log(f"unavailable {det} @ {gps:.4f}: {exc}")
                    break
                except GlitchFetchError as exc:
                    if attempt == retries:
                        counts["failed"] += 1
                        log(f"FAILED (transient, will retry on next run) "
                            f"{det} @ {gps:.4f}: {exc}")
                    else:
                        time.sleep(max(sleep, 1.0))
            if sleep:
                time.sleep(sleep)
    return counts


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pool", required=True, help="Gravity Spy pool CSV (gps,ifo,label,...)")
    p.add_argument("--cache-dir", required=True,
                   help="Target glitch cache (the build's real_glitch_cache_dir).")
    p.add_argument("--sample-rate", type=int, default=4096)
    p.add_argument("--halfwin", type=float, default=4.0,
                   help="Seconds fetched each side of the glitch GPS (must match the build).")
    p.add_argument("--detectors", nargs="+", default=None,
                   help="Subset of detectors to prefetch (default: all in the pool).")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after this many new fetches (connectivity check).")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--sleep", type=float, default=0.0,
                   help="Pause between fetches (politeness / rate limiting).")
    args = p.parse_args(argv)

    cfg = DatasetConfig(
        num_events=1,
        glitch_source="gwosc",
        glitch_metadata_csv=args.pool,
        real_glitch_cache_dir=Path(args.cache_dir),
        sample_rate=args.sample_rate,
        glitch_fetch_halfwin=args.halfwin,
    )
    provider = RealGlitchProvider(cfg)
    counts = prefetch_pool(
        provider,
        detectors=args.detectors,
        limit=args.limit,
        retries=args.retries,
        sleep=args.sleep,
    )
    print(
        f"\ndone: {counts['fetched']} fetched, {counts['cached']} already cached, "
        f"{counts['unavailable']} newly unavailable, "
        f"{counts['skipped_unavailable']} blacklisted, {counts['failed']} transient failures."
    )
    if counts["failed"]:
        print("re-run this script to retry the transient failures.")


if __name__ == "__main__":
    main()
