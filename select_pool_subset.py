"""Select a download-efficient subset of a Gravity Spy pool CSV.

GWOSC serves strain in 4096-second files, and gwpy caches downloads, so the
prefetch cost is the number of DISTINCT files touched -- not the number of
glitches. This script greedily picks, per detector, the files containing the
most pool glitches until ``--per-detector`` glitches are covered, then tops up
any glitch class that would otherwise be missing. The result is written sorted
by (ifo, gps) so consecutive prefetches hit the warm gwpy cache.

Example
-------
    python select_pool_subset.py --pool gravityspy_pool.csv \
        --out gravityspy_pool_3000.csv --per-detector 550
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

FILE_SECONDS = 4096


def select_subset(rows: list[dict], per_detector: int) -> list[dict]:
    by_det_file: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        fid = int(float(r["gps"]) // FILE_SECONDS)
        by_det_file[r["ifo"]][fid].append(r)

    selected: list[dict] = []
    for det, files in by_det_file.items():
        ranked = sorted(files.values(), key=len, reverse=True)
        picked: list[dict] = []
        picked_fids: set = set()
        for bucket in ranked:
            if len(picked) >= per_detector:
                break
            picked.extend(bucket)
            picked_fids.add(int(float(bucket[0]["gps"]) // FILE_SECONDS))
        # Top up any glitch class the greedy pass missed: add the single file
        # holding the most rows of that class.
        all_classes = {r["label"] for bucket in files.values() for r in bucket}
        have = {r["label"] for r in picked}
        for cls in sorted(all_classes - have):
            best = max(
                (b for fid, b in files.items() if fid not in picked_fids),
                key=lambda b: sum(r["label"] == cls for r in b),
                default=None,
            )
            if best:
                picked.extend(best)
                picked_fids.add(int(float(best[0]["gps"]) // FILE_SECONDS))
        selected.extend(picked)

    selected.sort(key=lambda r: (r["ifo"], float(r["gps"])))
    return selected


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pool", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per-detector", type=int, default=550,
                   help="Minimum glitches to cover per detector.")
    args = p.parse_args(argv)

    with open(args.pool, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        rows = list(reader)

    subset = select_subset(rows, args.per_detector)

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(subset)

    n_files = len({(r["ifo"], int(float(r["gps"]) // FILE_SECONDS)) for r in subset})
    per_det: dict[str, int] = defaultdict(int)
    per_cls: dict[str, int] = defaultdict(int)
    for r in subset:
        per_det[r["ifo"]] += 1
        per_cls[r["label"]] += 1
    print(f"selected {len(subset)} glitches across {n_files} GWOSC files")
    print("per detector:", dict(per_det))
    print("per class:", dict(sorted(per_cls.items())))


if __name__ == "__main__":
    main()
