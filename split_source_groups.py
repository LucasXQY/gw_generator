"""GWOSC 4096 s source-file grouping and group-to-split assignment (G1/D1).

A *source group* identifies the GWOSC open-data file a strain segment comes
from::

    glitch_source_group = f"{ifo}:{floor(gps / 4096)}"

Groups are assigned to train/val/test **before** any event is built and a
group never serves more than one split. The assignment is deterministic in
``seed`` and uses its own RNG so it cannot perturb the builder's shared
random stream.

Stdlib-only so leakage audits can run in minimal environments.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterable, Mapping

# One GWOSC open-data strain file spans 4096 seconds. Must stay in sync with
# select_pool_subset.FILE_SECONDS (enforced by test_source_leakage).
FILE_SECONDS = 4096


def source_group(ifo: str, gps) -> str:
    """Stable group id for the 4096 s GWOSC file containing ``gps``."""
    return f"{ifo}:{int(float(gps) // FILE_SECONDS)}"


def assign_groups_to_splits(
    rows: Iterable[Mapping],
    ratios: Mapping[str, float],
    seed: int,
) -> Dict[str, str]:
    """Assign each (ifo, 4096 s file) group to exactly one split.

    ``rows`` need ``ifo`` and ``gps`` keys (the glitch-pool CSV schema).
    Assignment is per detector and weighted by each group's row count:
    groups are handed out largest-first to the split with the largest
    remaining row deficit, so every split's usable pool approximates
    ``ratios`` even though whole files are indivisible.
    """
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        ifo = str(r["ifo"])
        counts[ifo][source_group(ifo, r["gps"])] += 1

    split_order = list(ratios)
    rng = random.Random(seed)
    assignment: Dict[str, str] = {}
    for ifo in sorted(counts):
        groups = sorted(counts[ifo].items())
        rng.shuffle(groups)
        # Stable sort after the shuffle: descending size, random tie order.
        groups.sort(key=lambda kv: -kv[1])
        total = sum(counts[ifo].values())
        deficit = {s: total * ratios[s] for s in split_order}
        for group, n in groups:
            best = max(split_order, key=lambda s: deficit[s])
            assignment[group] = best
            deficit[best] -= n
    return assignment
