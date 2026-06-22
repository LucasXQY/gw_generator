"""Positive and negative pair generation (within-split, leakage-safe).

Consumes lightweight per-sample info dicts and produces:

* ``match_pairs``  : positive same-chirp cross-detector pairs (pair_label 1);
* ``negative_pairs``: all enabled negative categories (pair_label 0);
* ``pair_metadata`` : the combined positive + negative table.

Pairs are built independently inside each split, so they never cross
train/val/test boundaries unless ``allow_cross_split_negative_pairs`` is set.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import DatasetConfig

HARD_TYPES = (
    "chirp_vs_glitch",
    "similar_snr_different_chirp",
    "similar_frequency_different_chirp",
    "invalid_delay_same_chirp",
)
EASY_TYPES = (
    "different_chirp_cross_detector",
    "chirp_vs_noise",
    "same_detector_different_event",
)


def _freq_overlap(a: dict, b: dict) -> float:
    """IoU of the two label frequency ranges (0 if either is missing)."""
    al, ah = a.get("label_low"), a.get("label_high")
    bl, bh = b.get("label_low"), b.get("label_high")
    if None in (al, ah, bl, bh):
        return 0.0
    inter = max(0.0, min(ah, bh) - max(al, bl))
    union = max(ah, bh) - min(al, bl)
    return float(inter / union) if union > 0 else 0.0


def _time_delay_diff(a: dict, b: dict) -> float:
    ta = a.get("injection_time")
    tb = b.get("injection_time")
    if ta is None or tb is None:
        return float("nan")
    return abs(float(ta) - float(tb))


class PairBuilder:
    def __init__(self, config: DatasetConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self._neg_id = 0
        self._match_id = 0
        self._pair_id = 0

    # ----------------------------------------------------------------- public
    def generate(self, samples: List[dict]):
        match_rows: List[dict] = []
        negative_rows: List[dict] = []
        for split in ("train", "val", "test"):
            split_samples = [s for s in samples if s["split"] == split]
            if not split_samples:
                continue
            m = self._positive_pairs(split, split_samples)
            n = self._negative_pairs(split, split_samples)
            match_rows.extend(m)
            negative_rows.extend(n)
        pair_rows = self._combined(match_rows, negative_rows, samples)
        return match_rows, negative_rows, pair_rows

    # -------------------------------------------------------------- positives
    def _positive_pairs(self, split: str, samples: List[dict]) -> List[dict]:
        by_event: Dict[str, List[dict]] = {}
        for s in samples:
            if s["has_chirp"]:
                by_event.setdefault(s["event_id"], []).append(s)

        rows: List[dict] = []
        for event_id, group in by_event.items():
            for a, b in combinations(group, 2):  # one-to-one / one-to-many / many-to-many
                self._match_id += 1
                rows.append(
                    {
                        "match_id": self._match_id,
                        "event_id": event_id,
                        "chirp_id": a["chirp_id"],
                        "split": split,
                        "anchor_sample_id": a["sample_id"],
                        "positive_sample_id": b["sample_id"],
                        "anchor_detector": a["detector"],
                        "positive_detector": b["detector"],
                        "same_chirp": 1,
                        "same_noise": 0,
                        "notes": "cross-detector same-chirp",
                    }
                )
        return rows

    # -------------------------------------------------------------- negatives
    def _negative_pairs(self, split: str, samples: List[dict]) -> List[dict]:
        chirps = [s for s in samples if s["has_chirp"]]
        glitches = [s for s in samples if s["global_class"] == "glitch_only"]
        if not glitches:
            glitches = [s for s in samples if s["has_glitch"] and not s["has_chirp"]]
        noises = [s for s in samples if s["global_class"] == "pure_noise"]

        enabled = self._enabled_types(glitches, noises, chirps)
        rows: List[dict] = []
        seen: set = set()

        for anchor in chirps:
            count = self.config.negative_pairs_per_positive
            n_hard = int(round(count * self.config.hard_negative_fraction))
            order = ["hard"] * n_hard + ["easy"] * (count - n_hard)
            for tier in order:
                pool = [t for t in (HARD_TYPES if tier == "hard" else EASY_TYPES) if t in enabled]
                if not pool:
                    pool = [t for t in enabled]
                if not pool:
                    break
                ntype = str(self.rng.choice(pool))
                row = self._make(split, ntype, anchor, chirps, glitches, noises, seen)
                if row is not None:
                    rows.append(row)

        # Coverage: guarantee at least one row per enabled type if possible.
        present = {r["negative_type"] for r in rows}
        for ntype in enabled:
            if ntype in present:
                continue
            for anchor in chirps:
                row = self._make(split, ntype, anchor, chirps, glitches, noises, seen)
                if row is not None:
                    rows.append(row)
                    break
        return rows

    def _enabled_types(self, glitches, noises, chirps) -> Tuple[str, ...]:
        cfg = self.config
        enabled = ["different_chirp_cross_detector", "same_detector_different_event",
                   "similar_snr_different_chirp", "similar_frequency_different_chirp"]
        if cfg.enable_chirp_vs_glitch_negatives and glitches:
            enabled.append("chirp_vs_glitch")
        if cfg.enable_chirp_vs_noise_negatives and noises:
            enabled.append("chirp_vs_noise")
        if cfg.enable_invalid_delay_negatives:
            enabled.append("invalid_delay_same_chirp")
        return tuple(enabled)

    def _make(self, split, ntype, anchor, chirps, glitches, noises, seen) -> Optional[dict]:
        cand = None
        force_invalid_delay = False
        if ntype == "different_chirp_cross_detector":
            cand = self._pick(
                chirps,
                lambda c: c["event_id"] != anchor["event_id"] and c["detector"] != anchor["detector"],
            )
        elif ntype == "same_detector_different_event":
            cand = self._pick(
                chirps + glitches + noises,
                lambda c: c["detector"] == anchor["detector"] and c["event_id"] != anchor["event_id"],
            )
        elif ntype == "chirp_vs_glitch":
            cand = self._pick(glitches, lambda c: c["event_id"] != anchor["event_id"])
        elif ntype == "chirp_vs_noise":
            cand = self._pick(noises, lambda c: c["event_id"] != anchor["event_id"])
        elif ntype == "similar_snr_different_chirp":
            cand = self._pick(
                chirps,
                lambda c: c["chirp_id"] != anchor["chirp_id"]
                and c["snr_bin"] == anchor["snr_bin"],
            )
        elif ntype == "similar_frequency_different_chirp":
            cand = self._pick(
                chirps,
                lambda c: c["chirp_id"] != anchor["chirp_id"] and _freq_overlap(anchor, c) > 0.3,
            )
        elif ntype == "invalid_delay_same_chirp":
            cand = self._pick(
                chirps,
                lambda c: c["chirp_id"] == anchor["chirp_id"]
                and c["sample_id"] != anchor["sample_id"],
            )
            force_invalid_delay = True

        if cand is None:
            return None
        key = (ntype, anchor["sample_id"], cand["sample_id"])
        if key in seen:
            return None
        seen.add(key)
        return self._row(split, ntype, anchor, cand, force_invalid_delay)

    def _pick(self, pool, predicate) -> Optional[dict]:
        candidates = [c for c in pool if predicate(c)]
        if not candidates:
            return None
        return candidates[int(self.rng.integers(0, len(candidates)))]

    def _row(self, split, ntype, anchor, cand, force_invalid_delay) -> dict:
        same_chirp = int(anchor["chirp_id"] == cand["chirp_id"] and bool(anchor["chirp_id"]))
        same_event = int(anchor["event_id"] == cand["event_id"])
        if force_invalid_delay:
            td_diff = self.config.max_physical_time_delay * 8.0
            valid_delay = 0
        else:
            td_diff = _time_delay_diff(anchor, cand)
            valid_delay = int(
                np.isfinite(td_diff) and td_diff <= self.config.max_physical_time_delay
            )
        self._neg_id += 1
        return {
            "negative_pair_id": self._neg_id,
            "split": split,
            "anchor_sample_id": anchor["sample_id"],
            "candidate_sample_id": cand["sample_id"],
            "anchor_event_id": anchor["event_id"],
            "candidate_event_id": cand["event_id"],
            "anchor_chirp_id": anchor["chirp_id"],
            "candidate_chirp_id": cand["chirp_id"],
            "anchor_detector": anchor["detector"],
            "candidate_detector": cand["detector"],
            "same_chirp": same_chirp,
            "same_event": same_event,
            "valid_time_delay": valid_delay,
            "time_delay_difference": _fmt(td_diff),
            "frequency_overlap_score": _fmt(_freq_overlap(anchor, cand)),
            "snr_bin_anchor": anchor["snr_bin"],
            "snr_bin_candidate": cand["snr_bin"],
            "negative_type": ntype,
            "notes": "",
        }

    # --------------------------------------------------------------- combined
    def _combined(self, match_rows, negative_rows, samples) -> List[dict]:
        by_id = {s["sample_id"]: s for s in samples}
        rows: List[dict] = []
        for m in match_rows:
            self._pair_id += 1
            a = by_id[m["anchor_sample_id"]]
            c = by_id[m["positive_sample_id"]]
            td = _time_delay_diff(a, c)
            rows.append(
                {
                    "pair_id": self._pair_id,
                    "pair_label": 1,
                    "pair_type": "positive_same_chirp",
                    "split": m["split"],
                    "anchor_sample_id": m["anchor_sample_id"],
                    "candidate_sample_id": m["positive_sample_id"],
                    "anchor_event_id": m["event_id"],
                    "candidate_event_id": m["event_id"],
                    "anchor_chirp_id": m["chirp_id"],
                    "candidate_chirp_id": m["chirp_id"],
                    "anchor_detector": m["anchor_detector"],
                    "candidate_detector": m["positive_detector"],
                    "same_chirp": 1,
                    "same_event": 1,
                    "valid_time_delay": int(np.isfinite(td) and td <= self.config.max_physical_time_delay),
                    "time_delay_difference": _fmt(td),
                    "frequency_overlap_score": _fmt(_freq_overlap(a, c)),
                    "snr_bin_anchor": a["snr_bin"],
                    "snr_bin_candidate": c["snr_bin"],
                    "notes": "positive_same_chirp",
                }
            )
        for n in negative_rows:
            self._pair_id += 1
            rows.append(
                {
                    "pair_id": self._pair_id,
                    "pair_label": 0,
                    "pair_type": n["negative_type"],
                    "split": n["split"],
                    "anchor_sample_id": n["anchor_sample_id"],
                    "candidate_sample_id": n["candidate_sample_id"],
                    "anchor_event_id": n["anchor_event_id"],
                    "candidate_event_id": n["candidate_event_id"],
                    "anchor_chirp_id": n["anchor_chirp_id"],
                    "candidate_chirp_id": n["candidate_chirp_id"],
                    "anchor_detector": n["anchor_detector"],
                    "candidate_detector": n["candidate_detector"],
                    "same_chirp": n["same_chirp"],
                    "same_event": n["same_event"],
                    "valid_time_delay": n["valid_time_delay"],
                    "time_delay_difference": n["time_delay_difference"],
                    "frequency_overlap_score": n["frequency_overlap_score"],
                    "snr_bin_anchor": n["snr_bin_anchor"],
                    "snr_bin_candidate": n["snr_bin_candidate"],
                    "notes": n["negative_type"],
                }
            )
        return rows


def _fmt(value) -> str:
    if value is None:
        return ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(v):
        return ""
    return f"{v:.6f}"
