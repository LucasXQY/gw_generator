"""Data-leakage and pair-consistency validators.

All functions accept either a pandas ``DataFrame`` or a list of dict rows
(as produced by the builder / read back with :mod:`csv`). They raise
``AssertionError`` on a violation and return ``True`` on success.
"""

from __future__ import annotations

from typing import Dict, List

from split_source_groups import source_group


def _rows(data) -> List[dict]:
    if hasattr(data, "to_dict"):  # pandas DataFrame
        return data.to_dict("records")
    return list(data)


def _offenders(bad: Dict[str, set], limit: int = 5) -> str:
    shown = {k: sorted(bad[k]) for k in list(bad)[:limit]}
    more = f" (+{len(bad) - limit} more)" if len(bad) > limit else ""
    return f"{len(bad)} offenders: {shown}{more}"


def _split_by_sample(metadata) -> Dict[str, str]:
    return {str(r["sample_id"]): str(r["split"]) for r in _rows(metadata)}


def validate_no_event_leakage(metadata) -> bool:
    """Assert that each ``event_id`` appears in only one split."""
    seen: Dict[str, set] = {}
    for r in _rows(metadata):
        seen.setdefault(str(r["event_id"]), set()).add(str(r["split"]))
    bad = {e: s for e, s in seen.items() if len(s) > 1}
    assert not bad, f"event_id leaks across splits: {bad}"
    return True


def validate_no_chirp_leakage(metadata) -> bool:
    """Assert that each (non-empty) ``chirp_id`` appears in only one split."""
    seen: Dict[str, set] = {}
    for r in _rows(metadata):
        chirp = str(r.get("chirp_id", "") or "")
        if not chirp:
            continue
        seen.setdefault(chirp, set()).add(str(r["split"]))
    bad = {c: s for c, s in seen.items() if len(s) > 1}
    assert not bad, f"chirp_id leaks across splits: {bad}"
    return True


_TRUTHY = {"1", "true", "True"}


def _glitch_rows(metadata):
    for r in _rows(metadata):
        gid = str(r.get("glitch_id", "") or "")
        if gid:
            yield r, gid


def _glitch_group(r: dict) -> str:
    """Explicit ``glitch_source_group`` column wins; else derive it from
    ``detector`` + ``glitch_gps`` so pre-G1 datasets can be audited."""
    explicit = str(r.get("glitch_source_group", "") or "")
    if explicit:
        return explicit
    gps = str(r.get("glitch_gps", "") or "")
    if not gps:
        return ""
    return source_group(str(r.get("detector", "") or ""), gps)


_BACKGROUND_DOMAINS = {
    "real_gwosc": "gwosc",
    "real_gwosc_offsource": "gwosc",
    "gaussian_aligo_colored": "synthetic",
}


def _background_domain(r: dict) -> str:
    """Explicit ``background_source`` column wins; else map ``noise_type``."""
    explicit = str(r.get("background_source", "") or "")
    if explicit:
        return explicit
    ntype = str(r.get("noise_type", "") or "")
    return _BACKGROUND_DOMAINS.get(ntype, ntype)


def validate_no_glitch_leakage(metadata) -> bool:
    """Assert that each (non-empty) ``glitch_id`` appears in only one split."""
    seen: Dict[str, set] = {}
    for r, gid in _glitch_rows(metadata):
        seen.setdefault(gid, set()).add(str(r["split"]))
    bad = {g: s for g, s in seen.items() if len(s) > 1}
    assert not bad, f"glitch_id leaks across splits: {_offenders(bad)}"
    return True


def validate_no_source_group_leakage(metadata) -> bool:
    """Assert each glitch 4096 s source_group appears in only one split."""
    seen: Dict[str, set] = {}
    for r, _gid in _glitch_rows(metadata):
        group = _glitch_group(r)
        if not group:
            continue
        seen.setdefault(group, set()).add(str(r["split"]))
    bad = {g: s for g, s in seen.items() if len(s) > 1}
    assert not bad, f"glitch source_group leaks across splits: {_offenders(bad)}"
    return True


def validate_background_domain_decoupled(metadata) -> bool:
    """Assert the background domain is not collinear with ``has_glitch``.

    Fails when glitch rows and non-glitch rows draw their backgrounds from
    different domains (e.g. real GWOSC noise iff has_glitch) -- a model
    could then classify glitches from background texture alone.
    """
    domains = {True: set(), False: set()}
    for r in _rows(metadata):
        domain = _background_domain(r)
        if not domain:
            continue
        has_glitch = str(r.get("has_glitch", "") or "").strip() in _TRUTHY
        domains[has_glitch].add(domain)
    if domains[True] and domains[False]:
        assert domains[True] == domains[False], (
            "background domain is collinear with has_glitch: "
            f"glitch rows use {sorted(domains[True])}, "
            f"non-glitch rows use {sorted(domains[False])}"
        )
    return True


def validate_no_background_group_leakage(metadata) -> bool:
    """Assert each ``background_source_group`` appears in only one split."""
    seen: Dict[str, set] = {}
    for r in _rows(metadata):
        group = str(r.get("background_source_group", "") or "")
        if not group:
            continue
        seen.setdefault(group, set()).add(str(r["split"]))
    bad = {g: s for g, s in seen.items() if len(s) > 1}
    assert not bad, (
        f"background source_group leaks across splits: {_offenders(bad)}"
    )
    return True


def validate_pair_split_consistency(pairs, metadata, allow_cross_split: bool = False) -> bool:
    """Assert anchor and candidate of each pair share the pair's split."""
    if allow_cross_split:
        return True
    sample_split = _split_by_sample(metadata)
    for r in _rows(pairs):
        anchor = str(r.get("anchor_sample_id"))
        cand = str(r.get("candidate_sample_id") or r.get("positive_sample_id"))
        split = str(r["split"])
        a_split = sample_split.get(anchor)
        c_split = sample_split.get(cand)
        assert a_split == split and c_split == split, (
            f"pair {r} crosses splits: anchor={a_split}, candidate={c_split}, "
            f"pair_split={split}"
        )
    return True


def validate_positive_pairs(match_pairs, metadata) -> bool:
    """Assert positive pairs connect the same chirp_id and same event_id."""
    by_id = {str(r["sample_id"]): r for r in _rows(metadata)}
    for r in _rows(match_pairs):
        a = by_id[str(r["anchor_sample_id"])]
        p = by_id[str(r["positive_sample_id"])]
        assert str(a["event_id"]) == str(p["event_id"]), f"positive pair event mismatch: {r}"
        assert str(a["chirp_id"]) == str(p["chirp_id"]) and a["chirp_id"], (
            f"positive pair chirp mismatch: {r}"
        )
    return True


def validate_negative_pairs(negative_pairs, metadata) -> bool:
    """Assert standard negatives differ in chirp; counterfactuals are flagged."""
    by_id = {str(r["sample_id"]): r for r in _rows(metadata)}
    for r in _rows(negative_pairs):
        ntype = str(r["negative_type"])
        a = by_id[str(r["anchor_sample_id"])]
        c = by_id[str(r["candidate_sample_id"])]
        if ntype == "invalid_delay_same_chirp":
            # Controlled counterfactual: may share chirp but must be a negative.
            assert int(r["same_chirp"]) == 1, f"invalid_delay must be flagged same_chirp=1: {r}"
            continue
        assert str(a["chirp_id"]) != str(c["chirp_id"]) or not a["chirp_id"], (
            f"standard negative {ntype} shares chirp_id: {r}"
        )
    return True


def run_all_validations(
    metadata, match_pairs, negative_pairs, pair_metadata, allow_cross_split: bool = False
) -> bool:
    validate_no_event_leakage(metadata)
    validate_no_chirp_leakage(metadata)
    # Source-isolation checks are unconditional: allow_cross_split exists
    # only for negative-pair experiments and must never relax these.
    validate_no_glitch_leakage(metadata)
    validate_no_source_group_leakage(metadata)
    validate_background_domain_decoupled(metadata)
    validate_no_background_group_leakage(metadata)
    validate_positive_pairs(match_pairs, metadata)
    validate_negative_pairs(negative_pairs, metadata)
    validate_pair_split_consistency(match_pairs, metadata, allow_cross_split)
    validate_pair_split_consistency(negative_pairs, metadata, allow_cross_split)
    validate_pair_split_consistency(pair_metadata, metadata, allow_cross_split)
    return True
