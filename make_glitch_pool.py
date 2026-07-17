"""Build a Gravity Spy glitch *pool* CSV for GWOSC real-glitch injection.

Fetches the Gravity Spy glitch catalog (via gwpy's ``GravitySpyTable``, which
queries the public Gravity Spy database) and writes a CSV in the schema expected
by ``real_glitch.RealGlitchProvider`` / ``build_dataset.py --glitch-source gwosc``:

    gps, ifo, label, snr, peak_frequency, central_freq, bandwidth, duration, gravityspy_id

Rows are filtered by ML confidence and by label (defaults to the pipeline's
``DEFAULT_GLITCH_TYPES``), and capped per (ifo, label) so the pool is balanced.

If the online fetch is unavailable or you already downloaded a Gravity Spy
metadata table (CSV / HDF5 / FITS readable by astropy), pass ``--from-file`` to
convert that instead — the same column-mapping logic applies.

Examples
--------
    python make_glitch_pool.py --out gravityspy_pool.csv
    python make_glitch_pool.py --ifos H1 L1 --min-confidence 0.9 --per-class 300
    python make_glitch_pool.py --from-file trainingset_v1d1.h5 --out gravityspy_pool.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from typing import Dict, List, Optional, Sequence

from config import DEFAULT_GLITCH_TYPES

# Candidate source-column names -> our canonical name. First match wins.
_COLUMN_CANDIDATES: Dict[str, Sequence[str]] = {
    "ifo": ("ifo", "detector", "Detector", "IFO"),
    "label": ("ml_label", "label", "ML_label", "gravityspy_label", "Label"),
    "confidence": ("ml_confidence", "confidence", "ml_conf", "ML_confidence"),
    "snr": ("snr", "SNR", "Snr"),
    "peak_frequency": ("peak_frequency", "peakFreq", "peak_freq", "peakFrequency"),
    "central_freq": ("central_freq", "centralFreq", "central_frequency"),
    "bandwidth": ("bandwidth", "Bandwidth", "bw"),
    "duration": ("duration", "Duration"),
    "gravityspy_id": ("gravityspy_id", "gravityspyID", "uniqueID", "id", "event_id"),
    # GPS is handled specially (peak_time[+peak_time_ns] or a single column).
    "gps_single": ("event_time", "GPStime", "gps", "peakGPS", "peak_time_gps", "time"),
}

_OUT_FIELDS = (
    "gps", "ifo", "label", "snr",
    "peak_frequency", "central_freq", "bandwidth", "duration", "gravityspy_id",
)


def _find(colnames: Sequence[str], key: str) -> Optional[str]:
    lower = {c.lower(): c for c in colnames}
    for cand in _COLUMN_CANDIDATES.get(key, ()):  # preserve priority order
        if cand in colnames:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _gps_getter(colnames: Sequence[str]):
    """Return a function row->float(gps), based on available columns."""
    if "peak_time" in colnames:
        has_ns = "peak_time_ns" in colnames
        def get(row):
            ns = float(row["peak_time_ns"]) if has_ns and row["peak_time_ns"] not in (None, "") else 0.0
            return float(row["peak_time"]) + ns * 1e-9
        return get
    single = _find(colnames, "gps_single")
    if single is not None:
        return lambda row, _c=single: float(row[_c])
    raise SystemExit(
        "Could not find a GPS-time column. Looked for 'peak_time'(+'peak_time_ns') "
        f"or one of {_COLUMN_CANDIDATES['gps_single']}. Available: {list(colnames)}"
    )


def _iter_source(args):
    """Yield (colnames, row-dict) from local files (streamed) or the online DB."""
    if args.from_file:
        for path in args.from_file:
            low = str(path).lower()
            print(f"[read] {path}", file=sys.stderr)
            if low.endswith((".csv", ".tsv", ".txt")):
                delim = "\t" if low.endswith(".tsv") else ","
                with open(path, newline="", encoding="utf-8") as fh:
                    rd = csv.DictReader(fh, delimiter=delim)
                    cols = rd.fieldnames or []
                    for r in rd:
                        yield cols, r
            else:
                from astropy.table import Table  # gwpy dependency; only for h5/fits
                tbl = Table.read(path)
                cols = list(tbl.colnames)
                for row in tbl:
                    yield cols, dict(zip(cols, row))
        return

    from gwpy.table import GravitySpyTable
    for ifo in args.ifos:
        selection = [f"ml_confidence>{args.min_confidence}", f"ifo={ifo}"]
        print(f"[fetch] GravitySpyTable '{args.table}' selection={selection}", file=sys.stderr)
        try:
            tbl = GravitySpyTable.fetch("gravityspy", args.table, selection=selection)
        except TypeError:
            # Older/newer signatures without a 'selection' kwarg: fetch then filter.
            tbl = GravitySpyTable.fetch("gravityspy", args.table)
        cols = list(tbl.colnames)
        for row in tbl:
            yield cols, dict(zip(cols, row))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="gravityspy_pool.csv", help="Output pool CSV path.")
    p.add_argument("--ifos", nargs="+", default=["H1", "L1"], help="Detectors to include.")
    p.add_argument("--labels", nargs="+", default=list(DEFAULT_GLITCH_TYPES),
                   help="Glitch classes to keep (default: pipeline DEFAULT_GLITCH_TYPES).")
    p.add_argument("--min-confidence", type=float, default=0.9,
                   help="Minimum ML confidence (0-1).")
    p.add_argument("--per-class", type=int, default=300,
                   help="Max rows per (ifo, label).")
    p.add_argument("--table", default="glitches_v2d0",
                   help="Gravity Spy DB table name (online fetch).")
    p.add_argument("--from-file", nargs="+", default=None,
                   help="Convert local Gravity Spy metadata file(s) instead of fetching "
                        "(e.g. H1_O3a.csv L1_O3a.csv). Streams CSV/TSV; astropy for h5/fits.")
    args = p.parse_args(argv)

    want_labels = set(args.labels)
    want_ifos = set(args.ifos)
    counts: Dict[tuple, int] = {}
    out_rows: List[dict] = []
    cols: Optional[list] = None
    resolved: Dict[str, Optional[str]] = {}
    get_gps = None

    try:
        for colnames, r in _iter_source(args):
            if cols is None:
                cols = list(colnames)
                for kkey in ("ifo", "label", "confidence", "snr", "peak_frequency",
                             "central_freq", "bandwidth", "duration", "gravityspy_id"):
                    resolved[kkey] = _find(cols, kkey)
                get_gps = _gps_getter(cols)
                for req in ("ifo", "label"):
                    if resolved[req] is None:
                        raise SystemExit(
                            f"Required column '{req}' not found. Available: {cols}"
                        )
            ifo = str(r[resolved["ifo"]]).strip()
            label = str(r[resolved["label"]]).strip()
            if ifo not in want_ifos or label not in want_labels:
                continue
            if resolved["confidence"] is not None and args.from_file:
                # local files are not pre-filtered server-side
                try:
                    if float(r[resolved["confidence"]]) < args.min_confidence:
                        continue
                except (TypeError, ValueError):
                    pass
            key = (ifo, label)
            if counts.get(key, 0) >= args.per_class:
                continue
            try:
                gps = get_gps(r)
            except (TypeError, ValueError):
                continue
            counts[key] = counts.get(key, 0) + 1

            def _g(name):
                col = resolved[name]
                return r[col] if col else ""
            out_rows.append({
                "gps": f"{gps:.6f}",
                "ifo": ifo,
                "label": label,
                "snr": _g("snr"),
                "peak_frequency": _g("peak_frequency"),
                "central_freq": _g("central_freq"),
                "bandwidth": _g("bandwidth"),
                "duration": _g("duration"),
                "gravityspy_id": _g("gravityspy_id"),
            })
    except SystemExit:
        raise
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency for fetch/convert: {exc}. "
            "Install gwpy (online) or astropy (local h5/fits)."
        )
    except Exception as exc:
        raise SystemExit(
            f"Failed to obtain the Gravity Spy catalog: {exc}\n"
            "If the online DB needs LVK credentials or is unreachable, download the "
            "public per-detector CSVs from Zenodo (record 5649212) and re-run with "
            "--from-file H1_*.csv L1_*.csv."
        )

    if cols is None or not out_rows:
        raise SystemExit(
            "No rows matched your --ifos / --labels filters. "
            f"Requested labels={sorted(want_labels)}, ifos={sorted(want_ifos)}."
        )

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_OUT_FIELDS)
        w.writeheader()
        w.writerows(out_rows)

    print(f"[done] wrote {len(out_rows)} rows -> {args.out}")
    by_class: Dict[tuple, int] = {}
    for r in out_rows:
        by_class[(r["ifo"], r["label"])] = by_class.get((r["ifo"], r["label"]), 0) + 1
    for (ifo, label), n in sorted(by_class.items()):
        print(f"   {ifo} {label}: {n}")


if __name__ == "__main__":
    main()
