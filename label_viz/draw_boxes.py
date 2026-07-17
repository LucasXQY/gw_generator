"""Draw chirp/glitch boxes onto the 640x640 training spectrograms for visual
label auditing, driven by ``metadata.csv``.

Sources drawn per sample:
  - chirp box from metadata (``chirp_yolo_*``)      -> solid blue
  - glitch box(es) from the YOLO label file (cls 1) -> solid red
  - chirp box(es) from the YOLO label file (cls 0)  -> dashed cyan

The dashed-vs-solid chirp overlay is a consistency cross-check: metadata and
label file should coincide; samples where they differ beyond ``--tol`` are
flagged MISMATCH in the title, listed in the console summary, and still
rendered so you can inspect them.

Outputs one annotated PNG per sample plus a per-``global_class`` contact
sheet, into ``--out`` (default: ``label_viz/output/<dataset>_<split>/``).

Usage::

    python label_viz/draw_boxes.py --dataset datasets/gw_dataset_3000_real \
        --split test --per-class 4
    python label_viz/draw_boxes.py --dataset datasets/gw_dataset_3000_real \
        --sample-ids test_000006_H1 test_000019_L1
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

Box = Tuple[int, float, float, float, float]  # (cls, cx, cy, w, h)

CHIRP_META = (42, 120, 214)    # solid blue
CHIRP_LABEL = (64, 220, 255)   # dashed cyan
GLITCH_LABEL = (227, 73, 72)   # solid red


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def boxes_from_metadata_row(row: Dict[str, str]) -> List[Box]:
    """Chirp box recorded in metadata.csv, as [(0, cx, cy, w, h)] or []."""
    if row.get("has_chirp") != "1":
        return []
    vals = [row.get(k, "") for k in ("chirp_yolo_cx", "chirp_yolo_cy",
                                     "chirp_yolo_w", "chirp_yolo_h")]
    if any(v in ("", None) for v in vals):
        return []
    cx, cy, w, h = map(float, vals)
    return [(0, cx, cy, w, h)]


def boxes_mismatch(meta: List[Box], label: List[Box], tol: float = 0.01) -> bool:
    """True if the chirp boxes from metadata and label file disagree.

    Compares the (single) metadata chirp box against the best-matching cls-0
    label box coordinate-wise; presence/absence disagreement is a mismatch.
    """
    label_chirps = [b for b in label if b[0] == 0]
    if not meta and not label_chirps:
        return False
    if bool(meta) != bool(label_chirps):
        return True
    m = meta[0]
    best = min(label_chirps,
               key=lambda b: max(abs(b[i] - m[i]) for i in range(1, 5)))
    return max(abs(best[i] - m[i]) for i in range(1, 5)) > tol


def load_label_file(path: Path) -> List[Box]:
    if not path or not path.exists():
        return []
    out: List[Box] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) == 5:
            out.append((int(p[0]), *map(float, p[1:])))
    return out


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def _rect(draw: ImageDraw.ImageDraw, box: Box, size: Tuple[int, int],
          color: Tuple[int, int, int], width: int = 3,
          dashed: bool = False) -> None:
    _, cx, cy, w, h = box
    W, H = size
    x0, y0 = (cx - w / 2) * W, (cy - h / 2) * H
    x1, y1 = (cx + w / 2) * W, (cy + h / 2) * H
    if not dashed:
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
        return
    dash, gap = 9, 6
    edges = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
             ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
    for (ax, ay), (bx, by) in edges:
        length = max(abs(bx - ax), abs(by - ay))
        steps = max(1, int(length // (dash + gap)))
        for s in range(steps + 1):
            t0 = s * (dash + gap) / max(1.0, length)
            t1 = min(1.0, t0 + dash / max(1.0, length))
            if t0 >= 1.0:
                break
            draw.line([ax + (bx - ax) * t0, ay + (by - ay) * t0,
                       ax + (bx - ax) * t1, ay + (by - ay) * t1],
                      fill=color, width=width)


def annotate_sample(dataset_dir: Path, row: Dict[str, str],
                    tol: float) -> Tuple[Optional[Image.Image], bool]:
    """Render one sample. Returns (annotated image or None, mismatch flag)."""
    rel = row.get("qtransform_normalized_path")
    if not rel:
        return None, False
    img_path = dataset_dir / rel
    if not img_path.exists():
        return None, False
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    meta_boxes = boxes_from_metadata_row(row)
    label_boxes = load_label_file(
        dataset_dir / row["yolo_label_path"] if row.get("yolo_label_path") else None)

    for b in label_boxes:
        if b[0] == 1:
            _rect(draw, b, img.size, GLITCH_LABEL, width=3)
    for b in label_boxes:
        if b[0] == 0:
            _rect(draw, b, img.size, CHIRP_LABEL, width=2, dashed=True)
    for b in meta_boxes:
        _rect(draw, b, img.size, CHIRP_META, width=3)

    return img, boxes_mismatch(meta_boxes, label_boxes, tol)


def contact_sheet(images: List[Tuple[str, Image.Image]], cols: int = 4,
                  thumb: int = 320, pad: int = 28) -> Image.Image:
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (thumb + pad) + pad,
                              rows * (thumb + pad) + pad), (250, 250, 249))
    d = ImageDraw.Draw(sheet)
    for i, (title, im) in enumerate(images):
        r, c = divmod(i, cols)
        x = pad + c * (thumb + pad)
        y = pad + r * (thumb + pad)
        sheet.paste(im.resize((thumb, thumb)), (x, y))
        d.text((x, y + thumb + 4), title, fill=(11, 11, 11))
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True,
                    help="dataset dir (contains metadata.csv)")
    ap.add_argument("--split", default="test",
                    choices=["train", "val", "test", "all"])
    ap.add_argument("--per-class", type=int, default=4,
                    help="samples drawn per global_class (ignored with --sample-ids)")
    ap.add_argument("--sample-ids", nargs="*", default=None)
    ap.add_argument("--tol", type=float, default=0.01,
                    help="metadata-vs-label-file mismatch tolerance")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = list(csv.DictReader((args.dataset / "metadata.csv").open(encoding="utf-8")))
    if args.sample_ids:
        chosen = [r for r in rows if r["sample_id"] in set(args.sample_ids)]
    else:
        rows = [r for r in rows
                if args.split == "all" or r["split"] == args.split]
        rng = random.Random(args.seed)
        by_class = defaultdict(list)
        for r in rows:
            by_class[r["global_class"]].append(r)
        chosen = []
        for cls in sorted(by_class):
            chosen += rng.sample(by_class[cls],
                                 min(args.per_class, len(by_class[cls])))
    if not chosen:
        raise SystemExit("no samples selected")

    out = args.out or (Path(__file__).resolve().parent / "output"
                       / f"{args.dataset.name}_{args.split}")
    out.mkdir(parents=True, exist_ok=True)

    mismatches: List[str] = []
    per_class_imgs: Dict[str, List[Tuple[str, Image.Image]]] = defaultdict(list)
    for r in chosen:
        img, bad = annotate_sample(args.dataset, r, args.tol)
        if img is None:
            continue
        tag = f"{r['sample_id']}" + ("  [MISMATCH]" if bad else "")
        if bad:
            mismatches.append(r["sample_id"])
        img.save(out / f"{r['sample_id']}.png")
        per_class_imgs[r["global_class"]].append((tag, img))

    for cls, imgs in per_class_imgs.items():
        contact_sheet(imgs).save(out / f"_sheet_{cls}.png")

    n = sum(len(v) for v in per_class_imgs.values())
    print(f"annotated {n} samples -> {out}")
    print("legend: solid blue = chirp (metadata) | dashed cyan = chirp "
          "(label file) | solid red = glitch (label file)")
    print(f"metadata-vs-label mismatches: {len(mismatches)}"
          + (f"  {mismatches}" if mismatches else ""))


if __name__ == "__main__":
    main()
