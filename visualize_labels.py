"""Overlay YOLO bounding-box labels onto the *display* Q-transform images.

The YOLO labels in ``labels_yolo/`` are normalized to the pure 640x640
spectrogram (``qtransform_normalized/``). The *display* images
(``qtransform_display_normalized/``) are larger (e.g. 832x640) because
matplotlib added axis ticks, ``Time``/``Frequency`` labels and a colorbar
around that 640x640 plot. Drawing the boxes directly with the display image's
full width/height would therefore misplace them.

This script reconstructs the exact pixel rectangle of the plot (data) region
inside the display figure by rebuilding the same matplotlib layout used in
``qtransform.QTransformRenderer._save_display`` (identical figsize, dpi, axis
labels, limits, scale and colorbar) and reading ``ax.get_position()``. The
layout depends only on the dataset config -- not on per-image energy values --
so the rectangle is computed once and reused for every image.

Boxes are then mapped linearly into that rectangle (both the train image and
the display axes have *high frequency at the top*, so the YOLO y-axis maps
straight through), drawn with PIL, and saved mirroring the
``<split>/<detector>/`` folder structure.

Usage::

    python visualize_labels.py --dataset gw_dataset_1000 --out boxed_labels
    python visualize_labels.py --dataset gw_dataset_2500 --out /path/out \
        --splits train val --detectors H1
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

# Per-class box colours (RGB). Falls back to a cycle for unknown class ids.
_CLASS_COLORS = [
    (255, 64, 64),    # class 0 -> red
    (64, 200, 255),   # class 1 -> cyan
    (64, 255, 96),    # class 2 -> green
    (255, 200, 32),   # class 3 -> amber
]


def _load_config(dataset_dir: Path) -> dict:
    cfg_path = dataset_dir / "dataset_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"{cfg_path} not found -- needed to reconstruct the display layout."
        )
    with cfg_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_class_names(dataset_dir: Path) -> List[str]:
    """Read class names from gw_data.yaml without requiring PyYAML."""
    yaml_path = dataset_dir / "gw_data.yaml"
    if not yaml_path.exists():
        return []
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("names:"):
            value = line.split(":", 1)[1].strip()
            # Format: names: ['chirp', 'glitch']
            value = value.strip("[]")
            names = [n.strip().strip("'\"") for n in value.split(",") if n.strip()]
            return names
    return []


def compute_plot_rect(cfg: dict) -> Tuple[float, float, float, float, int, int]:
    """Return the data-region rectangle (left, top, width, height) in display
    pixels, plus the full display image (width, height) in pixels.

    Rebuilds the exact figure layout from ``QTransformRenderer._save_display``.
    """
    width = int(cfg["qtransform_image_width"])
    height = int(cfg["qtransform_image_height"])
    dpi = int(cfg["qtransform_display_dpi"])
    duration = float(cfg["duration"])
    flow = float(cfg["frange_low"])
    fhigh = float(cfg["frange_high"])
    scale = str(cfg["frequency_axis_scale"])
    vmin = float(cfg["energy_vmin"])
    vmax = float(cfg["energy_vmax"])

    fig = plt.figure(figsize=(width / dpi * 1.3, height / dpi), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)

    # Dummy mesh spanning the real limits -- tick layout depends on the limits
    # and labels, not on the energy values, so this reproduces the exact axes.
    t_edges = np.array([0.0, duration / 2.0, duration])
    if scale == "log":
        lo = max(flow, 1e-6)
        f_edges = np.array([lo, (lo * fhigh) ** 0.5, fhigh])
    else:
        f_edges = np.array([flow, (flow + fhigh) / 2.0, fhigh])
    dummy = np.zeros((2, 2))

    mesh = ax.pcolormesh(
        t_edges, f_edges, dummy, cmap="viridis", vmin=vmin, vmax=vmax, shading="auto"
    )
    ax.set_yscale(scale if scale == "log" else "linear")
    ax.set_ylim(flow, fhigh)
    ax.set_xlim(0.0, duration)
    ax.set_xlabel("Time [secs]")
    ax.set_ylabel("Frequency [Hz]")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Normalized energy")
    fig.tight_layout()
    fig.canvas.draw()

    fig_w_px = width / dpi * 1.3 * dpi  # == figsize_w_inches * dpi
    fig_h_px = height / dpi * dpi
    pos = ax.get_position()  # figure-fraction bbox, y from bottom
    plt.close(fig)

    left = pos.x0 * fig_w_px
    right = pos.x1 * fig_w_px
    top = (1.0 - pos.y1) * fig_h_px  # convert bottom-origin -> top-origin
    bottom = (1.0 - pos.y0) * fig_h_px
    return left, top, (right - left), (bottom - top), round(fig_w_px), round(fig_h_px)


def _parse_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    boxes: List[Tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = (float(p) for p in parts[1:])
        boxes.append((cls, xc, yc, w, h))
    return boxes


def _draw_boxes(
    image: Image.Image,
    boxes: List[Tuple[int, float, float, float, float]],
    rect: Tuple[float, float, float, float],
    class_names: List[str],
    font: ImageFont.ImageFont,
) -> Image.Image:
    left, top, rect_w, rect_h = rect
    img = image.convert("RGBA")
    draw = ImageDraw.Draw(img)

    for cls, xc, yc, w, h in boxes:
        color = _CLASS_COLORS[cls % len(_CLASS_COLORS)]
        cx = left + xc * rect_w
        cy = top + yc * rect_h
        bw = w * rect_w
        bh = h * rect_h
        x0, y0 = cx - bw / 2.0, cy - bh / 2.0
        x1, y1 = cx + bw / 2.0, cy + bh / 2.0
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)

        name = class_names[cls] if cls < len(class_names) else str(cls)
        tb = draw.textbbox((0, 0), name, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = max(y0 - th - 3, 0)
        draw.rectangle([x0, ty, x0 + tw + 4, ty + th + 3], fill=color)
        draw.text((x0 + 2, ty + 1), name, fill=(0, 0, 0, 255), font=font)

    return img


def process_dataset(
    dataset_dir: Path,
    out_dir: Path,
    splits: List[str],
    detectors: List[str],
) -> None:
    cfg = _load_config(dataset_dir)
    class_names = _load_class_names(dataset_dir)
    left, top, rect_w, rect_h, fig_w, fig_h = compute_plot_rect(cfg)
    rect = (left, top, rect_w, rect_h)
    print(
        f"Display plot region: left={left:.1f} top={top:.1f} "
        f"w={rect_w:.1f} h={rect_h:.1f} (figure {fig_w}x{fig_h})"
    )

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    display_root = dataset_dir / "qtransform_display_normalized"
    label_root = dataset_dir / "labels_yolo"

    if not display_root.exists():
        raise FileNotFoundError(f"{display_root} not found.")

    total, boxed, copied = 0, 0, 0
    for split in splits:
        split_dir = display_root / split
        if not split_dir.is_dir():
            continue
        for det in detectors:
            det_dir = split_dir / det
            if not det_dir.is_dir():
                continue
            out_det = out_dir / split / det
            out_det.mkdir(parents=True, exist_ok=True)
            for img_path in sorted(det_dir.glob("*.png")):
                total += 1
                label_path = label_root / split / det / (img_path.stem + ".txt")
                boxes = _parse_labels(label_path)
                out_path = out_det / img_path.name
                if not boxes:
                    shutil.copyfile(img_path, out_path)
                    copied += 1
                    continue
                with Image.open(img_path) as im:
                    annotated = _draw_boxes(im, boxes, rect, class_names, font)
                annotated.convert("RGB").save(out_path)
                boxed += 1

    print(
        f"Done: {total} images -> {out_dir} "
        f"({boxed} with boxes, {copied} copied unchanged)."
    )


def _discover(root: Path, requested: List[str]) -> List[str]:
    if requested:
        return requested
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="gw_dataset_1000", help="Path to the dataset directory."
    )
    parser.add_argument(
        "--out", required=True, help="Output directory for annotated images."
    )
    parser.add_argument(
        "--splits", nargs="*", default=None,
        help="Splits to process (default: all found, e.g. train val test).",
    )
    parser.add_argument(
        "--detectors", nargs="*", default=None,
        help="Detectors to process (default: all found, e.g. H1 L1).",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    out_dir = Path(args.out)
    display_root = dataset_dir / "qtransform_display_normalized"

    splits = args.splits or _discover(display_root, [])
    # Detectors are discovered per-split; default to those under the first split.
    detectors = args.detectors
    if not detectors:
        for split in splits:
            detectors = _discover(display_root / split, [])
            if detectors:
                break
        detectors = detectors or []

    process_dataset(dataset_dir, out_dir, splits, detectors)


if __name__ == "__main__":
    main()
