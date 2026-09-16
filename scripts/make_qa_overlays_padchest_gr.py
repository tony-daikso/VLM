"""Render image+box overlay PNGs for a PadChest-GR-style manifest, for visual QA."""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_OUT_DIR = os.path.join(ROOT_DIR, "data", "X-ray", "PadChest-GR", "processed")


def load_manifest(out_dir):
    rows = []
    with open(os.path.join(out_dir, "manifest.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def render(row, out_dir, qa_dir):
    study_id = row["study_id"]
    img = Image.open(os.path.join(out_dir, row["image_relpath"]))
    w, h = img.size

    # Images are 16-bit ("I;16" mode, values up to ~65519), not display-ready
    # 8-bit -- Image.convert("L") on this mode just clips instead of rescaling
    # and produces a washed-out, near-white image. Percentile-stretch instead.
    arr = np.array(img).astype(np.float32)
    lo, hi = np.percentile(arr, [0.5, 99.5])
    disp = np.clip(arr, lo, hi)
    disp = (disp - lo) / max(hi - lo, 1e-6)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(disp, cmap="gray")

    rng_colors = ["#ff4040", "#40c0ff", "#40ff80", "#ffd040", "#c060ff", "#ff8040"]
    labels = []
    for i, finding in enumerate(row["findings"]):
        color = rng_colors[i % len(rng_colors)]
        for box in finding["boxes"]:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1 * w, y1 * h), (x2 - x1) * w, (y2 - y1) * h, linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
        tag = "abnormal" if finding["abnormal"] else "normal"
        labels.append(f"[{tag}] {(finding['sentence_en'] or '')[:60]}")

    ax.set_title(f"{study_id} ({row['split']})\n" + "\n".join(labels[:6]), fontsize=7)
    ax.axis("off")

    fig.tight_layout()
    out_path = os.path.join(qa_dir, f"{study_id}_qa.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int, nargs="?", default=20, help="how many studies to render")
    parser.add_argument("--data-dir", default=DEFAULT_OUT_DIR, help="a processed/ dir with manifest.jsonl + images/")
    args = parser.parse_args()

    qa_dir = os.path.join(args.data_dir, "qa")
    os.makedirs(qa_dir, exist_ok=True)

    rows = load_manifest(args.data_dir)
    for row in rows[: args.n]:
        p = render(row, args.data_dir, qa_dir)
        print("wrote", p)
