"""Render image+mask overlay PNGs for a handful of manifest rows, for visual QA."""
import base64
import io
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(ROOT_DIR, "data", "processed")
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.jsonl")
QA_DIR = os.path.join(OUT_DIR, "qa")
os.makedirs(QA_DIR, exist_ok=True)


def load_manifest():
    rows = []
    with open(MANIFEST_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def render(row):
    volume_id = row["volume_id"]
    img = np.load(os.path.join(OUT_DIR, row["image_npy"]))
    npz = np.load(os.path.join(OUT_DIR, row["segmentation_mask_npz"]))
    mask_stack = npz["mask"]
    organ_names = list(npz["organ_names"])

    lo, hi = -600 - 1500 / 2, -600 + 1500 / 2
    disp = np.clip(img, lo, hi)
    disp = (disp - lo) / (hi - lo)
    disp = np.rot90(disp)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].imshow(disp, cmap="gray")
    axes[0].set_title(f"{volume_id}\nslice {row['slice_index']} ({row['selection_method']})", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(disp, cmap="gray")
    rng = np.random.RandomState(0)
    colors = rng.rand(len(organ_names), 3) * 0.6 + 0.4
    for i, name in enumerate(organ_names):
        m = np.rot90(mask_stack[i])
        overlay = np.zeros((*m.shape, 4))
        overlay[..., :3] = colors[i]
        overlay[..., 3] = m * 0.5
        axes[1].imshow(overlay)
    used = row.get("selection_masks_used") or []
    highlight = [n for n in organ_names if n in used]
    axes[1].set_title(f"{len(organ_names)} organs on this slice\ndriving lesion mask: {highlight}", fontsize=8)
    axes[1].axis("off")

    fig.tight_layout()
    out_path = os.path.join(QA_DIR, f"{volume_id}_qa.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    rows = load_manifest()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(rows)
    for row in rows[:n]:
        p = render(row)
        print("wrote", p)
