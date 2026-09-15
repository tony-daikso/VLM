"""Render image+mask overlay PNGs for ReXGroundingCT-pipeline manifest rows, for visual QA."""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(ROOT_DIR, "data", "processed_rex")
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
    findings = row["findings_on_slice"]

    lo, hi = -600 - 1500 / 2, -600 + 1500 / 2
    disp = np.clip(img, lo, hi)
    disp = (disp - lo) / (hi - lo)
    disp = np.rot90(disp)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(disp, cmap="gray")
    axes[0].set_title(f"{volume_id}\nslice {row['slice_index']} (grid {row['grid_shape'][:2]})", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(disp, cmap="gray")
    rng = np.random.RandomState(0)
    colors = rng.rand(len(findings), 3) * 0.6 + 0.4
    labels = []
    for c, m2d, f in zip(colors, mask_stack, findings):
        m = np.rot90(m2d > 0)
        overlay = np.zeros((*m.shape, 4))
        overlay[..., :3] = c
        overlay[..., 3] = m * 0.55
        axes[1].imshow(overlay)
        labels.append(f"[{f['index']}] {f['text'][:50]}")
    axes[1].set_title("\n".join(labels) or "(no findings on this slice)", fontsize=7)
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
