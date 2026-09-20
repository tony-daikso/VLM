"""One-off: visualize the current checkpoint_unet_xray.pth's predicted mask
on a validation-set sample, next to the CheXmask ground-truth mask."""
import io
import json
import os
import random
import sys
import tarfile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_RADAR_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lavis", "models", "radar_models")
sys.path.insert(0, _RADAR_MODELS_DIR)
from vision_branch import VisionBranch

ROOT_DIR = "/root/Desktop/VLM/data/X-ray/PadChest-Origin"
TAR_PATH = os.path.join(ROOT_DIR, "PNG_tar", "PADCHEST_PA_AP.tar")
MASKS_TAR_PATH = os.path.join(ROOT_DIR, "processed_chexmask", "masks.tar")
SPLIT_PATH = "/root/Desktop/VLM/RADAR/X-ray/data_split.json"
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pred_vis.png")

COLORS = {1: (255, 80, 80, 110), 2: (80, 160, 255, 110), 3: (80, 220, 120, 140)}


def percentile_normalize_uint8(arr, lo_pct=0.5, hi_pct=99.5):
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    disp = np.clip(arr, lo, hi)
    disp = (disp - lo) / max(hi - lo, 1e-6)
    return (disp * 255.0).astype(np.uint8)


def overlay(base_img, mask):
    overlay_px = np.zeros((*mask.shape, 4), dtype=np.uint8)
    for value, color in COLORS.items():
        overlay_px[mask == value] = color
    overlay_rgba = Image.fromarray(overlay_px, mode="RGBA")
    return Image.alpha_composite(base_img.convert("RGBA"), overlay_rgba).convert("RGB")


def main():
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    random.seed(0)
    image_id = random.choice(split["val_image_ids"])
    print("sample:", image_id)

    with tarfile.open(TAR_PATH) as tf:
        member = next(m for m in tf.getmembers() if os.path.basename(m.name) == image_id)
        raw = np.array(Image.open(io.BytesIO(tf.extractfile(member).read())))
    with tarfile.open(MASKS_TAR_PATH) as tf:
        member = next(m for m in tf.getmembers() if os.path.basename(m.name) == image_id)
        gt_mask = np.array(Image.open(io.BytesIO(tf.extractfile(member).read())))

    img8 = percentile_normalize_uint8(raw)
    img_512 = Image.fromarray(img8).resize((512, 512), Image.BILINEAR)
    image_t = torch.from_numpy(np.array(img_512, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vb = VisionBranch().to(device)
    vb.eval()
    with torch.no_grad():
        _, segs = vb.UNet(image_t.to(device))
        seg_probs = torch.softmax(segs[0], 1)
        pred = F.interpolate(seg_probs, size=(512, 512), mode="bilinear", align_corners=False)
        pred_mask = pred.argmax(1)[0].cpu().numpy().astype(np.uint8)

    base_rgb = img_512.convert("RGB")
    gt_overlay = overlay(base_rgb, gt_mask)
    pred_overlay = overlay(base_rgb, pred_mask)

    canvas = Image.new("RGB", (512 * 3, 512))
    canvas.paste(base_rgb, (0, 0))
    canvas.paste(gt_overlay, (512, 0))
    canvas.paste(pred_overlay, (1024, 0))
    canvas.save(OUT_PATH)
    print("saved", OUT_PATH, "(left=original, middle=CheXmask ground truth, right=model prediction)")


if __name__ == "__main__":
    main()
