"""
RADAR X-ray Phase 5: assemble the full RadarPretrain model (VisionBranch +
XBertEncoder text encoder + RADAR+ anatomy/whole-image ITC) and run a
small-scale verification pass -- confirm loss_seg/loss_itc actually trend
down on real data before committing to a full 96k-image training run.

Does NOT go through LAVIS's registry/from_config/Runner (see
RADAR/X-ray/phase3/README.md and phase4/dataset.py for why this track avoids
the full LAVIS import chain): constructs RadarPretrain directly via its
__init__, mirroring radar_config.yaml's values but skipping from_config's
`radar_ft` path (which would try to load the CT track's own
checkpoint_radar_pretrain.pth -- not applicable here, different organs/
vision encoder).

Two import-time compatibility issues had to be worked around (both isolated
to this file, vendored lavis/ code under phase3/ is untouched):
  1. `lavis/__init__.py` eagerly imports the full LAVIS framework (decord,
     fairscale, ...) that this plain-Dataset/plain-model construction
     doesn't need. Worked around with a stub `lavis` module in sys.modules
     carrying just the real package's __path__, so `lavis.models.med`'s own
     `from lavis.common.utils import ...`-style imports still resolve
     without lavis/__init__.py actually executing -- except
     `registry.register_path("library_root", ...)`, which IS needed (used
     by `get_abs_path` inside XBertEncoder.from_config), so it's replicated
     by hand below.
  2. `lavis/models/med.py` was vendored against `transformers==4.25`.
     Confirmed this repo's own RADAR/CT/requirements.txt already pins that
     exact version -- installed it here too (this shared env otherwise had
     a much newer transformers, incompatible with this code's BertModel
     subclassing). No shim needed once the right version is installed.

Usage:
    cd RADAR/X-ray/phase5
    python3 train.py [--limit 500] [--steps 100] [--batch-size 8] [--lr 1e-4]
"""
import argparse
import os
import sys
import types

import numpy as np
import torch

# --- bypass lavis/__init__.py's heavy import chain (see module docstring) ---
_PHASE3_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase3")
_LAVIS_DIR = os.path.join(_PHASE3_DIR, "lavis")
sys.path.insert(0, _PHASE3_DIR)

_fake_lavis = types.ModuleType("lavis")
_fake_lavis.__path__ = [os.path.abspath(_LAVIS_DIR)]
sys.modules["lavis"] = _fake_lavis

from lavis.common.registry import registry  # noqa: E402
registry.register_path("library_root", os.path.abspath(_LAVIS_DIR))

from lavis.models.med import XBertEncoder  # noqa: E402
from lavis.models.radar_models.radar_pretrain import RadarPretrain  # noqa: E402

sys.path.insert(0, os.path.join(_LAVIS_DIR, "models", "radar_models"))
from vision_branch import VisionBranch  # noqa: E402

_PHASE4_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase4")
sys.path.insert(0, _PHASE4_DIR)
from dataset import RadarXrayDataset  # noqa: E402

from torch.utils.data import DataLoader, Subset


def build_model(device):
    image_encoder = VisionBranch()  # auto-loads Phase 3's checkpoint_unet_xray.pth if present
    text_encoder = XBertEncoder.from_config({"med_config_path": "unused"}, from_pretrained=True)
    model = RadarPretrain(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        text_decoder=None,
        queue_size=0,       # matches radar_config.yaml
        alpha=0.4,
        embed_dim=256,
        momentum=0.995,
        tie_enc_dec_weights=False,
        max_txt_len=512,
        radar_plus=True,
    )
    return model.to(device)


CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ckpt")
CKPT_OUT_PATH = os.path.join(CKPT_DIR, "checkpoint_radar_pretrain_xray.pth")


def get_loss_itc_value(output):
    # on an anatomy-wise step, radar_pretrain.py returns loss_itc as a plain
    # python 0 (not a tensor) when zero organs in this batch pass its
    # intact+abnormal-flagged filter -- more likely with small batch sizes.
    # backward() still works fine (0 + tensor promotes normally), just guard
    # our own .item() calls.
    return output.loss_itc.item() if isinstance(output.loss_itc, torch.Tensor) else float(output.loss_itc)


def run_step(model, optimizer, batch, device, global_step):
    batch["image"] = batch["image"].to(device)
    batch["seg"] = batch["seg"].to(device)
    batch["organ_abnormal_flags"] = batch["organ_abnormal_flags"].to(device)
    batch["iters"] = global_step  # RADAR+ alternation (radar_pretrain.py:153) -- the training
                                   # loop's job, not the dataset's (see phase4/dataset.py docstring)

    output = model(batch)
    loss = output.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), output.loss_seg.item(), get_loss_itc_value(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500, help="cap the train dataset to N records (quick-verify mode only)")
    parser.add_argument("--steps", type=int, default=100, help="quick-verify mode: run this many steps total, not full epochs")
    parser.add_argument("--epochs", type=int, default=None, help="full-training mode: run this many passes over the WHOLE train split, checkpointing every epoch")
    parser.add_argument("--batch-size", type=int, default=8)  # measured fastest -- see phase5/README.md, larger batches are slower here
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = build_model(device)
    print(f"RadarPretrain params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    ds = RadarXrayDataset(split="train")
    if args.epochs is None and args.limit and args.limit < len(ds):
        idx = np.random.RandomState(args.seed).choice(len(ds), size=args.limit, replace=False)
        ds = Subset(ds, idx.tolist())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    print(f"train set: {len(ds)} records, {len(loader)} batches/epoch at batch_size={args.batch_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    model.train()

    if args.epochs is not None:
        os.makedirs(CKPT_DIR, exist_ok=True)
        n_batches = len(loader)
        log_every = max(1, n_batches // 20)  # ~20 progress lines per epoch
        global_step = 0
        for epoch in range(args.epochs):
            epoch_losses, epoch_seg, epoch_itc = [], [], []
            for i, batch in enumerate(loader):
                loss_val, seg_val, itc_val = run_step(model, optimizer, batch, device, global_step)
                epoch_losses.append(loss_val)
                epoch_seg.append(seg_val)
                epoch_itc.append(itc_val)
                if (i + 1) % log_every == 0:
                    print(f"  epoch {epoch+1}/{args.epochs} batch {i+1}/{n_batches}: "
                          f"running_loss={np.mean(epoch_losses[-log_every:]):.4f} "
                          f"running_seg={np.mean(epoch_seg[-log_every:]):.4f} "
                          f"running_itc={np.mean(epoch_itc[-log_every:]):.4f}")
                global_step += 1
            print(f"epoch {epoch+1}/{args.epochs}: loss={np.mean(epoch_losses):.4f} "
                  f"loss_seg={np.mean(epoch_seg):.4f} loss_itc={np.mean(epoch_itc):.4f}")
            torch.save({"model": model.state_dict()}, CKPT_OUT_PATH)
            print(f"saved {CKPT_OUT_PATH} (epoch {epoch+1})")
        return

    # quick-verify mode (original behavior): fixed step count over a capped subset
    step = 0
    loader_iter = iter(loader)
    history = []
    while step < args.steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        loss_val, seg_val, itc_val = run_step(model, optimizer, batch, device, step)
        branch = "whole" if step % 2 == 0 else "anatomy"
        history.append((step, loss_val, seg_val, itc_val, branch))
        print(f"step {step:4d} [{branch:7s}] loss={loss_val:.4f} loss_seg={seg_val:.4f} loss_itc={itc_val:.4f}")
        step += 1

    print("\ndone. loss_seg / loss_itc by branch, first vs last 10 steps:")
    seg_first10 = np.mean([h[2] for h in history[:10]])
    seg_last10 = np.mean([h[2] for h in history[-10:]])
    print(f"  loss_seg: first10={seg_first10:.4f} last10={seg_last10:.4f}")
    for b in ("whole", "anatomy"):
        itc_first = [h[3] for h in history if h[4] == b][:5]
        itc_last = [h[3] for h in history if h[4] == b][-5:]
        if itc_first and itc_last:
            print(f"  loss_itc[{b}]: first5={np.mean(itc_first):.4f} last5={np.mean(itc_last):.4f}")


if __name__ == "__main__":
    main()
