"""Frozen nnU-Net / TotalSegmentator encoder behind a small interface.

Input : normalized patch tensor [B, 1, Z, Y, X] (nnU-Net array order, see NibabelIOWithReorient)
Output: list of per-stage feature maps (encoder skips), highest resolution first.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


class FrozenEncoder:
    """Abstract interface so a different nnU-Net / self-trained encoder can be swapped in."""

    patch_size: tuple[int, int, int]
    spacing: tuple[float, float, float]
    norm: dict  # percentile_00_5, percentile_99_5, mean, std

    def stage_features(self, x: torch.Tensor, upto: int | None = None) -> list[torch.Tensor]:
        raise NotImplementedError

    def full_network(self) -> torch.nn.Module:
        raise NotImplementedError


class NnUNetEncoder(FrozenEncoder):
    def __init__(self, model_folder: str | Path, fold: int = 0, checkpoint: str = "checkpoint_final.pth",
                 device: str = "cuda"):
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        self.model_folder = Path(model_folder)
        self.device = torch.device(device)
        pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                               perform_everything_on_device=True, device=self.device,
                               verbose=False, allow_tqdm=False)
        pred.initialize_from_trained_model_folder(str(self.model_folder), use_folds=(fold,),
                                                  checkpoint_name=checkpoint)
        self.predictor = pred
        net = pred.network
        # predictor stores weights in list_of_parameters; load fold 0 into the network
        net.load_state_dict(pred.list_of_parameters[0])
        net.to(self.device).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.network = net
        self.encoder = net.encoder
        if not getattr(self.encoder, "return_skips", False):
            raise RuntimeError("encoder does not return skips; need forward hooks")

        plans = json.loads((self.model_folder / "plans.json").read_text())
        cfg = plans["configurations"]["3d_fullres"]
        self.patch_size = tuple(cfg["patch_size"])
        self.spacing = tuple(cfg["spacing"])
        self.norm = plans["foreground_intensity_properties_per_channel"]["0"]
        self.labels = json.loads((self.model_folder / "dataset.json").read_text())["labels"]

    @torch.no_grad()
    def stage_features(self, x: torch.Tensor, upto: int | None = None) -> list[torch.Tensor]:
        """per-stage outputs; with upto=s only stages 0..s are computed (skip the unused deeper stages)."""
        x = x.to(self.device)
        if upto is None:
            return self.encoder(x)
        if getattr(self.encoder, "stem", None) is not None:
            x = self.encoder.stem(x)
        out = []
        for st in self.encoder.stages[:upto + 1]:
            x = st(x)
            out.append(x)
        return out

    def full_network(self) -> torch.nn.Module:
        return self.network

    def stage_channels(self, stages) -> list[int]:
        return [int(self.encoder.output_channels[s]) for s in stages]


def build_encoder(cfg: dict) -> FrozenEncoder:
    kind = cfg.get("encoder_type", "nnunet")
    if kind == "nnunet":
        return NnUNetEncoder(cfg["ts_model_dir"], device=cfg.get("device", "cuda"))
    raise ValueError(kind)


if __name__ == "__main__":
    import sys, time
    enc = NnUNetEncoder(sys.argv[1])
    print("patch", enc.patch_size, "spacing", enc.spacing, "norm", enc.norm)
    x = torch.randn(1, 1, *enc.patch_size)
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    with torch.autocast("cuda", dtype=torch.float16):
        feats = enc.stage_features(x)
    torch.cuda.synchronize()
    for i, f in enumerate(feats):
        print(f"stage {i}: {tuple(f.shape)}")
    print(f"time {time.time()-t:.2f}s peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    print("trainable params:", sum(p.numel() for p in enc.network.parameters() if p.requires_grad))
