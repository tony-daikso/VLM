"""Config loading and small shared helpers. Nothing here hard-codes organ names or HU values."""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_cfg(path: str | Path | None = None) -> dict:
    cfg = yaml.safe_load(open(path or ROOT / "config.yaml"))
    cfg["organ_map"] = yaml.safe_load(open(ROOT / cfg["organ_map_file"]))
    w = yaml.safe_load(open(ROOT / cfg["windows_file"]))
    cfg["windows"] = w["windows"]
    cfg["presets"] = w.get("presets", {})
    cfg["finding_map"] = yaml.safe_load(open(ROOT / cfg["finding_map_file"]))
    cfg["out"] = Path(cfg["output_dir"])
    for sub in ("masks", "prep", "patches", "emb", "qc", "results"):
        (cfg["out"] / sub).mkdir(parents=True, exist_ok=True)
    return cfg


def read_plans(cfg: dict) -> dict:
    """patch_size / spacing / CT normalization from the nnU-Net model folder (no network needed)."""
    folder = Path(cfg["ts_model_dir"])
    plans = json.loads((folder / "plans.json").read_text())
    c = plans["configurations"]["3d_fullres"]
    return {"patch_size": list(c["patch_size"]), "spacing": list(c["spacing"]),
            "norm": plans["foreground_intensity_properties_per_channel"]["0"],
            "n_stages": len(c["pool_op_kernel_sizes"])}


def unit_names(cfg: dict) -> list[str]:
    return list(cfg["organ_map"].keys())


def window_names(cfg: dict) -> list[str]:
    return list(cfg["windows"].keys())


def case_ids(cfg: dict) -> list[str]:
    ids = sorted(p.name[:-7] for p in Path(cfg["ct_dir"]).glob("*.nii.gz"))
    n = cfg.get("subset_size")
    return ids[:n] if n else ids


def shard(ids: list[str], spec: str | None) -> list[str]:
    if not spec:
        return ids
    i, n = map(int, spec.split("/"))
    return ids[i::n]


def log_progress(msg: str) -> None:
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(ROOT / "PROGRESS.md", "a") as f:
        f.write(f"- [{ts}] {msg}\n")
