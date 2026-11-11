"""Option B (arm P_gt): Pillar-0 features as LLM image tokens for Merlin method 4.

Image block = [256 global tokens | 52 unit tokens] inside the usual <|vision_start|> ... <|vision_end|> (order fixed).
  global : 256 learnable queries attention-pool (2 Perceiver layers, m4_model.ResamplerLayer) over the 4096 grid16 tokens
           (LayerNorm + Linear 384 -> 1024, + fixed 3-D sinusoidal position of the (z, y, x) cell); grid cells entirely in
           the z padding are masked; an always-valid null key avoids empty attention -> LayerNorm -> MLP -> LLM hidden.
  unit   : one token per pillar_units.UNITS entry (OWL unit order): LayerNorm(vec16 | vec64) -> Linear, + Linear(info)
           + learned unit-type embedding; a unit with no voxels in the crop uses a learned missing vector (+ type
           embedding) -> LayerNorm -> MLP -> LLM hidden.
  modes  : both (P_gt, 308 tokens) | global (256) | unit (52)  -- --pillar-mode, ablations only.
Everything else (prompt, cue switches, LoRA recipe, saving) is the unchanged method-4 code: train_p.py / infer_p.py
patch train.py / infer.py (ImageAdapter, read_token_meta, M4Dataset, N_IMG_TOKENS) and then call their main().

Data path: PillarStore opens the pack_pillar.py arrays with np.load(mmap_mode="r") lazily in each process (DataLoader
workers included) and copies one row per item -> RAM stays at a few MB per batch. The dataset hands the adapter
(tok, mask) = (grid [4096, 385] fp16: 384 features + z-valid flag, units [52, 771] fp16: 768 features + 3 info) through
the existing collate / ReportGenModel code (the second tensor is NOT a mask in this arm).
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from m4_data import M4Dataset  # noqa: E402
from m4_model import ResamplerLayer  # noqa: E402
from pillar_units import GRID, INFO_DIM, N_UNITS, UNIT_DIM, UNIT_NAMES  # noqa: E402

GRID_DIM = 384
MODES = ("both", "global", "unit")


@dataclass
class PillarCfg:
    root: str = ""
    mode: str = "both"
    n_global: int = 256
    allow_missing: bool = True        # a case without a packed row -> all-missing image (counted, printed)

    @property
    def n_tokens(self):
        return (self.n_global if self.mode in ("both", "global") else 0) + (N_UNITS if self.mode in ("both", "unit") else 0)

    def to_json(self):
        return {**asdict(self), "n_tokens": self.n_tokens, "n_units": N_UNITS, "unit_names": UNIT_NAMES,
                "token_order": "global then units"}


CFG = PillarCfg()                     # set by train_p.py / infer_p.py before main(); forked workers inherit it


# ---------------------------------------------------------------------------------------------------- data
class PillarStore:
    def __init__(self, root):
        self.root = Path(root)
        self.ids = (self.root / "ids.txt").read_text().split()
        self.index = {c: i for i, c in enumerate(self.ids)}
        self._a = None

    def arrays(self):
        if self._a is None:                                     # lazily, once per process
            self._a = {k: np.load(self.root / f"{k}.npy", mmap_mode="r")
                       for k in ("grid16", "units", "uinfo", "zvalid", "done")}
        return self._a

    def has(self, cid):
        i = self.index.get(cid)
        return i is not None and bool(self.arrays()["done"][i])

    def row(self, cid, need_grid=True):
        """-> (grid [4096, 385] fp16 or None, units [52, 771] fp16); missing case -> (zeros, all-missing units)."""
        a = self.arrays()
        i = self.index.get(cid)
        ok = i is not None and bool(a["done"][i])
        units = np.zeros((N_UNITS, UNIT_DIM + INFO_DIM), np.float16)
        grid = np.zeros((GRID ** 3, GRID_DIM + 1), np.float16) if need_grid else None
        if ok:
            units[:, :UNIT_DIM] = a["units"][i]
            units[:, UNIT_DIM:] = a["uinfo"][i]
            if need_grid:
                grid[:, :GRID_DIM] = a["grid16"][i]
                z0, z1 = (int(v) for v in a["zvalid"][i])
                zc = np.arange(GRID ** 3) // (GRID * GRID)
                grid[:, GRID_DIM] = (zc >= z0) & (zc <= z1)
        return grid, units, ok


class PDataset(M4Dataset):
    """M4Dataset with Pillar features; same prompt / cue / target handling (n_img = CFG.n_tokens)."""

    def __init__(self, *a, **kw):
        kw["n_img"] = CFG.n_tokens
        if kw.get("lr_flip_prob") or kw.get("unit_labels"):
            raise SystemExit("P arm: --lr-flip-prob / --aux-unit-loss are not implemented for Pillar tokens")
        kw["meta_rows"] = None
        super().__init__(*a, **kw)
        if not self.use_image:
            return
        self.store = PillarStore(CFG.root)
        miss = [r["case_id"] for r in self.rows if not self.store.has(r["case_id"])]
        self.store._a = None                                    # re-open lazily in each worker
        print(f"[pillar] {len(self.rows)} rows, {len(miss)} without Pillar features"
              + (f" (e.g. {miss[:3]})" if miss else ""), flush=True)
        if miss and not CFG.allow_missing:
            raise SystemExit("cases without Pillar features (pass --pillar-allow-missing to use all-missing images)")
        self.n_missing = len(miss)

    def features(self, r):
        grid, units, _ = self.store.row(r["case_id"], need_grid=CFG.mode != "unit")
        if grid is None:
            grid = np.zeros((1, GRID_DIM + 1), np.float16)
        return torch.from_numpy(grid), torch.from_numpy(units)


# ---------------------------------------------------------------------------------------------------- model
def sincos_3d(n, dim):
    """fixed 3-D sinusoidal position code of an n^3 raster (z, y, x) -> [n^3, dim] (dim // 6 frequencies per axis)."""
    k = dim // 6
    freq = torch.exp(-math.log(10000.0) * torch.arange(k) / k)
    idx = torch.arange(n ** 3)
    parts = []
    for c in (idx // (n * n), (idx // n) % n, idx % n):
        ang = c[:, None].float() * freq[None]
        parts += [ang.sin(), ang.cos()]
    pe = torch.cat(parts, 1)
    return torch.cat([pe, torch.zeros(n ** 3, dim - pe.shape[1])], 1)


def _mlp(dim, hidden):
    return nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))


class PillarAdapter(nn.Module):
    def __init__(self, llm_hidden, mode="both", n_global=256, dim=1024, depth=2, heads=8, ff_mult=4):
        super().__init__()
        assert mode in MODES
        self.mode, self.n_global = mode, n_global
        self.use_g, self.use_u = mode in ("both", "global"), mode in ("both", "unit")
        if self.use_g:
            self.g_inp = nn.Sequential(nn.LayerNorm(GRID_DIM), nn.Linear(GRID_DIM, dim))
            self.register_buffer("pos", sincos_3d(GRID, dim), persistent=False)
            self.null = nn.Parameter(torch.zeros(1, 1, dim))
            self.latents = nn.Parameter(torch.randn(n_global, dim) * 0.02)
            self.layers = nn.ModuleList(ResamplerLayer(dim, heads, ff_mult) for _ in range(depth))
            self.g_ln = nn.LayerNorm(dim)
            self.g_proj = _mlp(dim, llm_hidden)
        if self.use_u:
            self.u_inp = nn.Sequential(nn.LayerNorm(UNIT_DIM), nn.Linear(UNIT_DIM, dim))
            self.u_info = nn.Linear(INFO_DIM, dim)
            self.u_type = nn.Parameter(torch.randn(N_UNITS, dim) * 0.02)
            self.u_missing = nn.Parameter(torch.randn(dim) * 0.02)
            self.u_ln = nn.LayerNorm(dim)
            self.u_proj = _mlp(dim, llm_hidden)
        self.n_out = (n_global if self.use_g else 0) + (N_UNITS if self.use_u else 0)
        self.last_aux_logits = None

    def forward_global(self, grid):
        B = grid.shape[0]
        g = grid.float()
        x = self.g_inp(g[..., :GRID_DIM]) + self.pos[None]
        x = torch.cat([self.null.expand(B, 1, -1), x], 1)
        key_pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=grid.device), g[..., GRID_DIM] < 0.5], 1)
        lat = self.latents[None].expand(B, -1, -1)
        for layer in self.layers:
            lat = layer(lat, x, key_pad)
        return self.g_proj(self.g_ln(lat))

    def forward_units(self, units):
        u = units.float()
        feat, info = u[..., :UNIT_DIM], u[..., UNIT_DIM:]
        present = info[..., :1] > 0.5
        x = torch.where(present, self.u_inp(feat) + self.u_info(info), self.u_missing.expand_as(self.u_type)[None])
        return self.u_proj(self.u_ln(x + self.u_type[None]))

    def forward(self, grid, units):
        out = []
        if self.use_g:
            out.append(self.forward_global(grid))
        if self.use_u:
            out.append(self.forward_units(units))
        return torch.cat(out, 1)


def adapter_factory(meta, llm_hidden, dim=1024, n_latents=None, depth=2, heads=8, aux_unit=False, groups=False, **_):
    """drop-in for m4_model.ImageAdapter(meta, hidden, dim=, n_latents=, depth=, heads=, aux_unit=, groups=)."""
    if aux_unit or groups:
        raise SystemExit("P arm: --aux-unit-loss / --resampler-groups are OWL-token options")
    return PillarAdapter(llm_hidden, CFG.mode, CFG.n_global, dim=dim, depth=depth, heads=heads)


def patch(module):
    """make a train.py / infer.py module build the Pillar arm."""
    module.ImageAdapter = adapter_factory
    module.read_token_meta = lambda path: None
    module.M4Dataset = PDataset
    module.N_IMG_TOKENS = CFG.n_tokens


def write_cfg(out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    root = Path(CFG.root)
    info = CFG.to_json()
    if (root / "ids.txt").exists():
        import hashlib
        info["store_ids_md5"] = hashlib.md5((root / "ids.txt").read_bytes()).hexdigest()
    if (root / "pack_meta.json").exists():
        pm = json.loads((root / "pack_meta.json").read_text())
        info["store_n"], info["store_n_done"] = pm.get("n"), pm.get("n_done")
    (Path(out_dir) / "pillar.json").write_text(json.dumps(info, indent=1))
