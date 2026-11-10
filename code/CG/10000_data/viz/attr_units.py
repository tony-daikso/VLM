"""Unit-level attribution for the OWL E3_late head (CPU only).

For every case x finding x unit (29 units, model order) and averaged over the 3 saved seeds:
  lin      [N,F,U,W]  exact linear decomposition of the logit (mean-pooled findings, 'late' mode):
                      logit = sum_{u,w} lin + const,  lin[u,w] = G[f,w] * A_u/cnt * <w_fw, z_uw>
                      (z = standardized embedding, 0 = Merlin-train mean -> lin is the evidence vs an average organ)
  occ_zero [N,F,U]    logit - logit(z_u := 0)          (= sum_w lin for mean findings; for MIL only the max units move)
  occ_abs  [N,F,U]    logit - logit(unit u marked absent: z_u=0, P_u=0)  (changes cnt/frac too)
  mil_L    [N,U]      fracture MIL unit score (site head + per-unit bias; site logit = max, finding = logsumexp over sites)
  mil_data [N,U]      data part of mil_L (sum_w G_w <ws_site,w, z_uw>), i.e. mil_L minus frac/bias terms
  logit, prob [N,F]   (prob = mean of per-seed sigmoids, = infer_head.py output)
Units absent / not mapped to a finding get NaN in lin/occ/mil.

usage: python attr_units.py --model E3_late.pt --infer-src <dir with infer_head.py> --emb-dir .. --bone-emb-dir .. \
          --ids ids.txt --out attr.npz [--threads 2]
"""
import argparse
import importlib.util
import sys

import numpy as np
import torch


def load_infer(src):
    spec = importlib.util.spec_from_file_location("infer_head", f"{src}/infer_head.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@torch.no_grad()
def attribute_seed(m, Z, P):
    """Z [N,U,W,D] standardized, P [N,U] -> dict of numpy arrays (one seed)."""
    N, U, W, D = Z.shape
    Fn = m.Mu.shape[0]
    gs = torch.softmax(m.gate, -1)                                   # [U,W]
    Mu = m.Mu
    A = P[:, None, :] * Mu[None]                                     # [N,F,U]
    cnt = A.sum(-1).clamp(min=1)
    G = (Mu @ gs) / Mu.sum(-1, keepdim=True)                         # [F,W]
    dot = torch.einsum("nuwd,fwd->nfuw", Z, m.w[..., :-1])
    lin = dot * G[None, :, None, :] * (A / cnt[..., None])[..., None]
    frac = A.sum(-1) / Mu.sum(-1)[None]
    const = ((m.w[..., -1][None] * frac[..., None] + m.b[None]) * G[None]).sum(-1)
    logit = m(Z, P)
    # sanity: decomposition is exact for mean findings
    mean_f = [f for f in range(Fn) if f not in m.fm]
    err = (lin.sum((2, 3)) + const - logit)[:, mean_f].abs().max().item()
    assert err < 1e-3, err
    occ_zero = torch.full((N, Fn, U), float("nan"))
    occ_abs = torch.full((N, Fn, U), float("nan"))
    for u in range(U):
        Z0 = Z.clone(); Z0[:, u] = 0
        P0 = P.clone(); P0[:, u] = 0
        occ_zero[:, :, u] = logit - m(Z0, P)
        occ_abs[:, :, u] = logit - m(Z0, P0)
    # MIL (fracture)
    mil_L = torch.full((N, U), float("nan"))
    mil_data = torch.full((N, U), float("nan"))
    for i, f in enumerate(m.fm):
        M = Mu[f]
        Wu = m.ws[i, m.site_of[i].clamp(min=0)]                       # [U,W,D+1]
        Gm = (M @ gs) / M.sum()                                       # [W]
        fr = (P * M).sum(-1) / M.sum()
        data = (torch.einsum("nuwd,uwd->nuw", Z, Wu[..., :-1]) * Gm).sum(-1)
        L = data + fr[:, None] * (Wu[..., -1] * Gm).sum(-1)[None] + m.ub[i][None]
        ok = (P > 0) & (M[None] > 0)
        mil_L = torch.where(ok, L, mil_L)
        mil_data = torch.where(ok, data, mil_data)
        lin[:, f] = float("nan")                                     # linear decomposition not defined for MIL
    nanmask = (A == 0)
    lin = lin.masked_fill(nanmask[..., None], float("nan"))
    occ_zero = occ_zero.masked_fill(nanmask, float("nan"))
    occ_abs = occ_abs.masked_fill(nanmask, float("nan"))
    return dict(lin=lin.numpy(), occ_zero=occ_zero.numpy(), occ_abs=occ_abs.numpy(), mil_L=mil_L.numpy(),
                mil_data=mil_data.numpy(), logit=logit.numpy(), prob=torch.sigmoid(logit).numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--infer-src", required=True)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--bone-emb-dir", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=256)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    ih = load_infer(a.infer_src)
    from pathlib import Path
    ck, models = ih.load_model(a.model, "cpu")
    meta = ck["meta"]
    ed, bd = Path(a.emb_dir), Path(a.bone_emb_dir)
    want = [l.strip() for l in open(a.ids) if l.strip()]
    ids, res = [], []
    for i in range(0, len(want), a.chunk):
        Es, Ps = [], []
        for c in want[i:i + a.chunk]:
            try:
                e, p = ih.load_case(c, meta, ed, bd)
            except FileNotFoundError:
                print("missing", c, file=sys.stderr)
                continue
            Es.append(e); Ps.append(p); ids.append(c)
        E = torch.from_numpy(np.stack(Es)); P = torch.from_numpy(np.stack(Ps))
        Z = ih.standardize(E.float(), P.float(), ck)
        per = [attribute_seed(m, Z, P.float()) for m in models]
        agg = {k: np.mean([d[k] for d in per], 0) for k in per[0] if k != "prob"}
        agg["prob"] = np.mean([d["prob"] for d in per], 0)
        agg["present"] = P.numpy()
        res.append(agg)
        print(f"{len(ids)}/{len(want)}", file=sys.stderr, flush=True)
    out = {k: np.concatenate([r[k] for r in res]).astype(np.float32) for k in res[0]}
    np.savez_compressed(a.out, ids=np.array(ids), findings=np.array(meta["findings"]), units=np.array(meta["units"]),
                        windows=np.array(meta["windows"]), **out)
    print("wrote", a.out, {k: v.shape for k, v in out.items()}, file=sys.stderr)


if __name__ == "__main__":
    main()
