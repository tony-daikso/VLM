"""gs15 probe (膽結石 1.5 mm GB crop unit, 5 regions x 6 windows; unit gs15 main units gallbladder/liver; variants
U (all 5 regions), U_org (+main TS gallbladder/liver units), U_gb3 (regions r0-r2 = Exp N-equivalent GB/fossa/dependent)).
Copied from bd15 probe:
bd15 probe = copy of Exp O no_probe.py with unit bd15 (main units pancreas/gallbladder/liver, as unit bd) and the
variants U (7 regions), U_org (+main TS organ units), U_bd5reg (only the 4 BD5-equivalent regions r0-r3).
Exp O (copy of Exp N nu_probe.py; new codes bd/fa2/asc2/frp/frpp/go; rows with y = NaN are kept: never used for
fitting / early stopping, but scored in OOF and val -- needed for frpp, the pelvis-site head that must score every case)
Exp N region-unit probes (copy of Exp M ru_probe.py; units gs = gallbladder/GB-fossa, ath = aorto-iliac wall,
fst = focal-steatosis liver regions; U_std5 = ablation without the extra windows (stone/calcium), the main units get
zeros for the extra windows in U_org).
Exp M region-unit probes (E3_late-style 'late' head: one linear head per window, gated; mean|max region pooling).
Same recipe as rlq_probe.py / hn_oof.py: standardisation on present TRAIN rows (per crop x region; absent regions = 0),
class-balanced BCE, Adam lr 1e-3 / wd 1e-4, early stopping on an inner 15 % TRAIN holdout, seeds 42/43, the finding's own
5 folds (fold column of the analysis E3_late OOF), val = refit on all train -> val once.
Test is never loaded: the labels csv holds train/val rows only, the main-embedding npz is the *_notest matrix subset and
crop embeddings exist for train+val ids only.
Variants by unit:
  hn  : U_mil      = per-kidney instance (4 regions concatenated), MIL max over the two kidneys (shared head + side bias)
        U_mean     = same instances, mean (no MIL)
        U_org_mil  = per-kidney instance = crop regions + main TS kidney embedding (mean 448), MIL max
  fa/asc: U        = all crops x regions concatenated per window (one unit)
        U_org      = U + main TS organ units of the finding (free air: small_bowel/colon/stomach; ascites: liver/spleen);
                     the extra air window (fa) has zeros for the main units.
Usage: ru_probe.py UNIT EMB_DIR MAIN_NPZ LABELS_CSV OUT_DIR [--device cuda] [--variants a,b]
Writes OUT_DIR/oof_train.csv (study_id, fold, <variant>) and OUT_DIR/val_scores.csv (study_id, <variant>)."""
import sys, os, argparse, numpy as np, pandas as pd, torch, torch.nn.functional as F
torch.set_num_threads(int(os.environ.get("NT", "4")))
sys.path.insert(0, os.environ.get("MW_SRC", "/root/OWL/mw_exp/src"))
from probe import Probe, auroc

ap = argparse.ArgumentParser()
ap.add_argument("unit"); ap.add_argument("emb_dir"); ap.add_argument("main"); ap.add_argument("labels"); ap.add_argument("out")
ap.add_argument("--device", default="cpu"); ap.add_argument("--variants", default=None)
ap.add_argument("--vram-gb", type=float, default=10.0)
a = ap.parse_args(); dev = torch.device(a.device)
if dev.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb / (torch.cuda.get_device_properties(0).total_memory / 2 ** 30)), 0)
os.makedirs(a.out, exist_ok=True)
lab = pd.read_csv(a.labels)
assert set(lab.split) <= {"train", "val"}
lab = lab.reset_index(drop=True); HAS = lab.y.notna().values
z = np.load(a.main); mids = list(z["ids"]); units = list(z["units"]); pos = {s: i for i, s in enumerate(mids)}
keep = [pos[s] for s in lab.study_id]
Em = z["E"][keep].astype(np.float32); Pm = z["P"][keep].astype(np.float32); del z
N = len(lab); ids = list(lab.study_id)
emb0 = None
for s in ids:
    f = f"{a.emb_dir}/{s}.npz"
    if os.path.exists(f):
        emb0 = np.load(f)["emb"]; break
K, Rn, W, D = emb0.shape
EP = np.zeros((N, K, Rn, W, D), np.float16); RP = np.zeros((N, K, Rn), np.float32); miss = 0
for i, s in enumerate(ids):
    f = f"{a.emb_dir}/{s}.npz"
    if not os.path.exists(f):
        miss += 1; continue
    q = np.load(f)
    pr = np.asarray(q["present"], bool)
    EP[i] = q["emb"]; RP[i] = (q["region_vox"] > 0) & pr[:, None]
print(f"N {N} missing emb {miss}; crop x region presence:\n{RP.mean(0).round(4)}", flush=True)
tr = (lab.split == "train").values


def standardize(X, Pr):
    """X [N,U,R,W,D] (float16 ok), Pr [N,U,R] -> float32 standardised, zero where absent"""
    out = np.zeros(X.shape, np.float32)
    for u in range(X.shape[1]):
        for r in range(X.shape[2]):
            m = tr & (Pr[:, u, r] > 0)
            if m.sum() < 10:
                continue
            x = X[:, u, r].astype(np.float32)
            mu, sd = x[m].mean(0), np.maximum(x[m].std(0), 1e-6)
            out[:, u, r] = ((x - mu) / sd) * (Pr[:, u, r] > 0)[:, None, None]
    return out


Us = standardize(EP, RP); del EP                                   # [N,K,R,W,D]
ui = {"hn": ["kidney_right", "kidney_left"], "fa": ["small_bowel", "colon", "stomach"], "asc": ["liver", "spleen"],
      "gs": ["gallbladder", "liver"], "ath": ["aorta"], "fst": ["liver"], "bd": ["pancreas", "gallbladder", "liver"], "bd15": ["pancreas", "gallbladder", "liver"], "gs15": ["gallbladder", "liver"],
      "fa2": ["small_bowel", "colon", "stomach"], "asc2": ["liver", "spleen"], "go": ["stomach", "small_bowel"],
      "frp": ["sacrum", "hip_left", "hip_right", "femur_left", "femur_right"], "frpp": ["sacrum", "hip_left", "hip_right", "femur_left", "femur_right"]}[a.unit]
Mo = Em[:, [units.index(u) for u in ui]]; Mp = Pm[:, [units.index(u) for u in ui]]
Ms = standardize(Mo[:, :, None], Mp[:, :, None])[:, :, 0]           # [N,Uo,Wm,448]
del Em, Mo
Wm = Ms.shape[2]
if Wm < W:                                                          # extra (air) window: zeros for main units
    Ms = np.concatenate([Ms, np.zeros((N, Ms.shape[1], W - Wm, Ms.shape[3]), np.float32)], 2)


def inst(org):
    """hn: units = kidneys; per unit concat regions (+ main kidney emb)"""
    X = Us.transpose(0, 1, 3, 2, 4).reshape(N, K, W, Rn * D)
    P = (RP.max(2) > 0).astype(np.float32)
    if org:
        X = np.concatenate([X, Ms], 3); P = np.maximum(P, Mp)
    return torch.from_numpy(X), torch.from_numpy(P)


def single(org):
    X = Us.transpose(0, 3, 1, 2, 4).reshape(N, W, K * Rn * D)
    P = (RP.max((1, 2)) > 0).astype(np.float32)
    if org:
        X = np.concatenate([X, Ms.transpose(0, 2, 1, 3).reshape(N, W, -1)], 2); P = np.maximum(P, Mp.max(1))
    return torch.from_numpy(X)[:, None], torch.from_numpy(P)[:, None]


if a.unit in ("hn",):
    VIEWS = {"U_mil": (lambda: inst(False), True), "U_mean": (lambda: inst(False), False), "U_org_mil": (lambda: inst(True), True)}
else:
    VIEWS = {"U": (lambda: single(False), False), "U_org": (lambda: single(True), False)}
    if a.unit == "gs15":
        VIEWS["U_gb3"] = (lambda: (lambda X, P: (X.reshape(N, 1, W, K, Rn, D)[:, :, :, :, :3].reshape(N, 1, W, -1).contiguous(), P))(*single(False)), False)
    if a.unit == "bd15":
        VIEWS["U_bd5reg"] = (lambda: (lambda X, P: (X.reshape(N, 1, W, K, Rn, D)[:, :, :, :, :4].reshape(N, 1, W, -1).contiguous(), P))(*single(False)), False)
    if W > 5:                                   # ablation: drop the extra windows
        VIEWS["U_std5"] = (lambda: tuple(t[:, :, :5].contiguous() if i == 0 else t for i, t in enumerate(single(False))), False)
if a.variants:
    VIEWS = {k: VIEWS[k] for k in a.variants.split(",")}


def fit(X, P, Y, mil, idx_tr, idx_es, idx_pred, seed, lr=1e-3, wd=1e-4, max_ep=100, pat=10, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    U = X.shape[1]
    m = Probe("late", U, X.shape[2], X.shape[3], [list(range(U))], None,
              finding_sites=[[list(range(U))]] if mil else None).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    Et, Pt, Yt = X[idx_tr].to(dev), P[idx_tr].to(dev), torch.from_numpy(Y[idx_tr]).to(dev)
    Ee, Pe = X[idx_es].to(dev), P[idx_es].to(dev)
    npos = Yt.sum(); nneg = len(Yt) - npos; wpos = len(Yt) / (2 * npos); wneg = len(Yt) / (2 * nneg)
    best, bst, bad = -1, None, 0
    for ep in range(max_ep):
        m.train(); perm = torch.randperm(len(Et), device=dev)
        for i in range(0, len(Et), bs):
            j = perm[i:i + bs]; y = Yt[j]
            lg = m(Et[j], Pt[j]); w = torch.where(y == 1, wpos, wneg)
            loss = (F.binary_cross_entropy_with_logits(lg, y, reduction="none") * w).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            sc = auroc(Y[idx_es][:, 0], m(Ee, Pe)[:, 0].float().cpu().numpy())
        if sc > best + 1e-5: best, bad, bst = sc, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= pat: break
    m.load_state_dict(bst); m.eval()
    with torch.no_grad():
        out = []
        for i in range(0, len(idx_pred), 2048):
            j = idx_pred[i:i + 2048]
            out.append(m(X[j].to(dev), P[j].to(dev))[:, 0].float().cpu().numpy())
    del Et, Pt, Ee, Pe, m, opt
    return np.concatenate(out)


fo, fv = a.out + "/oof_train.csv", a.out + "/val_scores.csv"
tri = np.where(tr)[0]; vai = np.where(~tr)[0]
oof = pd.read_csv(fo) if os.path.exists(fo) else pd.DataFrame({"study_id": [ids[i] for i in tri], "fold": lab.fold.values[tri].astype(int)})
val = pd.read_csv(fv) if os.path.exists(fv) else pd.DataFrame({"study_id": [ids[i] for i in vai]})
Y = np.nan_to_num(lab[["y"]].values.astype(np.float32)); fid = oof.fold.values; HT = HAS[tri]
for vname, (mk, mil) in VIEWS.items():
    if vname in oof.columns and vname in val.columns:
        continue
    X, P = mk()
    cc = np.zeros(len(tri))
    for k in sorted(np.unique(fid)):
        a_ = np.where((fid != k) & HT)[0]; b_ = np.where(fid == k)[0]
        a_ = np.random.default_rng(int(k)).permutation(a_); nes = int(0.15 * len(a_))
        cc[b_] = np.mean([fit(X, P, Y, mil, tri[a_[nes:]], tri[a_[:nes]], tri[b_], s) for s in (42, 43)], 0)
    rr = np.random.default_rng(0).permutation(np.where(HT)[0]); nes = int(0.15 * len(rr))
    vv = np.mean([fit(X, P, Y, mil, tri[rr[nes:]], tri[rr[:nes]], vai, s) for s in (42, 43)], 0)
    oof[vname] = cc; val[vname] = vv
    hv = HAS[vai]
    print(f"{a.unit} {vname}: train OOF {auroc(Y[tri[HT], 0], cc[HT]):.4f} | val {auroc(Y[vai[hv], 0], vv[hv]):.4f}", flush=True)
    oof.to_csv(fo, index=False); val.to_csv(fv, index=False)
    del X, P
    if dev.type == "cuda": torch.cuda.empty_cache()
print("RU_PROBE_DONE", flush=True)
