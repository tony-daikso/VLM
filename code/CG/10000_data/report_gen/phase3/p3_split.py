#!/usr/bin/env python3
"""Phase 3 split: PID-grouped train 70% / val 10% / test 20% of the 6,970 OWL-scored C+ main exams.

Design (2026-11-09):
- Phase 3 test is drawn ONLY from Phase 1 test PIDs (report_gen/split.csv split == 'test'), so every Phase 3
  test exam is also a Phase 1 test exam -> the Phase 1 vs Phase 3 comparison can use the whole Phase 3 test set.
  (Phase 1 dev was only used for threshold selection; Phase 1 test was never used for any fitting, and Phase 1 has
  no trained generator, so putting Phase 1 dev exams into Phase 3 train is harmless.)
- Greedy stratification over def-B report labels (rarest positive first: appendicitis, free_air,
  bowel_obstruction, abdominal_aortic_aneurysm, then the other 10 classes), then remaining PIDs stratified by LOC.
- Fixed seed. Output: data/.../report_gen/phase3/split_p3.csv (EXAMID, PID, LOC, split, p1_split) + stats.
  --no-nest writes the unconstrained variant (for reference only) to split_p3_nonest.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DIS, OUT, PROBS, ZH, defb_labels, read_reports  # noqa: E402

SEED = 20261109
FRAC = {"train": 0.70, "val": 0.10, "test": 0.20}
RARE_FIRST = ["appendicitis", "free_air", "bowel_obstruction", "abdominal_aortic_aneurysm"]
P3 = OUT / "phase3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-nest", action="store_true", help="do not restrict test to Phase 1 test PIDs")
    a = ap.parse_args()
    ids = pd.read_csv(PROBS, usecols=["study_id"]).study_id
    rep = read_reports().set_index("EXAMID").loc[ids, ["PID", "LOC"]].reset_index()
    p1 = pd.read_csv(OUT / "split.csv").set_index("EXAMID").split
    rep["p1_split"] = rep.EXAMID.map(p1)
    lab = pd.read_csv(Path(OUT).parent / "disease14_qwen.csv", encoding="utf-8-sig").set_index("EXAMID")
    y = pd.DataFrame({k: lab[k + "_cat"].astype(str).isin({"yes", "probably_yes"}).astype(int) for k in DIS}).reindex(ids)
    y.index.name = "EXAMID"
    # PID-level positives (a PID may have several exams; here every PID has exactly one)
    g = rep.join(y, on="EXAMID").groupby("PID")
    pid = pd.DataFrame({"n": g.size(), "LOC": g.LOC.first(), "p1test": g.p1_split.apply(lambda s: (s == "test").all())})
    for k in DIS:
        pid[k] = g[k].max()
    rng = np.random.default_rng(SEED)
    pid = pid.loc[rng.permutation(pid.index.values)]
    order = RARE_FIRST + sorted([k for k in DIS if k not in RARE_FIRST], key=lambda k: pid[k].sum())
    N = pid.n.sum()
    target_n = {s: FRAC[s] * N for s in FRAC}
    cnt = {s: 0 for s in FRAC}
    cls_cnt = {s: {k: 0 for k in DIS} for s in FRAC}
    loc_cnt = {s: {} for s in FRAC}
    assign = {}

    def allowed(row):
        return [s for s in FRAC if s != "test" or a.no_nest or row.p1test]

    def pick(row, key):
        opts = allowed(row)
        tot = pid[key].sum() if key in DIS else (pid.LOC == key).sum()
        best, bd = None, None
        for s in opts:
            have = cls_cnt[s][key] if key in DIS else loc_cnt[s].get(key, 0)
            deficit = (FRAC[s] * tot - have) / (FRAC[s] * tot)   # relative shortfall vs its share
            size_def = (target_n[s] - cnt[s]) / target_n[s]
            score = (deficit, size_def)
            if bd is None or score > bd:
                best, bd = s, score
        return best

    def put(p, s):
        row = pid.loc[p]
        assign[p] = s
        cnt[s] += row.n
        for k in DIS:
            cls_cnt[s][k] += row[k]
        loc_cnt[s][row.LOC] = loc_cnt[s].get(row.LOC, 0) + row.n

    for k in order:                       # positives of each class, rarest first
        for p in pid.index[(pid[k] == 1)]:
            if p not in assign:
                put(p, pick(pid.loc[p], k))
    for p in pid.index:                   # everything else stratified by LOC, size-capped
        if p in assign:
            continue
        row = pid.loc[p]
        opts = [s for s in allowed(row) if cnt[s] < target_n[s]] or allowed(row)
        tot = (pid.LOC == row.LOC).sum()
        s = max(opts, key=lambda s: ((FRAC[s] * tot - loc_cnt[s].get(row.LOC, 0)) / (FRAC[s] * tot),
                                      (target_n[s] - cnt[s]) / target_n[s]))
        put(p, s)
    rep["split"] = rep.PID.map(assign)
    assert rep.groupby("PID").split.nunique().max() == 1 and rep.split.notna().all()
    tag = "_nonest" if a.no_nest else ""
    P3.mkdir(parents=True, exist_ok=True)
    rep[["EXAMID", "PID", "LOC", "split", "p1_split"]].to_csv(P3 / f"split_p3{tag}.csv", index=False)

    # ---- stats
    yb = defb_labels().reindex(ids)
    yb.index.name = "EXAMID"
    d = rep.set_index("EXAMID").join(yb)
    rows = []
    for k in DIS:
        r = {"disease": k, "中文": ZH[k]}
        for s in FRAC:
            sub = d[d.split == s][k]
            r[f"{s}_pos"] = int((sub == 1).sum())
            r[f"{s}_neg"] = int((sub == 0).sum())
        both = d[(d.split == "test") & (d.p1_split == "test")][k]
        r["both_test_pos"] = int((both == 1).sum())
        r["both_test_neg"] = int((both == 0).sum())
        rows.append(r)
    st = pd.DataFrame(rows)
    st.to_csv(P3 / f"split_p3{tag}_stats.csv", index=False)
    print(rep.groupby(["LOC", "split"]).size().unstack())
    print(pd.crosstab(rep.split, rep.p1_split, margins=True))
    print(st.to_string(index=False))


if __name__ == "__main__":
    main()
