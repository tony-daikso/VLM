"""Pick viz cases (high-score TP + FP) per target finding and write work/viz_spec.json for make_itk.py.

Per unit value ('contrib', model unit order, None = absent / not used by the finding):
  mean findings : occ_zero = exact linear contribution of the unit to the logit (vs Merlin-train average organ)
  fracture      : MIL unit score mil_L (site head + unit bias); heatmap = mil_L - min over present bone units (>=0)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common_viz import BONE, DIS_ZH

ROOT = Path("/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data")
W = Path(__file__).resolve().parent / "work"
N_TP, N_FP = 7, 2


def main():
    z = np.load(W / "attr.npz")
    ids, units, F = list(z["ids"]), list(z["units"]), list(z["findings"])
    row = {c: i for i, c in enumerate(ids)}
    lab = {}
    for l in open(ROOT / "disease14_qwen.jsonl"):
        x = json.loads(l)
        if x.get("parse_ok") and x["study_id"] in row:
            lab[x["study_id"]] = x["answer"]
    spec = []
    for fn in DIS_ZH:
        f = F.index(fn)
        df = pd.DataFrame({"id": ids, "prob": z["prob"][:, f], "cat": [lab[c][fn]["category"] for c in ids]})
        tp = df[df.cat == "yes"].sort_values("prob", ascending=False)
        if fn == "fracture":
            loc = pd.read_csv(W / "loc_cases_fracture.csv").set_index("id")
            tp = tp[tp.id.isin(loc.index)]
            s = loc.loc[tp.id, "sites"].values
            pick = list(tp[s == "vertebra"].id[:4]) + list(tp[s == "rib"].id[:2]) + \
                list(tp[np.isin(s, ["pelvis", "hip_femur", "pelvis|rib", "hip_femur|pelvis"])].id[:1])
        elif fn == "hydronephrosis":
            side = tp.id.map(lambda c: lab[c][fn]["detail"].get("side"))
            pick = list(tp[side == "left"].id[:4]) + list(tp[side == "right"].id[:3])
        else:
            pick = list(tp.id[:N_TP])
        fp = list(df[df.cat.isin(["no", "not_mentioned"])].sort_values("prob", ascending=False).id[:N_FP])
        for kind, lst in (("TP", pick), ("FP", fp)):
            for cid in lst:
                n = row[cid]
                if fn == "fracture":
                    v = z["mil_L"][n].astype(float)
                    contrib = [None if np.isnan(x) else float(x) for x in v]
                    bmin = np.nanmin([v[units.index(u)] for u in BONE])
                    heat = [None if c is None else c - bmin for c in contrib]
                    metric = "MIL 單元分數 (logit 尺度)"
                else:
                    v = z["occ_zero"][n, f].astype(float)
                    contrib = [None if np.isnan(x) else float(x) for x in v]
                    heat = contrib
                    metric = "單元對 logit 的線性貢獻 (= occlusion→train 平均)"
                a = lab[cid][fn]
                spec.append(dict(disease=fn, disease_zh=DIS_ZH[fn], id=cid, kind=kind, prob=float(z["prob"][n, f]),
                                 category=a["category"], quote=" || ".join(a["quotes"]) or f"（報告無相關句；標註說明：{a.get('why', '')}）",
                                 detail=a.get("detail", {}), contrib=contrib, heat=heat, metric=metric,
                                 logit=float(z["logit"][n, f]), units=units))
    json.dump(spec, open(W / "viz_spec.json", "w"), ensure_ascii=False, indent=1)
    print(len(spec), "case x finding")
    print(pd.DataFrame(spec)[["disease", "id", "kind", "prob", "quote"]].to_string(max_colwidth=70))


if __name__ == "__main__":
    main()
