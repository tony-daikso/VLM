#!/usr/bin/env python3
"""Unit-level labels for --aux-unit-loss: which OWL units are abnormal for which disease, derived from agent B's
target statuses + attributes (side, vertebral levels, rib descriptions, fracture sites, obstruction type).

Rule per disease (B status pos -> 1, neg -> 0, uncertain / not_mentioned -> no label):
  side-specific   hydronephrosis: side left/right -> that kidney (organ3 kidney_*, ru_hydronephrosis k1/k0) = 1 and
                  the other kidney = 0; bilateral -> both 1; no side / transplant -> unlabelled ("unmapped");
                  negative -> both 0
  level-specific  fracture: vertebral_levels T8-L5 -> vert15/<L> + bone3/<L>; rib_desc with side + number -> rib3 /
                  rib15 rib_<side>_<n> (1-4 share one unit) + bone3 ribs_<side>; sites sacrum -> bone3 sacrum +
                  ru_pelvis_fx k3. Other units of a positive case stay unlabelled (an unnamed second fracture is possible).
                  Negative -> every vertebra / rib / pelvis unit 0. Positive with nothing mappable -> "unmapped".
  type-specific   bowel_obstruction: SBO -> small_bowel (+ ru_bowel), LBO -> colon, both -> both; no type -> unmapped
  organ-level     the finding's organ / crop units take the study label (liver, spleen, gallbladder, aorta, RLQ,
                  peritoneal pockets, hilum, lumbar spine ...): coarse but "as far as the units allow".
Output jsonl per split: {case_id, units: {"group/unit": [[disease_index, 0|1], ...]}} + unit_label_stats.json/.md
(per disease: positives, positives mapped to a specific unit, rate). Train + val only.
usage: python build_unit_labels.py --targets targets_train.jsonl targets_val.jsonl --meta token_meta.csv --out DIR
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m4_common import DISEASES, ZH  # noqa: E402

VERT_LV = ["T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5"]
RIB_UNITS = ["1_4", "5", "6", "7", "8", "9", "10", "11", "12"]
ORGAN = {  # organ-level mapping (unit keys "group/unit")
    "hepatic_steatosis": ["organ3/liver"],
    "surgically_absent_gallbladder": ["organ3/gallbladder", "ru_gallstones/k0"],
    "gallstones": ["organ3/gallbladder", "ru_gallstones/k0"],
    "splenomegaly": ["organ3/spleen"],
    "abdominal_aortic_aneurysm": ["organ3/aorta", "ru_atherosclerosis/k0", "ru_atherosclerosis/k1"],
    "atherosclerosis": ["organ3/aorta"] + [f"ru_atherosclerosis/k{i}" for i in range(4)],
    "osteopenia": ["organ3/lumbar_spine"] + [f"vert15/{l}" for l in VERT_LV] + [f"bone3/{l}" for l in VERT_LV],
    "appendicitis": ["ru_rlq/k0"],
    "ascites": [f"ru_ascites/k{i}" for i in range(3)],
    "free_air": ["ru_free_air/k0", "ru_free_air/k1"],
    "biliary_ductal_dilation": ["ru_bile_duct/k0"],
}
KID = {"left": ["organ3/kidney_left", "ru_hydronephrosis/k1"], "right": ["organ3/kidney_right", "ru_hydronephrosis/k0"]}
BOWEL = {"SBO": ["organ3/small_bowel", "ru_bowel/k1"], "LBO": ["organ3/colon"]}
FX_ALL = [f"vert15/{l}" for l in VERT_LV] + [f"bone3/{l}" for l in VERT_LV] + \
         [f"{g}/rib_{s}_{n}" for g in ("rib3", "rib15") for s in ("left", "right") for n in RIB_UNITS] + \
         ["bone3/ribs_left", "bone3/ribs_right", "bone3/sacrum", "bone3/hip_left", "bone3/hip_right",
          "bone3/femur_left", "bone3/femur_right"] + [f"ru_pelvis_fx/k{i}" for i in range(4)]
RIB_NUM = re.compile(r"\d{1,2}")


def rib_units(desc):
    """'left 5th-7th ribs' -> {'rib_left_5', 'rib_left_6', 'rib_left_7'}; no side -> set()."""
    t = desc.lower()
    sides = [s for s in ("left", "right") if s in t] or (["left", "right"] if "bilateral" in t else [])
    nums = [int(x) for x in RIB_NUM.findall(t) if 1 <= int(x) <= 12]
    if re.search(r"\d\s*(?:-|through|to)\s*\d", t) and len(nums) >= 2:
        nums = list(range(min(nums[0], nums[1]), max(nums[0], nums[1]) + 1)) + nums[2:]
    out = set()
    for s in sides:
        for n in nums:
            out.add(f"rib_{s}_{'1_4' if n <= 4 else n}")
    return out


def per_disease(r):
    return r.get("diseases") or {}


def case_units(r):
    """-> ({unit_key: {disease_index: 0/1}}, {disease: 'pos_mapped'|'pos_unmapped'|'neg'|None})"""
    lab = defaultdict(dict)
    how = {}
    per = per_disease(r)

    def put(keys, d, y):
        for k in keys:
            lab[k][DISEASES.index(d)] = y

    for d in DISEASES:
        v = per.get(d) or {}
        st, at = v.get("status"), v.get("attributes") or {}
        if st not in ("pos", "neg"):
            how[d] = None
            continue
        y = int(st == "pos")
        if d == "hydronephrosis":
            if not y:
                put(KID["left"] + KID["right"], d, 0)
                how[d] = "neg"
            elif at.get("side") in ("left", "right"):
                other = "right" if at["side"] == "left" else "left"
                put(KID[at["side"]], d, 1)
                put(KID[other], d, 0)
                how[d] = "pos_mapped"
            elif at.get("side") == "bilateral":
                put(KID["left"] + KID["right"], d, 1)
                how[d] = "pos_mapped"
            else:
                how[d] = "pos_unmapped"
        elif d == "fracture":
            if not y:
                put(FX_ALL, d, 0)
                how[d] = "neg"
                continue
            keys = []
            for lv in at.get("vertebral_levels") or []:
                if lv in VERT_LV:
                    keys += [f"vert15/{lv}", f"bone3/{lv}"]
            for desc in at.get("rib_desc") or []:
                for u in rib_units(desc):
                    keys += [f"rib3/{u}", f"rib15/{u}", f"bone3/ribs_{u.split('_')[1]}"]
            if "sacrum" in (at.get("sites") or []):
                keys += ["bone3/sacrum", "ru_pelvis_fx/k3"]
            put(keys, d, 1)
            how[d] = "pos_mapped" if keys else "pos_unmapped"
        elif d == "bowel_obstruction":
            if not y:
                put(BOWEL["SBO"] + BOWEL["LBO"], d, 0)
                how[d] = "neg"
            elif at.get("type") in ("SBO", "LBO", "both"):
                put(BOWEL["SBO"] + BOWEL["LBO"] if at["type"] == "both" else BOWEL[at["type"]], d, 1)
                how[d] = "pos_mapped"
            else:
                how[d] = "pos_unmapped"
        else:
            put(ORGAN[d], d, y)
            how[d] = "pos_mapped" if y else "neg"
    return lab, how


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--meta", required=True, help="features/token_meta.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    keys = {f"{r['group']}/{r['unit']}" for r in csv.DictReader(open(a.meta))}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {d: defaultdict(int) for d in DISEASES}
    unknown = set()
    for tf in a.targets:
        fh = {}
        for line in open(tf):
            r = json.loads(line)
            sp = r.get("split")
            if sp not in ("train", "val"):
                raise SystemExit(f"split {sp!r} refused (train/val only)")
            lab, how = case_units(r)
            for k in list(lab):
                if k not in keys:
                    unknown.add(k)
                    lab.pop(k)
            for d, h in how.items():
                if h:
                    stats[d][f"{sp}_{h}"] += 1
            if sp not in fh:
                fh[sp] = open(out / f"unit_labels_{sp}.jsonl", "w")
            fh[sp].write(json.dumps({"case_id": r["case_id"],
                                     "units": {k: sorted(v.items()) for k, v in lab.items()}}) + "\n")
        for f in fh.values():
            f.close()
    if unknown:
        print("unit keys not in token_meta (dropped):", sorted(unknown)[:10])
    json.dump({d: dict(v) for d, v in stats.items()}, open(out / "unit_label_stats.json", "w"), indent=1)
    md = ["# 單元層級標籤對應率（train）", "", "| 疾病 | 陽性 | 對應到特定單元 | 對應率 | 陰性 | 說明 |", "|---|---|---|---|---|---|"]
    note = {"hydronephrosis": "依側別 → 左/右腎", "fracture": "依椎體節段（T8–L5）、肋骨側別＋第幾根、薦椎",
            "bowel_obstruction": "依 SBO/LBO → 小腸/大腸"}
    for d in DISEASES:
        s = stats[d]
        pm, pu, n = s.get("train_pos_mapped", 0), s.get("train_pos_unmapped", 0), s.get("train_neg", 0)
        md.append(f"| {ZH[d]} | {pm + pu} | {pm} | {pm / (pm + pu):.3f} | {n} | {note.get(d, '器官層級（研究標籤直接給該器官/區域單元）')} |"
                  if pm + pu else f"| {ZH[d]} | 0 | 0 | — | {n} | {note.get(d, '器官層級')} |")
    (out / "unit_label_stats.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
