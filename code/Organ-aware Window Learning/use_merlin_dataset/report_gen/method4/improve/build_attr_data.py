#!/usr/bin/env python3
"""Agent N (11-12), arm T_unit_gt_pil_attr: A+ (T_unit_gt_pil) + the "attributes to state" line.

Why: the A+ attribute errors are mostly omissions (the report does not state the attribute). The line tells the model
which attributes to state for a finding that is present (m4_common.ATTR_ITEMS, switches only, never a value).

build (default):
  train rows  "attr_state" {key: 1 / 0 / None} from the structured TRAIN targets (inputs/targets_v2 targets_train.jsonl,
              attribute fields of the cleaned target sentence): finding positive in the target (status == pos and the
              row has a target line for it) -> 1 / 0 = the sentence states / does not state the attribute; any other
              finding -> None. The stored prompt gets one fixed-seed draw of the line (m4_common.sample_attr_switches);
              train.py --unit-cue-gt <cue file> redraws the None switches at every read.
  val rows    every switch on. The val targets are never opened: the line cannot depend on a val reference report.
  cue file    = the A+ cue file + "attr_patterns" {finding: joint on / off patterns of its train-positive cases}.
  Everything else in a row is the A+ row, byte for byte (asserted). Test rows are refused.
--check: sanity checks on the built data (aggregates only, no ids / text) -> <out>/attr_checks.json
  - no test row; val / val600 hold exactly one line, all on; each row with the line and attr_state removed == the A+ row;
    cue file without attr_patterns == the A+ cue file;
  - the train.py recipe itself (m4_data.M4Dataset.__getitem__ with CueGT + UnitCueGT, character-level stand-in
    tokenizer, --check-passes passes over all train rows): the prompt without the attribute line equals the prompt the
    A+ data gives under the same random state; a positive finding's switches equal attr_state;
  - no-leak: per attribute the on-rate for target-positive vs other rows (also vs the cue label), and cross-fitted
    AUROC of predicting each finding's label from the whole switch vector (pattern table; logistic regression when
    sklearn is there);
  - how often every switch / every relevant switch is on in training (inference = always all on);
  - with --llm: prompt + target token lengths vs --max-len (train rows, the stored prompt with every switch on = the
    longest form), next to the A+ rows.
usage (A6000): python build_attr_data.py --aplus-data improve/data_unit_pil --aplus-val600 improve/data/val600_unit_pil.jsonl \
   --targets-train inputs/targets_v2/targets_train.jsonl --out improve/data_unit_pil_attr \
   --out-val600 improve/data/val600_unit_pil_attr.jsonl [--check --cue-gt cue/cue_gt.json --llm <tokenizer dir>]
"""
import argparse
import json
import random
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from m4_common import (ATTR_DISEASES, ATTR_ITEMS, ATTR_PREFIX, attr_line_all_on, parse_attr_line,  # noqa: E402
                       render_attr_line, replace_attr_line, sample_attr_switches)

# key -> attribute field of targets_v2 diseases.<finding>.attributes
FIELD = {"hepatic_steatosis:grade": "grade", "hydronephrosis:grade": "grade", "hydronephrosis:side": "side",
         "splenomegaly:length": "length_mm", "abdominal_aortic_aneurysm:diameter": "diameter_mm",
         "atherosclerosis:grade": "grade", "bowel_obstruction:type": "type", "ascites:amount": "amount",
         "free_air:postoperative": "postoperative", "biliary_ductal_dilation:site": "site", "fracture:sites": "sites",
         "fracture:levels": "vertebral_levels", "fracture:acuity": "acuity"}
assert list(FIELD) == [k for k, _, _ in ATTR_ITEMS]
KEYS = [k for k, _, _ in ATTR_ITEMS]


def stated(v):
    return not (v is None or v == "not_stated" or v == "" or v == [])


def read_rows(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def target_states(targets_train, rows_by_id):
    """-> {case_id: {key: 1 / 0 / None}} for the train rows, from structured fields only (no sentence is read)."""
    out = {}
    for line in open(targets_train):
        r = json.loads(line)
        if r["split"] != "train":
            raise SystemExit("attr_state is built from TRAIN targets only")
        row = rows_by_id.get(r["case_id"])
        if row is None:
            continue
        st = {}
        for k, d, _ in ATTR_ITEMS:
            x = r["diseases"][d]
            if x["status"] == "pos" and d in row["target_diseases"]:
                st[k] = int(stated(x["attributes"].get(FIELD[k])) and FIELD[k] not in (x.get("attributes_raw_only") or {}))
            else:
                st[k] = None
        out[r["case_id"]] = st
    return out


def pattern_tables(states):
    """joint on / off patterns per finding among its target-positive train cases."""
    tabs = {}
    for d in ATTR_DISEASES:
        keys = [k for k, dd, _ in ATTR_ITEMS if dd == d]
        cnt = {}
        for st in states.values():
            if st[keys[0]] is not None:
                p = tuple(st[k] for k in keys)
                cnt[p] = cnt.get(p, 0) + 1
        if not cnt:
            raise SystemExit(f"{d}: no target-positive train case, cannot build its switch patterns")
        pats = sorted(cnt)
        cum, c = [], 0
        for p in pats:
            c += cnt[p]
            cum.append(c)
        tabs[d] = {"keys": keys, "patterns": [list(p) for p in pats], "cum": cum}
    return tabs


def build(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ap_tr = read_rows(Path(a.aplus_data) / "train.jsonl")
    ap_va = read_rows(Path(a.aplus_data) / "val.jsonl")
    ap_600 = read_rows(a.aplus_val600)
    assert all(r["split"] == "train" for r in ap_tr) and all(r["split"] == "val" for r in ap_va + ap_600), "split"
    states = target_states(a.targets_train, {r["case_id"]: r for r in ap_tr})
    assert len(states) == len(ap_tr), "a train row without train targets"
    tabs = pattern_tables(states)
    uc = json.load(open(Path(a.aplus_data) / a.aplus_cue_name))
    assert "attr_patterns" not in uc
    uc["attr_patterns"] = tabs
    uc["attr_note"] = ("agent N 11-12: attribute line switches; a target-positive finding follows row attr_state, any "
                       "other finding gets a pattern drawn from attr_patterns (train-positive cases)")
    (out / a.cue_name).write_text(json.dumps(uc))
    rng = random.Random(a.seed)
    st = {"rows": {}, "seed": a.seed, "all_on_line_items": len(KEYS)}
    with open(out / "train.jsonl", "w") as fo:
        for r in ap_tr:
            assert ATTR_PREFIX not in r["prompt"] and "attr_state" not in r
            r = dict(r)
            r["attr_state"] = states[r["case_id"]]
            r["prompt"] = replace_attr_line(r["prompt"], render_attr_line(sample_attr_switches(r["attr_state"], tabs, rng)))
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    st["rows"]["train"] = len(ap_tr)
    allon = attr_line_all_on()
    for name, rows, path in (("val", ap_va, out / "val.jsonl"), ("val600", ap_600, Path(a.out_val600))):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fo:
            for r in rows:
                assert ATTR_PREFIX not in r["prompt"]
                r = dict(r)
                r["prompt"] = replace_attr_line(r["prompt"], allon)
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        st["rows"][name] = len(rows)
    st["train_positive_on_rate"] = {}
    for d, t in tabs.items():
        n = t["cum"][-1]
        cnt = [c - (t["cum"][i - 1] if i else 0) for i, c in enumerate(t["cum"])]
        for j, k in enumerate(t["keys"]):
            on = sum(c for p, c in zip(t["patterns"], cnt) if p[j])
            st["train_positive_on_rate"][k] = {"positive_rows": n, "on": on, "rate": on / n}
    (out / "build_stats_attr.json").write_text(json.dumps(st, indent=1))
    print(json.dumps(st, indent=1))


# ------------------------------------------------------------------------------------------------ checks
class CharTok:
    """character-level stand-in tokenizer: lets the real M4Dataset.__getitem__ run on CPU without a model, and the
    prompt it built is read back from the ids."""
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, **kw):
        return {"input_ids": [ord(c) for c in text]}


def auc(y, s):
    """rank AUROC with ties (pure python); y 0/1 list."""
    order = sorted(range(len(s)), key=lambda i: s[i])
    rank = [0.0] * len(s)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and s[order[j + 1]] == s[order[i]]:
            j += 1
        for k in range(i, j + 1):
            rank[order[k]] = (i + j) / 2 + 1
        i = j + 1
    n1 = sum(y)
    n0 = len(y) - n1
    if not n1 or not n0:
        return float("nan")
    return (sum(r for r, v in zip(rank, y) if v) - n1 * (n1 + 1) / 2) / (n1 * n0)


def crossfit_table_auc(X, y, grp):
    """predict y from the whole switch vector with a pattern table (smoothed positive rate per distinct vector), fitted
    on one half of the cases and scored on the other (both directions). Captures any interaction between switches."""
    s = [0.0] * len(y)
    for h in (0, 1):
        fit = [i for i in range(len(y)) if grp[i] != h]
        prior = sum(y[i] for i in fit) / max(len(fit), 1)
        tab = {}
        for i in fit:
            c = tab.setdefault(X[i], [0, 0])
            c[0] += y[i]
            c[1] += 1
        for i in range(len(y)):
            if grp[i] == h:
                c = tab.get(X[i], [0, 0])
                s[i] = (c[0] + 5 * prior) / (c[1] + 5)
    return auc(y, s)


def check(a):
    out = Path(a.out)
    chk = {}
    rows = {sp: read_rows(out / f"{sp}.jsonl") for sp in ("train", "val")}
    rows["val600"] = read_rows(a.out_val600)
    aplus = {sp: read_rows(Path(a.aplus_data) / f"{sp}.jsonl") for sp in ("train", "val")}
    aplus["val600"] = read_rows(a.aplus_val600)
    # ---- 1. splits, equality with A+ once the new parts are removed, val lines
    eq = {}
    for sp in ("train", "val", "val600"):
        want = "train" if sp == "train" else "val"
        assert all(r["split"] == want for r in rows[sp]), "split"
        assert len(rows[sp]) == len(aplus[sp])
        for r, q in zip(rows[sp], aplus[sp]):
            assert sum(ln.startswith(ATTR_PREFIX) for ln in r["prompt"].split("\n")) == 1
            s = {k: v for k, v in r.items() if k != "attr_state"}
            s["prompt"] = replace_attr_line(r["prompt"], "")
            assert s == q and json.dumps(s, ensure_ascii=False) == json.dumps(q, ensure_ascii=False), f"{sp}: differs from A+"
            assert ("attr_state" in r) == (sp == "train")
        eq[sp] = len(rows[sp])
    chk["equal_to_aplus_without_attr_line_rows"] = eq
    allon = attr_line_all_on()
    for sp in ("val", "val600"):
        lines = {ln for r in rows[sp] for ln in r["prompt"].split("\n") if ln.startswith(ATTR_PREFIX)}
        assert lines == {allon}, f"{sp}: attribute line is not the all-on line"
    vby = {r["case_id"]: r for r in rows["val"]}
    assert all(r == vby[r["case_id"]] for r in rows["val600"]), "val600 rows != val.jsonl rows"
    chk["val_lines"] = {"distinct_lines_val": 1, "distinct_lines_val600": 1, "all_on_items": len(KEYS),
                        "val_targets_opened": False, "val600_rows_identical_to_val_rows": True}
    chk["line_format"] = {"prefix": ATTR_PREFIX, "items": [txt for _, _, txt in ATTR_ITEMS],
                          "all_on_line_chars": len(allon)}
    uc = json.load(open(out / a.cue_name))
    base = json.load(open(Path(a.aplus_data) / a.aplus_cue_name))
    tabs = uc["attr_patterns"]
    assert {k: v for k, v in uc.items() if k not in ("attr_patterns", "attr_note")} == base, "cue file != A+ cue file"
    chk["cue_file_equals_aplus_plus_attr_patterns"] = True
    chk["attr_patterns"] = {d: {"patterns": t["patterns"], "cum": t["cum"]} for d, t in tabs.items()}

    # ---- 2. the train.py recipe (real dataset code), several passes
    from m4_data import CueGT, M4Dataset, UnitCueGT, chatml      # noqa: E402  (torch import; CPU only)
    tk = CharTok()
    mk = lambda data, cue: M4Dataset(Path(data) / "train.jsonl", tk, use_image=False, max_len=10 ** 9,  # noqa: E731
                                     cue_gt=CueGT(a.cue_gt, 0.3, "sample", unclear="oof"),
                                     unit_cue=UnitCueGT(str(cue), 0.3, grades=True))
    ds_new = mk(out, out / a.cue_name)
    ds_old = mk(a.aplus_data, Path(a.aplus_data) / a.aplus_cue_name)
    n = len(ds_new)
    cue_lab = json.load(open(a.cue_gt))["labels_train"]
    from m4_common import DISEASES
    ypos = {d: [int(r["attr_state"][next(k for k, dd, _ in ATTR_ITEMS if dd == d)] is not None) for r in rows["train"]]
            for d in ATTR_DISEASES}
    ycue = {d: [cue_lab[r["case_id"]][DISEASES.index(d)] for r in rows["train"]] for d in ATTR_DISEASES}
    grp = [zlib.crc32(r["case_id"].encode()) % 2 for r in rows["train"]]
    X = []                                                      # [pass][row] -> tuple of 13 switches
    same = 0
    for p in range(a.check_passes):
        Xp = []
        for i in range(n):
            seed = a.seed * 1000 + p * 100003 + i
            random.seed(seed)
            new = "".join(map(chr, ds_new[i]["prompt_ids"]))
            random.seed(seed)
            old = "".join(map(chr, ds_old[i]["prompt_ids"]))
            on = parse_attr_line(new)
            assert on is not None and new.count(ATTR_PREFIX) == 1
            # without the line: the A+ prompt under the same random state (the line is drawn last)
            assert new.replace(render_attr_line(on) + "\n", "", 1) == old, "rebuilt prompt differs from A+ beyond the line"
            same += 1
            st = rows["train"][i]["attr_state"]
            for k in KEYS:
                assert st[k] is None or (k in on) == bool(st[k]), "positive finding: switch != target sentence"
            Xp.append(tuple(int(k in on) for k in KEYS))
        X.append(Xp)
    chk["train_recipe"] = {"passes": a.check_passes, "rows_per_pass": n,
                           "prompts_equal_to_aplus_recipe_without_line": same,
                           "positive_switch_equals_target_statement": True}
    # stored (fixed-seed) line as pass "stored"
    Xs = [tuple(int(k in parse_attr_line(r["prompt"])) for k in KEYS) for r in rows["train"]]

    def rates(Xp, ylab):
        o = {}
        for j, (k, d, _) in enumerate(ATTR_ITEMS):
            y = ylab[d]
            a1 = [x[j] for x, v in zip(Xp, y) if v == 1]
            a0 = [x[j] for x, v in zip(Xp, y) if v == 0]
            p1, p0 = sum(a1) / max(len(a1), 1), sum(a0) / max(len(a0), 1)
            se = (p1 * (1 - p1) / max(len(a1), 1) + p0 * (1 - p0) / max(len(a0), 1)) ** 0.5
            o[k] = {"n_pos": len(a1), "n_other": len(a0), "on_rate_pos": p1, "on_rate_other": p0, "diff": p0 - p1,
                    "diff_se": se, "z": (p0 - p1) / se if se else 0.0,
                    "switch_auroc_for_label": auc([int(v == 1) for v in y if v is not None],
                                                  [x[j] for x, v in zip(Xp, y) if v is not None])}
        return o
    leak = {"by_target_positive": {"stored_prompt": rates(Xs, ypos)}, "by_cue_label": {"stored_prompt": rates(Xs, ycue)}}
    for p in range(a.check_passes):
        leak["by_target_positive"][f"pass{p}"] = rates(X[p], ypos)
        leak["by_cue_label"][f"pass{p}"] = rates(X[p], ycue)
    # pooled over passes (the positive rows repeat: their switches are fixed; the others are redrawn)
    pooled = [x for Xp in X for x in Xp]
    leak["by_target_positive"]["pooled"] = rates(pooled, {d: ypos[d] * a.check_passes for d in ypos})
    leak["by_cue_label"]["pooled"] = rates(pooled, {d: ycue[d] * a.check_passes for d in ycue})
    # whole switch vector -> each finding's label
    vec = {}
    for d in ATTR_DISEASES:
        e = {"table_auroc_target_positive": [crossfit_table_auc(X[p], ypos[d], grp) for p in range(a.check_passes)]}
        yc = ycue[d]
        keep = [i for i in range(n) if yc[i] is not None]
        e["table_auroc_cue_label"] = [crossfit_table_auc([X[p][i] for i in keep], [yc[i] for i in keep],
                                                         [grp[i] for i in keep]) for p in range(a.check_passes)]
        try:
            import numpy as np
            from sklearn.linear_model import LogisticRegression
            g = np.array(grp)
            A = np.array(X[0], float)
            A = np.c_[A, A.sum(1, keepdims=True)]
            y = np.array(ypos[d])
            s = np.zeros(n)
            for h in (0, 1):
                s[g == h] = LogisticRegression(max_iter=500).fit(A[g != h], y[g != h]).predict_proba(A[g == h])[:, 1]
            e["logreg_auroc_target_positive_pass0"] = auc(list(map(int, y)), list(map(float, s)))
        except ImportError:
            pass
        vec[d] = e
    leak["whole_vector"] = vec
    chk["no_leak"] = leak
    # ---- 3. how often everything is on (train recipe) vs inference (always)
    rel = []
    for Xp in X:
        for x, r in zip(Xp, rows["train"]):
            ks = [j for j, k in enumerate(KEYS) if r["attr_state"][k] is not None]
            if ks:
                rel.append(all(x[j] for j in ks))
    chk["all_on"] = {"train_rows_every_switch_on": sum(all(x) for x in pooled) / len(pooled),
                     "train_rows_every_switch_on_count": sum(all(x) for x in pooled), "rows_checked": len(pooled),
                     "rows_with_a_positive_attribute_finding": len(rel) / len(pooled),
                     "of_those_every_relevant_switch_on": sum(rel) / max(len(rel), 1),
                     "mean_switches_on_train": sum(map(sum, pooled)) / len(pooled), "switches_on_inference": len(KEYS),
                     "train_on_rate_per_switch": {k: sum(x[j] for x in pooled) / len(pooled) for j, k in enumerate(KEYS)}}
    # ---- 4. token lengths
    if a.llm:
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained(a.llm)
        end = len(t("<|im_end|>", add_special_tokens=False)["input_ids"])
        L = {}
        for nm, rr, f in (("aplus", aplus["train"], lambda r: r["prompt"]),
                          ("attr_all_on", rows["train"], lambda r: replace_attr_line(r["prompt"], allon)),
                          ("attr_stored", rows["train"], lambda r: r["prompt"])):
            lp = [len(x) for x in t([chatml(f(r), False) for r in rr], add_special_tokens=False)["input_ids"]]
            if nm == "aplus":
                lt = [len(x) + end for x in t([r["target"] for r in rr], add_special_tokens=False)["input_ids"]]
            tot = [x + y for x, y in zip(lp, lt)]
            L[nm] = {"prompt_max": max(lp), "prompt_mean": sum(lp) / len(lp), "total_max": max(tot),
                     "total_mean": sum(tot) / len(tot), "rows_over_max_len": sum(x > a.max_len for x in tot),
                     "target_tokens_cut": sum(max(x - a.max_len, 0) for x in tot)}
        lv = [len(x) for x in t([chatml(r["prompt"], False) for r in rows["val"]], add_special_tokens=False)["input_ids"]]
        L["val_prompt_max"] = max(lv)
        L["all_on_line_tokens"] = len(t(allon + "\n", add_special_tokens=False)["input_ids"])
        L["max_len"] = a.max_len
        L["note"] = "text-only prompts (no image tokens), train rows, stored OOF-cue prompt (the gt-rebuilt cue is shorter or equal)"
        chk["token_lengths"] = L
    chk["note"] = "aggregates only; switches of val / inference are all on and never read a val report"
    (out / a.checks_name).write_text(json.dumps(chk, indent=1))
    print(json.dumps(chk, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplus-data", required=True)
    ap.add_argument("--aplus-val600", required=True)
    ap.add_argument("--aplus-cue-name", default="unit_cue_pil.json")
    ap.add_argument("--targets-train", default="", help="targets_v2/targets_train.jsonl (build only; TRAIN file)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-val600", required=True)
    ap.add_argument("--cue-name", default="unit_cue_pil_attr.json")
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--check", action="store_true", help="run the checks on the built data instead of building")
    ap.add_argument("--checks-name", default="attr_checks.json")
    ap.add_argument("--check-passes", type=int, default=3)
    ap.add_argument("--cue-gt", default="", help="--check: cue/cue_gt.json (train.py --cue-gt)")
    ap.add_argument("--llm", default="", help="--check: tokenizer dir for the token-length check")
    ap.add_argument("--max-len", type=int, default=1024)
    a = ap.parse_args()
    if a.check:
        if not a.cue_gt:
            raise SystemExit("--check needs --cue-gt")
        check(a)
    else:
        if not a.targets_train:
            raise SystemExit("build needs --targets-train")
        build(a)


if __name__ == "__main__":
    main()
