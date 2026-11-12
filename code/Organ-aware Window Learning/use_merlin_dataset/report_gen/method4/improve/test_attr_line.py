#!/usr/bin/env python3
"""Agent N (11-12): unit tests for the "attributes to state" line (m4_common attribute helpers, build_attr_data.py,
m4_data.UnitCueGT / M4Dataset) and the no-change guarantee for arms without it. Synthetic rows only.
usage: python -m pytest -q test_attr_line.py (or python test_attr_line.py)."""
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from m4_common import (ATTR_DISEASES, ATTR_ITEMS, ATTR_PREFIX, DISEASES, GRADE_PREFIX, UNIT_PREFIX,  # noqa: E402
                       attr_line_all_on, parse_attr_line, render_attr_line, replace_attr_line, replace_cue,
                       replace_grade_line, replace_unit_line, sample_attr_switches)
import build_attr_data as bad  # noqa: E402

KEYS = [k for k, _, _ in ATTR_ITEMS]
CUE = "Image classifier probabilities (0-1): hepatic steatosis 0.31; ascites 0.90; atherosclerosis 0.90; fracture 0.10."
PROMPT = "\n".join(["CT abdomen/pelvis", "Patient: x.", "<image>", CUE,
                    UNIT_PREFIX + "right kidney hydronephrosis 0.10; left kidney hydronephrosis 0.20.",
                    GRADE_PREFIX + "ascites moderate or large 0.50.",
                    "Image measurements: liver 50 HU.", "Write the report."])
TABS = {"hepatic_steatosis": {"keys": ["hepatic_steatosis:grade"], "patterns": [[0], [1]], "cum": [80, 100]},
        "hydronephrosis": {"keys": ["hydronephrosis:grade", "hydronephrosis:side"],
                           "patterns": [[0, 0], [0, 1], [1, 1]], "cum": [10, 40, 100]},
        "splenomegaly": {"keys": ["splenomegaly:length"], "patterns": [[0], [1]], "cum": [50, 100]},
        "abdominal_aortic_aneurysm": {"keys": ["abdominal_aortic_aneurysm:diameter"], "patterns": [[0], [1]], "cum": [30, 100]},
        "atherosclerosis": {"keys": ["atherosclerosis:grade"], "patterns": [[0], [1]], "cum": [66, 100]},
        "bowel_obstruction": {"keys": ["bowel_obstruction:type"], "patterns": [[0], [1]], "cum": [10, 100]},
        "ascites": {"keys": ["ascites:amount"], "patterns": [[0], [1]], "cum": [25, 100]},
        "free_air": {"keys": ["free_air:postoperative"], "patterns": [[0], [1]], "cum": [60, 100]},
        "biliary_ductal_dilation": {"keys": ["biliary_ductal_dilation:site"], "patterns": [[0], [1]], "cum": [5, 100]},
        "fracture": {"keys": ["fracture:sites", "fracture:levels", "fracture:acuity"],
                     "patterns": [[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], "cum": [10, 20, 45, 75, 100]}}
NONE = {k: None for k in KEYS}


def tgt_row(cid, pos, split="train"):
    """synthetic targets_v2 row: pos = {disease: attributes}; every other finding is negative."""
    return {"case_id": cid, "split": split,
            "diseases": {d: {"status": "pos" if d in pos else "neg", "attributes": pos.get(d, {}),
                             "attributes_raw_only": {}} for d in DISEASES}}


class T(unittest.TestCase):
    def test_render_place_remove(self):
        self.assertEqual(render_attr_line(set()), ATTR_PREFIX + "none.")
        line = render_attr_line({"fracture:acuity", "ascites:amount"})
        self.assertEqual(line, ATTR_PREFIX + "ascites amount; fracture acuity.")          # fixed order, fixed wording
        allon = attr_line_all_on()
        self.assertEqual(allon, ATTR_PREFIX + "; ".join(t for _, _, t in ATTR_ITEMS) + ".")
        for k, _, txt in ATTR_ITEMS:                                                       # an item reads the same alone
            self.assertIn(txt, allon)
            self.assertEqual(render_attr_line({k}), ATTR_PREFIX + txt + ".")
        self.assertFalse(any(ch.isdigit() for ch in allon))                               # switches only, no value
        p = replace_attr_line(PROMPT, line)
        ls = p.split("\n")
        self.assertEqual(ls[-2], line)                                                     # right after the measurements
        self.assertTrue(ls[-3].startswith("Image measurements"))
        self.assertEqual(replace_attr_line(p, ""), PROMPT)
        self.assertEqual(replace_attr_line(PROMPT, ""), PROMPT)
        self.assertEqual(replace_attr_line(p, allon).split("\n")[-2], allon)
        self.assertEqual(parse_attr_line(p), {"fracture:acuity", "ascites:amount"})
        self.assertEqual(parse_attr_line(replace_attr_line(PROMPT, allon)), set(KEYS))
        self.assertEqual(parse_attr_line(replace_attr_line(PROMPT, render_attr_line(set()))), set())
        self.assertIsNone(parse_attr_line(PROMPT))
        # the line survives the other rewrites, and they are unchanged by it
        for f in (lambda q: replace_cue(q, {"fracture": 0.5}), lambda q: replace_unit_line(q, ""),
                  lambda q: replace_grade_line(q, "")):
            self.assertEqual(replace_attr_line(f(p), ""), f(PROMPT))

    def test_sample_positive_follows_target(self):
        st = dict(NONE)
        st.update({"hydronephrosis:grade": 0, "hydronephrosis:side": 1, "atherosclerosis:grade": 1, "ascites:amount": 0})
        rng = random.Random(3)
        for _ in range(200):
            on = sample_attr_switches(st, TABS, rng)
            self.assertNotIn("hydronephrosis:grade", on)
            self.assertIn("hydronephrosis:side", on)
            self.assertIn("atherosclerosis:grade", on)
            self.assertNotIn("ascites:amount", on)
        # one rng draw per non-positive finding, none for a positive one
        r1, r2 = random.Random(9), random.Random(9)
        sample_attr_switches(st, TABS, r1)
        for _ in range(len(ATTR_DISEASES) - 3):
            r2.random()
        self.assertEqual(r1.random(), r2.random())
        allpos = {k: 1 for k in KEYS}
        r1, r2 = random.Random(4), random.Random(4)
        self.assertEqual(sample_attr_switches(allpos, TABS, r1), set(KEYS))
        self.assertEqual(r1.random(), r2.random())

    def test_no_leak(self):
        """switches of non-positive findings follow the positive cases' joint pattern distribution: the on-rate (and the
        joint pattern rate) of a switch is the same whether or not the finding is positive."""
        rng = random.Random(11)
        n = 40000
        # positives: real states drawn from TABS (as real reports would be); negatives: None -> sampled
        rows = []
        for i in range(n):
            st = {}
            for d in ATTR_DISEASES:
                t = TABS[d]
                if rng.random() < 0.3:
                    u = rng.random() * t["cum"][-1]
                    j = next(q for q, c in enumerate(t["cum"]) if u < c)
                    st.update(dict(zip(t["keys"], t["patterns"][j])))
                else:
                    st.update({k: None for k in t["keys"]})
            rows.append(st)
        srng = random.Random(5)
        X = [sample_attr_switches(st, TABS, srng) for st in rows]
        for k, d, _ in ATTR_ITEMS:
            pos = [k in on for on, st in zip(X, rows) if st[k] is not None]
            neg = [k in on for on, st in zip(X, rows) if st[k] is None]
            p1, p0 = sum(pos) / len(pos), sum(neg) / len(neg)
            se = (p1 * (1 - p1) / len(pos) + p0 * (1 - p0) / len(neg)) ** 0.5
            self.assertLess(abs(p1 - p0), 4 * se + 1e-9, k)
            y = [int(st[k] is not None) for st in rows]
            self.assertLess(abs(bad.auc(y, [int(k in on) for on in X]) - 0.5), 0.02, k)
        # joint pattern (fracture: 3 switches)
        fk = TABS["fracture"]["keys"]
        for pat in TABS["fracture"]["patterns"]:
            f = lambda on: all((k in on) == bool(b) for k, b in zip(fk, pat))   # noqa: E731
            pos = [f(on) for on, st in zip(X, rows) if st[fk[0]] is not None]
            neg = [f(on) for on, st in zip(X, rows) if st[fk[0]] is None]
            self.assertLess(abs(sum(pos) / len(pos) - sum(neg) / len(neg)), 0.02)
        # whole switch vector -> label: no predictive power
        vec = [tuple(int(k in on) for k in KEYS) for on in X]
        grp = [i % 2 for i in range(n)]
        for d in ATTR_DISEASES:
            y = [int(st[TABS[d]["keys"][0]] is not None) for st in rows]
            self.assertLess(abs(bad.crossfit_table_auc(vec, y, grp) - 0.5), 0.03, d)
        # a leaky rule (off for every non-positive finding) is caught by the same check
        leaky = [tuple(int(st[k] is not None and st[k] == 1) for k in KEYS) for st in rows]
        y = [int(st["ascites:amount"] is not None) for st in rows]
        self.assertGreater(bad.crossfit_table_auc(leaky, y, grp), 0.8)

    def _build(self, d):
        rows = []
        for i in range(60):
            rows.append({"case_id": f"c{i}", "split": "train", "prompt": PROMPT, "target": "[ascites] t",
                         "target_diseases": list(DISEASES) if i == 7 else ["ascites", "fracture"] if i % 2 else ["ascites"],
                         "grade_scores": {"asc_amount": 0.5}})
        (d / "ap").mkdir()
        (d / "ap/train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        val = [{"case_id": f"v{i}", "split": "val", "prompt": PROMPT, "target": "x", "target_diseases": []} for i in range(5)]
        (d / "ap/val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in val))
        (d / "v600.jsonl").write_text("".join(json.dumps(r) + "\n" for r in val[:3]))
        (d / "ap/unit_cue_pil.json").write_text(json.dumps({"labels_train": {}, "pools": {}, "thresholds": {}}))
        tg = []
        for i in range(60):
            pos = {"ascites": {"amount": "small" if i % 3 else "not_stated"}}
            if i % 2:                                           # fracture positive with a target line
                pos["fracture"] = {"sites": ["vertebra"], "vertebral_levels": ["L1"] if i % 4 == 1 else [], "acuity": "not_stated"}
            else:                                               # positive status but no target line -> not positive
                pos["fracture"] = {"sites": ["rib"], "vertebral_levels": [], "acuity": "old"}
            if i == 7:                                          # one case with every other finding positive
                pos.update({"splenomegaly": {"length_mm": 150.0}, "hepatic_steatosis": {"grade": "not_stated"},
                            "hydronephrosis": {"grade": "mild", "side": "not_stated"},
                            "abdominal_aortic_aneurysm": {"diameter_mm": None}, "atherosclerosis": {"grade": "severe"},
                            "bowel_obstruction": {"type": "SBO"}, "free_air": {"postoperative": "yes"},
                            "biliary_ductal_dilation": {"site": "both"}})
            if i == 9:
                pos["splenomegaly"] = {"length_mm": 150.0}      # positive status, no target line
            tg.append(tgt_row(f"c{i}", pos))
        tg.append(tgt_row("not_in_data", {}))
        (d / "tt.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tg))
        base = [sys.executable, str(HERE / "build_attr_data.py"), "--aplus-data", str(d / "ap"), "--aplus-val600",
                str(d / "v600.jsonl"), "--out", str(d / "out"), "--out-val600", str(d / "o600.jsonl")]
        return base, rows, val

    def test_build(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            base, rows, val = self._build(d)
            subprocess.run(base + ["--targets-train", str(d / "tt.jsonl")], check=True, capture_output=True)
            tr = bad.read_rows(d / "out/train.jsonl")
            self.assertEqual(len(tr), 60)
            for i, (r, q) in enumerate(zip(tr, rows)):
                st = r["attr_state"]
                self.assertEqual(st["ascites:amount"], 1 if i % 3 else 0)
                if i % 2:
                    self.assertEqual([st["fracture:sites"], st["fracture:levels"], st["fracture:acuity"]],
                                     [1, 1 if i % 4 == 1 else 0, 0])
                else:
                    self.assertIsNone(st["fracture:sites"])                      # no target line -> sampled
                if i == 7:
                    self.assertEqual([st[k] for k in KEYS[:10]], [0, 1, 0, 1, 0, 1, 1, 1, 1, 1])
                else:
                    self.assertIsNone(st["splenomegaly:length"])                 # i == 9: not in target_diseases
                    self.assertIsNone(st["hepatic_steatosis:grade"])
                on = parse_attr_line(r["prompt"])
                self.assertEqual("ascites:amount" in on, bool(st["ascites:amount"]))
                s = {k: v for k, v in r.items() if k != "attr_state"}
                s["prompt"] = replace_attr_line(r["prompt"], "")
                self.assertEqual(json.dumps(s), json.dumps(q))                   # = the A+ row
            uc = json.loads((d / "out/unit_cue_pil_attr.json").read_text())
            t = uc["attr_patterns"]
            self.assertEqual(t["ascites"], {"keys": ["ascites:amount"], "patterns": [[0], [1]], "cum": [20, 60]})
            self.assertEqual(t["fracture"]["cum"][-1], 30)
            self.assertEqual(t["fracture"]["patterns"], [[1, 0, 0], [1, 1, 0]])
            # val / val600: all on, the val rows otherwise untouched, no targets needed
            allon = attr_line_all_on()
            for path, src in ((d / "out/val.jsonl", val), (d / "o600.jsonl", val[:3])):
                got = bad.read_rows(path)
                self.assertEqual(len(got), len(src))
                for r, q in zip(got, src):
                    self.assertNotIn("attr_state", r)
                    self.assertEqual(parse_attr_line(r["prompt"]), set(KEYS))
                    self.assertIn(allon, r["prompt"].split("\n"))
                    self.assertEqual(replace_attr_line(r["prompt"], ""), q["prompt"])
            # refusals: a val / test target file, a test row
            (d / "tv.jsonl").write_text(json.dumps(tgt_row("c0", {}, "val")) + "\n")
            r = subprocess.run(base + ["--targets-train", str(d / "tv.jsonl")], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            (d / "ap/val.jsonl").write_text(json.dumps({"case_id": "t", "split": "test", "prompt": PROMPT}) + "\n")
            r = subprocess.run(base + ["--targets-train", str(d / "tt.jsonl")], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)

    def test_dataset(self):
        """the real train-time path: M4Dataset rebuilds the line at every read; cue files without attr_patterns (every
        earlier arm) give exactly the old prompt."""
        try:
            from m4_data import M4Dataset, UnitCueGT
        except ImportError:
            self.skipTest("torch not available")
        tk = bad.CharTok()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            lab = {"c0": {"kid": [0, 0], "vert": [0] * 10, "asc_amount": 1}}
            cue = {"labels_train": lab, "thresholds": {"kidney": 0.3, "vertebra": 0.3, "asc_amount": 0.4},
                   "pools": {"kidney": {"pos": [0.9], "neg": [0.1]}, "vertebra": {"pos": [0.9], "neg": [0.1]},
                             "asc_amount": {"pos": [0.8], "neg": [0.2]}, "hs_grade": {"pos": [0.8], "neg": [0.2]},
                             "ath_grade": {"pos": [0.8], "neg": [0.2]}},
                   "grade_cue_thr": {"hepatic_steatosis": 0.5, "ascites": 0.5, "atherosclerosis": 0.95}}
            (d / "old.json").write_text(json.dumps(cue))
            (d / "new.json").write_text(json.dumps(dict(cue, attr_patterns=TABS)))
            st = dict(NONE)
            st.update({"ascites:amount": 1, "fracture:sites": 1, "fracture:levels": 0, "fracture:acuity": 0})
            row = {"case_id": "c0", "split": "train", "prompt": PROMPT, "target": "[ascites] t",
                   "unit_scores": {"kid": [0.1, 0.2], "vert": [0.1] * 10},
                   "grade_scores": {"hs_grade": 0.1, "asc_amount": 0.5, "ath_grade": 0.3}}
            (d / "a.jsonl").write_text(json.dumps(row) + "\n")
            (d / "n.jsonl").write_text(json.dumps(dict(row, attr_state=st, prompt=replace_attr_line(PROMPT, attr_line_all_on()))) + "\n")
            ds_a = M4Dataset(d / "a.jsonl", tk, use_image=False, unit_cue=UnitCueGT(str(d / "old.json"), 0.3, grades=True))
            ds_n = M4Dataset(d / "n.jsonl", tk, use_image=False, unit_cue=UnitCueGT(str(d / "new.json"), 0.3, grades=True))
            seen = set()
            for s in range(60):
                random.seed(s)
                a = "".join(map(chr, ds_a[0]["prompt_ids"]))
                random.seed(s)
                n = "".join(map(chr, ds_n[0]["prompt_ids"]))
                on = parse_attr_line(n)
                self.assertEqual(n.replace(render_attr_line(on) + "\n", "", 1), a)      # the rest = the old recipe
                self.assertNotIn(ATTR_PREFIX, a)
                self.assertIn("ascites:amount", on)
                self.assertIn("fracture:sites", on)
                self.assertNotIn("fracture:levels", on)
                self.assertNotIn("fracture:acuity", on)
                seen.add(frozenset(on))
            self.assertGreater(len(seen), 10)                                           # redrawn at every read
            # inference / val: the stored line is used as is
            ds_v = M4Dataset(d / "n.jsonl", tk, use_image=False, train=False,
                             unit_cue=UnitCueGT(str(d / "new.json"), 0.3, grades=True))
            self.assertIn(attr_line_all_on(), "".join(map(chr, ds_v[0]["prompt_ids"])))
            # rows with attr_state but a cue file without patterns -> refused
            with self.assertRaises(SystemExit):
                M4Dataset(d / "n.jsonl", tk, use_image=False, unit_cue=UnitCueGT(str(d / "old.json"), 0.3, grades=True))


if __name__ == "__main__":
    unittest.main()
