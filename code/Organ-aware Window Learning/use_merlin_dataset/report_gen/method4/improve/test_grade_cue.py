#!/usr/bin/env python3
"""Agent H (11-12): unit tests for the A+ grade line (m4_common grade helpers, build_unit_data.py --grade-cue-gt) and
the no-change guarantee for arms without it. Synthetic rows only. usage: python -m pytest -q test_grade_cue.py
(or python test_grade_cue.py)."""
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from m4_common import (GRADE_KEYS_HY, GRADE_PREFIX, UNIT_PREFIX, grade_keys_shown, grade_line_from_scores,  # noqa: E402
                       gt_grade_values, render_grade_line, replace_cue, replace_grade_line, replace_unit_line,
                       render_unit_line)

THR = {"hepatic_steatosis": 0.30355, "ascites": 0.34422, "atherosclerosis": 0.38539}
CUE = ("Image classifier probabilities (0-1): hepatic steatosis 0.31; ascites 0.34; atherosclerosis 0.90; "
       "fracture 0.10.")
PROMPT = "\n".join(["CT abdomen/pelvis", "Patient: x.", "<image>", CUE,
                    UNIT_PREFIX + "right kidney hydronephrosis 0.10; left kidney hydronephrosis 0.20.",
                    "Image measurements: liver 50 HU.", "Write the report."])
POOLS = {"hs_grade": {"pos": [0.5, 0.6], "neg": [0.1]}, "asc_amount": {"pos": [0.8], "neg": [0.2]},
         "ath_grade": {"pos": [0.7], "neg": [0.3, 0.4]}}


class T(unittest.TestCase):
    def test_gate(self):
        self.assertEqual(grade_keys_shown(PROMPT, THR), ["hs_grade", "ath_grade"])   # ascites 0.34 < .34422
        self.assertEqual(grade_keys_shown(PROMPT.replace(CUE + "\n", ""), THR), [])   # no cue line -> nothing

    def test_render_place_remove(self):
        line = grade_line_from_scores(PROMPT, {"hs_grade": 0.71, "asc_amount": 0.9, "ath_grade": 0.05}, THR)
        self.assertEqual(line, GRADE_PREFIX + "hepatic steatosis moderate or severe 0.71; "
                                              "atherosclerosis moderate or severe 0.05.")
        p = replace_grade_line(PROMPT, line)
        ls = p.split("\n")
        self.assertTrue(ls[ls.index(next(l for l in ls if l.startswith(UNIT_PREFIX))) + 1].startswith(GRADE_PREFIX))
        self.assertEqual(replace_grade_line(p, ""), PROMPT)
        self.assertEqual(replace_grade_line(PROMPT, ""), PROMPT)
        self.assertEqual(render_grade_line({}), "")
        # no unit line (both parts dropped) -> right after the classifier line
        q = replace_grade_line(replace_unit_line(PROMPT, ""), line)
        self.assertEqual(q.split("\n")[4], line)
        # grade line survives a unit line rewrite and a cue rewrite
        p2 = replace_unit_line(p, render_unit_line([0.5, None], None))
        self.assertEqual(sum(l.startswith(GRADE_PREFIX) for l in p2.split("\n")), 1)
        self.assertEqual(sum(l.startswith(GRADE_PREFIX) for l in replace_cue(p, {"fracture": 0.5}).split("\n")), 1)

    def test_gt_values(self):
        rng = random.Random(0)
        lab = {"hs_grade": 1, "asc_amount": 0, "ath_grade": None}
        oof = {"hs_grade": 0.2, "asc_amount": 0.99, "ath_grade": 0.42}
        for _ in range(50):
            v = gt_grade_values(["hs_grade", "asc_amount", "ath_grade"], lab, oof, POOLS, rng)
            self.assertIn(v["hs_grade"], POOLS["hs_grade"]["pos"])
            self.assertEqual(v["asc_amount"], 0.2)
            self.assertEqual(v["ath_grade"], 0.42)                  # null label -> own OOF score
        r1, r2 = random.Random(5), random.Random(5)
        self.assertEqual(gt_grade_values([], lab, oof, POOLS, r1), {})
        self.assertEqual(r1.random(), r2.random())                  # no key shown -> no rng draw

    def test_build_unit_data_default_unchanged(self):
        """without --grade-cue-gt the output equals the old behaviour (no grade line, no grade_scores)."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "s.csv").write_text("case_id,split,kid_right,kid_left,T8,T9,T10,T11,T12,L1,L2,L3,L4,L5,hs_grade,asc_amount,ath_grade\n"
                                     "c1,train,0.1,0.2,,0.3,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.81,0.5,0.25\n")
            (d / "in.jsonl").write_text(json.dumps({"case_id": "c1", "split": "train", "prompt": PROMPT, "target": "t"}) + "\n")
            (d / "cg.json").write_text(json.dumps({"thresholds": THR}))
            bud = str(HERE / "build_unit_data.py")
            subprocess.run([sys.executable, bud, "--scores", str(d / "s.csv"), "--in", str(d / "in.jsonl"),
                            "--out", str(d / "a.jsonl")], check=True, capture_output=True)
            subprocess.run([sys.executable, bud, "--scores", str(d / "s.csv"), "--in", str(d / "in.jsonl"),
                            "--out", str(d / "b.jsonl"), "--grade-cue-gt", str(d / "cg.json"),
                            "--stats-out", str(d / "st.json")], check=True, capture_output=True)
            a = json.loads((d / "a.jsonl").read_text())
            b = json.loads((d / "b.jsonl").read_text())
            self.assertNotIn("grade_scores", a)
            self.assertNotIn(GRADE_PREFIX, a["prompt"])
            self.assertEqual(b["grade_scores"], {"hs_grade": 0.81, "asc_amount": 0.5, "ath_grade": 0.25})
            self.assertIn(GRADE_PREFIX + "hepatic steatosis moderate or severe 0.81; atherosclerosis moderate or severe 0.25.",
                          b["prompt"])
            self.assertEqual(replace_grade_line(b["prompt"], ""), a["prompt"])
            st = json.loads((d / "st.json").read_text())
            self.assertEqual(st["grade_shown"], {"hs_grade": 1, "asc_amount": 0, "ath_grade": 1})
            # test rows refused
            (d / "t.jsonl").write_text(json.dumps({"case_id": "c1", "split": "test", "prompt": PROMPT}) + "\n")
            r = subprocess.run([sys.executable, bud, "--scores", str(d / "s.csv"), "--in", str(d / "t.jsonl"),
                                "--out", str(d / "c.jsonl")], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)

    # ---- agent I (11-12): optional hydronephrosis grade item
    def test_hy_key_off_by_default(self):
        thr = dict(THR, hydronephrosis=0.33165)
        p = PROMPT.replace("fracture 0.10.", "fracture 0.10; hydronephrosis 0.80.")
        sc = {"hs_grade": 0.71, "asc_amount": 0.9, "ath_grade": 0.05, "hy_grade": 0.66}
        # default keys: the hydronephrosis item never appears, even with a score and a threshold present
        self.assertEqual(grade_keys_shown(p, thr), ["hs_grade", "ath_grade"])
        self.assertEqual(grade_line_from_scores(p, sc, thr), grade_line_from_scores(PROMPT, sc, THR))
        self.assertEqual(render_grade_line(sc), render_grade_line({k: v for k, v in sc.items() if k != "hy_grade"}))
        # with the key list: last item, gated by the hydronephrosis cue
        self.assertEqual(grade_keys_shown(p, thr, GRADE_KEYS_HY), ["hs_grade", "ath_grade", "hy_grade"])
        self.assertEqual(grade_line_from_scores(p, sc, thr, GRADE_KEYS_HY),
                         grade_line_from_scores(PROMPT, sc, THR)[:-1] + "; hydronephrosis moderate or severe 0.66.")
        low = p.replace("hydronephrosis 0.80", "hydronephrosis 0.33")
        self.assertEqual(grade_line_from_scores(low, sc, thr, GRADE_KEYS_HY), grade_line_from_scores(PROMPT, sc, THR))
        self.assertEqual(grade_line_from_scores(p, dict(sc, hy_grade=None), thr, GRADE_KEYS_HY),
                         grade_line_from_scores(PROMPT, sc, THR))                      # no score -> item left out
        only = "\n".join(["CT", "<image>", "Image classifier probabilities (0-1): hydronephrosis 0.90.", "Write."])
        self.assertEqual(grade_line_from_scores(only, sc, thr, GRADE_KEYS_HY),
                         GRADE_PREFIX + "hydronephrosis moderate or severe 0.66.")
        rng = random.Random(1)
        pools = dict(POOLS, hy_grade={"pos": [0.9], "neg": [0.1]})
        v = gt_grade_values(["hs_grade", "hy_grade"], {"hs_grade": None, "hy_grade": 0}, {"hs_grade": 0.4, "hy_grade": 0.7},
                            pools, rng)
        self.assertEqual(v, {"hs_grade": 0.4, "hy_grade": 0.1})
        self.assertEqual(gt_grade_values(["hy_grade"], {"hy_grade": None}, {"hy_grade": 0.7}, pools, rng), {"hy_grade": 0.7})

    def test_build_unit_data_hy(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            p = PROMPT.replace("fracture 0.10.", "fracture 0.10; hydronephrosis 0.80.")
            (d / "s.csv").write_text("case_id,split,kid_right,kid_left,T8,T9,T10,T11,T12,L1,L2,L3,L4,L5,hs_grade,asc_amount,ath_grade\n"
                                     "c1,train,0.1,0.2,,0.3,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.81,0.5,0.25\n"
                                     "c2,train,0.1,0.2,,0.3,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.81,0.5,0.25\n")
            (d / "h.csv").write_text("case_id,split,hy_grade,hy_grade_right,hy_grade_left,hy_severe\n"
                                     "c1,train,0.664,0.1,0.664,0.2\nc2,train,,,,\n")
            (d / "in.jsonl").write_text("".join(json.dumps({"case_id": c, "split": "train", "prompt": p, "target": "t"}) + "\n"
                                                for c in ("c1", "c2")))
            (d / "cg.json").write_text(json.dumps({"thresholds": dict(THR, hydronephrosis=0.33165)}))
            bud = str(HERE / "build_unit_data.py")
            base = [sys.executable, bud, "--scores", str(d / "s.csv"), "--in", str(d / "in.jsonl"), "--grade-cue-gt",
                    str(d / "cg.json")]
            subprocess.run(base + ["--out", str(d / "a.jsonl")], check=True, capture_output=True)
            subprocess.run(base + ["--out", str(d / "b.jsonl"), "--hy-grade-scores", str(d / "h.csv"), "--stats-out",
                                   str(d / "st.json")], check=True, capture_output=True)
            a = [json.loads(l) for l in (d / "a.jsonl").read_text().splitlines()]
            b = [json.loads(l) for l in (d / "b.jsonl").read_text().splitlines()]
            self.assertNotIn("hy_grade", a[0]["grade_scores"])
            self.assertNotIn("hydronephrosis moderate", a[0]["prompt"])
            self.assertEqual(b[0]["grade_scores"]["hy_grade"], 0.66)
            self.assertIn("atherosclerosis moderate or severe 0.25; hydronephrosis moderate or severe 0.66.", b[0]["prompt"])
            self.assertIsNone(b[1]["grade_scores"]["hy_grade"])
            self.assertEqual(b[1]["prompt"], a[1]["prompt"])                           # no score -> the A+ prompt
            st = json.loads((d / "st.json").read_text())
            self.assertEqual(st["grade_shown"]["hy_grade"], 1)
            self.assertEqual(st["grade_scored"]["hy_grade"], 1)
            r = subprocess.run([sys.executable, bud, "--scores", str(d / "s.csv"), "--in", str(d / "in.jsonl"), "--out",
                                str(d / "c.jsonl"), "--hy-grade-scores", str(d / "h.csv")], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)                                       # needs --grade-cue-gt


if __name__ == "__main__":
    unittest.main()
