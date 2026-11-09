"""Probe-head scores for NEW studies with the exact analysis OOF script (same code, seeds, early-stopping split).
DO NOT RUN ON TEST until the user has locked the configuration.

usage: ib_test_heads.py <analysis oof script> <mw_exp/src dir> <script args ...>
The analysis OOF scripts train 5-fold OOF on train and then refit on all train and score the rows whose split is
"val" (vai = np.where(S == "val")...). This wrapper runs the same source with only three textual changes:
  1. the row selector of the refit-and-score step becomes split == "test";
  2. output names: oof_train*.csv -> oof_train*_testrun.csv, scores_val*.csv -> scores_test*.csv (nothing is overwritten);
  3. AUROC prints on unlabeled test rows return NaN instead of raising.
The embedding npz / labels csv given to the script must contain the train rows (with labels) plus the test rows
(split = "test", label NaN), produced with the same dump script (e.g. hn_dump.py) on train+test ids.
The train folds, seeds and refit are identical to the val run, so the head is the same recipe that produced the
val scores used by the integrated stacks."""
import re, sys, runpy, os, tempfile
script, src = sys.argv[1], sys.argv[2]
code = open(script).read()
pat = re.compile(r"vai = np\.where\((?:\(S == \"val\"\)[^\n]*?|S == \"val\")\)\[0\]")
assert pat.search(code), "row selector not found - check the script before using this wrapper"
code = pat.sub('vai = np.where(S == "test")[0]', code)
code = re.sub(r"(oof_train[a-z_]*)\.csv", r"\1_testrun.csv", code)
code = code.replace("scores_val", "scores_test")
code = code.replace('in ("train", "val")', 'in ("train", "val", "test")')   # sbo_oof keeps only train/val rows
sys.path.insert(0, src)
import numpy as np, probe                      # wrap probe.auroc before the script imports it
_a = probe.auroc
def _safe(y, s):
    y = np.asarray(y, float)
    return float("nan") if np.isnan(y).all() else _a(y[~np.isnan(y)], np.asarray(s)[~np.isnan(y)])
probe.auroc = _safe
tmp = os.path.join(tempfile.mkdtemp(), os.path.basename(script))
open(tmp, "w").write(code)
sys.argv = [script] + sys.argv[3:]
runpy.run_path(tmp, run_name="__main__")
