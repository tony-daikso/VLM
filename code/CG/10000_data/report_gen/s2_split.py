#!/usr/bin/env python3
"""S2: PID-grouped dev (30%) / test (70%) split of the 6,970 OWL-scored CG exams. Fixed seed.
Output: report_gen/split.csv (EXAMID, PID, LOC, split)."""
import numpy as np
import pandas as pd

from common import OUT, PROBS, read_reports

SEED = 20261109
DEV_FRAC = 0.30


def main():
    ids = pd.read_csv(PROBS, usecols=["study_id"]).study_id
    rep = read_reports().set_index("EXAMID").loc[ids, ["PID", "LOC"]].reset_index()
    pids = np.array(sorted(rep.PID.unique()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(pids)
    dev = set(pids[: int(round(DEV_FRAC * len(pids)))])
    rep["split"] = np.where(rep.PID.isin(dev), "dev", "test")
    assert rep.groupby("PID").split.nunique().max() == 1
    rep.to_csv(OUT / "split.csv", index=False)
    print(rep.groupby(["LOC", "split"]).size().unstack(), "\nPIDs", len(pids), "exams", len(rep))


if __name__ == "__main__":
    main()
