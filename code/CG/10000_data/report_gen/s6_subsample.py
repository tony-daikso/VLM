#!/usr/bin/env python3
"""S6 subsample: A6000-only Qwen time for all 4,879 test drafts was projected at ~4.2 h alone (3.1 s/report at
32 concurrent) and is shared with another agent, so S6 re-extraction uses N_TARGET test exams:
all test exams that are def-B positive for the rare classes (appendicitis, bowel obstruction, free air, AAA),
the 64 random smoke-test exams, then LOC-stratified random fill. Order is shuffled so a partial run is still mixed.
Labels are used only to choose which exams to extract, never for thresholds. Output: ids_s6_subsample.txt."""
import numpy as np
import pandas as pd

from common import OUT, defb_labels

N_TARGET, SEED = 2400, 20261109
RARE = ["appendicitis", "bowel_obstruction", "free_air", "abdominal_aortic_aneurysm"]
SCR = "/private/tmp/claude-501/-Users-tony-tu-Desktop------VLM-data-CT-CG-10000-data/ff0f36b2-4d0f-420b-b826-27f6225f000d/scratchpad"


def main():
    sp = pd.read_csv(OUT / "split.csv").set_index("EXAMID")
    test = sp[sp.split == "test"]
    lab = defb_labels().loc[test.index]
    must = set(test.index[(lab[RARE] == 1).any(axis=1)])
    smoke = [l.strip() for l in open(f"{SCR}/ids_test_order.txt")][:64]
    must |= set(smoke)
    rng = np.random.default_rng(SEED)
    rest = test.drop(index=list(must))
    need = N_TARGET - len(must)
    fill = []
    for loc, g in rest.groupby("LOC"):
        n = int(round(need * len(g) / len(rest)))
        fill += list(rng.choice(g.index.values, size=min(n, len(g)), replace=False))
    ids = list(must) + fill
    rng.shuffle(ids)
    s = [i for i in ids if i in smoke] + [i for i in ids if i not in smoke]
    open(OUT / "ids_s6_subsample.txt", "w").write("\n".join(s) + "\n")
    print("must", len(must), "fill", len(fill), "total", len(s))
    print(sp.loc[s].LOC.value_counts())


if __name__ == "__main__":
    main()
