import pandas as pd, numpy as np, sys
M={"hn":"analysis_hydronephrosis/hu_ct.csv","ath":"analysis_atherosclerosis/hu_ct.csv","fa":"analysis_free_air/air_ct.csv","asc":"analysis_ascites/hu_ct.csv",
"aaa":"analysis_aaa/geom_ct.csv","ag":"analysis_absent_gb/feat_ct.csv","sbo":"analysis_bowel_obstruction/feat_ct.csv","sp":"analysis_splenomegaly/geom_ct.csv",
"op":"analysis_osteopenia/hu_ct.csv","hs":"analysis_steatosis/hs_ct.csv","ap":"analysis_appendicitis/hu_ct.csv"}
for k,f in M.items():
    try: o=pd.read_csv(f"out_{k}.csv")
    except Exception as e: print(k,"no output"); continue
    r=pd.read_csv(f).drop_duplicates("study_id").set_index("study_id"); o=o.set_index("study_id")
    ids=[i for i in o.index if i in r.index]; cols=[c for c in o.columns if c in r.columns and pd.api.types.is_numeric_dtype(o[c])]
    a=o.loc[ids,cols].astype(float).values; b=r.loc[ids,cols].astype(float).values
    bad=~(np.isclose(a,b,rtol=1e-4,atol=1e-4)|(np.isnan(a)&np.isnan(b)))
    bc=[cols[j] for j in np.where(bad.any(0))[0]]
    print(f"{k}: n={len(ids)} cols={len(cols)} missing_cols={len(set(r.columns)-set(o.columns))} mismatched_cells={bad.sum()} cols={bc[:8]}")
