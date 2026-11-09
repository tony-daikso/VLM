import numpy as np, pandas as pd
def auc(y, s):
    y, s = np.asarray(y, float), np.asarray(s, float)
    m = ~np.isnan(y) & ~np.isnan(s); y, s = y[m], s[m]
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    if n1 == 0 or n0 == 0: return np.nan
    r = pd.Series(s).rank().values
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
