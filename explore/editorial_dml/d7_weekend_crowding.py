"""D7: is the weekend premium a crowding effect?

Weekend articles get more shares, and weekends are also the days with the fewest articles. Test: estimate the
effect of the day's volume (same section, other sections) using Monday-Friday articles only, with the day of
the week held fixed, then multiply by how much emptier weekends are. If crowding is the mechanism, this
out-of-sample prediction should match the weekend premium estimated directly.

Output: out/d7_weekend_crowding.csv
"""
import numpy as np
import pandas as pd

import common as C

df, T, S, Y = C.build()
W = C.control_sets(df)
wk = df.week.to_numpy()
dm = lambda a: a - a.mean(0)
wd = (T.weekend == 0).to_numpy()
gap = np.array([T.vol_same[~wd].mean() - T.vol_same[wd].mean(), T.vol_other[~wd].mean() - T.vol_other[wd].mean()])
rows = []
for rep in range(5):
    ry = C.residuals(df, "Y", Y, "title", rep, W)
    R = np.column_stack([C.residuals(df, l, T[l].to_numpy(), "title", rep, W) for l in ("vol_same", "vol_other")])
    b, V = C.cluster_ols(dm(ry[wd]), dm(R[wd]), wk[wd])
    ryb = C.residuals(df, "Y", Y, "title", rep, W, lever="weekend"); rw = C.residuals(df, "weekend", T.weekend.to_numpy(), "title", rep, W, lever="weekend")
    bw, Vw = C.cluster_ols(dm(ryb), dm(rw), wk)
    rows.append(dict(rep=rep, b_same=b[0], b_other=b[1], pred=gap @ b, pred_se=np.sqrt(gap @ V @ gap), direct=bw[0], direct_se=np.sqrt(Vw[0, 0])))
E = pd.DataFrame(rows)
pred, pse = C.combine(E.pred, E.pred_se); dirc, dse = C.combine(E.direct, E.direct_se)
out = pd.DataFrame([
    dict(quantity="articles per section-day, weekend / weekday", value=np.exp(gap[0])),
    dict(quantity="other-section articles per day, weekend / weekday", value=np.exp(gap[1])),
    dict(quantity="volume effect from weekdays only: same section (log-log slope)", value=E.b_same.median()),
    dict(quantity="volume effect from weekdays only: other sections (log-log slope)", value=E.b_other.median()),
    dict(quantity="weekend premium predicted from crowding alone (%)", value=100 * (np.exp(pred) - 1),
         lo=100 * (np.exp(pred - 1.96 * pse) - 1), hi=100 * (np.exp(pred + 1.96 * pse) - 1)),
    dict(quantity="weekend premium estimated directly (%)", value=100 * (np.exp(dirc) - 1),
         lo=100 * (np.exp(dirc - 1.96 * dse) - 1), hi=100 * (np.exp(dirc + 1.96 * dse) - 1))])
out.round(3).to_csv(C.OUT / "d7_weekend_crowding.csv", index=False)
print(out.round(3).to_string(index=False))
