"""D8: compact, citable tables assembled from the other outputs (for the report integrator).

out/cite_levers.csv            one row per lever: raw, model what-if, adjusted (double ML), negative control
out/cite_weekend_crowding.csv  weekend premium three ways + the identification, neighbour, composition checks
"""
import numpy as np
import pandas as pd

import common as C

df, T, S, Y = C.build()
L = pd.read_csv(C.OUT / "d2_ladder.csv"); N = pd.read_csv(C.OUT / "d1_naive_levers.csv").set_index("lever")
NC = pd.read_csv(C.OUT / "d5_negative_control.csv"); SA = pd.read_csv(C.OUT / "d5_sensitivity.csv").set_index("lever")
g = lambda spec, mode: L[(L.spec == spec) & (L["mode"] == mode)].set_index("lever")
raw, adj, one = g("none", "single"), g("title", "joint"), g("title", "single")
nc = NC[NC.outcome.str.startswith("linked") & (NC["mode"] == "joint")].set_index("lever")
own = NC[NC.outcome.str.startswith("own shares, same") & (NC["mode"] == "joint")].set_index("lever")
rows = []
for l in ["imgs", "videos", "length", "title_len", "links_out", "links_self", "subj_title", "pol_title", "subj_body", "vol_same", "vol_other"]:
    unit = "doubling (1 + count)" if l in C.LEVERS else "+1 standard deviation"
    r = dict(lever=l, label=(C.LEVERS.get(l) or C.STYLE[l])[0], unit=unit,
             raw_pct=raw.pct[l], raw_lo=raw.pct_lo[l], raw_hi=raw.pct_hi[l],
             model_whatif_pct=N.gbdt_whatif_pct.get(l, np.nan),
             adjusted_pct=adj.pct[l], adjusted_lo=adj.pct_lo[l], adjusted_hi=adj.pct_hi[l],
             adjusted_alone_pct=one.pct[l], adjusted_alone_lo=one.pct_lo[l], adjusted_alone_hi=one.pct_hi[l])
    if l in nc.index and l != "links_self":   # the control only exists when there is an internal link
        r.update(negative_control_pct=nc.pct[l], negative_control_lo=nc.pct_lo[l], negative_control_hi=nc.pct_hi[l],
                 adjusted_same_articles_pct=own.pct[l], adjusted_same_articles_lo=own.pct_lo[l], adjusted_same_articles_hi=own.pct_hi[l])
    if l in SA.index:
        r.update(robustness_value=SA.RV[l])
    rows.append(r)
A = pd.DataFrame(rows); A.round(3).to_csv(C.OUT / "cite_levers.csv", index=False)
print(A[["lever", "raw_pct", "model_whatif_pct", "adjusted_pct", "adjusted_lo", "adjusted_hi", "negative_control_pct", "negative_control_lo", "negative_control_hi"]].round(1).to_string(index=False))

Wk = pd.read_csv(C.OUT / "d7_weekend_crowding.csv").set_index("quantity"); K = pd.read_csv(C.OUT / "d7_crowding_checks.csv")
rows = [dict(item="weekend premium, raw", pct=raw.pct["weekend"], lo=raw.pct_lo["weekend"], hi=raw.pct_hi["weekend"]),
        dict(item="weekend premium, adjusted for W (double ML)", pct=one.pct["weekend"], lo=one.pct_lo["weekend"], hi=one.pct_hi["weekend"]),
        dict(item="weekend premium predicted from the Monday-Friday volume slopes", **dict(zip(["pct", "lo", "hi"], Wk.loc["weekend premium predicted from weekday crowding (%)", ["value", "lo", "hi"]]))),
        dict(item="same, slopes identified within week only", **dict(zip(["pct", "lo", "hi"], Wk.loc["same, slopes identified within week only (%)", ["value", "lo", "hi"]]))),
        dict(item="weekend premium with the day's volume held fixed", pct=adj.pct["weekend"], lo=adj.pct_lo["weekend"], hi=adj.pct_hi["weekend"]),
        dict(item="same-section volume, doubling (main)", pct=adj.pct["vol_same"], lo=adj.pct_lo["vol_same"], hi=adj.pct_hi["vol_same"]),
        dict(item="other-section volume, doubling (main)", pct=adj.pct["vol_other"], lo=adj.pct_lo["vol_other"], hi=adj.pct_hi["vol_other"])]
pl = NC[NC.lever.str.contains("placebo") & (NC["mode"] == "single")].iloc[0]
rows.append(dict(item="placebo: same-section volume 7 days later", pct=pl.pct, lo=pl.pct_lo, hi=pl.pct_hi))
lab = {"vol_same": "same-section", "vol_other": "other-section", "vol_same_lag1": "same-section, day t-1", "vol_same_lead1": "same-section, day t+1",
       "vol_other_lag1": "other-section, day t-1", "vol_other_lead1": "other-section, day t+1"}
for _, r in K.dropna(subset=["se"]).iterrows():
    rows.append(dict(item=f"check [{r.check}] {r.spec}: {lab[r.term]} volume, doubling", pct=r.pct_per_doubling, lo=r.pct_lo, hi=r.pct_hi))
for q in Wk.index[Wk.index.str.startswith("quiet weekdays")]:
    rows.append(dict(item=q.replace(" (%)", ""), pct=Wk.value[q], lo=Wk.lo[q], hi=Wk.hi[q]))
rows.append(dict(item="weekend / weekday ratio of same-section articles per day", pct=np.nan, lo=np.nan, hi=np.nan, value=Wk.value["articles per section-day, weekend / weekday"]))
rows.append(dict(item="weekend / weekday ratio of other-section articles per day", pct=np.nan, lo=np.nan, hi=np.nan, value=Wk.value["other-section articles per day, weekend / weekday"]))
B = pd.DataFrame(rows); B.round(3).to_csv(C.OUT / "cite_weekend_crowding.csv", index=False)
print(B.round(1).to_string(index=False))
print("sample:", len(df), df.publish_date.min().date(), df.publish_date.max().date(), "days:", df.day.nunique(), "weeks:", df.week.nunique())
