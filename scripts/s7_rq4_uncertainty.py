"""S7 (RQ4): what can a model promise? Conformalised quantile regression (CQR, Romano et al. 2019)
with and without adaptive conformal inference (ACI, Gibbs & Candes 2021) under drift, and selective
classification (risk-coverage).

CQR: HistGB quantile regressors (10% / 90%) of log(1+shares) trained on the oldest 70%, conformal
calibration on the next 10%, target coverage 80% on the newest 20%. ACI: same base intervals,
quantile of the most recent 3,000 scores, step gamma = 0.005, updated article by article in time
order (the true shares are assumed known before the next article is scored, as in an online setting;
in practice shares arrive with a delay, so this is optimistic).
Selective prediction: past-only HistGB (and L1 logistic regression) scores (monthly rolling origin, cached by s2); predict only the
fraction of articles with the largest |p - 0.5|.

Outputs: results/tables/s7_coverage.csv, results/tables/s7_selective.csv
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR

from newspop.data import ROOT, feature_set, load
from newspop.splits import temporal_split

df = load()
n = len(df)
yl = np.log1p(df.shares.to_numpy())
X = df[feature_set(df, "all")].to_numpy(float)
tr, ca, te = temporal_split(n, 0.7, 0.1)
alpha = 0.2
q = {a: HGBR(loss="quantile", quantile=a, max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
             random_state=0).fit(X[tr], yl[tr]).predict(X) for a in (alpha / 2, 1 - alpha / 2)}
L, H = q[alpha / 2], q[1 - alpha / 2]
score = np.maximum(L - yl, yl - H)
qhat = np.quantile(score[ca], np.ceil((len(ca) + 1) * (1 - alpha)) / len(ca))
cov_raw = (yl[te] >= L[te]) & (yl[te] <= H[te])
cov_cqr = (yl[te] >= L[te] - qhat) & (yl[te] <= H[te] + qhat)
gamma, a_t, hist, cov_aci, width_aci = 0.005, alpha, list(score[ca]), [], []
for t in te:
    qt = np.quantile(hist[-3000:], min(1.0, max(0.0, 1 - a_t)))
    c = (yl[t] >= L[t] - qt) & (yl[t] <= H[t] + qt)
    cov_aci.append(c)
    width_aci.append(H[t] - L[t] + 2 * qt)
    a_t += gamma * (alpha - (0 if c else 1))
    hist.append(score[t])
month = df.publish_date.dt.to_period("M").astype(str).to_numpy()[te]
C = pd.DataFrame({"month": month, "quantile regression": cov_raw, "CQR": cov_cqr, "CQR + ACI": cov_aci})
by = C.groupby("month").mean()
by.loc["all test"] = C.drop(columns="month").mean()
by["n"] = C.groupby("month").size().reindex(by.index).fillna(len(C)).astype(int)
w = np.exp(H[te] - L[te] + 2 * qhat)
by["median width ratio (CQR)"] = np.nan
by.loc["all test", "median width ratio (CQR)"] = np.median(w)
lo_sh, hi_sh = np.expm1(np.median(L[te] - qhat)), np.expm1(np.median(H[te] + qhat))
by.round(4).to_csv(ROOT / "results" / "tables" / "s7_coverage.csv")
print(by.round(3).to_string())
print(f"median 80% CQR interval: {lo_sh:.0f} - {hi_sh:.0f} shares; upper/lower ratio median {np.median(w):.1f}")

S = pd.read_csv(ROOT / "data" / "processed" / "scores.csv")
rows = []
for model in ["hgb_all", "logreg_all"]:  # L1 logistic regression as a baseline
    p = S[model].to_numpy()
    m = ~np.isnan(p)
    yb = (df.shares.to_numpy() >= 1400)[m]
    p = p[m]
    order = np.argsort(-np.abs(p - 0.5))
    correct = ((p > 0.5) == yb)[order]
    for cov in [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]:
        k = int(round(cov * len(p)))
        rows.append({"model": model, "coverage": cov, "accuracy": correct[:k].mean(), "n": k})
R = pd.DataFrame(rows).round(4)
R.to_csv(ROOT / "results" / "tables" / "s7_selective.csv", index=False)
print(R.to_string(index=False))
