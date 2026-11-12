"""S13: sensitivity to unobserved confounding (Cinelli & Hazlett 2020, JRSS-B) for the two main effects.

For an OLS coefficient with t-statistic t and residual degrees of freedom df, the robustness value
RV = 0.5 * (sqrt(f^4 + 4 f^2) - f^2), f = |t| / sqrt(df), is the share of residual variance that an
unobserved confounder would have to explain of BOTH the treatment and the outcome to bring the estimate
to zero; RV_alpha does the same for losing significance at 5%. Benchmarks: the partial R^2 with the
outcome (and with the treatment) of the strongest observed covariate.
Models (plain OLS t, as in the framework):
  1. within-twin subjectivity: twin-difference regression of s4 (main sample);
  2. daily volume: s5 with week, weekday, channel FE and all content controls.

Output: results/tables/s13_sensitivity.csv
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from newspop.data import ROOT, feature_groups, load
from newspop import stories as st


def rv(t, df, alpha=None):
    f = abs(t) / np.sqrt(df)
    if alpha is not None:
        f = max(0.0, f - stats.t.ppf(1 - alpha / 2, df - 1) / np.sqrt(df - 1))
    return 0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2)


def partial_r2(t, df):
    return t ** 2 / (t ** 2 + df)


rows = []
df = load()

# 1. twins
ls = np.log1p(df.shares.to_numpy())
chan = st.channel(df)
F = pd.DataFrame({
    "log body length": np.log1p(df.n_tokens_content), "log title length": np.log1p(df.n_tokens_title),
    "log images": np.log1p(df.num_imgs), "log videos": np.log1p(df.num_videos),
    "log links": np.log1p(df.num_hrefs), "log internal links": np.log1p(df.num_self_hrefs),
    "average word length": df.average_token_length, "number of keywords": df.num_keywords,
    "body subjectivity": df.global_subjectivity, "share positive words": df.global_rate_positive_words,
    "share negative words": df.global_rate_negative_words, "title subjectivity": df.title_subjectivity,
    "title polarity": df.title_sentiment_polarity})
F = (F - F.mean()) / F.std()
P = st.clean_twins(df)
later = np.where(P["first"] == P.i, P.j, P.i)
first = P["first"].to_numpy()
D = pd.DataFrame(F.to_numpy()[first] - F.to_numpy()[later], columns=[c.replace(" ", "_") for c in F.columns])
CH = pd.get_dummies(pd.Series(chan[first]), dtype=float) .reindex(columns=sorted(set(chan)), fill_value=0) \
    - pd.get_dummies(pd.Series(chan[later]), dtype=float).reindex(columns=sorted(set(chan)), fill_value=0)
W = pd.concat([D, CH.iloc[:, 1:].reset_index(drop=True).add_prefix("ch_")], axis=1)
m = sm.OLS(ls[first] - ls[later], sm.add_constant(W)).fit()
treat = "body_subjectivity"
covs = [c for c in D.columns if c != treat]
tt = m.tvalues[treat]
# strongest observed covariate: partial R2 with outcome, and with the treatment
po = {c: partial_r2(m.tvalues[c], m.df_resid) for c in covs}
mt = sm.OLS(W[treat], sm.add_constant(W.drop(columns=treat))).fit()
pt = {c: partial_r2(mt.tvalues[c], mt.df_resid) for c in covs}
best = max(covs, key=lambda c: po[c])
rows.append(dict(effect="within-twin body subjectivity", estimate=m.params[treat], t=tt, df=m.df_resid,
                 partial_r2_treatment=partial_r2(tt, m.df_resid), RV=rv(tt, m.df_resid), RV_alpha05=rv(tt, m.df_resid, 0.05),
                 strongest_covariate=best, cov_r2_outcome=po[best], cov_r2_treatment=pt[best],
                 max_cov_r2_treatment=max(pt.values())))

# 2. volume
d = pd.DataFrame({"ls": ls, "day": st.day_index(df), "week": st.week_index(df),
                  "wd": df.publish_date.dt.dayofweek, "chan": chan})
d["log_n_day"] = np.log(d.groupby("day").ls.transform("size"))
g = feature_groups(df)
content = [c for c in g["content"] + g["lda"] + g["sentiment"]
           if c not in ("LDA_04", "rate_negative_words", "n_non_stop_words")]
for c in content:
    d[c] = df[c].to_numpy()
fe = "C(week) + C(wd) + C(chan)"
mv = smf.ols(f"ls ~ log_n_day + {fe} + {' + '.join(content)}", d).fit()
tt = mv.tvalues["log_n_day"]
po = {c: partial_r2(mv.tvalues[c], mv.df_resid) for c in content}
best = max(po, key=po.get)
rows.append(dict(effect="daily volume (FE + content)", estimate=mv.params["log_n_day"], t=tt, df=mv.df_resid,
                 partial_r2_treatment=partial_r2(tt, mv.df_resid), RV=rv(tt, mv.df_resid), RV_alpha05=rv(tt, mv.df_resid, 0.05),
                 strongest_covariate=best, cov_r2_outcome=po[best], cov_r2_treatment=np.nan, max_cov_r2_treatment=np.nan))
R = pd.DataFrame(rows).round(4)
R.to_csv(ROOT / "results" / "tables" / "s13_sensitivity.csv", index=False)
print(R.T.to_string())
