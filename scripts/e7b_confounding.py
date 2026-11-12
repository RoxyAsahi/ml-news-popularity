"""E7b (RQ4, review item R6) - Is the sentiment effect on popularity just a channel effect?
And what do the Isolation-Forest outliers look like?

1. Article-level: negative overall polarity (global_sentiment_polarity < 0) vs popular (>= 1400),
   crude and stratified by channel (Cochran-Mantel-Haenszel), plus a logistic regression
   popular ~ negative + channel dummies.
2. Cluster-level: the negative-polarity KMeans cluster (k = 9) vs the other clusters, same tests.
3. Isolation Forest with contamination 1%, 2%, 5%: popularity of the outliers, and their
   profile (feature medians vs inliers, channel mix).

Run from the repo root:  python scripts/e7b_confounding.py
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.contingency_tables import StratifiedTable

from newspop.data import ROOT, feature_groups, load, make_label

warnings.filterwarnings("ignore")
TAB = ROOT / "results" / "tables"
SEED = 0


def channel_of(df, g):
    return df[g["channel"]].idxmax(axis=1).str.replace("data_channel_is_", "").where(
        df[g["channel"]].sum(axis=1) > 0, "none")


def stratified(exposure, y, strata, name):
    rows = []
    tables = []
    for s in sorted(pd.unique(strata)):
        m = strata == s
        t = pd.crosstab(pd.Categorical(exposure[m], [1, 0]), pd.Categorical(y[m], [1, 0]), dropna=False).to_numpy()
        tables.append(t)
        rows.append({"exposure": name, "channel": s, "n_exposed": int(exposure[m].sum()),
                     "popular_exposed": y[m][exposure[m] == 1].mean() if exposure[m].sum() else np.nan,
                     "popular_unexposed": y[m][exposure[m] == 0].mean()})
    st = StratifiedTable([t.astype(float) for t in tables if t.sum(axis=1).min() > 0])
    test = st.test_null_odds()
    crude = pd.crosstab(exposure, y)
    crude_or = (crude.loc[1, 1] * crude.loc[0, 0]) / (crude.loc[1, 0] * crude.loc[0, 1])
    summary = {"exposure": name, "crude_OR": crude_or, "MH_pooled_OR": st.oddsratio_pooled,
               "MH_OR_lo": st.oddsratio_pooled_confint()[0], "MH_OR_hi": st.oddsratio_pooled_confint()[1],
               "CMH_stat": test.statistic, "CMH_p": test.pvalue,
               "breslow_day_p": st.test_equal_odds().pvalue}
    return pd.DataFrame(rows), summary


def logit(exposure, y, chan, name):
    X = pd.get_dummies(pd.DataFrame({"exposure": exposure, "channel": chan}), columns=["channel"],
                       drop_first=True, dtype=float)
    res = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    b, (lo, hi) = res.params["exposure"], res.conf_int().loc["exposure"]
    return {"exposure": name, "logit_OR": np.exp(b), "logit_OR_lo": np.exp(lo), "logit_OR_hi": np.exp(hi),
            "logit_p": res.pvalues["exposure"]}


def main():
    df = load()
    df = df[df.empty_content == 0].reset_index(drop=True)
    y = make_label(df.shares, 1400)
    g = feature_groups(df)
    chan = channel_of(df, g).to_numpy()

    # exposures
    neg_article = (df["global_sentiment_polarity"] < 0).astype(int).to_numpy()
    Z = StandardScaler().fit_transform(df[g["lda"] + g["sentiment"]])
    lab = KMeans(9, n_init=10, random_state=SEED).fit_predict(Z)
    pol = pd.Series(df["global_sentiment_polarity"].to_numpy()).groupby(lab).mean()
    neg_cluster = (lab == pol.idxmin()).astype(int)
    print("negative cluster mean polarity:", round(pol.min(), 3), "n =", neg_cluster.sum())

    strata_rows, summaries = [], []
    for exp, name in [(neg_article, "negative_polarity_article"), (neg_cluster, "negative_polarity_cluster")]:
        r, sm_ = stratified(exp, y, chan, name)
        sm_.update(logit(exp, y, chan, name))
        strata_rows.append(r); summaries.append(sm_)
    strata = pd.concat(strata_rows)
    strata.to_csv(TAB / "e7b_sentiment_by_channel.csv", index=False)
    summ = pd.DataFrame(summaries)
    summ.to_csv(TAB / "e7b_sentiment_cmh.csv", index=False)
    print(strata.round(3).to_string()); print(summ.round(4).T.to_string())

    # Isolation Forest at several contamination levels
    top5 = df.shares >= df.shares.quantile(0.95)
    iso_rows, prof_rows = [], []
    for c in [0.01, 0.02, 0.05]:
        out = IsolationForest(n_estimators=300, contamination=c, random_state=SEED).fit(Z).predict(Z) == -1
        t = pd.crosstab(out, top5)
        from scipy.stats import fisher_exact
        orr, p = fisher_exact(t.to_numpy())
        iso_rows.append({"contamination": c, "n_outliers": int(out.sum()),
                         "popular_out": y[out].mean(), "popular_in": y[~out].mean(),
                         "top5_out": top5[out].mean(), "top5_in": top5[~out].mean(),
                         "top5_ratio": top5[out].mean() / top5[~out].mean(), "fisher_p_top5": p})
        r_, s_ = stratified(out.astype(int), top5.astype(int).to_numpy(), chan, f"outlier_c{c}_top5")
        iso_rows[-1].update({"MH_OR_top5": s_["MH_pooled_OR"], "MH_OR_lo": s_["MH_OR_lo"],
                             "MH_OR_hi": s_["MH_OR_hi"], "CMH_p": s_["CMH_p"]})
        if c == 0.02:
            cols = g["lda"] + g["sentiment"] + ["n_tokens_content", "num_imgs", "num_videos", "num_hrefs"]
            for col in cols:
                prof_rows.append({"feature": col, "median_outliers": df.loc[out, col].median(),
                                  "median_inliers": df.loc[~out, col].median(),
                                  "std_shift": (df.loc[out, col].mean() - df.loc[~out, col].mean()) / df[col].std()})
            mix = pd.crosstab(chan, out, normalize="columns")
            mix.columns = ["inliers", "outliers"]
            mix.to_csv(TAB / "e7b_outlier_channel_mix.csv")
            print(mix.round(3))
    iso = pd.DataFrame(iso_rows)
    iso.to_csv(TAB / "e7b_isolation_forest.csv", index=False)
    prof = pd.DataFrame(prof_rows).sort_values("std_shift", key=np.abs, ascending=False)
    prof.to_csv(TAB / "e7b_outlier_profile.csv", index=False)
    print(iso.round(4).to_string()); print(prof.head(10).round(3).to_string())


if __name__ == "__main__":
    main()
