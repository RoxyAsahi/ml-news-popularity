"""D1: what the usual predictive workflow says about the editorial levers.

(a) raw association: slope of log shares on each lever, nothing held fixed;
(b) a boosted-tree model on the dataset's 58 features plus the two crowding counts, cross-fitted on
    whole weeks: TreeSHAP importance of every feature, and the model's own "what if": change one lever
    for every held-out article (double 1 + count, or raise it by one standard deviation, or move the article to a Saturday) and average the change in
    the prediction. This is the number a practitioner reads off a partial-dependence plot.

Outputs: out/d1_naive_levers.csv, out/d1_shap_features.csv, out/d1_shap_groups.csv
"""
import numpy as np
import pandas as pd

import common as C

df, T, S, Y = C.build()
g = C.groups(df)
feats = [c for grp in g.values() for c in grp if c not in ("empty_content",)]
X = df[feats].assign(vol_same=df.vol_same, vol_other=df.vol_other)
cols = list(X.columns)
days = df.day.to_numpy()
LN2 = np.log(2)
SD = T.std()


def what_if(Xte, lever, delta):
    """Copy of the held-out rows with one lever raised by `delta` on its log(1 + count) scale."""
    Z = Xte.copy()
    up = lambda v: np.exp(np.log1p(v) + delta) - 1
    if lever == "imgs": Z["num_imgs"] = up(Z.num_imgs)
    elif lever == "videos": Z["num_videos"] = up(Z.num_videos)
    elif lever == "length": Z["n_tokens_content"] = up(Z.n_tokens_content)
    elif lever == "title_len": Z["n_tokens_title"] = up(Z.n_tokens_title)
    elif lever == "links_out":
        out = (Z.num_hrefs - Z.num_self_hrefs).clip(lower=0); Z["num_hrefs"] = Z.num_self_hrefs + up(out)
    elif lever == "links_self":
        add = up(Z.num_self_hrefs) - Z.num_self_hrefs; Z["num_self_hrefs"] += add; Z["num_hrefs"] += add
    elif lever in ("vol_same", "vol_other"): Z[lever] = Z[lever] + delta
    return Z


def set_day(Xte, weekend):
    Z = Xte.copy()
    for c in g["weekday"]: Z[c] = 0.0
    Z["is_weekend"] = float(weekend); Z["weekday_is_saturday" if weekend else "weekday_is_wednesday"] = 1.0
    return Z


rows, shap_sum, r2s = {k: [] for k in C.LEVERS}, np.zeros(len(cols)), []
for rep in range(3):
    folds = C.week_folds(df, rep)
    pred = np.zeros(len(Y))
    for k in range(C.N_FOLDS):
        tr, te = folds != k, folds == k
        m = C.make_learner(seed=k).fit(X[tr], Y[tr])
        pred[te] = m.predict(X[te])
        shap_sum += np.abs(m.predict(X[te], pred_contrib=True)[:, :-1]).sum(0)
        for lv in C.LEVERS:
            if lv == "weekend":
                diff = m.predict(set_day(X[te], 1)) - m.predict(set_day(X[te], 0)); dsd = diff * SD[lv]
            else:
                diff = m.predict(what_if(X[te], lv, LN2)) - pred[te]; dsd = m.predict(what_if(X[te], lv, SD[lv])) - pred[te]
            rows[lv].append(pd.DataFrame({"rep": rep, "d": diff, "dsd": dsd}))
    r2s.append(1 - np.mean((Y - pred) ** 2) / np.var(Y))
    print("rep", rep, "out-of-fold R2", round(r2s[-1], 4), flush=True)

shap = pd.DataFrame({"feature": cols, "mean_abs_shap": shap_sum / (3 * len(Y))}).sort_values("mean_abs_shap", ascending=False)
shap["rank"] = np.arange(1, len(shap) + 1)
shap["share"] = shap.mean_abs_shap / shap.mean_abs_shap.sum()
lever_cols = {"imgs": ["num_imgs"], "videos": ["num_videos"], "length": ["n_tokens_content"], "title_len": ["n_tokens_title"],
              "links_out": ["num_hrefs"], "links_self": ["num_self_hrefs"], "weekend": g["weekday"],
              "vol_same": ["vol_same"], "vol_other": ["vol_other"]}
grp = {c: "other text statistics" for c in cols}
for name, cs in [("keyword history (kw_*)", g["keyword"]), ("linked articles' shares (self_reference_*)", g["self_reference"]),
                 ("section + LDA topics", g["channel"] + g["lda"]), ("dictionary sentiment", g["sentiment"])]:
    grp.update({c: name for c in cs})
for lv, cs in lever_cols.items():
    grp.update({c: "editorial levers" for c in cs})
shap["group"] = shap.feature.map(grp)
shap.round(5).to_csv(C.OUT / "d1_shap_features.csv", index=False)
G = shap.groupby("group").agg(n_features=("feature", "size"), share=("share", "sum")).sort_values("share", ascending=False)
G.round(4).to_csv(C.OUT / "d1_shap_groups.csv")
print(G.round(3).to_string()); print(shap.head(15).round(4).to_string(index=False))

out = []
for lv, (label, kind) in C.LEVERS.items():
    t = T[lv].to_numpy()
    b, V = C.cluster_ols(Y - Y.mean(), t - t.mean(), days)
    D = pd.concat(rows[lv]).d; Dsd = pd.concat(rows[lv]).dsd.mean()
    sh = shap[shap.feature.isin(lever_cols[lv])]
    out.append(dict(lever=lv, label=label, raw_slope=b[0], raw_se=np.sqrt(V[0, 0]), raw_pct=C.pct(b[0], kind), raw_per_sd=b[0] * SD[lv], gbdt_whatif_per_sd=Dsd,
                    gbdt_whatif=D.mean(), gbdt_whatif_pct=100 * (np.exp(D.mean()) - 1),
                    shap_rank=int(sh["rank"].min()), shap_share=sh.share.sum()))
R = pd.DataFrame(out)
R["gbdt_r2"] = np.mean(r2s)
R.round(4).to_csv(C.OUT / "d1_naive_levers.csv", index=False)
print(R.round(3).to_string(index=False))
