"""S10: does a story's audience get split among its articles? (links RQ1 twins and RQ3 competition)

Story = connected component of the clean twin graph (singletons are stories of size 1).
(a) Article level: log(1+shares) ~ log(story size k) + week, weekday, channel FE [+ content].
    Pure splitting of a fixed audience predicts a slope near -1; independent articles predict 0.
    Bigger stories are also more newsworthy, which biases the slope upwards, so a negative slope is
    conservative evidence for splitting.
(b) Story level: log(total shares of the story) ~ log k + FE of its first article. Fixed audience ->
    slope 0; independent articles -> slope 1.
(c) Same-story vs other-story volume: log(1 + same-story articles within +-2 days, excluding the article)
    and log(other articles published that day), with the FE and content controls of s5.
Story-clustered SEs. Robustness: stories with k <= 10 only (large components can chain loosely).

Output: results/tables/s10_split_attention.csv
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from newspop.data import ROOT, feature_groups, load
from newspop import stories as st

df = load()
P = st.clean_twins(df)
sid = st.story_ids(len(df), P)
d = pd.DataFrame({"ls": np.log1p(df.shares), "shares": df.shares, "day": st.day_index(df), "week": st.week_index(df),
                  "wd": df.publish_date.dt.dayofweek, "chan": st.channel(df), "story": sid})
d["k"] = d.groupby("story").ls.transform("size")
d["log_k"] = np.log(d.k)
# same-story articles within +-2 days (excluding self)
near = np.zeros(len(d), int)
for _, g in d[d.k > 1].groupby("story"):
    days = g.day.to_numpy()
    near[g.index.to_numpy()] = (np.abs(days[:, None] - days[None, :]) <= 2).sum(1) - 1
d["log_near"] = np.log1p(near)
d["n_day"] = d.groupby("day").ls.transform("size")
d["n_same_day_story"] = d.groupby(["day", "story"]).ls.transform("size")
d["log_other"] = np.log(d.n_day - d.n_same_day_story + 1)
g = feature_groups(df)
content = [c for c in g["content"] + g["lda"] + g["sentiment"]
           if c not in ("LDA_04", "rate_negative_words", "n_non_stop_words")]
for c in content:
    d[c] = df[c].to_numpy()
ctrl = " + ".join(content)
fe = "C(week) + C(wd) + C(chan)"
rows = []


def fit(name, formula, data, term, level="article"):
    m = smf.ols(formula, data).fit(cov_type="cluster", cov_kwds={"groups": data["story"]})
    lo, hi = m.conf_int().loc[term]
    rows.append(dict(level=level, spec=name, term=term, coef=m.params[term], lo=lo, hi=hi, n=int(m.nobs)))
    print(level, name, term, round(m.params[term], 3), [round(lo, 3), round(hi, 3)], flush=True)


fit("story size, FE", f"ls ~ log_k + {fe}", d, "log_k")
fit("story size, FE + content", f"ls ~ log_k + {fe} + {ctrl}", d, "log_k")
fit("story size, FE + content, k <= 10", f"ls ~ log_k + {fe} + {ctrl}", d[d.k <= 10], "log_k")
m = f"ls ~ log_near + log_other + {fe} + {ctrl}"
fit("same-story articles within 2 days", m, d, "log_near")
fit("other articles that day", m, d, "log_other")

# story level
first = d.sort_values(["day"]).groupby("story").head(1).set_index("story")
S = d.groupby("story").agg(total=("shares", "sum"), k=("k", "first"))
S = S.join(first[["week", "wd", "chan"]] + 0 if False else first[["week", "wd", "chan"]])
S["log_total"] = np.log1p(S.total)
S["log_k"] = np.log(S.k)
S["story"] = S.index
fit("story total vs size, FE of first article", f"log_total ~ log_k + {fe}", S, "log_k", level="story")
fit("story total vs size, k <= 10", f"log_total ~ log_k + {fe}", S[S.k <= 10], "log_k", level="story")
R = pd.DataFrame(rows).round(4)
R.to_csv(ROOT / "results" / "tables" / "s10_split_attention.csv", index=False)
print(R.to_string(index=False))
print("articles in multi-article stories:", int((d.k > 1).sum()), "stories with k>1:", int((S.k > 1).sum()))
