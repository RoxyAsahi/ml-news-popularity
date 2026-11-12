"""S5 (RQ3): attention competition. Do articles get fewer shares on days when more articles appear?

log(1+shares) ~ log(articles in the dataset that day) + week FE + weekday FE + channel FE
[+ all content features], day-clustered SEs. Robustness: placebo (volume of the same weekday one
week later), long articles only (>= 600 words), dropping the 5% busiest days, by year, same-channel
volume vs other-channel volume, and within twins (does the twin published on the busier day do worse?).
Note: n_day counts articles in this dataset, which may differ from Mashable's true daily output.

Output: results/tables/s5_competition.csv
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from newspop.data import ROOT, feature_groups, load
from newspop import stories as st

df = load()
d = pd.DataFrame({"ls": np.log1p(df.shares), "day": st.day_index(df), "week": st.week_index(df),
                  "wd": df.publish_date.dt.dayofweek, "chan": st.channel(df), "year": df.publish_date.dt.year,
                  "words": df.n_tokens_content})
d["n_day"] = d.groupby("day").ls.transform("size")
d["n_chday"] = d.groupby(["day", "chan"]).ls.transform("size")
d["log_n_day"] = np.log(d.n_day)
d["log_n_same"] = np.log(d.n_chday)
d["log_n_other"] = np.log((d.n_day - d.n_chday).clip(lower=1))
daily = d.groupby("day").n_day.first()
d["log_n_placebo"] = np.log(d.day.add(7).map(daily))
g = feature_groups(df)
content = [c for c in g["content"] + g["lda"] + g["sentiment"]
           if c not in ("LDA_04", "rate_negative_words", "n_non_stop_words")]  # drop exact collinearities
for c in content:
    d[c] = df[c].to_numpy()
ctrl = " + ".join(content)
fe = "C(week) + C(wd) + C(chan)"

rows = []


def fit(name, formula, data, term="log_n_day"):
    m = smf.ols(formula, data).fit(cov_type="cluster", cov_kwds={"groups": data["day"]})
    lo, hi = m.conf_int().loc[term]
    rows.append(dict(spec=name, term=term, coef=m.params[term], lo=lo, hi=hi, n=int(m.nobs),
                     pct_change_if_volume_doubles=100 * (2 ** m.params[term] - 1)))
    print(name, round(m.params[term], 3), [round(lo, 3), round(hi, 3)], flush=True)


fit("main: week + weekday + channel FE", f"ls ~ log_n_day + {fe}", d)
fit("+ all content features", f"ls ~ log_n_day + {fe} + {ctrl}", d)
dp = d.dropna(subset=["log_n_placebo"])
fit("placebo: volume one week later", f"ls ~ log_n_placebo + {fe}", dp, term="log_n_placebo")
fit("placebo jointly with own volume", f"ls ~ log_n_day + log_n_placebo + {fe}", dp)
fit("long articles (>= 600 words)", f"ls ~ log_n_day + {fe}", d[d.words >= 600])
busy = daily.quantile(0.95)
fit("drop 5% busiest days", f"ls ~ log_n_day + {fe}", d[d.n_day < busy])
for y in [2013, 2014]:
    fit(f"year {y}", f"ls ~ log_n_day + {fe}", d[d.year == y])
m = f"ls ~ log_n_same + log_n_other + {fe}"
fit("same-channel volume", m, d, term="log_n_same")
fit("other-channel volume", m, d, term="log_n_other")

# within twins: first minus later, difference in log volume
P = st.clean_twins(df)
P = P[P.dt > 0]
later = np.where(P["first"] == P.i, P.j, P.i)
first = P["first"].to_numpy()
t = pd.DataFrame({"dy": d.ls.to_numpy()[first] - d.ls.to_numpy()[later],
                  "dvol": d.log_n_day.to_numpy()[first] - d.log_n_day.to_numpy()[later],
                  "story": st.story_ids(len(df), st.clean_twins(df))[first]})
mt = smf.ols("dy ~ dvol", t).fit(cov_type="cluster", cov_kwds={"groups": t.story})
lo, hi = mt.conf_int().loc["dvol"]
rows.append(dict(spec="within twins (story-clustered)", term="dvol", coef=mt.params["dvol"], lo=lo, hi=hi,
                 n=int(mt.nobs), pct_change_if_volume_doubles=100 * (2 ** mt.params["dvol"] - 1)))
print("twins", round(mt.params["dvol"], 3), [round(lo, 3), round(hi, 3)])
print("daily volume: mean", round(daily.mean(), 1), "sd", round(daily.std(), 1), "busiest 5% >=", busy)
pd.DataFrame(rows).round(4).to_csv(ROOT / "results" / "tables" / "s5_competition.csv", index=False)
