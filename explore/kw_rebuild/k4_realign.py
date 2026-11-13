"""The url column of OnlineNewsPopularity.csv is shifted against every other column from June 2013 on.
This script measures the shift and recovers, for each row, the URL (and headline) that its features and shares
actually belong to.

Evidence used (all inside the course CSV, plus the real headlines of data/external/mashable_sitemap_titles.csv):
  * content date: timedelta counts days back from 2015-01-08, and agrees with the weekday_is_* flags on every row
  * url date: the yyyy/mm/dd in the URL
  * title fingerprint: n_tokens_title equals (number of alphanumeric runs in the HTML-escaped headline) + 1 for the
    headline that really belongs to the row; title_subjectivity / title_sentiment_polarity are reproduced by
    TextBlob for most headlines
In the original file order, row i's content matches url[i + k_i] with an offset k_i that is 0 until early June 2013
and then only grows. k_i is estimated by a Viterbi alignment: states k = 0..KMAX, non-decreasing along the file,
emission = date agreement (hard) + token-count agreement + sentiment agreement, cost JUMP per unit increase.

Output: data/interim/kw_rebuild/realigned.csv  (file row, url as given, offset, true url, true headline, flags)
        results/tables/k4_alignment_quality.csv (agreement of the fingerprints before / after re-alignment, by period)
        results/tables/k4_offset_by_month.csv
        results/tables/k4_offset_steps.csv (compact, complete: file row where each new offset starts; the true URL of
        file row i is the url of file row i + offset)
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from newspop.data import add_publish_date, load_raw  # noqa: E402
from newspop.titles import aligned_titles  # noqa: E402

OUT = ROOT / "data" / "interim" / "kw_rebuild"
TAB = ROOT / "results" / "tables"
KMAX, JUMP = 1200, 6.0
ACQUIRED = pd.Timestamp("2015-01-08")


def ntok(s) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", html.escape(s, quote=False))) + 1 if isinstance(s, str) else -1


def sentiment(titles: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    cache = OUT / "title_textblob.npy"
    if cache.exists():
        a = np.load(cache)
        if len(a) == len(titles):
            return a[:, 0], a[:, 1]
    from textblob import TextBlob
    a = np.array([TextBlob(t).sentiment if isinstance(t, str) else (np.nan, np.nan) for t in titles], dtype=float)
    np.save(cache, a)
    return a[:, 0], a[:, 1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = add_publish_date(load_raw()).reset_index(drop=True)
    N = len(raw)
    title = aligned_titles(raw)
    u_tok = np.array([ntok(t) for t in title]); u_has = title.notna().to_numpy()
    u_pol, u_sub = sentiment(title)
    u_day = (raw.publish_date - pd.Timestamp("2013-01-01")).dt.days.to_numpy()
    c_day = ((ACQUIRED - pd.to_timedelta(raw.timedelta, unit="D")) - pd.Timestamp("2013-01-01")).dt.days.to_numpy()
    c_tok = raw.n_tokens_title.to_numpy().astype(int)
    c_pol, c_sub = raw.title_sentiment_polarity.to_numpy(), raw.title_subjectivity.to_numpy()

    # agreement rates where the file is known to be aligned (first four months) and at a wrong pairing (shift 50)
    cal = np.flatnonzero(c_day < (pd.Timestamp("2013-05-01") - pd.Timestamp("2013-01-01")).days)
    cal = cal[u_has[cal]]

    def rates(i, j):
        ok = u_has[j]
        return (np.mean(c_tok[i][ok] == u_tok[j][ok]),
                np.mean((np.abs(c_pol[i] - u_pol[j]) < 1e-6)[ok] & (np.abs(c_sub[i] - u_sub[j]) < 1e-6)[ok]))

    p_tok, p_sen = rates(cal, cal)
    q_tok, q_sen = rates(cal[:-50], cal[50:])
    print(f"aligned rows: token count equal {p_tok:.3f}, sentiment pair equal {p_sen:.3f}; "
          f"wrong pairing: {q_tok:.3f}, {q_sen:.3f}", flush=True)
    s_tok = (np.log(p_tok / q_tok), np.log((1 - p_tok) / (1 - q_tok)))
    s_sen = (np.log(p_sen / q_sen), np.log((1 - p_sen) / (1 - q_sen)))

    ks = np.arange(KMAX + 1)
    dp = np.full(KMAX + 1, -1e9); dp[0] = 0.0
    back = np.zeros((N, KMAX + 1), dtype=np.int16)
    for i in range(N):
        a = dp + JUMP * ks
        run = np.maximum.accumulate(a)
        arg = np.maximum.accumulate(np.where(a >= run, ks, 0))
        back[i] = arg
        prev = run - JUMP * ks
        j = i + ks; inside = j < N; jj = np.minimum(j, N - 1)
        em = np.where(c_tok[i] == u_tok[jj], s_tok[0], s_tok[1])
        em = em + np.where((np.abs(c_pol[i] - u_pol[jj]) < 1e-6) & (np.abs(c_sub[i] - u_sub[jj]) < 1e-6), s_sen[0], s_sen[1])
        em = np.where(u_has[jj], em, 0.0)
        em = np.where(u_day[jj] == c_day[i], em, -25.0)
        em = np.where(inside, em, -1.0)  # beyond the end of the file: the true URL is not in the CSV
        dp = prev + em
    k = np.zeros(N, dtype=int); k[-1] = int(np.argmax(dp))
    for i in range(N - 1, 0, -1):
        k[i - 1] = back[i, k[i]]
    j = np.arange(N) + k; inside = j < N; jj = np.minimum(j, N - 1)

    true_url = np.where(inside, raw.url.to_numpy()[jj], None)
    true_title = np.where(inside, title.to_numpy()[jj], None)
    tok_ok = inside & (c_tok == u_tok[jj])
    sen_ok = inside & (np.abs(c_pol - u_pol[jj]) < 1e-6) & (np.abs(c_sub - u_sub[jj]) < 1e-6)
    date_ok = inside & (u_day[jj] == c_day)
    R = pd.DataFrame({"file_row": np.arange(N), "url_in_file": raw.url, "offset": k, "true_url": true_url, "true_title": true_title,
                      "content_date": (pd.Timestamp("2013-01-01") + pd.to_timedelta(c_day, unit="D")).strftime("%Y-%m-%d"),
                      "url_in_file_date": raw.publish_date.dt.strftime("%Y-%m-%d"),
                      "date_agrees": date_ok, "token_count_agrees": tok_ok, "sentiment_agrees": sen_ok})
    R.to_csv(OUT / "realigned.csv", index=False)
    step = np.flatnonzero(np.r_[True, np.diff(k) != 0])  # compact form: the offset holds from first_file_row on
    pd.DataFrame({"first_file_row": step, "offset": k[step], "content_date": R.content_date[step]}).to_csv(TAB / "k4_offset_steps.csv", index=False)

    cd = pd.to_datetime(R.content_date)
    per = pd.cut(cd, pd.to_datetime(["2013-01-01", "2013-06-01", "2013-07-01", "2014-01-01", "2014-07-01", "2014-10-01", "2015-01-01"]), right=False).astype(str)
    Q = pd.DataFrame({"period": per, "rows": 1,
                      "as_given_date_agrees": (u_day == c_day), "as_given_token_agrees": (c_tok == u_tok),
                      "as_given_sentiment_agrees": (np.abs(c_pol - u_pol) < 1e-6) & (np.abs(c_sub - u_sub) < 1e-6),
                      "realigned_has_url": inside, "realigned_date_agrees": date_ok, "realigned_token_agrees": tok_ok,
                      "realigned_sentiment_agrees": sen_ok, "offset_min": k, "offset_max": k})
    agg = {c: "mean" for c in Q.columns if c not in ("period", "rows", "offset_min", "offset_max")}
    agg.update(rows="sum", offset_min="min", offset_max="max")
    Qs = Q.groupby("period").agg(agg); Qs.to_csv(TAB / "k4_alignment_quality.csv")
    M = pd.DataFrame({"month": cd.dt.to_period("M").astype(str), "offset": k}).groupby("month").offset.agg(["min", "median", "max"])
    M.to_csv(TAB / "k4_offset_by_month.csv")
    pd.set_option("display.width", 250)
    print(Qs.round(3).T.to_string()); print(M.T.to_string())
    first = int(np.flatnonzero(k > 0)[0])
    print("first shifted row:", first, R.content_date[first], "| rows with a recovered URL:", int(inside.sum()),
          "| rows whose URL as given is already right:", int((k == 0).sum()),
          "| URLs in the file that belong to no row:", int(N - len(set(jj[inside]))))


if __name__ == "__main__":
    main()
