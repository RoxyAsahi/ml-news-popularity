"""The url column of OnlineNewsPopularity.csv is shifted against every other column from 6 June 2013 on.
This script measures the shift and recovers, for each row, the URL that its features and shares belong to.

Evidence used (the CSV itself plus the real headlines in data/external/mashable_sitemap_titles.csv):
  * content date: timedelta counts days back from 2015-01-08 and agrees with the weekday_is_* flags on every row
  * url date: the yyyy/mm/dd in the URL
  * title fingerprints: n_tokens_title equals (number of alphanumeric runs in the HTML-escaped headline) + 1 for the
    headline that belongs to the row; title_sentiment_polarity / title_subjectivity are reproduced by TextBlob for
    most headlines
In the original file order, row i's content matches url[i + k_i]; k_i is 0 until June 2013 and then only grows
(rows were removed from the feature side but not from the url list). k_i is estimated by a Viterbi alignment:
states k = 0..KMAX, non-decreasing along the file, emission = date agreement (hard) + log-likelihood ratios of
token-count and sentiment agreement (calibrated on January-April 2013 against a deliberately wrong pairing),
cost JUMP per unit increase.

Hold-out check: the alignment is repeated with the date and the token count only; the sentiment fingerprint, which
that alignment never saw, must then return to its January-May level in every later period.

Output: data/external/url_realignment.csv   (file_row, offset; offset -1 = the row's URL is not in the file;
                                             whether the two title fingerprints agree after the repair)
        results/tables/a1_alignment_quality.csv, a1_offset_by_month.csv
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from newspop.data import ACQUIRED, REALIGN_CSV, load_raw  # noqa: E402
from newspop.titles import aligned_titles  # noqa: E402

TAB = ROOT / "results" / "tables"
CACHE = ROOT / "data" / "interim" / "title_textblob.npy"
KMAX, JUMP = 400, 6.0
T0 = pd.Timestamp("2013-01-01")


def ntok(s) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", html.escape(s, quote=False))) + 1 if isinstance(s, str) else -1


def sentiment(titles: pd.Series) -> np.ndarray:
    if CACHE.exists() and len(np.load(CACHE)) == len(titles):
        return np.load(CACHE)
    from textblob import TextBlob
    a = np.array([TextBlob(t).sentiment if isinstance(t, str) else (np.nan, np.nan) for t in titles], dtype=float)
    CACHE.parent.mkdir(parents=True, exist_ok=True); np.save(CACHE, a)
    return a


def main():
    raw = load_raw().reset_index(drop=True); N = len(raw)
    title = aligned_titles(raw)                       # headline of the URL written in each row
    u_has = title.notna().to_numpy(); u_tok = np.array([ntok(t) for t in title])
    S = sentiment(title); u_pol, u_sub = S[:, 0], S[:, 1]
    u_day = (pd.to_datetime(raw.url.str.extract(r"mashable\.com/(\d{4}/\d{2}/\d{2})/")[0]) - T0).dt.days.to_numpy()
    c_date = ACQUIRED - pd.to_timedelta(raw.timedelta, unit="D")
    c_day = (c_date - T0).dt.days.to_numpy()
    wd = np.argmax(raw[[f"weekday_is_{d}" for d in ("monday", "tuesday", "wednesday", "thursday", "friday",
                                                    "saturday", "sunday")]].to_numpy(), axis=1)
    print(f"content date (2015-01-08 - timedelta) agrees with weekday_is_*: {(c_date.dt.weekday.to_numpy() == wd).mean():.4f}; "
          f"file sorted by content date: {bool((np.diff(c_day) >= 0).all())}")
    c_tok = raw.n_tokens_title.to_numpy().astype(int)
    c_pol, c_sub = raw.title_sentiment_polarity.to_numpy(), raw.title_subjectivity.to_numpy()

    def tok_eq(i, j):
        return c_tok[i] == u_tok[j]

    def sen_eq(i, j):
        return (np.abs(c_pol[i] - u_pol[j]) < 1e-6) & (np.abs(c_sub[i] - u_sub[j]) < 1e-6)

    cal = np.flatnonzero((c_day < (pd.Timestamp("2013-05-01") - T0).days) & u_has)     # known to be aligned
    p_tok, p_sen = tok_eq(cal, cal).mean(), sen_eq(cal, cal).mean()
    w = cal[:-50][u_has[cal[50:]]]; q_tok, q_sen = tok_eq(w, w + 50).mean(), sen_eq(w, w + 50).mean()
    print(f"aligned rows: token count equal {p_tok:.3f}, sentiment equal {p_sen:.3f}; wrong pairing: {q_tok:.3f}, {q_sen:.3f}")
    s_tok = (np.log(p_tok / q_tok), np.log((1 - p_tok) / (1 - q_tok)))
    s_sen = (np.log(p_sen / q_sen), np.log((1 - p_sen) / (1 - q_sen)))
    ks = np.arange(KMAX + 1)

    def viterbi(use_sentiment: bool) -> np.ndarray:
        dp = np.full(KMAX + 1, -1e9); dp[0] = 0.0
        back = np.zeros((N, KMAX + 1), dtype=np.int16)
        for i in range(N):
            a = dp + JUMP * ks
            run = np.maximum.accumulate(a)
            back[i] = np.maximum.accumulate(np.where(a >= run, ks, 0))
            j = i + ks; inside = j < N; jj = np.minimum(j, N - 1)
            em = np.where(tok_eq(i, jj), s_tok[0], s_tok[1])
            if use_sentiment:
                em = em + np.where(sen_eq(i, jj), s_sen[0], s_sen[1])
            em = np.where(u_has[jj], em, 0.0)
            em = np.where(u_day[jj] == c_day[i], em, -25.0)
            em = np.where(inside, em, -1.0)            # beyond the end of the file: the true URL is not in the CSV
            dp = run - JUMP * ks + em
        k = np.zeros(N, dtype=int); k[-1] = int(np.argmax(dp))
        for i in range(N - 1, 0, -1):
            k[i - 1] = back[i, k[i]]
        return k

    k = viterbi(True); k_hold = viterbi(False)
    idx = np.arange(N)

    def flags(kk):
        j = idx + kk; inside = j < N; jj = np.minimum(j, N - 1)
        return inside, jj, inside & (u_day[jj] == c_day), inside & u_has[jj] & tok_eq(idx, jj), inside & u_has[jj] & sen_eq(idx, jj)

    inside, jj, date_ok, tok_ok, sen_ok = flags(k)
    inside_h, _, _, _, sen_ok_h = flags(k_hold)
    REALIGN_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"file_row": idx, "offset": np.where(inside, k, -1), "title_length_agrees": tok_ok.astype(int),
                  "title_sentiment_agrees": sen_ok.astype(int)}).to_csv(REALIGN_CSV, index=False)

    per = pd.cut(c_date, pd.to_datetime(["2013-01-01", "2013-06-06", "2014-01-01", "2014-07-01", "2015-01-01"]), right=False,
                 labels=["2013-01-01..06-05", "2013-06-06..12-31", "2014 H1", "2014 H2"]).astype(str)
    has0 = u_has; has1 = inside & u_has[jj]
    Q = pd.DataFrame({"period": per, "rows": 1, "as_given_date": u_day == c_day,
                      "as_given_token": np.where(has0, tok_eq(idx, idx), np.nan), "as_given_sentiment": np.where(has0, sen_eq(idx, idx), np.nan),
                      "repaired_has_url": inside, "repaired_date": np.where(inside, date_ok, np.nan),
                      "repaired_token": np.where(has1, tok_ok, np.nan), "repaired_sentiment": np.where(has1, sen_ok, np.nan),
                      "holdout_sentiment": np.where(inside_h & u_has[np.minimum(idx + k_hold, N - 1)], sen_ok_h, np.nan),
                      "holdout_same_offset": k_hold == k, "offset_min": k, "offset_max": k})
    agg = {c: "mean" for c in Q.columns if c not in ("period", "rows", "offset_min", "offset_max")}
    agg.update(rows="sum", offset_min="min", offset_max="max")
    Qs = Q.groupby("period").agg(agg); Qs.to_csv(TAB / "a1_alignment_quality.csv")
    M = pd.DataFrame({"month": c_date.dt.to_period("M").astype(str), "offset": k}).groupby("month").offset.agg(["min", "median", "max"])
    M.to_csv(TAB / "a1_offset_by_month.csv")
    pd.set_option("display.width", 250); print(Qs.round(3).T.to_string()); print(M.T.to_string())
    first = int(np.flatnonzero(k > 0)[0])
    print("first shifted row:", first, c_date[first].date(), "| rows with a recovered URL:", int(inside.sum()),
          "| rows whose URL as given is right:", int((k == 0).sum()), "| shifted rows:", int((k > 0).sum()), f"({(k > 0).mean():.3f})",
          "| URLs in the file that belong to no row:", int(N - len(set(jj[inside]))),
          "| hold-out alignment gives the same offset on", f"{(k_hold == k).mean():.4f}", "of rows",
          "| both fingerprints disagree after the repair on", f"{(inside & u_has[jj] & ~tok_ok & ~sen_ok).mean():.4f}",
          "| offset steps:", int((np.diff(k) > 0).sum()))


if __name__ == "__main__":
    main()
