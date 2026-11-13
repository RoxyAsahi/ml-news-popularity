"""Shared setup for the Upworthy ranking experiments (u3+): headline representations, article clusters,
natural pairs, within-cell evaluation pairs and the metrics. See scripts/u2_interventional_ranking.py for the
definitions; this module factors them out so later experiments use identical data and metrics."""
from __future__ import annotations

import gzip
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

from . import upworthy as up
from .data import ROOT


@dataclass
class Setup:
    d: pd.DataFrame
    H: pd.Series          # unique headlines
    hid: np.ndarray       # headline id per package
    Z: np.ndarray         # dense representation per unique headline (GloVe-100 mean + 15 features)
    tests: pd.DataFrame
    tix: np.ndarray       # test index per package
    pk: list              # packages per test
    article: np.ndarray   # article cluster per test
    day: np.ndarray
    ri: np.ndarray        # natural pairs (tests of the same article on different days)
    rj: np.ndarray
    ev_a: np.ndarray      # within-cell evaluation pairs (package indices)
    ev_b: np.ndarray
    zz: np.ndarray        # two-proportion z of the pair (a minus b)
    y: np.ndarray         # logit CTR per package


def load_glove(vocab: set[str]) -> dict[str, np.ndarray]:
    out = {}
    with gzip.open(ROOT / "data" / "external" / "glove" / "glove-wiki-gigaword-100.gz", "rt", encoding="utf8") as fh:
        for line in fh:
            w, rest = line.split(" ", 1)
            if w in vocab:
                out[w] = np.array(rest.split(), dtype=float)
    return out


def setup(split: str) -> Setup:
    d = up.load(split)
    d["ctr"] = d.clicks / d.impressions
    d["cell"] = up.cell_key(d)
    H = pd.Series(d.headline.astype(str).unique())
    hid = pd.Index(H).get_indexer(d.headline.astype(str))
    F15 = up.headline_features(H)
    F15 = ((F15 - F15.mean()) / F15.std()).to_numpy()
    glove = load_glove(set(w for h in H for w in up.tokens(h)))
    G = np.array([np.mean([glove[w] for w in up.tokens(h) if w in glove] or [np.zeros(100)], 0) for h in H])
    G = (G - G.mean(0)) / (G.std(0) + 1e-9)
    tests = d.groupby("test").agg(t=("t", "min"), week=("test_week", "first"), lede=("lede_clean", "first")).reset_index()
    tix = pd.Index(tests.test).get_indexer(d.test)
    pk = [np.where(tix == k)[0] for k in range(len(tests))]
    day = (tests.t - tests.t.min()).dt.days.to_numpy()
    has = tests.lede.to_numpy() != ""
    L = TfidfVectorizer(sublinear_tf=True).fit_transform(tests.lede)
    ri, rj = [], []
    for a0 in range(0, len(tests), 3000):
        M = (L[a0:a0 + 3000] @ L.T).tocoo()
        r, c = M.row + a0, M.col
        keep = (r < c) & (M.data >= 0.8) & has[r] & has[c]
        ri.append(r[keep]), rj.append(c[keep])
    ri, rj = np.concatenate(ri), np.concatenate(rj)
    article = connected_components(coo_matrix((np.ones(len(ri)), (ri, rj)), shape=(len(tests),) * 2),
                                   directed=False)[1]
    nat = day[ri] != day[rj]
    ea, eb = [], []
    for _, idx in d.groupby("cell").indices.items():
        if len(idx) < 2:
            continue
        a, b = np.triu_indices(len(idx), 1)
        keep = hid[idx[a]] != hid[idx[b]]
        ea.append(idx[a][keep]), eb.append(idx[b][keep])
    ea, eb = np.concatenate(ea), np.concatenate(eb)
    assert (tix[ea] == tix[eb]).all(), "evaluation pairs must lie within one test"
    c, n = d.clicks.to_numpy(), d.impressions.to_numpy()
    p1, p2 = c[ea] / n[ea], c[eb] / n[eb]
    pp = (c[ea] + c[eb]) / (n[ea] + n[eb])
    zz = (p1 - p2) / np.sqrt(pp * (1 - pp) * (1 / n[ea] + 1 / n[eb]) + 1e-12)
    return Setup(d, H, hid, np.c_[G, F15], tests, tix, pk, article, day, ri[nat], rj[nat], ea, eb, zz,
                 d.logit_ctr.to_numpy())


def one_per_test(S: Setup, test_ids, rng) -> np.ndarray:
    return np.array([S.pk[k][rng.integers(0, len(S.pk[k]))] for k in test_ids])


def week_demeaned(S: Setup, rows: np.ndarray) -> np.ndarray:
    wk = S.tests.week.to_numpy()[S.tix[rows]]
    return S.y[rows] - pd.Series(S.y[rows]).groupby(wk).transform("mean").to_numpy()


def within_rows(S: Setup, test_ids) -> tuple[np.ndarray, np.ndarray]:
    """Packages of the given tests in cells with >= 2 distinct headlines, and their cell ids."""
    rows = np.where(np.isin(S.tix, test_ids))[0]
    ci = S.d.cell.to_numpy()[rows]
    multi = pd.Series(S.hid[rows]).groupby(ci).transform("nunique").to_numpy() >= 2
    return rows[multi], pd.factorize(ci[multi])[0]


def demean(A: np.ndarray, g: np.ndarray) -> np.ndarray:
    return A - pd.DataFrame(A).groupby(g).transform("mean").to_numpy()


def evaluate(S: Setup, score_rows: np.ndarray, pred_rows: np.ndarray, te_tests) -> dict:
    """Within-cell pairwise accuracy (all pairs, significant pairs) and decision lift on held-out tests."""
    pred = np.full(len(S.d), np.nan)
    pred[score_rows] = pred_rows
    te = np.isin(S.tix[S.ev_a], te_tests)
    dp = pred[S.ev_a[te]] - pred[S.ev_b[te]]
    assert not np.isnan(dp).any(), "every evaluation pair must be scored"
    ok = np.where(dp == 0, 0.5, (np.sign(dp) == np.sign(S.zz[te])).astype(float))
    sig = np.abs(S.zz[te]) > 1.96
    sub = S.d.iloc[score_rows][["cell", "ctr"]].assign(p=pred_rows, h=S.hid[score_rows])
    sub = sub[sub.groupby("cell").h.transform("nunique") >= 2]
    g = sub.groupby("cell")
    pick = sub.loc[g.p.idxmax(), ["cell", "ctr"]].set_index("cell").ctr
    return dict(within_acc=float(ok.mean()), within_acc_sig=float(ok[sig].mean()),
                lift_pct=float((pick / g.ctr.mean() - 1).mean() * 100), n_sig=int(sig.sum()))
