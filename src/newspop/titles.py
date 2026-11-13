"""Real Mashable headlines from the monthly sitemap (scripts/fetch_mashable_titles.py), aligned to the
dataset by URL slug. 39,619 of 39,643 articles match; the rest get NaN."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SITEMAP_CSV = ROOT / "data" / "external" / "mashable_sitemap_titles.csv"
BGE_CACHE = ROOT / "data" / "interim" / "mashable_title_bge.npy"


def aligned_titles(df: pd.DataFrame, path: str | Path = SITEMAP_CSV) -> pd.Series:
    """Headline per dataset row (same order as df), matched on the URL slug."""
    T = pd.read_csv(path)
    T["slug"] = T.url.str.extract(r"/archive/([^/?#]+)")[0]
    T = T.dropna(subset=["slug"]).drop_duplicates("slug")
    slug = df.url.str.extract(r"/\d{4}/\d{2}/\d{2}/([^/]+)/?\s*$")[0]
    return pd.Series(slug.map(T.set_index("slug").title).to_numpy(), index=df.index, name="title")


def bge_embeddings(titles: pd.Series, cache: str | Path = BGE_CACHE, threads: int = 2) -> np.ndarray:
    """bge-base-en-v1.5 [CLS] vectors (max 48 tokens) of the headlines, cached on disk."""
    cache = Path(cache)
    if cache.exists():
        return np.load(cache)
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(threads)
    name = "BAAI/bge-base-en-v1.5"
    tok, m = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval()
    T = titles.fillna("").tolist()
    order = np.argsort([len(t) for t in T]); out = np.zeros((len(T), 768), np.float32)
    with torch.no_grad():
        for b in range(0, len(T), 128):
            ix = order[b:b + 128]
            enc = tok([T[i] for i in ix], padding=True, truncation=True, max_length=48, return_tensors="pt")
            out[ix] = m(**enc).last_hidden_state[:, 0].numpy()
    cache.parent.mkdir(parents=True, exist_ok=True); np.save(cache, out)
    return out
