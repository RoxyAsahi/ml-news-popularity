"""Real Mashable headlines from the monthly sitemap (scripts/fetch_mashable_titles.py), aligned to the
dataset by URL slug. With the repaired url column (newspop.data.load) 39,467 of 39,643 articles get a headline:
152 rows have no URL in the file and 24 slugs are not in the sitemap; those get NaN."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SITEMAP_CSV = ROOT / "data" / "external" / "mashable_sitemap_titles.csv"
BGE_CACHE = ROOT / "data" / "interim" / "mashable_title_bge.npz"


def aligned_titles(df: pd.DataFrame, path: str | Path = SITEMAP_CSV) -> pd.Series:
    """Headline per dataset row (same order as df), matched on the URL slug."""
    T = pd.read_csv(path)
    T["slug"] = T.url.str.extract(r"/archive/([^/?#]+)")[0]
    T = T.dropna(subset=["slug"]).drop_duplicates("slug")
    slug = df.url.str.extract(r"/\d{4}/\d{2}/\d{2}/([^/]+)/?\s*$")[0]
    return pd.Series(slug.map(T.set_index("slug").title).to_numpy(), index=df.index, name="title")


def bge_embeddings(titles: pd.Series, cache: str | Path = BGE_CACHE, threads: int = 2) -> np.ndarray:
    """bge-base-en-v1.5 [CLS] vectors (max 48 tokens) of the headlines, one row per element of `titles`.
    Vectors are cached on disk by headline text, so any row order (repaired or as distributed) can be served."""
    cache = Path(cache)
    T = pd.Series(titles).fillna("").astype(str).tolist()
    known: dict[str, np.ndarray] = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        known = dict(zip(z["titles"].tolist(), z["E"]))
    todo = sorted({t for t in T if t not in known}, key=len)
    if todo:
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(threads)
        name = "BAAI/bge-base-en-v1.5"
        tok, m = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval()
        with torch.no_grad():
            for b in range(0, len(todo), 128):
                enc = tok(todo[b:b + 128], padding=True, truncation=True, max_length=48, return_tensors="pt")
                for t, v in zip(todo[b:b + 128], m(**enc).last_hidden_state[:, 0].numpy()):
                    known[t] = v.astype(np.float32)
        cache.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(known)
        np.savez(cache, titles=np.array(keys), E=np.stack([known[k] for k in keys]))
    return np.stack([known[t] for t in T])
