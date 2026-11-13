"""D0: bge-base-en-v1.5 [CLS] vectors of the real Mashable headlines, in the row order of the raw CSV.

The url column of OnlineNewsPopularity.csv is shifted against the feature rows from June 2013 on, so
headlines are taken from data/external/realigned.csv (file_row -> true_url, true_title; built by the
main research thread, /mnt/project-files/kw_rebuild/). Same model, pooling and max length (48 tokens) as
newspop.titles.bge_embeddings, run through onnxruntime (no torch needed). If the older cache
data/interim/mashable_title_bge.npy exists, vectors of identical headline strings are reused.
Resumable: finished batches are appended to a .part file.

Output: data/interim/dml/title_bge_true.npy  (39,644 x 768; rows without a headline are NaN)
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

import common as C
from newspop.titles import BGE_CACHE

OUTF = C.CACHE / "title_bge_true.npy"
if OUTF.exists():
    print("exists:", OUTF); raise SystemExit
R = pd.read_csv(C.REALIGNED)
titles = R.true_title
E = np.full((len(R), 768), np.nan, np.float32)
if BGE_CACHE.exists():  # reuse: the old cache is in newspop.data.load() order, keyed by the url in the file
    from newspop.data import load
    from newspop.titles import aligned_titles
    old = load(); old_t = aligned_titles(old); old_E = np.load(BGE_CACHE)
    first = pd.Series(np.arange(len(old)), index=old_t.to_numpy())
    first = first[~first.index.duplicated() & first.index.notna()]
    pos = titles.map(first)
    ok = pos.notna().to_numpy()
    E[ok] = old_E[pos[ok].astype(int).to_numpy()]
    print("reused", ok.sum(), "of", titles.notna().sum(), flush=True)
todo = np.where(titles.notna().to_numpy() & np.isnan(E[:, 0]))[0]
if len(todo):
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    name = "BAAI/bge-base-en-v1.5"
    tok = Tokenizer.from_file(hf_hub_download(name, "tokenizer.json"))
    tok.enable_truncation(max_length=48); tok.enable_padding()
    so = ort.SessionOptions(); so.intra_op_num_threads = C.THREADS
    sess = ort.InferenceSession(hf_hub_download(name, "onnx/model.onnx"), so, providers=["CPUExecutionProvider"])
    T = titles.tolist()
    todo = todo[np.argsort([len(T[i]) for i in todo])]
    part = Path(str(OUTF) + ".part.npz")
    done = dict(np.load(part)) if part.exists() else {}
    for b in range(0, len(todo), 128):
        if str(b) not in done:
            ix = todo[b:b + 128]
            enc = tok.encode_batch([T[i] for i in ix])
            feed = {"input_ids": np.array([e.ids for e in enc], np.int64),
                    "attention_mask": np.array([e.attention_mask for e in enc], np.int64),
                    "token_type_ids": np.array([e.type_ids for e in enc], np.int64)}
            done[str(b)] = sess.run(None, feed)[0][:, 0]
            if (b // 128) % 20 == 0:
                np.savez(part, **done); print(b, len(todo), flush=True)
        E[todo[b:b + 128]] = done[str(b)]
    part.unlink(missing_ok=True)
np.save(OUTF, E)
print("saved", OUTF, "rows with a vector:", int((~np.isnan(E[:, 0])).sum()))
