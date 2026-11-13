"""Compact the parsed archive captures into two small CSV files (standard library only).

Reads <root>/parsed/lists*.jsonl and articles*.jsonl (output of wb_fetch.py; several shard files are fine,
a broken last line is skipped, repeated captures are de-duplicated) and writes
  <out>/wb_obs.csv.gz    one row per sighting of an article with a share count
  <out>/wb_meta.csv.gz   one row per article (publication time, author, channel, title, text length)
  <out>/wb_summary.json  counts, for a quick look
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import os

NETS = ["twitter", "facebook", "linked_in", "google_plus", "pinterest", "stumble_upon"]


def lines(root, stem):
    for fn in sorted(glob.glob(os.path.join(root, "parsed", stem + "*.jsonl"))):
        with open(fn, encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta, seen, n_obs, n_cap, kinds = {}, set(), 0, 0, {}
    with gzip.open(os.path.join(a.out, "wb_obs.csv.gz"), "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["art_id", "ts", "src", "col", "rank", "total"] + NETS + ["comments", "vel_n", "vel_last", "vel_first",
                                                                           "vel_min", "vel_max", "vel_slope"])
        for r in lines(a.root, "lists"):
            key = (r.get("kind"), r.get("ts"), r.get("orig"))
            if key in seen:
                continue
            seen.add(key)
            n_cap += 1
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
            for col, items in (r.get("lists") or {}).items():
                for rank, it in enumerate(items):
                    aid = (it.get("id") or "").lower().strip("/")
                    sh = it.get("shares")
                    if aid:
                        m = meta.setdefault(aid, {})
                        for k in ("post_date", "author", "channel", "title", "n_chars"):
                            if it.get(k) is not None and k not in m:
                                m[k] = it[k]
                    if not aid or not isinstance(sh, dict):
                        continue
                    v = [x for x in (it.get("velocity") or []) if isinstance(x, (int, float))]
                    w.writerow([aid, r["ts"], r["kind"], col, rank, num(sh.get("total"))] + [num(sh.get(k, 0)) for k in NETS]
                               + [it.get("comments_count", ""), len(v), v[-1] if v else "", v[0] if v else "",
                                  min(v) if v else "", max(v) if v else "", (v[-1] - v[max(0, len(v) - 6)]) if v else ""])
                    n_obs += 1
        n_art = 0
        for r in lines(a.root, "articles"):
            key = ("article", r.get("ts"), r.get("id"))
            if key in seen:
                continue
            seen.add(key)
            aid = (r.get("id") or "").lower()
            net = {k: num(v) for k, v in (r.get("net") or {}).items()}
            m = meta.setdefault(aid, {})
            if r.get("published") and "post_date" not in m:
                m["post_date"] = r["published"]
            if "author" not in m and (r.get("author_name") or r.get("byline")):
                m["author"] = r.get("author_name") or r.get("byline")
            if "title" not in m and r.get("og_title"):
                m["title"] = r["og_title"]
            o = r.get("o") or {}
            if "channel" not in m and o.get("channel"):
                m["channel"] = o["channel"]
            for k in ("topics", "canonical", "byline"):
                if r.get(k) and k not in m:
                    m[k] = r[k]
            if not net:
                continue
            v = [num(x) for x in (r.get("velocity") or "").split(",") if x]
            v = [x for x in v if x != ""]
            tot = sum(x for x in net.values() if x != "")
            w.writerow([aid, r["ts"], "article", "page", -1, tot] + [net.get(k, 0) for k in NETS]
                       + ["", len(v), v[-1] if v else "", v[0] if v else "", min(v) if v else "", max(v) if v else "",
                          (v[-1] - v[max(0, len(v) - 6)]) if v else ""]
                       )
            n_obs += 1
            n_art += 1
    with gzip.open(os.path.join(a.out, "wb_meta.csv.gz"), "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        cols = ["post_date", "author", "channel", "title", "n_chars", "topics", "canonical", "byline"]
        w.writerow(["art_id"] + cols)
        for aid in sorted(meta):
            w.writerow([aid] + [meta[aid].get(k, "") for k in cols])
    summary = dict(list_captures=n_cap, by_kind=kinds, article_captures=n_art, sightings=n_obs, articles=len(meta))
    with open(os.path.join(a.out, "wb_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
