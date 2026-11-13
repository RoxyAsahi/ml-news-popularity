"""Build the observation panel from the Internet Archive captures (scripts/wayback/wb_fetch.py output).

Input  (--src, default data/external/wayback/parsed): lists.jsonl, articles.jsonl (optional), texts.jsonl (optional)
Output (data/interim/):
  wb_obs.parquet    one row per sighting of an article with a share count:
                    art_id, ts (capture time, UTC), src (home / stories / ch-* / article), col (new / rising / hot / page),
                    rank (position in the column, 0 = top), age_h, total, twitter, facebook, linked_in, google_plus,
                    pinterest, stumble_upon, vel_last, vel_n
  wb_meta.parquet   one row per article: art_id, post_date (UTC, to the second), author, channel, title, n_chars
Sightings of the same article in the same capture in several columns are all kept (col differs).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NETS = ["twitter", "facebook", "linked_in", "google_plus", "pinterest", "stumble_upon"]


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return np.nan


def lines(src, stem):
    """Parsed JSON lines of every <stem>*.jsonl file (one per fetch process); broken last lines are skipped."""
    for fn in sorted(src.glob(stem + "*.jsonl")):
        with open(fn, encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "external" / "wayback" / "parsed"))
    a = ap.parse_args()
    src = Path(a.src)
    obs, meta = [], {}
    seen_cap = set()
    for line in lines(src, "lists"):
        if True:
            r = line
            if (r["kind"], r["ts"], r["orig"]) in seen_cap:
                continue
            seen_cap.add((r["kind"], r["ts"], r["orig"]))
            for col, items in r["lists"].items():
                for rank, it in enumerate(items):
                    aid = it.get("id")
                    if not aid or not isinstance(it.get("shares"), dict):
                        continue
                    aid = aid.lower().strip("/")
                    sh = it["shares"]
                    v = it.get("velocity") or []
                    row = dict(art_id=aid, ts=r["ts"], src=r["kind"], col=col, rank=rank, total=num(sh.get("total")),
                               vel_last=float(v[-1]) if v else np.nan, vel_n=len(v),
                               vel_slope=float(v[-1] - v[max(0, len(v) - 6)]) if v else np.nan)
                    for k in NETS:
                        row[k] = num(sh.get(k, 0))
                    obs.append(row)
                    m = meta.setdefault(aid, {})
                    for k in ("post_date", "author", "channel", "title", "n_chars"):
                        if it.get(k) is not None and k not in m:
                            m[k] = it[k]
    n_list = len(obs)
    if True:
        if True:
            for r in lines(src, "articles"):
                net = {k: num(v) for k, v in (r.get("net") or {}).items()}
                if not net:
                    continue
                aid = r["id"].lower()
                vel = [num(x) for x in (r.get("velocity") or "").split(",") if x]
                row = dict(art_id=aid, ts=r["ts"], src="article", col="page", rank=-1,
                           total=float(np.nansum(list(net.values()))),
                           vel_last=vel[-1] if vel else np.nan, vel_n=len(vel),
                           vel_slope=vel[-1] - vel[max(0, len(vel) - 6)] if vel else np.nan)
                for k in NETS:
                    row[k] = net.get(k, 0.0)
                obs.append(row)
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
    O = pd.DataFrame(obs)
    M = pd.DataFrame.from_dict(meta, orient="index").rename_axis("art_id").reset_index()
    M["post_date"] = pd.to_datetime(M.post_date, utc=True, errors="coerce").dt.tz_localize(None)
    O["ts"] = pd.to_datetime(O.ts, format="%Y%m%d%H%M%S")
    O = O.merge(M[["art_id", "post_date"]], on="art_id", how="left")
    O["age_h"] = (O.ts - O.post_date).dt.total_seconds() / 3600
    O = O.drop(columns=["post_date"]).drop_duplicates(["art_id", "ts", "src", "col"]).sort_values(["art_id", "ts"])
    out = ROOT / "data" / "interim"
    out.mkdir(parents=True, exist_ok=True)
    O.to_parquet(out / "wb_obs.parquet", index=False)
    M.to_parquet(out / "wb_meta.parquet", index=False)
    print(f"sightings {len(O):,} (lists {n_list:,}); articles {O.art_id.nunique():,}; captures {O.ts.nunique():,}")
    print(O.groupby("src").size().to_string())
    print(O.age_h.describe().round(1).to_string())


if __name__ == "__main__":
    main()
