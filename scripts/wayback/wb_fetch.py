"""Reconstruct early share trajectories and homepage exposure for Mashable 2013-2014 from the Internet Archive.

Standard library only (runs on any Python >= 3.8, Windows or Linux). Everything is resumable: re-running
skips whatever is already on disk. Raw responses are kept (gzip) so parsing can be redone offline.

Stages (run in this order by --stage all)
  lists     CDX index + download of list pages that embed exact share counts as JSON:
            the homepage, /stories.json (all parameter variants) and the channel pages.
            Each capture holds the "new" / "rising" / "hot" columns (The New Stuff / The Next Big Thing /
            What's Hot) with per-network share counts, author, exact publication time and, for "rising",
            Mashable's Velocity series.
  artcdx    CDX index of every archived article page, one query per publication day.
  articles  Download selected article-page captures, in passes:
              pass 1  the earliest capture of every article
              pass 2  the capture closest to 2015-01-08 (when the UCI dataset collected its share counts)
              pass 3  up to two more captures from the first four days

Outputs under --root
  cdx/*.json                      raw CDX answers
  raw/<kind>/<yyyymm>/<key>.gz    raw archived bytes
  parsed/lists.jsonl              one line per list capture (items without the article text)
  parsed/texts.jsonl              article id -> plain text, first sighting only
  parsed/articles.jsonl           one line per article-page capture
  log.txt, status.json            progress

Politeness: one request at a time, --delay seconds between requests (default 1.0), exponential backoff on
429 / 5xx / network errors, generic User-Agent.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

UA = "research-archive-reader/0.2 (academic coursework; polite, 1 req/s)"
CDX = "https://web.archive.org/cdx/search/cdx"
CHANNELS = ["tech", "social-media", "business", "entertainment", "world", "us-world", "lifestyle", "watercooler"]
T_UCI = dt.datetime(2015, 1, 8)
ART_RE = re.compile(r"^https?://(?:www\.)?mashable\.com(?::80)?/(\d{4}/\d{2}/\d{2}/[^/?#]+)/?(\?.*)?$", re.I)

ARGS = None
_last = [0.0]


def log(*a):
    line = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(os.path.join(ARGS.root, "log.txt"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def http_get(url, tries=6):
    """Polite GET with backoff. Returns bytes, or None after repeated failure / on 404."""
    for k in range(tries):
        wait = ARGS.delay - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
                if r.headers.get("Content-Encoding", "") == "gzip" or data[:2] == b"\x1f\x8b":
                    try:
                        data = gzip.decompress(data)
                    except Exception:
                        pass
                return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            pause = min(600, 30 * 2 ** k) if e.code in (429, 503) else 10 * (k + 1)
            log("HTTP", e.code, "pause", pause, url[:120])
            time.sleep(pause)
        except Exception as e:  # network error, timeout, reset
            pause = min(300, 15 * 2 ** k)
            log("ERR", type(e).__name__, str(e)[:80], "pause", pause, url[:120])
            time.sleep(pause)
    log("GIVEUP", url[:160])
    return None


def cdx_query(name, params):
    """CDX query cached in cdx/<name>.json; follows resume keys. Returns list of rows (lists of str)."""
    path = os.path.join(ARGS.root, "cdx", name + ".json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    rows, resume = [], None
    while True:
        p = dict(params, output="json", showResumeKey="true", limit="50000")
        if resume:
            p["resumeKey"] = resume
        raw = http_get(CDX + "?" + urllib.parse.urlencode(p, doseq=True))
        if raw is None:
            log("CDX failed", name)
            return None
        txt = raw.decode("utf-8", "replace").strip()
        data = json.loads(txt) if txt else []
        resume = None
        if data and data[0] and data[0][0] in ("timestamp", "urlkey"):
            data = data[1:]
        if len(data) >= 2 and data[-2] == []:
            resume = data[-1][0]
            data = data[:-2]
        rows.extend(data)
        if not resume:
            break
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(rows, f)
    os.replace(path + ".tmp", path)
    return rows


# ------------------------------------------------------------------ parsing
KEEP = ["id", "title", "author", "post_date", "channel", "shares", "comments_count", "velocity", "sort_key",
        "choice", "spotlight", "short_url"]


def parse_lists(text):
    """Return {'new': [...], 'rising': [...], 'hot': [...], ...} from a homepage/channel page or stories.json."""
    out = {}
    t = text.lstrip()
    if t[:1] == "{":
        try:
            obj = json.loads(t)
            out = {k: v for k, v in obj.items() if isinstance(v, list)}
            if isinstance(obj.get("channel"), dict):
                out["_channel"] = {k: obj["channel"].get(k) for k in ("id", "name", "follower_count", "posts_count")}
            return out
        except Exception:
            pass
    dec = json.JSONDecoder()
    for m in re.finditer(r'window\.postData\[\s*["\'](\w+)["\']\s*\]\s*=\s*', text):
        try:
            obj, _ = dec.raw_decode(text, m.end())
        except Exception:
            continue
        if isinstance(obj, list):
            out[m.group(1)] = obj
        elif isinstance(obj, dict):  # e.g. "feature": {"hero": [...], ...}
            for k, v in obj.items():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    out[m.group(1) + "." + k] = v
    return out


def slim(item):
    d = {k: item.get(k) for k in KEEP if k in item}
    c = item.get("content")
    plain = c.get("plain") if isinstance(c, dict) else None
    if plain:
        d["n_chars"] = len(plain)
    return d, plain


SHARE_RE = re.compile(r'social-share\s+(\w+)"[^>]*?data-shares=["\']([^"\']*)["\']')


def parse_article(text):
    d = {}
    m = re.search(r"class=['\"]total-shares['\"][^>]*>\s*<em>([^<]*)</em>", text)
    d["total_shown"] = m.group(1).strip() if m else None
    i = text.find("share-buttons")
    block = text[i:i + 4000] if i >= 0 else ""
    d["net"] = {k: v for k, v in SHARE_RE.findall(block)}
    m = re.search(r'<meta content="([^"]+)"[^>]*property="og:article:published_time"', text)
    d["published"] = m.group(1) if m else None
    m = re.search(r'<meta content="([^"]+)"[^>]*property="og:article:modified_time"', text)
    d["modified"] = m.group(1) if m else None
    m = re.search(r'<link[^>]*href="([^"]+)"[^>]*rel="canonical"', text)
    d["canonical"] = m.group(1) if m else None
    m = re.search(r'<meta content="([^"]*)"[^>]*property="og:title"', text)
    d["og_title"] = m.group(1) if m else None
    m = re.search(r'"author_name":"((?:[^"\\]|\\.)*)"', text)
    d["author_name"] = m.group(1) if m else None
    m = re.search(r"class=['\"]byline basic['\"]>\s*([^<]*)<|class=['\"]author_name['\"]>\s*([^<]*)<", text)
    d["byline"] = (m.group(1) or m.group(2) or "").strip() if m else None
    m = re.search(r'"topics":"((?:[^"\\]|\\.)*)"', text)
    d["topics"] = m.group(1) if m else None
    m = re.search(r"window\.__o\s*=\s*(\{[^\n]*?\});?\s*\n", text)
    if m:
        try:
            o = json.loads(m.group(1))
            d["o"] = {k: o.get(k) for k in ("channel", "content_type", "h_pub", "v_cur", "v_max", "v_cur_i", "v_max_i")}
        except Exception:
            pass
    m = re.search(r"<article[^>]*id=['\"]story['\"][^>]*>|<article[^>]*data-wp-id[^>]*>", text)
    if m:
        tag = m.group(0)
        mv = re.search(r"data-velocity=['\"]([^'\"]*)['\"]", tag) or re.search(r"=['\"]([\d,]{20,})['\"]", tag)
        d["velocity"] = mv.group(1) if mv else None
        mc = re.search(r"data-channel=['\"]([^'\"]*)['\"]", tag)
        d["data_channel"] = mc.group(1) if mc else None
    d["n_bytes"] = len(text)
    return d


# ------------------------------------------------------------------ storage helpers
def load_done(name):
    path = os.path.join(ARGS.root, "parsed", name)
    s = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                s.add(line.strip())
    return s


def append(name, text):
    path = os.path.join(ARGS.root, "parsed", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def save_raw(kind, ts, key, data):
    d = os.path.join(ARGS.root, "raw", kind, ts[:6])
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, key + ".gz"), "wb") as f:
        f.write(gzip.compress(data, 6))


def status(**kw):
    path = os.path.join(ARGS.root, "status.json")
    cur = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                cur = json.load(f)
        except Exception:
            cur = {}
    cur.update(kw, updated=dt.datetime.utcnow().isoformat())
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(cur, f, indent=1)
    os.replace(path + ".tmp", path)


def months(a="201301", b="201502"):
    y, m = int(a[:4]), int(a[4:])
    while "%04d%02d" % (y, m) <= b:
        yield "%04d%02d" % (y, m)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def month_end(ym):
    y, m = int(ym[:4]), int(ym[4:])
    nxt = dt.date(y + 1, 1, 1) if m == 12 else dt.date(y, m + 1, 1)
    return (nxt - dt.timedelta(days=1)).strftime("%Y%m%d")


# ------------------------------------------------------------------ stage: lists
def stage_lists():
    plan = []  # (kind, timestamp, original)
    rows = cdx_query("home", {"url": "mashable.com/", "from": "20130101", "to": "20150215",
                              "filter": "statuscode:200", "fl": "timestamp,original,digest", "collapse": "digest"})
    plan += [("home", r[0], r[1]) for r in rows or []]
    for ym in months():
        rows = cdx_query("stories_" + ym, {"url": "mashable.com/stories.json", "matchType": "prefix",
                                           "from": ym + "01", "to": month_end(ym), "filter": "statuscode:200",
                                           "fl": "timestamp,original,digest", "collapse": "digest"})
        plan += [("stories", r[0], r[1]) for r in rows or []]
    for ch in CHANNELS:
        rows = cdx_query("channel_" + ch, {"url": "mashable.com/%s/" % ch, "from": "20130101", "to": "20150215",
                                           "filter": "statuscode:200", "fl": "timestamp,original,digest",
                                           "collapse": "digest"})
        plan += [("ch-" + ch, r[0], r[1]) for r in rows or []]
    # homepage first (chronological), then stories.json, then channel pages
    order = {"home": 0, "stories": 1}
    plan.sort(key=lambda p: (order.get(p[0], 2), p[1]))
    done = load_done("lists_done.txt")
    seen_text = load_done("texts_ids.txt")
    log("lists plan", len(plan), "already done", len(done))
    n_ok = n_empty = 0
    for i, (kind, ts, orig) in enumerate(plan):
        key = "%s_%s_%s" % (kind, ts, hashlib.md5(orig.encode()).hexdigest()[:8])
        if key in done:
            continue
        data = http_get("https://web.archive.org/web/%sid_/%s" % (ts, orig))
        if data is None:
            append("lists_done.txt", key)
            continue
        save_raw("lists", ts, key, data)
        text = data.decode("utf-8", "replace")
        lists = parse_lists(text)
        rec = {"kind": kind, "ts": ts, "orig": orig, "lists": {}}
        for name, items in lists.items():
            if name == "_channel":
                rec["channel"] = items
                continue
            out = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                d, plain = slim(it)
                out.append(d)
                if plain and d.get("id") and d["id"] not in seen_text:
                    seen_text.add(d["id"])
                    append("texts.jsonl", json.dumps({"id": d["id"], "ts": ts, "text": plain}, ensure_ascii=False))
                    append("texts_ids.txt", d["id"])
            rec["lists"][name] = out
        n_items = sum(len(v) for v in rec["lists"].values())
        n_ok += n_items > 0
        n_empty += n_items == 0
        append("lists.jsonl", json.dumps(rec, ensure_ascii=False))
        append("lists_done.txt", key)
        if (n_ok + n_empty) % 50 == 0:
            log("lists", i + 1, "/", len(plan), "ok", n_ok, "empty", n_empty, "last", kind, ts)
            status(lists_pos=i + 1, lists_total=len(plan), lists_ok=n_ok, lists_empty=n_empty)
    status(lists_pos=len(plan), lists_total=len(plan), lists_finished=True)
    log("lists finished")


# ------------------------------------------------------------------ stage: artcdx / articles
def stage_artcdx():
    d = dt.date(2013, 1, 1)
    n = 0
    while d <= dt.date(2014, 12, 31):
        name = "art_" + d.strftime("%Y%m%d")
        rows = cdx_query(name, {"url": "mashable.com/%s/" % d.strftime("%Y/%m/%d"), "matchType": "prefix",
                                "from": d.strftime("%Y%m%d"), "to": "20151231", "filter": "statuscode:200",
                                "fl": "timestamp,original,digest"})
        n += len(rows or [])
        if d.day == 1:
            log("artcdx", d, "rows so far", n)
            status(artcdx_day=str(d), artcdx_rows=n)
        d += dt.timedelta(days=1)
    status(artcdx_finished=True, artcdx_rows=n)
    log("artcdx finished", n)


def article_plan():
    caps = {}  # article id -> list of (ts, original, digest)
    d = dt.date(2013, 1, 1)
    while d <= dt.date(2014, 12, 31):
        path = os.path.join(ARGS.root, "cdx", "art_%s.json" % d.strftime("%Y%m%d"))
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for ts, orig, dig in json.load(f):
                    m = ART_RE.match(orig)
                    if not m:
                        continue
                    aid = m.group(1).lower()
                    if aid.endswith((".json", ".xml", "/feed", "/amp")):
                        continue
                    caps.setdefault(aid, []).append((ts, orig, dig, bool(m.group(2))))
        d += dt.timedelta(days=1)
    p1, p2, p3 = [], [], []
    for aid, lst in caps.items():
        lst.sort(key=lambda c: (c[0], c[3]))  # by time, clean URL before ?variant
        uniq, seen = [], set()
        for c in lst:  # one capture per digest
            if c[2] in seen:
                continue
            seen.add(c[2]); uniq.append(c)
        t0 = dt.datetime.strptime(aid[:10], "%Y/%m/%d")
        when = [dt.datetime.strptime(c[0], "%Y%m%d%H%M%S") for c in uniq]
        p1.append((aid, uniq[0]))
        j = min(range(len(uniq)), key=lambda k: abs((when[k] - T_UCI).total_seconds()))
        if j != 0:
            p2.append((aid, uniq[j]))
        extra = [k for k in range(1, len(uniq)) if k != j and (when[k] - t0).days <= 4][:2]
        p3 += [(aid, uniq[k]) for k in extra]
    for p in (p1, p2, p3):
        p.sort(key=lambda x: x[0])
    return p1, p2, p3


def stage_articles():
    p1, p2, p3 = article_plan()
    done = load_done("articles_done.txt")
    log("articles plan: pass1", len(p1), "pass2", len(p2), "pass3", len(p3), "done", len(done))
    n = 0
    for pno, plan in ((1, p1), (2, p2), (3, p3)):
        if ARGS.max_pass and pno > ARGS.max_pass:
            break
        for i, (aid, (ts, orig, dig, _)) in enumerate(plan):
            key = "%s_%s" % (ts, hashlib.md5(aid.encode()).hexdigest()[:10])
            if key in done:
                continue
            data = http_get("https://web.archive.org/web/%sid_/%s" % (ts, orig))
            if data is not None:
                save_raw("articles", aid[:4] + aid[5:7], key, data)
                rec = parse_article(data.decode("utf-8", "replace"))
                rec.update(id=aid, ts=ts, p=pno)
                append("articles.jsonl", json.dumps(rec, ensure_ascii=False))
            append("articles_done.txt", key)
            done.add(key)
            n += 1
            if n % 100 == 0:
                log("articles pass", pno, i + 1, "/", len(plan), "fetched this run", n)
                status(articles_pass=pno, articles_pos=i + 1, articles_total=len(plan), articles_fetched_run=n)
        status(**{"articles_pass%d_finished" % pno: True})
    log("articles finished")


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--stage", default="all", choices=["all", "lists", "artcdx", "articles"])
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--max-pass", type=int, default=0)
    ARGS = ap.parse_args()
    os.makedirs(os.path.join(ARGS.root, "parsed"), exist_ok=True)
    log("start", ARGS.stage, "delay", ARGS.delay, "python", sys.version.split()[0])
    if ARGS.stage in ("all", "lists"):
        stage_lists()
    if ARGS.stage in ("all", "artcdx"):
        stage_artcdx()
    if ARGS.stage in ("all", "articles"):
        stage_articles()
    log("all done")


if __name__ == "__main__":
    main()
