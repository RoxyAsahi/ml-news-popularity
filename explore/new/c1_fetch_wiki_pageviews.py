"""Probe C data: daily English-Wikipedia pageviews 2012-12 .. 2014-12 from the Wikimedia monthly dumps.

The monthly file (about 5 GB, one bzip2 stream, sorted by wiki) is not downloaded whole. bzip2 blocks start with a
48-bit magic number at an arbitrary bit offset, so we (1) binary-search the byte range that holds the en.wikipedia
rows with small HTTP range requests, decoding a block from the middle of the file, and (2) stream only that range
in chunks, re-aligning the bit stream on the fly. Resumable per month; output:
  <out>/pv_YYYYMM.tsv.gz   title <tab> access <tab> monthly_total <tab> daily string (A=day 1, B=day 2, ...)
for pages with at least --min-total views in the month.
"""
from __future__ import annotations

import argparse
import bz2
import gzip
import os
import sys
import time
import urllib.request

import numpy as np

MAGIC = 0x314159265359
URL = "https://dumps.wikimedia.org/other/pageview_complete/monthly/{y}/{y}-{m}/pageviews-{y}{m}-user.bz2"
UA = "research-script/0.1 (academic coursework)"


def http_range(url, a, n, tries=8):
    for k in range(tries):
        try:
            r = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (a, a + n - 1), "User-Agent": UA})
            with urllib.request.urlopen(r, timeout=180) as f:
                return f.read()
        except Exception as e:
            print("  retry", k, type(e).__name__, str(e)[:60], flush=True)
            time.sleep(min(120, 5 * 2 ** k))
    raise RuntimeError("range request failed")


def size_of(url):
    r = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    with urllib.request.urlopen(r, timeout=60) as f:
        return int(f.headers["Content-Length"])


def find_magic(b, start=0):
    """First (byte, bit) position of the block magic in bytes b at or after byte `start`."""
    a = np.frombuffer(b, np.uint8).astype(np.uint64)
    n = len(a) - 7
    w = np.zeros(n, np.uint64)
    for j in range(7):  # 56-bit window starting at each byte
        w = (w << np.uint64(8)) | a[j:n + j]
    best = None
    for k in range(8):
        hit = np.flatnonzero(((w >> np.uint64(8 - k)) & np.uint64(0xFFFFFFFFFFFF)) == np.uint64(MAGIC))
        hit = hit[hit >= start]
        if len(hit) and (best is None or (hit[0], k) < best):
            best = (int(hit[0]), k)
    return best


def shift(b, k, carry=b""):
    """Left-shift the byte string carry+b by k bits; returns (aligned bytes, new 1-byte carry)."""
    a = np.frombuffer(carry + b, np.uint8)
    if k == 0:
        return a[:-1].tobytes(), a[-1:].tobytes()
    out = (a[:-1] << k) | (a[1:] >> (8 - k))
    return out.tobytes(), a[-1:].tobytes()


def first_wiki_at(url, off):
    """Wiki code of the first complete row decoded from the first block at/after byte offset off."""
    b = http_range(url, off, 4_000_000)
    pos = find_magic(b)
    if pos is None:
        return None
    i, k = pos
    data, _ = shift(b[i:], k)
    out = bz2.BZ2Decompressor().decompress(b"BZh9" + data, 300_000)
    lines = out.split(b"\n")
    return lines[1].split(b" ", 1)[0].decode("utf-8", "replace") if len(lines) > 2 else None


def bisect(url, size, pred):
    """Smallest probe offset (on a 16 MB grid) whose first decoded wiki satisfies pred."""
    lo, hi = 0, size // (16 << 20)
    while lo < hi:
        mid = (lo + hi) // 2
        w = first_wiki_at(url, mid * (16 << 20))
        if w is not None and pred(w):
            hi = mid
        else:
            lo = mid + 1
    return lo * (16 << 20)


SEG, OVER = 96 << 20, 8 << 20


def segment(url, size, start, min_total):
    """Decode bytes [start, start+SEG+OVER) from the first block boundary; return (rows, saw_end_of_en)."""
    rows, past = [], False
    dec, k, carry, tail = None, 0, b"", b""
    off, end = start, min(size, start + SEG + OVER)
    while off < end and not past:
        chunk = http_range(url, off, min(32 << 20, end - off))
        off += len(chunk)
        if dec is None:
            pos = find_magic(chunk)
            if pos is None:
                continue
            i, k = pos
            data, carry = shift(chunk[i:], k)
            dec = bz2.BZ2Decompressor()
            text = dec.decompress(b"BZh9" + data)
            text = text[text.find(b"\n") + 1:]  # drop the partial first row
        else:
            data, carry = shift(chunk, k, carry)
            text = dec.decompress(data)
        lines = (tail + text).split(b"\n")
        tail = lines.pop()  # partial last row: covered by the next segment's overlap
        for ln in lines:
            if not ln.startswith(b"en.wikipedia "):
                if ln[:13] > b"en.wikipedia ":
                    past = True
                    break
                continue
            p = ln.split(b" ")
            if len(p) != 6:
                continue
            try:
                tot = int(p[4])
            except ValueError:
                continue
            if tot >= min_total:
                rows.append("%s\t%s\t%d\t%s" % (p[1].decode("utf-8", "replace"), p[3].decode(), tot, p[5].decode()))
    return rows, past


def month(ym, out, min_total):
    """Resumable at segment level: each 96 MB segment is written atomically, then all parts are merged."""
    import json
    y, m = ym[:4], ym[4:]
    dst = os.path.join(out, "pv_%s.tsv.gz" % ym)
    if os.path.exists(dst):
        return
    url = URL.format(y=y, m=m)
    rj = os.path.join(out, "pv_%s.range.json" % ym)
    if os.path.exists(rj):
        size, a, b = json.load(open(rj))
    else:
        size = size_of(url)
        a = max(0, bisect(url, size, lambda w: w >= "en.wikipedia") - (32 << 20))
        b = min(size, bisect(url, size, lambda w: w > "en.wikipedia" and not w.startswith("en.wikipedia")) + (32 << 20))
        json.dump([size, a, b], open(rj, "w"))
    print(ym, "size %.2f GB, en.wikipedia in [%.2f, %.2f] GB" % (size / 2**30, a / 2**30, b / 2**30), flush=True)
    t0 = time.time()
    i, parts, ended = 0, [], False
    while a + i * SEG < b and not ended:
        part = os.path.join(out, "pv_%s.part%03d.tsv.gz" % (ym, i))
        flag = part + ".end"
        if not os.path.exists(part):
            rows, past = segment(url, size, a + i * SEG, min_total)
            with gzip.open(part + ".tmp", "wt", encoding="utf-8") as g:
                g.write("\n".join(rows) + ("\n" if rows else ""))
            if past:
                open(flag, "w").close()
            os.replace(part + ".tmp", part)
            print("  %s segment %d: %d rows, %.0f s" % (ym, i, len(rows), time.time() - t0), flush=True)
        ended = os.path.exists(flag)
        parts.append(part)
        i += 1
    if not ended and b < size:
        print(ym, "WARNING: did not see the end of en.wikipedia; widen the range", flush=True)
    seen = set()
    with gzip.open(dst + ".tmp", "wt", encoding="utf-8") as g:
        for part in parts:
            with gzip.open(part, "rt", encoding="utf-8") as f:
                for line in f:
                    if line not in seen:
                        seen.add(line)
                        g.write(line)
    os.replace(dst + ".tmp", dst)
    for part in parts:
        os.remove(part)
        if os.path.exists(part + ".end"):
            os.remove(part + ".end")
    print(ym, "done", len(seen), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/user/work/wiki")
    ap.add_argument("--months", default="")
    ap.add_argument("--min-total", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ms = a.months.split(",") if a.months else ["201212"] + ["2013%02d" % i for i in range(1, 13)] + ["2014%02d" % i for i in range(1, 13)]
    for ym in ms:
        try:
            month(ym, a.out, a.min_total)
        except Exception as e:
            print(ym, "FAILED", type(e).__name__, e, flush=True)
    print("ALLDONE", flush=True)
