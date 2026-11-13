"""Fetch Mashable headlines for 2013-2014 from the public monthly sitemap pages.

What it does
------------
1. Downloads the 24 monthly sitemap pages
       https://mashable.com/sitemap/YYYY/M      (YYYY in 2013, 2014; M in 1..12)
   one request per second (time.sleep(1) between requests), saving the raw HTML
   to ``--html-dir`` as ``YYYY-MM_p1.html``. Pages already on disk are reused,
   so re-running does not download again.
2. Parses each page for (date, title, link): every article is listed as a
   ``<span class="inline-block w-full lg:w-28">MM/DD/YYYY</span>`` followed by
   ``<a class="inline-block text-gray-dark" href="/archive/...">Title</a>``.
3. Writes ``date,title,url`` (ISO date, HTML-unescaped title, absolute URL) to
   ``--out`` (default: data/external/mashable_sitemap_titles.csv).

The committed CSV has 40,161 rows. The raw HTML is not kept in the repository.

Usage
-----
    python scripts/fetch_mashable_titles.py --html-dir /tmp/mashable_html
"""

import argparse
import csv
import html
import os
import re
import time
import urllib.request

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")
YEARS = (2013, 2014)
ROW_RE = re.compile(
    r'<span class="inline-block w-full lg:w-28">([^<]*)</span>\s*'
    r'<a class="inline-block text-gray-dark" href="([^"]*)">(.*?)</a>',
    re.S,
)


def download(html_dir):
    os.makedirs(html_dir, exist_ok=True)
    for y in YEARS:
        for m in range(1, 13):
            path = os.path.join(html_dir, f"{y}-{m:02d}_p1.html")
            if os.path.exists(path):
                continue
            url = f"https://mashable.com/sitemap/{y}/{m}"
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
            with open(path, "wb") as fh:
                fh.write(body)
            print(url, r.status, len(body), flush=True)
            time.sleep(1)  # one request per second


def parse(html_dir):
    rows = []
    for y in YEARS:
        for m in range(1, 13):
            path = os.path.join(html_dir, f"{y}-{m:02d}_p1.html")
            with open(path, encoding="utf-8") as fh:
                page = fh.read()
            for date, href, title in ROW_RE.findall(page):
                mm, dd, yy = date.strip().split("/")
                url = "https://mashable.com" + href if href.startswith("/") else href
                rows.append((f"{yy}-{mm}-{dd}", html.unescape(title).strip(), url))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--html-dir", required=True, help="folder for the raw monthly HTML pages")
    ap.add_argument("--out", default=os.path.join("data", "external", "mashable_sitemap_titles.csv"))
    ap.add_argument("--no-download", action="store_true", help="only parse pages already on disk")
    args = ap.parse_args()

    if not args.no_download:
        download(args.html_dir)
    rows = parse(args.html_dir)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "title", "url"])
        w.writerows(rows)
    print(f"{len(rows)} rows, {len({r[2] for r in rows})} unique urls -> {args.out}")


if __name__ == "__main__":
    main()
