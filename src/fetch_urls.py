"""Resumable, retrying downloader for IDA image-zip URL lists.

Usage: python src/fetch_urls.py <url_list.csv> <out_dir> [--name-prefix P]
- One URL per line. Each file is fetched with HTTP Range resume, retried with backoff
  (survives network drops / public-IP changes), size-checked against Content-Length and
  zip-tested before being marked done.
- Progress: <out_dir>/fetch_log.txt ; completion marker: <out_dir>/<file>.ok
No credentials are needed (IDA URLs are pre-signed); none are stored.
"""
from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path

import requests

CHUNK = 4 * 1024 * 1024


def log(out: Path, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with (out / "fetch_log.txt").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def fetch(url: str, dest: Path, out: Path, max_tries: int = 200) -> bool:
    ok = dest.with_suffix(dest.suffix + ".ok")
    if ok.exists():
        return True
    part = dest.with_suffix(dest.suffix + ".part")
    total = None
    for attempt in range(1, max_tries + 1):
        try:
            if total is None:
                h = requests.head(url, timeout=60, allow_redirects=True)
                h.raise_for_status()
                total = int(h.headers.get("Content-Length", 0)) or None
            have = part.stat().st_size if part.exists() else 0
            if total and have >= total:
                break
            headers = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as r:
                if r.status_code in (403, 404):
                    log(out, f"{dest.name}: HTTP {r.status_code} (URL expired or IP-restricted)")
                    time.sleep(min(600, 30 * attempt))
                    continue
                r.raise_for_status()
                mode = "ab" if have and r.status_code == 206 else "wb"
                with part.open(mode) as fh:
                    for chunk in r.iter_content(CHUNK):
                        fh.write(chunk)
            have = part.stat().st_size
            if total is None or have >= total:
                break
        except Exception as exc:  # network errors: back off and resume
            log(out, f"{dest.name}: attempt {attempt} error {type(exc).__name__}: {str(exc)[:120]}")
            time.sleep(min(300, 10 * attempt))
    else:
        log(out, f"{dest.name}: GAVE UP after {max_tries} attempts")
        return False
    try:
        with zipfile.ZipFile(part) as zf:
            bad = zf.testzip()
        if bad:
            raise zipfile.BadZipFile(f"corrupt member {bad}")
    except zipfile.BadZipFile as exc:
        log(out, f"{dest.name}: zip test failed ({exc}); deleting partial file to refetch")
        part.unlink(missing_ok=True)
        return fetch(url, dest, out, max_tries)
    part.replace(dest)
    ok.write_text(str(dest.stat().st_size), encoding="ascii")
    log(out, f"{dest.name}: DONE {dest.stat().st_size / 1e6:.1f} MB")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url_list")
    ap.add_argument("out_dir")
    ap.add_argument("--name-prefix", default="part")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    urls = [u.strip() for u in Path(args.url_list).read_text().splitlines() if u.strip()]
    log(out, f"START {len(urls)} files -> {out}")
    results = []
    for i, url in enumerate(urls, 1):
        name = url.rsplit("/", 1)[-1] or f"{args.name_prefix}_{i:02d}.zip"
        results.append(fetch(url, out / name, out))
    log(out, f"FINISHED {sum(results)}/{len(urls)} ok")


if __name__ == "__main__":
    main()
