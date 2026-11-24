#!/usr/bin/env python3
"""Download authorized OASIS-3 clinical/cohort resources from 0AS_data_files.

The data never leave this project folder. No credentials are persisted.
"""

from __future__ import annotations

import argparse
import csv
import getpass
import os
import re
import shutil
import time
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from tqdm import tqdm

from download_oasis3_t1w import (
    BASE_URL, DownloadError, NITRCClient, retry_delay, response_preview,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "oasis3_clinical"
SUBJECT = "0AS_data_files"


def safe_segment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))[:100]


def listing(client: NITRCClient, path: str) -> list[dict]:
    with client.get(path, params={"format": "json"}) as response:
        try:
            return response.json()["ResultSet"]["Result"]
        except (KeyError, TypeError, ValueError) as exc:
            raise DownloadError(f"Unexpected JSON listing: {response_preview(response)}") from exc


def safe_extract(archive: Path, target: Path) -> int:
    """Extract only safe relative paths, preserving nested clinical filenames."""
    count = 0
    with zipfile.ZipFile(archive) as zip_file:
        for item in zip_file.infolist():
            pure = PurePosixPath(item.filename.replace("\\", "/"))
            if pure.is_absolute() or ".." in pure.parts or not pure.parts:
                raise DownloadError(f"Unsafe ZIP path: {item.filename!r}")
            dest = target.joinpath(*pure.parts)
            if item.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zip_file.open(item) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)
            count += 1
    return count


def download_resource(client: NITRCClient, experiment: str, scan_id: str,
                      folder: Path) -> int:
    path = (
        f"/data/projects/OASIS3/subjects/{SUBJECT}/experiments/"
        f"{quote(experiment, safe='')}/scans/{quote(scan_id, safe='')}/files"
    )
    zip_path = folder.with_suffix(".zip")
    partial = folder.with_suffix(".zip.part")
    folder.parent.mkdir(parents=True, exist_ok=True)
    try:
        for attempt in range(1, 4):
            try:
                with client.get(path, params={"format": "zip"}, stream=True) as response:
                    with partial.open("wb") as output:
                        for chunk in response.iter_content(1024 * 1024):
                            if chunk:
                                output.write(chunk)
                if not zipfile.is_zipfile(partial):
                    raise DownloadError(f"{BASE_URL}{path} returned non-ZIP bytes")
                partial.replace(zip_path)
                break
            except (OSError, DownloadError) as exc:
                partial.unlink(missing_ok=True)
                if attempt == 3:
                    raise DownloadError(f"Resource transfer failed: {exc}") from exc
                retry_delay(attempt)
        return safe_extract(zip_path, folder)
    finally:
        partial.unlink(missing_ok=True)
        zip_path.unlink(missing_ok=True)


def fetch_clinical(client: NITRCClient, *, limit: int | None = None) -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    experiments = listing(
        client, f"/data/projects/OASIS3/subjects/{SUBJECT}/experiments"
    )
    resources: list[tuple[str, str]] = []
    for exp in experiments:
        label = str(exp.get("label") or exp.get("ID") or "")
        if not label:
            continue
        scans = listing(
            client,
            f"/data/projects/OASIS3/subjects/{SUBJECT}/experiments/"
            f"{quote(label, safe='')}/scans",
        )
        for scan in scans:
            scan_id = str(scan.get("ID") or "")
            if scan_id:
                resources.append((label, scan_id))
    if limit:
        resources = resources[:limit]

    manifest = OUTPUT / "clinical_download_manifest.csv"
    rows = []
    counters = {"downloaded": 0, "skipped": 0, "failed": 0}
    for label, scan_id in tqdm(resources, desc="Clinical", unit="resource"):
        folder = OUTPUT / safe_segment(label) / safe_segment(scan_id)
        existing = [p for p in folder.rglob("*") if p.is_file()] if folder.exists() else []
        if existing:
            status, count, error = "skipped", len(existing), ""
        else:
            try:
                count = download_resource(client, label, scan_id, folder)
                status, error = "downloaded", ""
                if count == 0:
                    raise DownloadError("ZIP contained no files")
            except (DownloadError, OSError, zipfile.BadZipFile) as exc:
                status, count, error = "failed", 0, str(exc)
                print(f"Clinical {label}/{scan_id}: {error}", flush=True)
        counters[status] += 1
        rows.append([label, scan_id, status, count, str(folder), error])
        # Rebuild status after each entry to make interruption recoverable.
        with manifest.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["experiment", "scan ID", "status", "file_count", "folder", "error"])
            writer.writerows(rows)
    counters["resources"] = len(resources)
    counters["csv_files"] = len(list(OUTPUT.rglob("*.csv"))) - 1
    return counters


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, help="Test the first N clinical resources")
    args = p.parse_args()
    client = NITRCClient(input("NITRC username: ").strip(), getpass.getpass("NITRC password: "))
    try:
        client.authenticate()
        print(fetch_clinical(client, limit=args.limit))
    finally:
        client.close()


if __name__ == "__main__":
    main()
