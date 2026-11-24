#!/usr/bin/env python3
"""Resumable, per-subject OASIS-3 downloader, QC and clinical label pipeline.

This is the unattended full-run worker. Start via start_background_download.py,
or run interactively. No credentials are written to disk or output logs.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import getpass
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from download_oasis3_t1w import (
    DownloadError, NITRCClient, append_log, choose_t1w_scan, extract_zip_flat,
    select_sessions,
)
from download_oasis_clinical import fetch_clinical
from make_labels import build_labels
from qc_oasis3 import is_valid_session, run_qc


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "oasis3_t1w"
CSV = ROOT / "akhileshrvce_11_8_2026_13_30_10.csv"
LOG = ROOT / "download_log.csv"
MANIFEST = ROOT / "download_manifest.csv"
FAILURES = ROOT / "failed_after_retry.csv"
STATUS = ROOT / "STATUS.md"
MANIFEST_FIELDS = ["Subject", "MR ID", "scan ID", "status", "file path", "error message"]


def keep_windows_awake(enable: bool) -> None:
    if os.name == "nt":
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if enable else 0)
        result = ctypes.windll.kernel32.SetThreadExecutionState(flags)
        if not result:
            print("WARNING: SetThreadExecutionState failed; prevent PC sleep manually.", flush=True)


def save_table(path: Path, rows: list[dict], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def size_gb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / (1024 ** 3) if path.exists() else 0.0


def status_markdown(phase: str, *, total: int, done: int,
                    rows: list[dict], final_errors: list[dict], notes: list[str]) -> None:
    free = shutil.disk_usage(ROOT).free / (1024 ** 3)
    counts = Counter(row["status"] for row in rows)
    qc_path = ROOT / "qc_report.csv"
    valid = invalid = 0
    if qc_path.exists():
        with qc_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                valid += row["ok"].lower() == "true"
                invalid += row["ok"].lower() != "true"
    label_counts = Counter()
    labels_path = ROOT / "labels.csv"
    if labels_path.exists():
        with labels_path.open(encoding="utf-8", newline="") as f:
            label_counts.update(row["label"] or "unmatched" for row in csv.DictReader(f))
    output_nifti = len(list(OUTPUT.glob("*/*/*.nii.gz"))) if OUTPUT.exists() else 0
    lines = [
        "# OASIS-3 Project Status", "",
        f"Updated (local time): {datetime.now().isoformat(timespec='seconds')}",
        f"Worker PID: {os.getpid()}", f"Phase: **{phase}**", "",
        "## Disk and downloads", "",
        f"- D: available: **{free:.2f} GiB** (warn if <60 GiB)",
        "- Estimate for 1,376 compressed NIfTIs: approximately 14-21 GB, "
        "plus atlas, temporary ZIPs and substantial preprocessing space.",
        f"- Subjects processed: **{done}/{total}**",
        f"- Completed subjects in current manifest: **{counts.get('downloaded',0)+counts.get('skipped',0)}**",
        f"- Newly downloaded subjects in current manifest: {counts.get('downloaded',0)}",
        f"- Resumed/skipped subjects: {counts.get('skipped',0)}",
        f"- Failed subjects after retry: {len(final_errors)}",
        f"- NIfTI files currently on disk: **{output_nifti}**",
        f"- MRI folder size: {size_gb(OUTPUT):.3f} GiB",
        "", "## NIfTI QC and clinical labels", "",
        f"- Valid 3D NIfTI with JSON and checksum: {valid}",
        f"- Invalid NIfTI/sidecars: {invalid}",
        f"- Proxy class counts (on-disk MRIs): {dict(label_counts)}",
        "- Mapping: CDRTOT 0=CN, 0.5=MCI-like, >=1=AD; clinical assessment within 365 days.",
        "- This is **not a clinician-confirmed AD diagnosis**; CDR alone cannot establish AD etiology or MCI.",
        "", "## Files and output", "",
        "- `download_oasis3_t1w.py`, `run_oasis_pipeline.py`, `start_background_download.py`",
        "- `download_oasis_clinical.py`, `make_labels.py`, `qc_oasis3.py`, `preprocess_oasis3.py`",
        "- `selected_sessions.csv`, `download_manifest.csv`, `download_log.csv`, `failed_after_retry.csv`",
        "- `qc_report.csv`, `scanner_info.csv`, `labels.csv`, `labels_summary.md`, `split_subjects.csv`",
        "- `oasis3_t1w/`, `oasis3_clinical/`, `oasis3_preprocessed/`, `adni/`, `aibl/`",
        "- `pipeline.log`, `download_worker.pid`, `requirements.txt`, `requirements-ml.txt`, `README.md`",
        "", "## Issues and manual follow-up", "",
        "- First unauthorised sandbox connectivity check failed DNS; approved network execution fixed it.",
        "- Original scan URL returned nested JSON; changed to project/subject/experiment path.",
        "- The five-scan live test succeeded: 5 NIfTI + 5 JSON, all opened with nibabel, 3D.",
        "- Clinical 0AS_data_files: 33 resources downloaded, 0 failed in the live test.",
        "- No head or face reconstructions are generated; all restricted files stay local.",
        "- Full preprocessing was intentionally not started. It needs a local MNI152 1mm template, "
        "ANTsPy and a supported skull-stripper; test just three MRIs once installed.",
        "- CPU preprocessing estimate: approximately 15-60 min/scan for N4 + brain extraction + SyN, "
        "or approximately 14-55 days serial for 1,376 subjects (hardware and settings dependent).",
    ]
    for item in notes:
        lines.append(f"- {item}")
    if final_errors:
        lines.extend(["", "## Persistent download failures (first 30)", ""])
        for item in final_errors[:30]:
            lines.append(f"- {item.get('Subject','?')}: {str(item.get('error message',''))[:220]}")
    lines.append("")
    STATUS.write_text("\n".join(lines), encoding="utf-8")


def candidates_by_subject() -> dict[str, list[str]]:
    sessions = select_sessions(CSV, all_sessions=True)
    return {
        subject: list(group["MR ID"])
        for subject, group in sessions.groupby("Subject", sort=False)
    }


def with_access_cache_retry(callback, attempts: int = 6):
    """Handle newly approved project access cache denials, max six tries."""
    for attempt in range(1, attempts + 1):
        try:
            return callback()
        except DownloadError as exc:
            error = str(exc)
            if "HTTP 401" not in error and "HTTP 403" not in error:
                raise
            if attempt == attempts:
                raise
            print(f"HTTP access denied; possible NITRC access cache. "
                  f"Waiting 10 min before retry {attempt+1}/{attempts}.", flush=True)
            time.sleep(600)


def attempt_subject(client: NITRCClient, subject: str, candidates: list[str]) -> dict:
    # Resume previously downloaded *valid* fallback sessions before issuing
    # API calls for an earlier candidate that failed on a previous run.
    for mr_id in candidates:
        folder = OUTPUT / subject / mr_id
        if is_valid_session(folder):
            path = next(folder.glob("*.nii.gz"))
            append_log(LOG, mr_id, "", "skipped", str(path), "Valid 3D NIfTI + sidecar")
            return {"Subject": subject, "MR ID": mr_id, "scan ID": "",
                    "status": "skipped", "file path": str(path), "error message": ""}

    errors = []
    for mr_id in candidates:
        folder = OUTPUT / subject / mr_id
        scan_id, zip_path = "", None
        try:
            scans = with_access_cache_retry(lambda: client.list_scans(mr_id))
            scan_id = choose_t1w_scan(scans)
            zip_path = client.download_zip(mr_id, scan_id, folder)
            paths = extract_zip_flat(zip_path, folder)
            if not is_valid_session(folder):
                raise DownloadError("Downloaded NIfTI is not 3D, lacks JSON, or cannot open")
            path = paths[0]
            append_log(LOG, mr_id, scan_id, "downloaded", str(path))
            return {"Subject": subject, "MR ID": mr_id, "scan ID": scan_id,
                    "status": "downloaded", "file path": str(path), "error message": ""}
        except (DownloadError, OSError, ValueError) as exc:
            message = str(exc)
            errors.append(f"{mr_id}: {message[:250]}")
            append_log(LOG, mr_id, scan_id, "failed", str(folder), message)
            print(f"{subject}: {mr_id} unavailable ({message[:180]}); trying next session.", flush=True)
        finally:
            if zip_path is not None:
                zip_path.unlink(missing_ok=True)
    return {"Subject": subject, "MR ID": candidates[0] if candidates else "",
            "scan ID": "", "status": "failed", "file path": "",
            "error message": " | ".join(errors)[-2000:]}


def run(client: NITRCClient, limit: int | None = None, skip_clinical: bool = False) -> int:
    notes = []
    selected = select_sessions(CSV, all_sessions=False)
    selected.to_csv(ROOT / "selected_sessions.csv", index=False)
    subjects = list(selected["Subject"])
    if limit:
        subjects = subjects[:limit]
    candidate_groups = candidates_by_subject()
    rows, failures = [], []

    status_markdown("starting", total=len(subjects), done=0,
                    rows=rows, final_errors=[], notes=notes)
    if not skip_clinical:
        print("Downloading clinical and cohort resources first...", flush=True)
        try:
            clinical = fetch_clinical(client)
            notes.append(f"Clinical resource summary: {clinical}")
        except Exception as exc:
            notes.append(f"Clinical fetch failed: {type(exc).__name__}: {exc}")
            print(notes[-1], flush=True)

    for i, subject in enumerate(subjects, 1):
        result = attempt_subject(client, subject, candidate_groups.get(subject, []))
        rows.append(result)
        if result["status"] == "failed":
            failures.append(subject)
        print(f"[{i}/{len(subjects)}] {subject} {result['status']} {result['MR ID']}", flush=True)
        save_table(MANIFEST, rows, MANIFEST_FIELDS)
        if i % 25 == 0 or i == len(subjects):
            status_markdown("downloading", total=len(subjects), done=i,
                            rows=rows, final_errors=[], notes=notes)

    if failures:
        print(f"Retrying {len(failures)} failed subjects once more...", flush=True)
        lookup = {r["Subject"]: i for i, r in enumerate(rows)}
        for subject in failures:
            result = attempt_subject(client, subject, candidate_groups.get(subject, []))
            rows[lookup[subject]] = result
            save_table(MANIFEST, rows, MANIFEST_FIELDS)

    final_errors = [item for item in rows if item["status"] == "failed"]
    save_table(FAILURES, final_errors, MANIFEST_FIELDS)
    print("Running final 3D NIfTI integrity QC and scanner metadata...", flush=True)
    qc_result = run_qc(check_gzip=True)
    print(f"QC result: {qc_result}", flush=True)
    print("Building clinical labels and subject-disjoint splits...", flush=True)
    labels = build_labels(window_days=365)
    print(f"Labels: {labels['label'].value_counts().to_dict()}", flush=True)
    status_markdown("completed" if not final_errors else "completed with failures",
                    total=len(subjects), done=len(subjects), rows=rows,
                    final_errors=final_errors, notes=notes)
    return 1 if final_errors or qc_result["invalid"] else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Process first N subjects only")
    parser.add_argument("--skip-clinical", action="store_true")
    parser.add_argument("--credentials-stdin", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.credentials_stdin:
        username = sys.stdin.readline().rstrip("\r\n")
        password = sys.stdin.readline().rstrip("\r\n")
    else:
        username = input("NITRC username: ").strip()
        password = getpass.getpass("NITRC password: ")
    if not username or not password:
        print("Missing NITRC credentials", file=sys.stderr)
        return 1

    keep_windows_awake(True)
    client = NITRCClient(username, password)
    try:
        with_access_cache_retry(client.authenticate)
        return run(client, limit=args.limit, skip_clinical=args.skip_clinical)
    except Exception as exc:
        print(f"Pipeline aborted: {type(exc).__name__}: {exc}", flush=True)
        # Keep a persistent status even after unexpected exceptions.
        status_markdown("aborted", total=0, done=0, rows=[], final_errors=[],
                        notes=[f"Unexpected failure: {type(exc).__name__}: {exc}"])
        return 1
    finally:
        client.close()
        keep_windows_awake(False)


if __name__ == "__main__":
    raise SystemExit(main())
