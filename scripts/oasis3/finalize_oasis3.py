#!/usr/bin/env python3
"""Local completion monitor for the detached OASIS-3 worker.

Never accesses the network or handles credentials. The main downloader writes
STATUS.md with a terminal phase after completing its first QC/label pass; this
monitor then reruns local QC and labeling using the latest source code and
writes an accurate, durable final report. Only one monitor should be running.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import os
import re
import shutil
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import nibabel as nib
import pandas as pd

from make_labels import build_labels
from qc_oasis3 import run_qc


ROOT = Path(__file__).resolve().parents[2]


def keep_awake(on: bool) -> None:
    if os.name == "nt":
        flags = 0x80000000 | (0x00000001 if on else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)


def csv_rows(name: str) -> list[dict]:
    path = ROOT / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def phase_now() -> str:
    path = ROOT / "STATUS.md"
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        found = re.match(r"Phase: \*\*([^*]+)\*\*", line)
        if found:
            return found.group(1)
    return ""


def format_distribution(rows: list[dict]) -> list[str]:
    result = []
    eligible = [r for r in rows if r.get("label") in ("CN", "MCI-like", "AD")]
    for label in ("CN", "MCI-like", "AD"):
        group = [r for r in eligible if r["label"] == label]
        ages = pd.to_numeric(pd.Series([r.get("age", "") for r in group], dtype=str), errors="coerce").dropna()
        sex = Counter((r.get("sex") or "unknown") for r in group)
        age_text = f"mean {ages.mean():.1f}, median {ages.median():.1f} (n={len(ages)})" if len(ages) else "unavailable"
        result.append(f"- {label}: **{len(group)}**; age {age_text}; sex {dict(sex)}")
    return result


def preprocess_summary() -> list[str]:
    outputs = sorted((ROOT / "oasis3_preprocessed").glob("*/*/*_preproc.nii.gz"))
    valid = 0
    for path in outputs:
        try:
            img = nib.load(str(path))
            if img.ndim == 3 and all(abs(float(z) - 1.0) <= .05 for z in img.header.get_zooms()[:3]):
                valid += 1
        except (OSError, ValueError):
            pass
    timings = []
    for row in csv_rows("preprocess_log.csv"):
        try:
            seconds = float(row.get("seconds", 0))
        except ValueError:
            continue
        if row.get("status") == "ok" and seconds > 30:
            timings.append(seconds)
    lines = [f"- Pilot outputs at MNI152 1 mm / 3D: **{valid}/{len(outputs)}** (target: 3)."]
    if timings:
        avg = sum(timings) / len(timings)
        lines.append(f"- Measured full-pipeline CPU time: {avg / 60:.1f} min/scan across {len(timings)} new scan(s).")
        lines.append(f"- Very rough serial estimate for 1,376: **{avg * 1376 / 3600:.1f} hours**; actual throughput depends on hardware, scans and concurrency.")
    else:
        lines.append("- No successful full-pipeline timing yet; consult preprocess_log.csv.")
    return lines


def final_report(initial_phase: str) -> None:
    notes = []
    qc_result: dict = {}
    if initial_phase != "aborted":
        try:
            print("Rechecking every NIfTI + gzip CRC and refreshing scanner metadata...", flush=True)
            qc_result = run_qc(check_gzip=True)
        except Exception as exc:
            notes.append(f"Final QC failed: {type(exc).__name__}: {exc}")
        try:
            print("Rebuilding labels and subject-disjoint splits...", flush=True)
            build_labels(window_days=365)
        except Exception as exc:
            notes.append(f"Final label/split generation failed: {type(exc).__name__}: {exc}")
    else:
        notes.append("Downloader aborted before finalization; resume via the documented launcher.")

    manifest = csv_rows("download_manifest.csv")
    failures = csv_rows("failed_after_retry.csv")
    qc_rows = csv_rows("qc_report.csv")
    labels = csv_rows("labels.csv")
    split = csv_rows("split_subjects.csv")
    selected = csv_rows("selected_sessions.csv")
    raw_paths = list((ROOT / "oasis3_t1w").glob("*/*/*.nii.gz"))
    status_counts = Counter(row.get("status", "") for row in manifest)
    expected = {row["Subject"]: row["MR ID"] for row in selected}
    fallback = [row for row in manifest if row.get("status") in ("downloaded", "skipped")
                and expected.get(row["Subject"]) != row["MR ID"]]
    label_counts = Counter((row.get("label") or "unmatched") for row in labels)
    qc_valid = sum(row.get("ok", "").lower() == "true" for row in qc_rows)
    qc_invalid = len(qc_rows) - qc_valid
    disk = shutil.disk_usage(ROOT)
    mri_size = sum(p.stat().st_size for p in (ROOT / "oasis3_t1w").rglob("*") if p.is_file()) / 2**30
    final_phase = "completed" if initial_phase == "completed" else initial_phase
    if qc_invalid or len(failures) or notes:
        final_phase = "completed with issues" if initial_phase != "aborted" else "aborted"

    lines = [
        "# OASIS-3 Project Status", "",
        f"Updated (local time): {datetime.now().isoformat(timespec='seconds')}",
        f"Phase: **{final_phase}** (post-download local review)", "",
        "## Download and disk", "",
        f"- Subjects processed in manifest: **{len(manifest)}/1376**.",
        f"- Successfully downloaded: **{status_counts.get('downloaded', 0)}**; resumed/skipped: **{status_counts.get('skipped', 0)}**.",
        f"- Fallback to a later T1w session: **{len(fallback)}** participants.",
        f"- Persistent failed participants: **{len(failures)}** (see failed_after_retry.csv).",
        f"- Raw NIfTI files on disk: **{len(raw_paths)}**; raw MRI folder: **{mri_size:.2f} GiB**.",
        f"- D: free **{disk.free / 2**30:.2f} GiB** (60 GiB minimum warning threshold).",
        "- Approximate original compressed scan expectation: 14–21 GB; MNI registration and preprocessing require additional space.",
        "", "## NIfTI quality control", "",
        f"- Valid 3D NIfTI and JSON, full gzip CRC: **{qc_valid}/{len(qc_rows)}**.",
        f"- Failed QC: **{qc_invalid}**; consult qc_report.csv for paths/reasons.",
        f"- Last QC result: {qc_result or 'previous snapshot only / unavailable'}.",
        "- Scanner information: scanner_info.csv (field strength, manufacturer, model when present).",
        "", "## Clinical proxy labels and splits", "",
        f"- Labeled MR sessions: **{sum(label_counts.values()) - label_counts.get('unmatched', 0)}** of {len(labels)}; unmatched/excluded: **{label_counts.get('unmatched', 0)}**.",
        f"- Class counts: **CN {label_counts.get('CN', 0)}**, **MCI-like {label_counts.get('MCI-like', 0)}**, **AD proxy {label_counts.get('AD', 0)}**.",
    ]
    lines.extend(format_distribution(labels))
    lines.extend([
        f"- Subjects assigned exactly one split: **{len(split)}**; train/val/test distribution: {dict(Counter(row.get('split') for row in split))}.",
        "- Nearest CDRTOT assessment within ±365 days. CDR 0=CN, 0.5=MCI-like, ≥1=AD severity **proxy**.",
        "- CDR alone cannot confirm Alzheimer's etiology or distinguish all MCI from very mild dementia; verify clinical diagnoses before claiming AD/MCI accuracy.",
        "", "## Three-scan preprocessing pilot", "",
    ])
    lines.extend(preprocess_summary())
    lines.extend([
        "- N4 + HD-BET brain extraction + ANTs SyN registration + normalized MNI152 1 mm; no face/head reconstructions.",
        "- Full preprocessing has NOT been launched.",
        "", "## Clinical resources, scripts, files", "",
        "- Clinical resource download: 33 project resources from 0AS_data_files; local files in oasis3_clinical/.",
        "- Source: akhileshrvce_11_8_2026_13_30_10.csv (2,844 MR sessions); selected_sessions.csv (1,376 participants).",
        "- Scripts: download_oasis3_t1w.py, run_oasis_pipeline.py, start_background_download.py, download_oasis_clinical.py, probe_xnat.py, qc_oasis3.py, make_labels.py, preprocess_oasis3.py, hd_bet_local.py, download_hd_bet_weights.py, finalize_oasis3.py.",
        "- Logs/reports: pipeline.log, download_log.csv, download_manifest.csv, failed_after_retry.csv, qc_report.csv, scanner_info.csv, labels.csv, labels_summary.md, split_subjects.csv, preprocess_log.csv, finalizer.log.",
        "- Folders: oasis3_t1w/, oasis3_clinical/, oasis3_preprocessed/, models/, references/, adni/ and aibl/ (the latter two are empty placeholders).",
        "", "## Issues and morning steps", "",
        "- Verified real NITRC-IR login and successful first five-session download (5/5 valid). The XNAT project/subject/experiment scan URL works.",
        "- If persistent failures remain, check failed_after_retry.csv and resume: .\\.venv\\Scripts\\python.exe start_background_download.py --skip-clinical.",
        "- Inspect labels_summary.md and review CDR proxy limitations in README.md before training.",
        "- Review qc_report.csv for corrupt scans and rerun local QC if needed.",
        "- No ADNI or AIBL access was attempted. All restricted data stays local.",
    ])
    if disk.free < 60 * 2**30:
        lines.append("- **WARNING: fewer than 60 GiB remain on D:.**")
    if len(manifest) < 1376:
        lines.append("- **WARNING: download manifest is incomplete.** Resume the full pipeline after diagnosing pipeline.log.")
    for item in notes:
        lines.append(f"- {item}")
    for row in failures[:20]:
        lines.append(f"- Unresolved {row.get('Subject', '?')}: {(row.get('error message') or '')[:200]}")
    lines.append("")
    (ROOT / "STATUS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Final STATUS.md written: {final_phase}, {len(manifest)} subjects, {qc_valid} validated NIfTIs", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="Wait for completed/aborted worker status")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    keep_awake(True)
    try:
        if args.watch:
            print("Monitoring STATUS.md for terminal worker state; no credentials required.", flush=True)
            while True:
                phase = phase_now()
                if phase in ("completed", "completed with failures", "aborted"):
                    print(f"Detected worker phase {phase}; finalizing locally.", flush=True)
                    break
                time.sleep(max(args.poll_seconds, 10))
        else:
            phase = phase_now()
            if phase not in ("completed", "completed with failures", "aborted"):
                parser.error("Downloader is still running; use --watch or finalize after completion")
        final_report(phase)
    finally:
        keep_awake(False)


if __name__ == "__main__":
    main()
