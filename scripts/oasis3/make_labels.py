#!/usr/bin/env python3
"""Match OASIS-3 T1w sessions to nearest global CDR assessment (+/-365 d).

CDRTOT in OASIS3_UDSb4_cdr.csv is the global CDR score (0, 0.5, 1, 2, 3).
CDRSUM is the distinct Sum of Boxes and must NOT be used for this mapping.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CLINICAL = ROOT / "oasis3_clinical"
IMAGES = ROOT / "oasis3_t1w"
LABEL_COLUMNS = [
    "Subject", "MR ID", "age", "sex", "scanner field strength", "CDR",
    "MMSE", "days_between_scan_and_assessment", "label", "assessment_day",
    "match_status", "nifti_path",
]


def find_exact(filename: str) -> Path | None:
    return next(CLINICAL.rglob(filename), None) if CLINICAL.is_dir() else None


def scan_manifest() -> pd.DataFrame:
    source = ROOT / "akhileshrvce_11_8_2026_13_30_10.csv"
    table = pd.read_csv(source, dtype=str, keep_default_na=False)
    table = table[["MR ID", "Subject", "Age", "Scanner"]].drop_duplicates("MR ID")
    files = sorted(IMAGES.glob("*/*/*.nii.gz")) if IMAGES.exists() else []
    rows = []
    for path in files:
        rows.append({"Subject": path.parent.parent.name, "MR ID": path.parent.name,
                     "nifti_path": str(path)})
    if not rows:
        return pd.DataFrame(columns=["Subject", "MR ID", "nifti_path", "Age", "Scanner"])
    # One row per actual downloaded MRI session. Multiple T1w runs within a
    # session are represented by the first file here; QC covers every file.
    found = pd.DataFrame(rows).drop_duplicates("MR ID")
    return found.merge(table, on=["MR ID", "Subject"], how="left")


def global_cdr(value) -> tuple[str, str]:
    value = pd.to_numeric(value, errors="coerce")
    if pd.isna(value) or value < 0:
        return "", "missing_cdr"
    if value == 0:
        return "CN", "matched"
    if value == 0.5:
        return "MCI-like", "matched"
    if value >= 1:
        return "AD", "matched"
    return "", "unexpected_cdr"


def build_labels(window_days: int = 365) -> pd.DataFrame:
    scans = scan_manifest()
    cdr_path = find_exact("OASIS3_UDSb4_cdr.csv")
    demo_path = find_exact("OASIS3_demographics.csv")
    if cdr_path is None or demo_path is None:
        print("Clinical CDR/demographics are missing; labels will be empty.")
        result = pd.DataFrame(columns=LABEL_COLUMNS)
        result.to_csv(ROOT / "labels.csv", index=False)
        (ROOT / "split_subjects.csv").write_text("Subject,split,label\n", encoding="utf-8")
        write_summary(result, "CDR and/or demographics CSV unavailable.")
        return result

    cdr = pd.read_csv(cdr_path, dtype={"OASISID": str}, low_memory=False)
    demo = pd.read_csv(demo_path, dtype={"OASISID": str}, low_memory=False)
    need = {"OASISID", "days_to_visit", "CDRTOT", "MMSE"}
    if not need.issubset(cdr.columns):
        raise ValueError(f"CDR table lacks columns: {sorted(need - set(cdr.columns))}")
    cdr["days_to_visit"] = pd.to_numeric(cdr["days_to_visit"], errors="coerce")
    cdr = cdr[cdr["days_to_visit"].notna()].copy()
    by_subject = {subject: group for subject, group in cdr.groupby("OASISID", sort=False)}
    demo = demo.drop_duplicates("OASISID").set_index("OASISID")

    rows = []
    for scan in scans.itertuples(index=False):
        subject, mr_id, nifti, age, scanner = scan

        # Some source CSV Scanner entries are empty. The acquired scan's BIDS
        # sidecar is the stronger source for magnetic field strength.
        sidecar = Path(nifti).with_name(Path(nifti).name[:-7] + ".json")
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            field_strength = metadata.get("MagneticFieldStrength", "")
        except (OSError, ValueError, TypeError):
            field_strength = ""
        if field_strength != "" and field_strength is not None:
            scanner = f"{field_strength}T"
        day_match = re.search(r"_MR_d(\d+)$", mr_id)
        day = int(day_match.group(1)) if day_match else None
        sex = ""
        if subject in demo.index:
            code = pd.to_numeric(demo.at[subject, "GENDER"], errors="coerce")
            sex = {1: "M", 2: "F"}.get(code, "")
        record = {
            "Subject": subject, "MR ID": mr_id, "age": age,
            "sex": sex, "scanner field strength": scanner,
            "CDR": "", "MMSE": "", "days_between_scan_and_assessment": "",
            "label": "", "assessment_day": "", "match_status": "no_clinical_within_365d",
            "nifti_path": nifti,
        }
        if day is None:
            record["match_status"] = "invalid_mr_id"
        elif subject in by_subject:
            visits = by_subject[subject].copy()
            visits["delta"] = (visits["days_to_visit"] - day).abs()
            visits = visits[visits["delta"] <= window_days]
            if len(visits):
                # Closest clinical assessment; tie goes to earlier visit.
                visit = visits.sort_values(["delta", "days_to_visit"], kind="stable").iloc[0]
                label, status = global_cdr(visit["CDRTOT"])
                record.update({
                    "CDR": visit["CDRTOT"] if pd.notna(visit["CDRTOT"]) else "",
                    "MMSE": visit["MMSE"] if pd.notna(visit["MMSE"]) else "",
                    "days_between_scan_and_assessment": int(visit["delta"]),
                    "label": label, "assessment_day": int(visit["days_to_visit"]),
                    "match_status": status,
                })
        rows.append(record)
    result = pd.DataFrame(rows, columns=LABEL_COLUMNS)
    result.to_csv(ROOT / "labels.csv", index=False)
    build_splits(result)
    write_summary(result)
    return result


def build_splits(labels: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Stratify by label at the subject level: 70/15/15 approximate."""
    eligible = labels[labels["label"].isin(["CN", "MCI-like", "AD"])].copy()
    unique = eligible.drop_duplicates("Subject")[['Subject', 'label']]
    rng = np.random.default_rng(seed)
    assignments = []
    for label, group in unique.groupby("label", sort=True):
        subjects = group["Subject"].to_numpy(copy=True)
        rng.shuffle(subjects)
        n = len(subjects)
        n_train = int(round(n * 0.7))
        n_val = int(round(n * 0.15))
        if n:
            n_train = max(1, min(n, n_train))
        n_val = max(0, min(n - n_train, n_val))
        for idx, subject in enumerate(subjects):
            split = "train" if idx < n_train else "val" if idx < n_train+n_val else "test"
            assignments.append({"Subject": subject, "split": split, "label": label})
    result = pd.DataFrame(assignments, columns=["Subject", "split", "label"])
    if result["Subject"].duplicated().any():
        raise AssertionError("Subject leaked into more than one split")
    result.sort_values("Subject").to_csv(ROOT / "split_subjects.csv", index=False)
    return result


def write_summary(labels: pd.DataFrame, note: str = "") -> None:
    matched = labels[labels["label"].isin(["CN", "MCI-like", "AD"])]
    lines = ["# OASIS-3 Clinical Label Summary", "",
             "Source: OASIS3_UDSb4_cdr.csv (`CDRTOT`, global CDR); OASIS3_demographics.csv (`GENDER`).",
             "Matching: nearest visit within +/-365 days; ties prefer earlier clinical day.",
             "Proxy labels: CDR=0 -> CN, CDR=0.5 -> MCI-like, CDR>=1 -> AD.",
             "**These are research proxy labels, not clinician-confirmed AD or MCI diagnoses.**", "",
             f"MRI sessions on disk: {len(labels)}", f"Matched/labelled: {len(matched)}",
             f"Unmatched/excluded: {len(labels) - len(matched)}", "", "## Class counts", ""]
    for label in ("CN", "MCI-like", "AD"):
        lines.append(f"- {label}: {sum(matched['label'] == label)}")
    lines.extend(["", "## Age and sex by class", ""])
    for label, group in matched.groupby("label", sort=True):
        ages = pd.to_numeric(group["age"], errors="coerce").dropna()
        sex_counts = group["sex"].replace("", "unknown").value_counts().to_dict()
        age_str = f"mean {ages.mean():.1f}, median {ages.median():.1f}, n={len(ages)}" if len(ages) else "unavailable"
        lines.append(f"- {label}: age {age_str}; sex {sex_counts}")
    if note:
        lines.extend(["", f"Note: {note}"])
    lines.append("")
    (ROOT / "labels_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window", type=int, default=365)
    args = ap.parse_args()
    result = build_labels(args.window)
    print(f"Labelled {result['label'].ne('').sum()}/{len(result)} MRI sessions.")
    print("Counts:", result["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
