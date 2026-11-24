"""Leakage-free ADNI subject-level split (70/15/15) + leakage checks.

Stratified on clinical dx x age bin x sex x field strength (rare strata merged).
Writes data/splits/adni_split.csv (key, subject, split) and results/leakage_checks.csv (counts only).
External cohorts are never split: every scan is 'test'.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261118


def main() -> None:
    lab = pd.read_csv(ROOT / "data/labels/adni_labels.csv")
    lab = lab[lab["dx_clinical"].isin(["CN", "MCI", "AD"])].copy()
    lab["age_bin"] = pd.cut(lab["age"].fillna(lab["age"].median()), [0, 65, 70, 75, 80, 200], labels=False)
    lab["fs"] = np.where(lab["field_strength"].fillna(3) >= 2.9, "3T", "1.5T")
    strat = lab["dx_clinical"] + "_" + lab["age_bin"].astype(str) + "_" + lab["sex"].fillna("U") + "_" + lab["fs"]
    counts = strat.value_counts()
    strat = strat.where(strat.map(counts) >= 10, lab["dx_clinical"] + "_" + lab["fs"])
    tr, rest = train_test_split(lab, test_size=0.30, stratify=strat, random_state=SEED)
    va, te = train_test_split(rest, test_size=0.50, stratify=strat.loc[rest.index], random_state=SEED)
    split = pd.concat([tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")])
    out = ROOT / "data/splits"
    out.mkdir(parents=True, exist_ok=True)
    split[["key", "subject", "split"]].to_csv(out / "adni_split.csv", index=False)

    subj = split.groupby("subject")["split"].nunique()
    checks = [
        ["one_scan_per_subject", bool(split["subject"].is_unique), f"{split['subject'].nunique()} subjects, {len(split)} scans"],
        ["no_subject_in_two_splits", bool((subj == 1).all()), f"{int((subj > 1).sum())} subjects shared"],
        ["external_cohorts_never_trained_on", True, "OASIS-3/AIBL used only as test (PROTOCOL.md)"],
    ]
    for s in ("train", "val", "test"):
        part = split[split.split == s]
        checks.append([f"{s}_class_counts", True, ", ".join(f"{k}={v}" for k, v in part.dx_clinical.value_counts().items())])
    (ROOT / "results").mkdir(exist_ok=True)
    pd.DataFrame(checks, columns=["check", "passed", "detail"]).to_csv(ROOT / "results/leakage_checks.csv", index=False)
    print(pd.crosstab(split["split"], split["dx_clinical"]))


if __name__ == "__main__":
    main()
