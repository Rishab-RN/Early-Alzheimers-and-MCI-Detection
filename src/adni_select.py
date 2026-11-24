"""Select one baseline T1w image per ADNI subject and split into download batches.

Inputs (local, never committed):
  adni/tables/Clinical_T1w_Imaging_Cohort_Manifest_*.csv  (IDA ARC image manifest)
  adni/tables/DXSUM_*.csv                                  (diagnostic summary)
Outputs (adni/selection/, git-ignored):
  selected_images.csv   one row per subject: image_id, subject, date, description, dx, phase, batch
  batch_A_ids.txt / batch_B_ids.txt  comma-separated IDA image IDs
  selection_summary.md  aggregate counts only (safe to share)

Rules (PROTOCOL.md):
  * earliest T1 imaging date per subject (screening/baseline in practice)
  * within that date prefer: not a repeat, not "_ND" (non-distortion-corrected), then lowest image_id
  * diagnosis = DXSUM DIAGNOSIS (1 CN, 2 MCI, 3 AD) nearest to the scan date within +/-180 days
  * no vendor / scanner filtering
  * batch A = ADNI GO/2/3 subjects, class-stratified (<=200 per class); batch B = everyone else
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "adni" / "tables"
OUT = ROOT / "adni" / "selection"
DX_MAP = {1: "CN", 2: "MCI", 3: "AD"}


def latest(pattern: str) -> Path:
    files = sorted(glob.glob(str(TABLES / pattern)))
    if not files:
        raise FileNotFoundError(pattern)
    return Path(files[-1])


def series_rank(desc: str) -> int:
    d = str(desc).lower()
    rank = 0
    if "repeat" in d:
        rank += 2
    if "_nd" in d or d.endswith(" nd"):
        rank += 1
    return rank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class-a", type=int, default=200)
    ap.add_argument("--dx-window-days", type=int, default=180)
    ap.add_argument("--seed", type=int, default=20261118)
    args = ap.parse_args()

    man = pd.read_csv(latest("Clinical_T1w_Imaging_Cohort_Manifest_*.csv"))
    man["image_date"] = pd.to_datetime(man["image_date"])
    man["rank"] = man["image_description"].map(series_rank)
    first_date = man.groupby("subject_id")["image_date"].transform("min")
    base = man[man["image_date"] == first_date]
    base = base.sort_values(["subject_id", "rank", "image_id"]).groupby("subject_id").head(1)

    dx = pd.read_csv(latest("DXSUM_*.csv"), low_memory=False)
    dx = dx[dx["DIAGNOSIS"].isin(DX_MAP)].copy()
    dx["EXAMDATE"] = pd.to_datetime(dx["EXAMDATE"], errors="coerce")
    dx = dx.dropna(subset=["EXAMDATE"])
    phase_first = (dx.sort_values("EXAMDATE").groupby("PTID")["PHASE"].first())

    merged = base.merge(dx[["PTID", "EXAMDATE", "DIAGNOSIS"]], left_on="subject_id",
                        right_on="PTID", how="left")
    merged["gap"] = (merged["EXAMDATE"] - merged["image_date"]).dt.days.abs()
    merged = merged[merged["gap"] <= args.dx_window_days]
    merged = merged.sort_values(["subject_id", "gap"]).groupby("subject_id").head(1)
    merged["dx"] = merged["DIAGNOSIS"].map(DX_MAP)
    merged["phase"] = merged["subject_id"].map(phase_first)

    rng = np.random.default_rng(args.seed)
    merged["batch"] = "B"
    pool = merged[merged["phase"].isin(["ADNIGO", "ADNI2", "ADNI3"])]
    for cls, grp in pool.groupby("dx"):
        take = grp.index.to_numpy()
        if len(take) > args.per_class_a:
            take = rng.choice(take, args.per_class_a, replace=False)
        merged.loc[take, "batch"] = "A"

    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["image_id", "subject_id", "image_date", "image_visit", "image_description",
            "dx", "gap", "phase", "batch"]
    merged[cols].sort_values(["batch", "image_id"]).to_csv(OUT / "selected_images.csv", index=False)
    for b in ("A", "B"):
        ids = merged.loc[merged["batch"] == b, "image_id"].astype(int).sort_values()
        (OUT / f"batch_{b}_ids.txt").write_text(",".join(map(str, ids)), encoding="ascii")

    n_all = man["subject_id"].nunique()
    tab = pd.crosstab(merged["phase"], [merged["batch"], merged["dx"]], margins=True)
    lines = ["# ADNI baseline T1 selection (aggregate counts only)", "",
             f"- Subjects in IDA T1w manifest: {n_all}",
             f"- Subjects with a diagnosis within +/-{args.dx_window_days} d of first scan: {len(merged)}",
             f"- Excluded (no matching diagnosis): {n_all - len(merged)}",
             f"- Batch A: {(merged.batch == 'A').sum()}  Batch B: {(merged.batch == 'B').sum()}", "",
             "## Counts by phase x batch x diagnosis", "", tab.to_markdown(), "",
             "## Series descriptions chosen (top 15)", "",
             merged["image_description"].value_counts().head(15).to_markdown()]
    (OUT / "selection_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:8]))


if __name__ == "__main__":
    main()
