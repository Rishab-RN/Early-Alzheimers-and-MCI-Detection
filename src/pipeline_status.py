"""Write results/pipeline_status.json: aggregate progress counts only (no subject IDs)."""
from __future__ import annotations

import csv
import glob
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def count_rows(pattern: str, status: str | None = None) -> int:
    n = 0
    for f in glob.glob(str(ROOT / pattern)):
        with open(f, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if status is None or r.get("status") == status:
                    n += 1
    return n


def main() -> None:
    log = (ROOT / "pipeline.log").read_text(encoding="utf-8", errors="ignore") if (ROOT / "pipeline.log").exists() else ""
    m = re.findall(r"\[(\d+)/(\d+)\]", log)
    oasis_dl = (int(m[-1][0]), int(m[-1][1])) if m else (0, 1376)
    sel = ROOT / "adni" / "selection" / "selected_images.csv"
    n_sel = sum(1 for _ in open(sel)) - 1 if sel.exists() else 0
    zips = {b: (len(list((ROOT / "adni/raw_zips" / f"batch_{b}").glob("*.zip.ok"))), 10) for b in "AB"}
    man = ROOT / "data/manifests/adni.csv"
    status = {
        "updated": time.strftime("%Y-%m-%d %H:%M"),
        "oasis3": {"downloaded": oasis_dl[0], "target": oasis_dl[1],
                   "preprocessed_ok": count_rows("data/preprocessed/oasis3/qc_*.csv", "ok"),
                   "preprocess_failed": count_rows("data/preprocessed/oasis3/qc_*.csv", "failed")},
        "adni": {"selected": n_sel, "zip_parts_done": {f"batch_{k}": f"{v[0]}/{v[1]}" for k, v in zips.items()},
                 "converted": (sum(1 for _ in open(man)) - 1) if man.exists() else 0,
                 "preprocessed_ok": count_rows("data/preprocessed/adni/qc_*.csv", "ok"),
                 "preprocess_failed": count_rows("data/preprocessed/adni/qc_*.csv", "failed")},
        "aibl": {"access": "application pending"},
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "pipeline_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
