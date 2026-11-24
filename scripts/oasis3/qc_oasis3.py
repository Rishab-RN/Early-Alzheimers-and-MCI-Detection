#!/usr/bin/env python3
"""Run 3D NIfTI and gzip integrity QC; collect scanner metadata from JSON."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import nibabel as nib


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "oasis3_t1w"
QC_HEADER = ["Subject", "MR ID", "file_path", "shape", "voxel_size_mm",
             "file_size_bytes", "json_sidecar", "ok", "error"]
SCANNER_HEADER = ["Subject", "MR ID", "field_strength", "manufacturer", "model", "json_sidecar"]


def inspect_file(path: Path, *, check_gzip: bool = True) -> dict:
    subject, mr_id = path.parent.parent.name, path.parent.name
    sidecar = path.with_name(path.name[:-7] + ".json")
    row = {"Subject": subject, "MR ID": mr_id, "file_path": str(path),
           "shape": "", "voxel_size_mm": "", "file_size_bytes": path.stat().st_size,
           "json_sidecar": str(sidecar) if sidecar.exists() else "", "ok": False, "error": ""}
    try:
        img = nib.load(str(path))
        if img.ndim != 3:
            raise ValueError(f"Expected 3D scan, received {img.ndim}D shape {img.shape}")
        if min(img.shape) <= 0:
            raise ValueError(f"Invalid shape {img.shape}")
        row["shape"] = "x".join(map(str, img.shape))
        row["voxel_size_mm"] = "x".join(f"{float(z):.5f}" for z in img.header.get_zooms()[:3])
        if not sidecar.exists():
            raise ValueError("JSON sidecar missing")
        with sidecar.open(encoding="utf-8") as file:
            json.load(file)
        if check_gzip:
            with gzip.open(path, "rb") as file:
                while file.read(1024 * 1024):
                    pass
        row["ok"] = True
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def is_valid_session(folder: Path) -> bool:
    files = sorted(folder.glob("*.nii.gz"))
    # nibabel's header read is lazy. Require a complete gzip/CRC read before
    # accepting a downloaded session as resumable, in case power or network
    # failed mid-transfer on a prior run.
    return bool(files) and all(inspect_file(f, check_gzip=True)["ok"] for f in files)


def run_qc(*, check_gzip: bool = True) -> dict:
    rows, scanner_rows = [], []
    paths = sorted(OUTPUT.glob("*/*/*.nii.gz")) if OUTPUT.exists() else []
    for path in paths:
        item = inspect_file(path, check_gzip=check_gzip)
        rows.append(item)
        meta = {}
        if item["json_sidecar"]:
            try:
                with Path(item["json_sidecar"]).open(encoding="utf-8") as file:
                    meta = json.load(file)
            except (OSError, ValueError):
                pass
        scanner_rows.append({
            "Subject": item["Subject"], "MR ID": item["MR ID"],
            "field_strength": meta.get("MagneticFieldStrength", ""),
            "manufacturer": meta.get("Manufacturer", ""),
            "model": meta.get("ManufacturersModelName", ""),
            "json_sidecar": item["json_sidecar"],
        })
    for name, data, columns in (
        ("qc_report.csv", rows, QC_HEADER),
        ("scanner_info.csv", scanner_rows, SCANNER_HEADER),
    ):
        with (ROOT / name).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(data)
    return {"nifti_files": len(rows), "valid": sum(bool(r["ok"]) for r in rows),
            "invalid": sum(not r["ok"] for r in rows),
            "sessions": len(set((r["Subject"], r["MR ID"]) for r in rows))}


if __name__ == "__main__":
    print(run_qc())
