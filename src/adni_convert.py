"""Convert completed IDA image zips (DICOM) to one NIfTI per image ID.

Scans adni/raw_zips/batch_*/ for zips that have a .ok marker (fully downloaded and
zip-tested by fetch_urls.py), extracts each to a temporary folder, runs dcm2niix per
image (I<image_id> folder), and writes data/raw/adni/I<image_id>.nii.gz.
Appends new rows to data/manifests/adni.csv (key, cohort, path). Idempotent.
Zips that contain NIfTI instead of DICOM are handled too.
"""
from __future__ import annotations

import csv
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "adni"
MANIFEST = ROOT / "data" / "manifests" / "adni.csv"
DONE_LIST = ROOT / "data" / "raw" / "adni_converted_zips.txt"


def dcm2niix_bin() -> str:
    exe = shutil.which("dcm2niix")
    if exe:
        return exe
    cand = ROOT / ".venv-gpu" / "Scripts" / "dcm2niix.exe"
    if cand.exists():
        return str(cand)
    raise FileNotFoundError("dcm2niix not found")


def convert_zip(zp: Path, exe: str) -> list[tuple[str, Path]]:
    out = []
    with tempfile.TemporaryDirectory(dir=ROOT / "preprocess_tmp") as td:
        td = Path(td)
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(td)
        image_dirs: dict[str, Path] = {}
        for p in td.rglob("*"):
            if p.is_dir() and re.fullmatch(r"I\d+", p.name):
                image_dirs[p.name] = p
        for iid, d in sorted(image_dirs.items()):
            dest = RAW / f"{iid}.nii.gz"
            if dest.exists():
                out.append((iid, dest)); continue
            niis = list(d.rglob("*.nii")) + list(d.rglob("*.nii.gz"))
            tmp_out = td / f"out_{iid}"
            tmp_out.mkdir()
            if niis:
                src = max(niis, key=lambda q: q.stat().st_size)
                if src.suffix == ".nii":
                    import nibabel as nib
                    nib.save(nib.load(str(src)), str(dest))
                else:
                    shutil.copy2(src, dest)
            else:
                r = subprocess.run([exe, "-z", "y", "-b", "n", "-f", iid, "-o", str(tmp_out), str(d)],
                                   capture_output=True, text=True)
                made = sorted(tmp_out.glob("*.nii.gz"), key=lambda q: q.stat().st_size)
                if not made:
                    print(f"  {iid}: dcm2niix produced nothing ({r.stderr[-200:]})")
                    continue
                shutil.move(str(made[-1]), dest)       # largest = full volume
            out.append((iid, dest))
    return out


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    done = set(DONE_LIST.read_text().split()) if DONE_LIST.exists() else set()
    have = set()
    if MANIFEST.exists():
        with MANIFEST.open(newline="") as fh:
            have = {r["key"] for r in csv.DictReader(fh)}
    exe = dcm2niix_bin()
    added = 0

    def flush(rows):
        nonlocal added
        if not rows:
            return
        new_file = not MANIFEST.exists()
        with MANIFEST.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["key", "cohort", "path"])
            if new_file:
                w.writeheader()
            w.writerows(rows)
        added += len(rows)

    # recovery: NIfTIs already on disk but missing from the manifest (e.g. after a crash)
    flush([{"key": p.name[:-7], "cohort": "adni", "path": str(p)} for p in sorted(RAW.glob("I*.nii.gz"))
           if p.name[:-7] not in have and not have.add(p.name[:-7])])
    for ok in sorted((ROOT / "adni" / "raw_zips").glob("batch_*/*.zip.ok")):
        zp = ok.with_suffix("")
        if str(zp) in done:
            continue
        print(f"converting {zp.name} ...", flush=True)
        rows = []
        for iid, dest in convert_zip(zp, exe):
            if iid not in have:
                rows.append({"key": iid, "cohort": "adni", "path": str(dest)})
                have.add(iid)
        flush(rows)                                   # manifest first, then mark the zip done
        with DONE_LIST.open("a") as fh:
            fh.write(str(zp) + "\n")
    print(f"added {added} images; manifest total {len(have)}")


if __name__ == "__main__":
    main()
