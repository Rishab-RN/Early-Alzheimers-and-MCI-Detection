"""Crop preprocessed MNI volumes to the template brain box and resample to a fixed grid.

data/preprocessed/<cohort>/<key>.nii.gz  ->  data/cache/<cohort>_<res>mm/<key>.npy (float16)
The crop box and output shape are identical for every cohort (derived from the template mask).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
MASK = ROOT / "references" / "tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz"


def crop_box(margin: int = 4):
    m = nib.load(str(MASK)).get_fdata() > 0
    idx = np.argwhere(m)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + margin + 1, m.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def target_shape(box, res: float):
    # multiples of 8 so 3 stride-2 stages divide evenly
    return tuple(int(np.ceil((s.stop - s.start) / res / 8) * 8) for s in box)


def to_tensor(path: Path, box, shape) -> np.ndarray:
    a = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)[box]
    t = torch.from_numpy(np.ascontiguousarray(a))[None, None]
    t = F.interpolate(t, size=shape, mode="trilinear", align_corners=False)
    return t[0, 0].numpy().astype(np.float16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--res", type=float, default=1.5)
    args = ap.parse_args()
    box = crop_box()
    shape = target_shape(box, args.res)
    src = ROOT / "data" / "preprocessed" / args.cohort
    out = ROOT / "data" / "cache" / f"{args.cohort}_{args.res}mm"
    out.mkdir(parents=True, exist_ok=True)
    import csv
    ok = set()
    for q in src.glob("qc_*.csv"):
        with open(q, newline="", encoding="utf-8") as fh:
            ok |= {r["key"] for r in csv.DictReader(fh)
                   if r["status"] == "ok" and float(r["reg_corr"] or 0) >= 0.3}   # QC gate
    n = 0
    for p in sorted(src.glob("*.nii.gz")):
        if p.name[:-7] not in ok:
            continue
        dest = out / (p.name[:-7] + ".npy")
        if dest.exists():
            continue
        tmp = dest.with_name(dest.stem + ".tmp.npy")
        np.save(tmp, to_tensor(p, box, shape))
        os.replace(tmp, dest)
        n += 1
    print(f"{args.cohort}: cached {n} new volumes, shape {shape}, box {[(s.start, s.stop) for s in box]}")


if __name__ == "__main__":
    main()
