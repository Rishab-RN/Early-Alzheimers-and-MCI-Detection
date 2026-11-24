"""Download the public MNI152NLin2009cAsym template (TemplateFlow) and build the Harvard-Oxford ROI atlas.

  python src/fetch_references.py
Creates references/: template T1w + brain mask (1 mm), atlas_ho_mni2009c_1mm.nii.gz, atlas_ho_labels.csv.
"""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "references"
TF = "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/"
FILES = ["tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz", "tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz"]
AD_PRIOR = ["Hippocampus", "Amygdala", "Parahippocampal", "Temporal Fusiform", "Inferior Temporal",
            "Middle Temporal", "Temporal Pole", "Entorhinal"]
SUBCORT = ["Hippocampus", "Amygdala", "Thalamus", "Caudate", "Putamen", "Pallidum", "Accumbens", "Lateral Ventricle"]


def main() -> None:
    REF.mkdir(exist_ok=True)
    for f in FILES:
        if not (REF / f).exists():
            print("downloading", f); urlretrieve(TF + f, REF / f)
    from nilearn import datasets, image
    ref = nib.load(str(REF / FILES[1]))
    cort = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-1mm", symmetric_split=True)
    sub = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-1mm")
    c = image.resample_to_img(cort.maps, ref, interpolation="nearest", force_resample=True, copy_header=True).get_fdata().astype(int)
    s = image.resample_to_img(sub.maps, ref, interpolation="nearest", force_resample=True, copy_header=True).get_fdata().astype(int)
    lab, names = np.zeros(c.shape, np.int16), []
    for i, n in enumerate(cort.labels):
        if i == 0:
            continue
        lab[c == i] = len(names) + 1; names.append(n)
    for i, n in enumerate(sub.labels):
        if any(k in n for k in SUBCORT):
            lab[s == i] = len(names) + 1; names.append(n)
    nib.save(nib.Nifti1Image(lab, ref.affine), str(REF / "atlas_ho_mni2009c_1mm.nii.gz"))
    pd.DataFrame({"label": range(1, len(names) + 1), "roi": names,
                  "ad_prior": [any(a in n for a in AD_PRIOR) for n in names]}).to_csv(REF / "atlas_ho_labels.csv", index=False)
    print(f"references ready: {len(names)} ROIs")


if __name__ == "__main__":
    main()
