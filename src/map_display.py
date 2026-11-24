"""Display post-processing for attribution maps: mask to brain, robust contrast stretch."""
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn import image

ROOT = Path(__file__).resolve().parents[1]
MASK = ROOT / "references/tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz"


def display_map(img: nib.Nifti1Image, lo_pct: float = 60, hi_pct: float = 99.5) -> nib.Nifti1Image:
    m = np.asarray(img.dataobj, dtype=np.float32)
    mask = image.resample_to_img(str(MASK), img, interpolation="nearest", force_resample=True, copy_header=True).get_fdata() > 0
    v = m[mask]
    lo, hi = np.percentile(v, lo_pct), np.percentile(v, hi_pct)
    out = np.clip((m - lo) / max(hi - lo, 1e-8), 0, 1) * mask
    return nib.Nifti1Image(out.astype(np.float32), img.affine)


if __name__ == "__main__":                      # re-process existing group Grad-CAM maps
    for p in sorted((ROOT / "results/maps").glob("*_mean_gradcam.nii.gz")):
        nib.save(display_map(nib.load(str(p))), str(p)); print("fixed", p.name)
