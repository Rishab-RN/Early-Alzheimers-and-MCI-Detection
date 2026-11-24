"""Analyse one T1-weighted MRI for the app: preprocess -> CN/AD + CN/MCI risk -> explanation -> reliability.

  python src/analyze_scan.py <input.nii[.gz]> <job_dir>
Writes <job_dir>/status.json (progress), brain.nii.gz (preprocessed, MNI), attr_<task>.nii.gz, result.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("nnUNet_n_proc_DA", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import nibabel as nib  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_tensors import crop_box, target_shape, to_tensor  # noqa: E402
from train import build_model  # noqa: E402
from xai_reliability import atlas_on_grid, ig, gradcam, roi_profile, maha_fit, maha, EPS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TASKS = {"cn_ad": ("CN", "AD"), "cn_mci": ("CN", "MCI")}
DEV = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 and os.environ.get("ANN_APP_CPU") != "1" else "cpu"


def status(job: Path, step: str, pct: int, **extra) -> None:
    (job / "status.json").write_text(json.dumps({"step": step, "pct": pct, **extra}), encoding="utf-8")


def model_for(task: str):
    """Ensemble of all calibrated seeds (same as evaluation). Returns (models, temps, threshold, res, xai_run_id)."""
    for prefix in ("", "proto_"):
        cal = RES / "calibration" / f"{prefix}resnet3d_{task}.json"
        ck0 = ROOT / f"models/runs/{prefix}resnet3d_{task}_s0.pt"
        if not ck0.exists():
            continue
        c = json.loads(cal.read_text()) if cal.exists() else {"checkpoints": [ck0.name], "temperatures": [1.0], "threshold": 0.5}
        models, res = [], 2.0
        for name in c["checkpoints"]:
            state = torch.load(ROOT / "models/runs" / name, map_location=DEV)
            m = build_model("resnet3d").to(DEV); m.load_state_dict(state["model"]); m.eval()
            models.append(m); res = float(state["args"].get("res", 2.0))
        return models, [float(t) for t in c["temperatures"]], float(c.get("threshold", 0.5)), res, f"{prefix}resnet3d_{task}_s0"
    return None


def adrs_reference(run_id: str):
    """ADNI-validation attribution profiles saved by xai_reliability.py -> per-class reference."""
    npz = RES / "calibration" / f"{run_id}_adrs_ref.npz"           # anonymous aggregate reference (shareable)
    if npz.exists():
        r = np.load(npz)
        class P:                                                    # minimal PCA stand-in
            def transform(self, X): return (X - r["pca_mean"]) @ r["pca_components"].T
        return P(), {0: (r["mu0"], r["prec0"]), 1: (r["mu1"], r["prec1"])}, float(r["cutoff"])
    sc, pf = RES / "predictions" / f"{run_id}_xai_scores.csv", RES / "predictions" / f"{run_id}_roi_profiles.npy"
    if not (sc.exists() and pf.exists()):
        return None
    from sklearn.decomposition import PCA
    S, P = pd.read_csv(sc), np.load(pf)
    val = (S["set"] == "adni_val").values
    clr = np.log(P + EPS); clr -= clr.mean(1, keepdims=True)
    pca = PCA(n_components=min(10, val.sum() - 1)).fit(clr[val])
    Z = pca.transform(clr)
    pred = (S["z"].values >= 0).astype(int)
    ref = {c: maha_fit(Z[val][pred[val] == c]) for c in (0, 1)}
    dist_val = np.array([maha(Z[i:i + 1], *ref[pred[i]])[0] for i in np.where(val)[0]])
    cut = float(np.percentile(dist_val, 90))
    np.savez(npz, pca_mean=pca.mean_, pca_components=pca.components_, mu0=ref[0][0], prec0=ref[0][1],
             mu1=ref[1][0], prec1=ref[1][1], cutoff=cut)                 # export for cloners (no subject data)
    return pca, ref, cut


def main() -> None:
    src, job = Path(sys.argv[1]), Path(sys.argv[2])
    job.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        img = nib.load(str(src))
        if img.ndim not in (3, 4):
            raise ValueError("expected a 3D T1-weighted MRI")
        brain = job / "brain.nii.gz"
        if img.shape[:3] == (193, 229, 193):                     # already preprocessed to our MNI grid
            status(job, "Already in MNI space, skipping preprocessing", 35)
            nib.save(img, str(brain))
        else:
            status(job, "Preprocessing: bias correction, skull stripping, registration", 10)
            import ants
            from preprocess import load_hdbet, process, TEMPLATE, TEMPLATE_MASK, OUT_ROOT
            import preprocess as pp
            pp.OUT_ROOT = job                                    # write into the job folder
            predictor, apply_bet = load_hdbet(DEV)
            fixed = ants.image_read(str(TEMPLATE)); tmask = ants.image_read(str(TEMPLATE_MASK)); fixed = fixed * (tmask > 0)
            qc = process("brain", "pre", src, predictor, apply_bet, fixed, tmask)
            (job / "pre" / "brain.nii.gz").replace(brain)
            if qc["reg_corr"] < 0.3:
                raise ValueError("registration quality check failed; the image may not be a T1-weighted brain MRI")
        b = nib.load(str(brain)); arr = np.asarray(b.dataobj, dtype=np.float32)
        if (arr == 0).any() and (arr != 0).any():                # black background for display
            arr[arr == 0] = arr[arr != 0].min()
            nib.save(nib.Nifti1Image(arr, b.affine), str(job / "brain_view.nii.gz"))
            (job / "brain_view.nii.gz").replace(job / "brain_disp.nii.gz")
        out = {"tasks": {}, "seconds": None}
        labels = pd.read_csv(ROOT / "references/atlas_ho_labels.csv")
        for i, (task, (neg, pos)) in enumerate(TASKS.items()):
            status(job, f"Predicting {neg} vs {pos} and computing the explanation", 45 + 25 * i)
            mm = model_for(task)
            if mm is None:
                continue
            models, temps, thr, res, run_id = mm
            model = models[0]                                    # explanations from seed 0 (as in the XAI study)
            box = crop_box(); shape = target_shape(box, res)
            x = torch.from_numpy(to_tensor(brain, box, shape).astype(np.float32))[None, None].to(DEV)
            with torch.no_grad():
                zs = [float(mi(x).float().item()) for mi in models]
            z = zs[0]
            p = float(np.mean([1 / (1 + np.exp(-zi / ti)) for zi, ti in zip(zs, temps)]))
            attr = ig(model, x, sign=1.0 if z >= 0 else -1.0)
            atlas, _ = atlas_on_grid(res)
            prof = roi_profile(attr, atlas, len(labels))
            top = np.argsort(-prof)[:8]
            rel = {"available": False}
            ref = adrs_reference(run_id)
            if ref is not None:
                pca, refs, cut = ref
                clr = np.log(prof + EPS); clr -= clr.mean()
                d = float(maha(pca.transform(clr[None]), *refs[int(z >= 0)])[0])
                rel = {"available": True, "adrs": d, "cutoff": float(cut), "reliable": bool(d <= cut)}
            # attribution map on MNI grid for the viewer
            ext = np.array([s.stop - s.start for s in box], float); vs = ext / np.array(shape)
            tpl = nib.load(str(ROOT / "references/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz"))
            start = np.array([s.start for s in box], float) + vs / 2 - 0.5
            aff = tpl.affine.copy(); aff[:3, :3] = tpl.affine[:3, :3] @ np.diag(vs); aff[:3, 3] = tpl.affine[:3, :3] @ start + tpl.affine[:3, 3]
            from scipy.ndimage import gaussian_filter
            a = gaussian_filter(np.abs(attr), sigma=1.0); a = np.clip(a / (np.percentile(a[a > 0], 99.5) + 1e-8), 0, 1).astype(np.float32)
            nib.save(nib.Nifti1Image(a, aff), str(job / f"attr_{task}.nii.gz"))
            cam = gradcam(model, x, sign=1.0 if z >= 0 else -1.0)
            from map_display import display_map
            nib.save(display_map(nib.Nifti1Image(cam.astype(np.float32), aff)), str(job / f"gradcam_{task}.nii.gz"))
            out["tasks"][task] = {
                "negative": neg, "positive": pos, "probability": float(p), "threshold": thr,
                "prediction": pos if p >= thr else neg, "reliability": rel,
                "top_regions": [{"roi": labels.roi.iloc[j], "share": float(prof[j]), "ad_region": bool(labels.ad_prior.iloc[j])} for j in top],
            }
        if not out["tasks"]:
            raise ValueError("no trained model available yet")
        out["seconds"] = round(time.time() - t0, 1)
        (job / "result.json").write_text(json.dumps(out), encoding="utf-8")
        status(job, "Done", 100, done=True)
    except Exception as exc:  # report to the UI
        status(job, "Failed", 100, done=True, error=f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
