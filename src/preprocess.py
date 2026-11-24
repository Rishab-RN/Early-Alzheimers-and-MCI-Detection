"""Identical structural-MRI preprocessing for every cohort (PROTOCOL.md).

N4 bias correction -> HD-BET skull strip (GPU, model loaded once) -> AFFINE registration
to MNI152NLin2009cAsym 1 mm -> robust nonzero z-score. Resumable and shardable.

Usage:
  python src/preprocess.py --manifest data/manifests/oasis3.csv [--shard 0 --nshards 2]
Manifest columns: key, cohort, path   (path = raw T1w .nii/.nii.gz)
Outputs: data/preprocessed/<cohort>/<key>.nii.gz  and  data/preprocessed/<cohort>/qc_<shard>.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("nnUNet_n_proc_DA", "1")          # avoids a Windows 'hostname' call
SCRATCH = ROOT / "preprocess_tmp"
SCRATCH.mkdir(exist_ok=True)
os.environ["TEMP"] = os.environ["TMP"] = str(SCRATCH)
tempfile.tempdir = str(SCRATCH)

import numpy as np  # noqa: E402
import ants  # noqa: E402
import torch  # noqa: E402

TEMPLATE = ROOT / "references" / "tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz"
TEMPLATE_MASK = ROOT / "references" / "tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz"
HDBET_WEIGHTS = ROOT / "models" / "hd-bet_params" / "release_2.0.0"
OUT_ROOT = ROOT / "data" / "preprocessed"
CORR_RETRY = 0.3
QC_FIELDS = ["key", "cohort", "status", "seconds", "native_brain_ml", "reg_corr",
             "nonzero_frac", "error"]


def load_hdbet(device: str):
    from HD_BET import paths, checkpoint_download, hd_bet_prediction
    for mod in (paths, checkpoint_download, hd_bet_prediction):
        mod.folder_with_parameter_files = str(HDBET_WEIGHTS)
    predictor = hd_bet_prediction.get_hdbet_predictor(use_tta=False, device=torch.device(device))
    return predictor, hd_bet_prediction.apply_bet


def robust_z(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    m = np.isfinite(arr) & (arr > 0)
    if m.sum() < 1000:
        raise ValueError("too few brain voxels after registration")
    lo, hi = np.percentile(arr[m], [1.0, 99.0])
    v = np.clip(arr[m], lo, hi)
    out = np.zeros_like(arr)
    out[m] = (v - v.mean()) / max(float(v.std()), 1e-6)
    return out


def _rotations():
    import itertools
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            for r, (c, s) in enumerate(zip(perm, signs)):
                m[r, c] = s
            if np.isclose(np.linalg.det(m), 1.0):
                mats.append(m)
    return mats                                   # 24 proper rotations


def orientation_search(fixed, moving, tmask, td: Path) -> str:
    """Return path of the best axis-aligned initial transform (fixed -> moving)."""
    f3 = ants.resample_image(fixed, (3, 3, 3), use_voxels=False, interp_type=0)
    m3 = ants.resample_image(moving, (3, 3, 3), use_voxels=False, interp_type=0)
    sel = ants.resample_image(tmask, (3, 3, 3), use_voxels=False, interp_type=1).numpy() > 0
    fa = f3.numpy()
    cf = np.array(ants.get_center_of_mass(f3 * 0 + (f3 > 0)))
    cm = np.array(ants.get_center_of_mass(m3 * 0 + (m3 > 0)))
    best, best_path = -2.0, None
    for i, R in enumerate(_rotations()):
        tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3,
                                        matrix=R, center=cf, translation=cm - cf)
        p = str(td / f"rot{i}.mat")
        ants.write_transform(tx, p)
        try:
            r = ants.registration(fixed=f3, moving=m3, type_of_transform="Affine", initial_transform=p,
                                  aff_iterations=(40, 20, 0), aff_shrink_factors=(4, 2, 1),
                                  aff_smoothing_sigmas=(2, 1, 0))
            w = r["warpedmovout"].numpy()
            c = float(np.corrcoef(w[sel], fa[sel])[0, 1])
        except Exception:
            continue
        if np.isfinite(c) and c > best:
            best, best_path = c, p
    return best_path


def process(key: str, cohort: str, src: Path, predictor, apply_bet, fixed, tmask) -> dict:
    out_dir = OUT_ROOT / cohort
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{key}.nii.gz"
    with tempfile.TemporaryDirectory(dir=SCRATCH) as td:
        td = Path(td)
        raw = ants.image_read(str(src))
        if raw.dimension == 4:                       # rare: singleton 4th dim
            raw = ants.from_numpy(raw.numpy()[..., 0], origin=raw.origin[:3],
                                  spacing=raw.spacing[:3], direction=raw.direction[:3, :3])
        if raw.dimension != 3:
            raise ValueError(f"expected 3D image, got {raw.dimension}D")
        raw = ants.reorient_image2(raw, "RAS")
        n4 = ants.n4_bias_field_correction(raw, mask=ants.get_mask(raw), shrink_factor=4)
        n4_path, mask_path, brain_path = td / "n4.nii.gz", td / "mask.nii.gz", td / "brain.nii.gz"
        ants.image_write(n4, str(n4_path))
        predictor.predict_from_files_sequential([[str(n4_path)]], [str(mask_path)],
                                                overwrite=True, save_probabilities=False)
        apply_bet(str(n4_path), str(mask_path), str(brain_path))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()          # release GPU memory for concurrent training during CPU steps
        mask = ants.image_read(str(mask_path))
        native_ml = float((mask.numpy() > 0).sum() * np.prod(mask.spacing) / 1000.0)
        brain = ants.image_read(str(brain_path))
        t = fixed.numpy()
        sel = tmask.numpy() > 0

        def register(init=None):
            kw = {"initial_transform": init} if init is not None else {}
            r = ants.registration(fixed=fixed, moving=brain, type_of_transform="Affine",
                                  aff_metric="mattes", **kw)
            w = r["warpedmovout"]
            a = robust_z(w.numpy())
            return w, a, float(np.corrcoef(a[sel], t[sel])[0, 1])

        warped, arr, corr = register()
        if corr < CORR_RETRY:
            # Header orientation is wrong for some scans (e.g. sagittal volumes labelled axial).
            # Try all 24 axis-aligned rotations as initial transforms at 3 mm, keep the best.
            init_path = orientation_search(fixed, brain, tmask, td)
            w2, a2, c2 = register(init_path)
            if c2 > corr:
                warped, arr, corr = w2, a2, c2
        out = ants.from_numpy(arr, origin=warped.origin, spacing=warped.spacing,
                              direction=warped.direction)
        tmp_dest = dest.with_name(dest.name + ".tmp.nii.gz")
        ants.image_write(out, str(tmp_dest))
        os.replace(tmp_dest, dest)                    # atomic: no half-written outputs after a power cut
    return {"native_brain_ml": round(native_ml, 1), "reg_corr": round(corr, 4),
            "nonzero_frac": round(float((arr != 0).mean()), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows = [r for i, r in enumerate(rows) if i % args.nshards == args.shard]
    if args.limit:
        rows = rows[: args.limit]
    cohort = rows[0]["cohort"] if rows else "unknown"
    qc_path = OUT_ROOT / cohort / f"qc_{args.shard}.csv"
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if qc_path.exists():
        with qc_path.open(newline="", encoding="utf-8") as fh:
            done = {r["key"] for r in csv.DictReader(fh) if r["status"] == "ok"}
    todo = [r for r in rows if r["key"] not in done and not (OUT_ROOT / r["cohort"] / f"{r['key']}.nii.gz").exists()]
    print(f"shard {args.shard}/{args.nshards}: {len(rows)} rows, {len(todo)} to do, device={args.device}", flush=True)
    if not todo:
        return 0

    ants.set_ants_num_threads = None  # noqa (keep default threading)
    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "4")
    predictor, apply_bet = load_hdbet(args.device)
    fixed = ants.image_read(str(TEMPLATE))
    tmask = ants.image_read(str(TEMPLATE_MASK))
    fixed = fixed * (tmask > 0)

    new = not qc_path.exists()
    with qc_path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=QC_FIELDS)
        if new:
            w.writeheader()
        attempted = set()
        while todo:
          for i, r in enumerate(todo, 1):
            attempted.add(r["key"])
            t0 = time.monotonic()
            rec = {"key": r["key"], "cohort": r["cohort"], "status": "ok", "error": ""}
            try:
                rec.update(process(r["key"], r["cohort"], Path(r["path"]), predictor, apply_bet, fixed, tmask))
            except Exception as exc:  # keep going; failures are logged for QC
                rec.update(status="failed", error=f"{type(exc).__name__}: {str(exc)[:200]}")
            rec["seconds"] = round(time.monotonic() - t0, 1)
            w.writerow(rec)
            fh.flush()
            print(f"[{i}/{len(todo)}] {r['key']} {rec['status']} {rec['seconds']}s "
                  f"brain={rec.get('native_brain_ml')} corr={rec.get('reg_corr')} {rec['error']}", flush=True)
          # pick up rows appended to the manifest while we were working (e.g. newly converted zips)
          with args.manifest.open(newline="", encoding="utf-8") as mf:
              rows = [x for j, x in enumerate(csv.DictReader(mf)) if j % args.nshards == args.shard]
          todo = [x for x in rows if x["key"] not in attempted
                  and not (OUT_ROOT / x["cohort"] / f"{x['key']}.nii.gz").exists()]
          if todo:
              print(f"manifest grew: {len(todo)} more to do", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
