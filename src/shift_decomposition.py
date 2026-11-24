"""C1: decompose the ADNI -> external AUROC drop (pre-registered in PROTOCOL.md amendments, 2026-11-19).

Sequential matching on the frozen ensemble's predictions:
  D0 = AUROC_ADNI(clinical, all)            - AUROC_EXT(clinical, all)            total drop
  D1 = AUROC_ADNI(amyloid,  all)            - AUROC_EXT(amyloid,  all)            labels harmonised
  D2 = AUROC_ADNI(amyloid,  age/sex-matched)- AUROC_EXT(amyloid,  age/sex-matched) + demographics
  D3 = as D2 restricted to 3T scans                                                 + field strength
components: label_definition = D0-D1, age_sex = D1-D2, field_strength = D2-D3, residual = D3.
95% CIs: 1,000 joint subject-level bootstrap resamples (matching re-done inside each resample).

  python src/shift_decomposition.py --run ensemble_resnet3d_cn_ad --external oasis3
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from baselines import age_matched
from metrics import RES
from train import TASKS


def auc(d: pd.DataFrame, col: str, neg: str, pos: str, matched: bool, only3t: bool, seed: int) -> float:
    d = d[d[col].isin([neg, pos])].copy()
    d["y"] = (d[col] == pos).astype(int)
    if only3t:
        d = d[d.fs == "3T"]
    if matched:
        d = age_matched(d, seed=seed)
    if d.y.nunique() < 2 or len(d) < 10:
        return np.nan
    return roc_auc_score(d.y, d.p)


def components(a: pd.DataFrame, e: pd.DataFrame, neg: str, pos: str, seed: int = 0) -> dict:
    def drop(col, matched, only3t):
        return auc(a, col, neg, pos, matched, only3t, seed) - auc(e, col, neg, pos, matched, only3t, seed)
    d0 = drop("dx_clinical", False, False)
    d1 = drop("dx_amyloid", False, False)
    d2 = drop("dx_amyloid", True, False)
    d3 = drop("dx_amyloid", True, True)
    return {"total_drop": d0, "label_definition": d0 - d1, "age_sex": d1 - d2, "field_strength": d2 - d3, "residual": d3}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--external", default="oasis3")
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()
    task = "cn_ad" if "cn_ad" in args.run else "cn_mci"
    neg, pos = TASKS[task]
    a = pd.read_csv(RES / "predictions" / f"{args.run}_adni.csv")
    e = pd.read_csv(RES / "predictions" / f"{args.run}_{args.external}.csv")
    point = components(a, e, neg, pos)
    rng = np.random.default_rng(0)
    boots = []
    for b in range(args.n_boot):
        ab = a.iloc[rng.integers(0, len(a), len(a))].reset_index(drop=True)
        eb = e.iloc[rng.integers(0, len(e), len(e))].reset_index(drop=True)
        boots.append(components(ab, eb, neg, pos, seed=b))
    B = pd.DataFrame(boots)
    rows = [[task, args.run, args.external, k, v, np.nanpercentile(B[k], 2.5), np.nanpercentile(B[k], 97.5)]
            for k, v in point.items()]
    out = pd.DataFrame(rows, columns=["task", "run_id", "cohort", "component", "value", "ci_low", "ci_high"])
    path = RES / "shift_decomposition.csv"
    if path.exists():
        old = pd.read_csv(path)
        out = pd.concat([old[~((old.run_id == args.run) & (old.cohort == args.external))], out])
    out.round(5).to_csv(path, index=False)
    print(out[out.run_id == args.run].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
