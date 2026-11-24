"""Simple ANN baseline (faculty Level 1): fully-connected MLP on down-sampled MRI voxels.

Input: 2 mm tensors average-pooled to 8 mm (20x24x22 = 10,560 voxels), standardised, PCA to 128 components,
then MLP (256 -> 64 -> 1, ReLU, Adam, early stopping). Trained on ADNI train (clinical labels), threshold on ADNI
val, evaluated on ADNI test and OASIS-3 under all label definitions, same metrics as the CNN.
Writes metrics/curves/confusion and results/training_curves/<run>.csv (loss + accuracy per epoch).
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from baselines import DEFS, age_matched
from evaluate import load_cohort
from metrics import RES, append_metrics, best_threshold, bootstrap, write_curves
from train import TASKS

PFX = os.environ.get("ANN_RUN_PREFIX", "")
RESMM = 2.0


def feats(df: pd.DataFrame) -> np.ndarray:
    out = []
    for p in df.path:
        x = torch.from_numpy(np.load(p).astype(np.float32))[None, None]
        out.append(F.avg_pool3d(x, 4).flatten().numpy())
    return np.stack(out)


def main() -> None:
    adni = load_cohort("adni", "cn_ad", RESMM)
    oasis = load_cohort("oasis3", "cn_ad", RESMM)
    rows, runs = [], []
    for task, (neg, pos) in TASKS.items():
        rid = f"{PFX}mlp_{task}"
        tr = adni[(adni.split == "train") & adni.dx_clinical.isin([neg, pos])]
        va = adni[(adni.split == "val") & adni.dx_clinical.isin([neg, pos])]
        Xtr, Xva = feats(tr), feats(va)
        ytr, yva = (tr.dx_clinical == pos).astype(int).values, (va.dx_clinical == pos).astype(int).values
        sc = StandardScaler().fit(Xtr); pca = PCA(128, random_state=0).fit(sc.transform(Xtr))
        T = lambda X: pca.transform(sc.transform(X))
        w = np.where(ytr == 1, 0.5 / ytr.mean(), 0.5 / (1 - ytr.mean()))
        idx = np.random.default_rng(0).choice(len(ytr), size=len(ytr), p=w / w.sum())     # class-balanced resample
        clf = MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1e-3, learning_rate_init=1e-3, max_iter=1,
                            warm_start=True, random_state=0)
        hist, best, best_state, wait = [], -1, None, 0
        for ep in range(1, 201):
            clf.fit(T(Xtr)[idx], ytr[idx])
            pva = clf.predict_proba(T(Xva))[:, 1]
            from sklearn.metrics import log_loss, roc_auc_score
            vauc = roc_auc_score(yva, pva)
            hist.append({"epoch": ep, "train_loss": clf.loss_, "val_loss": log_loss(yva, np.clip(pva, 1e-6, 1 - 1e-6)),
                         "train_acc": clf.score(T(Xtr), ytr), "val_acc": float(((pva >= .5) == yva).mean()), "val_auc": vauc, "lr": 1e-3})
            if vauc > best:
                best, wait = vauc, 0
                best_state = ([c.copy() for c in clf.coefs_], [b.copy() for b in clf.intercepts_])
            else:
                wait += 1
                if wait >= 20:
                    break
        clf.coefs_, clf.intercepts_ = best_state
        (RES / "training_curves").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(hist).to_csv(RES / "training_curves" / f"{rid}.csv", index=False)
        thr = best_threshold(yva, clf.predict_proba(T(Xva))[:, 1])
        for cohort, df in (("adni", adni[adni.split == "test"]), ("oasis3", oasis)):
            for ld, col in DEFS.items():
                d = df[df[col].isin([neg, pos])].copy()
                if d[col].nunique() < 2 or len(d) < 20:
                    continue
                d["y"] = (d[col] == pos).astype(int)
                d["p"] = clf.predict_proba(T(feats(d)))[:, 1]
                for sname, s in {"all": d, "age_matched": age_matched(d)}.items():
                    if s.y.nunique() < 2 or len(s) < 20:
                        continue
                    for met, (v, lo, hi) in bootstrap(s.y, s.p, thr, n_boot=500).items():
                        rows.append([rid, cohort, task, ld, sname, met, v, lo, hi, len(s)])
                write_curves(rid, cohort, ld, d.y, d.p, thr)
        runs.append([rid, time.strftime("%Y-%m-%dT%H:%M"), "", "mlp", task, "clinical", 0, "src/mlp_baseline.py", True,
                     f"done best_val_auc={best:.4f}"])
        print(rid, "best val AUROC", round(best, 4), flush=True)
    append_metrics(rows)
    idx_path = RES / "runs_index.csv"
    old = pd.read_csv(idx_path) if idx_path.exists() else pd.DataFrame(columns=["run_id"])
    old = old[~old.run_id.isin([r[0] for r in runs])]
    pd.concat([old, pd.DataFrame(runs, columns=["run_id", "timestamp", "git_hash", "model", "task", "label_def", "seed",
                                                "config_path", "frozen", "status"])]).to_csv(idx_path, index=False)


if __name__ == "__main__":
    main()
