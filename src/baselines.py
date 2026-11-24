"""Pre-registered non-imaging baselines (PROTOCOL.md): B1 age-only, B2 hippocampal volume/ICV + age.

Fit on ADNI train (clinical labels), threshold on ADNI val (max balanced accuracy),
evaluated ONCE on ADNI test and on every external cohort, under all three label definitions
and subsets (all / age_matched / 1.5T / 3T). Configs are fully specified a priori -> frozen.
"""
from __future__ import annotations

import csv
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from metrics import ROOT, RES, append_metrics, best_threshold, bootstrap, write_curves

TASKS = {"cn_ad": ("CN", "AD"), "cn_mci": ("CN", "MCI")}
DEFS = {"clinical": "dx_clinical", "cdr": "dx_cdr", "amyloid": "dx_amyloid"}
FEATS = {"age_only": ["age"], "hv_age": ["hv_norm", "age"]}


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hv_norm"] = (df["hv_l"] + df["hv_r"]) / df["icv"] * 1000.0
    df["fs"] = np.where(df["field_strength"] >= 2.9, "3T", "1.5T")
    return df


def age_matched(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """1:1 nearest-age matching within sex (caliper 2 y) between negatives and positives."""
    rng = np.random.default_rng(seed)
    pos, neg = df[df.y == 1], df[df.y == 0].copy()
    keep = []
    for _, r in pos.sample(frac=1, random_state=seed).iterrows():
        c = neg[(neg.sex == r.sex)]
        if c.empty:
            continue
        d = (c.age - r.age).abs()
        if d.min() <= 2:
            j = d.idxmin(); keep += [r.name, j]; neg = neg.drop(j)
    return df.loc[keep]


def git_hash() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def main() -> None:
    adni = prep(pd.read_csv(ROOT / "data/labels/adni_labels.csv")).merge(
        pd.read_csv(ROOT / "data/splits/adni_split.csv")[["key", "split"]], on="key")
    ext = {"oasis3": prep(pd.read_csv(ROOT / "data/labels/oasis3_labels.csv"))}
    PFX = __import__("os").environ.get("ANN_RUN_PREFIX", "")
    if PFX:   # same OASIS-3 scans as the current CNN evaluation (QC-passed, preprocessed)
        import glob
        ok = set()
        for q in glob.glob(str(ROOT / "data/preprocessed/oasis3/qc_*.csv")):
            if __import__("os").path.getsize(q) > 10:
                d = pd.read_csv(q); ok |= set(d[(d.status == "ok") & (d.reg_corr >= 0.3)].key)
        ext["oasis3"] = ext["oasis3"][ext["oasis3"].key.isin(ok)]
    rows, runs = [], []
    for task, (neg, pos) in TASKS.items():
        for model, feats in FEATS.items():
            run_id = f"{PFX}{model}_{task}"
            tr = adni[(adni.split == "train") & adni.dx_clinical.isin([neg, pos])].dropna(subset=feats)
            va = adni[(adni.split == "val") & adni.dx_clinical.isin([neg, pos])].dropna(subset=feats)
            clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
            clf.fit(tr[feats], (tr.dx_clinical == pos).astype(int))
            thr = best_threshold((va.dx_clinical == pos).astype(int), clf.predict_proba(va[feats])[:, 1])
            evals = {"adni": adni[adni.split == "test"], **ext}
            for cohort, df in evals.items():
                for ld, col in DEFS.items():
                    d = df[df[col].isin([neg, pos])].dropna(subset=feats).copy()
                    if d[col].nunique() < 2 or len(d) < 20:
                        continue
                    d["y"] = (d[col] == pos).astype(int)
                    d["p"] = clf.predict_proba(d[feats])[:, 1]
                    subsets = {"all": d, "age_matched": age_matched(d), "1.5T": d[d.fs == "1.5T"], "3T": d[d.fs == "3T"]}
                    for sname, s in subsets.items():
                        if s.y.nunique() < 2 or len(s) < 20:
                            continue
                        for met, (v, lo, hi) in bootstrap(s.y, s.p, thr, n_boot=1000).items():
                            rows.append([run_id, cohort, task, ld, sname, met, v, lo, hi, len(s)])
                    write_curves(run_id, cohort, ld, d.y, d.p, thr)
            runs.append([run_id, time.strftime("%Y-%m-%dT%H:%M"), git_hash(), model, task, "clinical", 0,
                         "PROTOCOL.md baselines (logistic regression, balanced)", True, f"done thr={thr:.3f}"])
    append_metrics(rows)
    idx = RES / "runs_index.csv"
    old = pd.read_csv(idx) if idx.exists() else pd.DataFrame(columns=["run_id"])
    old = old[~old.run_id.isin([r[0] for r in runs])]
    new = pd.concat([old, pd.DataFrame(runs, columns=["run_id", "timestamp", "git_hash", "model", "task", "label_def",
                                                      "seed", "config_path", "frozen", "status"])])
    new.to_csv(idx, index=False)
    m = pd.DataFrame(rows, columns=["run_id", "cohort", "task", "label_def", "subset", "metric", "value", "lo", "hi", "n"])
    print(m[(m.metric == "auroc") & (m.subset == "all")].pivot_table(index=["run_id", "cohort"], columns="label_def", values="value").round(3))


if __name__ == "__main__":
    main()
