"""Shared evaluation utilities: metrics with subject-level bootstrap CIs, curves, contract writers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve, precision_recall_curve)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
METRIC_COLS = ["run_id", "cohort", "task", "label_def", "subset", "metric", "value", "ci_low", "ci_high", "n"]


def ece(y, p, bins: int = 15) -> float:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def point_metrics(y, p, thr: float) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    yh = (p >= thr).astype(int)
    out = {"accuracy": accuracy_score(y, yh), "bal_acc": balanced_accuracy_score(y, yh),
           "precision": precision_score(y, yh, zero_division=0), "recall": recall_score(y, yh, zero_division=0),
           "f1": f1_score(y, yh, zero_division=0), "ece": ece(y, p)}
    out["auroc"] = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    return out


def bootstrap(y, p, thr: float, n_boot: int = 2000, seed: int = 0) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    base = point_metrics(y, p, thr)
    draws = {k: [] for k in base}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        for k, v in point_metrics(y[i], p[i], thr).items():
            draws[k].append(v)
    return {k: (base[k], np.nanpercentile(draws[k], 2.5), np.nanpercentile(draws[k], 97.5)) for k in base}


def best_threshold(y, p) -> float:
    """Threshold maximising balanced accuracy (fit on validation only)."""
    ts = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 99)))
    return float(max(ts, key=lambda t: balanced_accuracy_score(y, (p >= t).astype(int))))


def append_metrics(rows: list[list]) -> None:
    RES.mkdir(exist_ok=True)
    path = RES / "metrics.csv"
    new = pd.DataFrame(rows, columns=METRIC_COLS)
    if path.exists():
        old = pd.read_csv(path)
        key = ["run_id", "cohort", "task", "label_def", "subset", "metric"]
        old = old.merge(new[key], on=key, how="left", indicator=True)
        old = old[old["_merge"] == "left_only"].drop(columns="_merge")
        new = pd.concat([old, new])
    new.round(5).to_csv(path, index=False)


def write_curves(run_id: str, cohort: str, label_def: str, y, p, thr: float) -> None:
    y, p = np.asarray(y), np.asarray(p)
    if len(np.unique(y)) < 2:
        return
    for sub in ("curves", "confusion"):
        (RES / sub).mkdir(parents=True, exist_ok=True)
    fpr, tpr, th = roc_curve(y, p)
    pd.DataFrame({"x": fpr, "y": tpr, "threshold": np.clip(th, 0, 1)}).to_csv(RES / f"curves/{run_id}_{cohort}_{label_def}_roc.csv", index=False)
    pr, rc, th2 = precision_recall_curve(y, p)
    pd.DataFrame({"x": rc, "y": pr, "threshold": np.r_[th2, 1.0]}).to_csv(RES / f"curves/{run_id}_{cohort}_{label_def}_pr.csv", index=False)
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
    cal = [(p[idx == b].mean(), y[idx == b].mean(), int((idx == b).sum())) for b in range(10) if (idx == b).any()]
    pd.DataFrame(cal, columns=["x", "y", "threshold"]).to_csv(RES / f"curves/{run_id}_{cohort}_{label_def}_calib.csv", index=False)
    yh = (p >= thr).astype(int)
    names = {0: "Neg", 1: "Pos"}
    conf = [[names[a], names[b], int(((y == a) & (yh == b)).sum())] for a in (0, 1) for b in (0, 1)]
    pd.DataFrame(conf, columns=["true", "pred", "count"]).to_csv(RES / f"confusion/{run_id}_{cohort}_{label_def}.csv", index=False)
