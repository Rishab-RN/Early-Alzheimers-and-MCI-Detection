"""Evaluate trained CNNs (single seeds + deep ensemble) on ADNI test and external cohorts.

  python src/evaluate.py --model resnet3d --task cn_ad [--external] [--tta none,bnadapt,tent]

* Temperature (per seed) and decision threshold (ensemble) are fitted on ADNI val only.
* External cohorts are evaluated only with --external AND only if the config id
  "<model>_<task>" is listed in PROTOCOL.md "Frozen configurations" (test-set lock).
* Writes results/metrics.csv rows, curves/confusion, predictions (local only).
"""
from __future__ import annotations

import argparse
import os
import copy
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from metrics import ROOT, RES, append_metrics, best_threshold, bootstrap, write_curves
from train import TASKS, VolumeDS, build_model
from baselines import DEFS, age_matched

PFX = __import__("os").environ.get("ANN_RUN_PREFIX", "")   # e.g. "proto_" for prototype runs
DEV = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 and os.environ.get("ANN_APP_CPU") != "1" else "cpu"


def frozen_ids() -> set[str]:
    txt = (ROOT / "PROTOCOL.md").read_text(encoding="utf-8")
    sec = txt.split("## Frozen configurations", 1)[-1].split("\n## ", 1)[0]
    return set(re.findall(r"\|\s*([a-z0-9_]+)\s*\|", sec))


def load_cohort(cohort: str, task: str, res: float) -> pd.DataFrame:
    from cache_tensors import crop_box, target_shape, to_tensor
    lab = pd.read_csv(ROOT / f"data/labels/{cohort}_labels.csv")
    lab["hv_norm"] = (lab["hv_l"] + lab["hv_r"]) / lab["icv"] * 1000
    lab["fs"] = np.where(lab["field_strength"] >= 2.9, "3T", "1.5T")
    if cohort == "adni":
        lab = lab.merge(pd.read_csv(ROOT / "data/splits/adni_split.csv")[["key", "split"]], on="key")
    else:
        lab["split"] = "test"
    cache = ROOT / f"data/cache/{cohort}_{res}mm"
    cache.mkdir(parents=True, exist_ok=True)
    box = crop_box(); shape = target_shape(box, res)
    qc_ok = set()
    for q in glob.glob(str(ROOT / f"data/preprocessed/{cohort}/qc_*.csv")):
        if Path(q).stat().st_size < 10:
            continue
        d = pd.read_csv(q)
        qc_ok |= set(d[(d.status == "ok") & (d.reg_corr >= 0.3)].key)
    lab = lab[lab.key.isin(qc_ok)].copy()
    for k in lab.key:                                    # cache on demand
        f = cache / f"{k}.npy"
        if not f.exists():
            np.save(f, to_tensor(ROOT / f"data/preprocessed/{cohort}/{k}.nii.gz", box, shape))
    lab["path"] = lab.key.map(lambda k: str(cache / f"{k}.npy"))
    return lab


@torch.no_grad()
def logits(model, df, bs=4) -> np.ndarray:
    model.eval(); out = []
    for x, _ in DataLoader(VolumeDS(df.path, np.zeros(len(df))), batch_size=bs):
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
            out.append(model(x.to(DEV)).float().squeeze(1).cpu())
    return torch.cat(out).numpy()


def bn_adapt(model, df, bs=4):
    m = copy.deepcopy(model)
    for mod in m.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            mod.reset_running_stats(); mod.momentum = None
    m.train()
    with torch.no_grad():
        for x, _ in DataLoader(VolumeDS(df.path, np.zeros(len(df))), batch_size=bs, shuffle=True):
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
                m(x.to(DEV))
    return m.eval()


def tent(model, df, bs=4, lr=1e-4):
    m = bn_adapt(model, df, bs)
    params = []
    for mod in m.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            params += [mod.weight, mod.bias]
    for p in m.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    m.train()
    for x, _ in DataLoader(VolumeDS(df.path, np.zeros(len(df))), batch_size=bs, shuffle=True):
        z = m(x.to(DEV)).float().squeeze(1)
        p = torch.sigmoid(z)
        ent = -(p * torch.log(p + 1e-6) + (1 - p) * torch.log(1 - p + 1e-6)).mean()
        opt.zero_grad(); ent.backward(); opt.step()
    return m.eval()


def fit_temperature(z, y) -> float:
    t = torch.ones(1, requires_grad=True)
    z_, y_ = torch.tensor(z, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=200)
    def closure():
        opt.zero_grad(); l = nn.functional.binary_cross_entropy_with_logits(z_ / t.clamp(min=0.05), y_); l.backward(); return l
    opt.step(closure)
    return float(t.detach().clamp(min=0.05))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet3d")
    ap.add_argument("--task", default="cn_ad")
    ap.add_argument("--res", type=float, default=1.5)
    ap.add_argument("--external", action="store_true")
    ap.add_argument("--cohorts", default="oasis3")
    ap.add_argument("--tta", default="none")
    args = ap.parse_args()
    neg, pos = TASKS[args.task]
    cfg = f"{args.model}_{args.task}"
    ckpts = sorted(glob.glob(str(ROOT / f"models/runs/{PFX}{cfg}_s[0-9]*.pt")))
    ckpts = [c for c in ckpts if "smoke" not in c]
    if not ckpts:
        raise SystemExit(f"no checkpoints for {cfg}")
    if args.external and cfg not in frozen_ids():
        raise SystemExit(f"{cfg} is not frozen in PROTOCOL.md; external evaluation refused (test-set lock)")

    adni = load_cohort("adni", args.task, args.res)
    cohorts = {"adni": adni[adni.split == "test"]}
    if args.external:
        for c in args.cohorts.split(","):
            cohorts[c] = load_cohort(c, args.task, args.res)
    val = adni[(adni.split == "val") & adni.dx_clinical.isin([neg, pos])]
    yv = (val.dx_clinical == pos).astype(int).values

    models, temps = [], []
    for c in ckpts:
        m = build_model(args.model).to(DEV)
        m.load_state_dict(torch.load(c, map_location=DEV)["model"]); m.eval()
        temps.append(fit_temperature(logits(m, val), yv)); models.append(m)
    pv = np.mean([1 / (1 + np.exp(-logits(m, val) / t)) for m, t in zip(models, temps)], axis=0)
    thr = best_threshold(yv, pv)
    print(f"{cfg}: {len(models)} seeds, temps {np.round(temps, 2)}, thr {thr:.3f}", flush=True)
    import json
    (RES / "calibration").mkdir(parents=True, exist_ok=True)
    (RES / "calibration" / f"{PFX}{cfg}.json").write_text(json.dumps(
        {"checkpoints": [Path(c).name for c in ckpts], "temperatures": temps, "threshold": thr, "res": args.res}), encoding="utf-8")

    rows = []
    pred_dir = RES / "predictions"; pred_dir.mkdir(parents=True, exist_ok=True)
    for cohort, df in cohorts.items():
        df = df.copy()
        for tta in args.tta.split(","):
            if tta != "none" and cohort == "adni":
                continue
            per_seed = []
            for m, t in zip(models, temps):
                mm = m if tta == "none" else (bn_adapt(m, df) if tta == "bnadapt" else tent(m, df))
                per_seed.append(1 / (1 + np.exp(-logits(mm, df) / t)))
            per_seed = np.array(per_seed)
            df["p"], df["ens_sd"] = per_seed.mean(0), per_seed.std(0)
            for i in range(len(models)):
                df[f"p_s{i}"] = per_seed[i]
            tag = "" if tta == "none" else f"_{tta}"
            run_ens = f"{PFX}ensemble_{cfg}{tag}"
            df.drop(columns=["path"]).to_csv(pred_dir / f"{run_ens}_{cohort}.csv", index=False)
            for ld, col in DEFS.items():
                d = df[df[col].isin([neg, pos])].copy()
                if d[col].nunique() < 2 or len(d) < 20:
                    continue
                d["y"] = (d[col] == pos).astype(int)
                subsets = {"all": d, "age_matched": age_matched(d), "1.5T": d[d.fs == "1.5T"], "3T": d[d.fs == "3T"]}
                for run_id, pcol in [(run_ens, "p")] + [(f"{PFX}{cfg}_s{i}{tag}", f"p_s{i}") for i in range(len(models))]:
                    for sname, s in subsets.items():
                        if s.y.nunique() < 2 or len(s) < 20:
                            continue
                        nb = 1000 if run_id == run_ens else 200
                        for met, (v, lo, hi) in bootstrap(s.y, s[pcol], thr, n_boot=nb).items():
                            rows.append([run_id, cohort, args.task, ld, sname, met, v, lo, hi, len(s)])
                    write_curves(run_id, cohort, ld, d.y, d[pcol], thr)
            print(f"  {cohort} {tta}: n={len(df)}", flush=True)
    append_metrics(rows)
    m = pd.DataFrame(rows, columns=["run_id", "cohort", "task", "label_def", "subset", "metric", "value", "lo", "hi", "n"])
    print(m[(m.metric == "auroc") & (m.subset == "all") & m.run_id.str.startswith("ensemble")]
          .pivot_table(index=["run_id", "cohort"], columns="label_def", values="value").round(3))


if __name__ == "__main__":
    main()
