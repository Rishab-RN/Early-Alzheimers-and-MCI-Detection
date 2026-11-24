"""Train one model / task / seed on ADNI (clinical-diagnosis labels only; PROTOCOL.md).

  python src/train.py --model resnet3d --task cn_ad --seed 0
  python src/train.py --smoke            # random tensors, checks GPU memory/speed

Writes: models/runs/<run_id>.pt (best val AUROC), results/training_curves/<run_id>.csv,
        results/predictions/<run_id>_adni.csv (val+test, local only), appends results/runs_index.csv
External cohorts are NOT touched here (see evaluate.py; frozen configs only).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
PFX = __import__("os").environ.get("ANN_RUN_PREFIX", "")   # e.g. "proto_" for prototype runs
TASKS = {"cn_ad": ("CN", "AD"), "cn_mci": ("CN", "MCI")}


# ----------------------------------------------------------------------------- data
class VolumeDS(Dataset):
    def __init__(self, paths, labels, augment=False, shape=(104, 128, 112)):
        self.paths, self.labels, self.augment, self.shape = list(paths), np.asarray(labels, np.float32), augment, shape

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        p = self.paths[i]
        x = np.load(p).astype(np.float32) if p else np.random.randn(*self.shape).astype(np.float32)
        x = torch.from_numpy(x)[None]
        if self.augment:
            # small, anatomy-preserving augmentation: integer shifts + intensity jitter (no L/R flip)
            sh = [int(s) for s in np.random.randint(-4, 5, 3)]
            x = torch.roll(x, shifts=sh, dims=(1, 2, 3))
            x = x * float(np.random.uniform(0.9, 1.1)) + float(np.random.uniform(-0.1, 0.1))
        return x, torch.tensor(self.labels[i])


def load_split(task: str, res: float):
    lab = pd.read_csv(ROOT / "data/labels/adni_labels.csv")
    spl = pd.read_csv(ROOT / "data/splits/adni_split.csv")
    df = lab.merge(spl[["key", "split"]], on="key")
    neg, pos = TASKS[task]
    df = df[df["dx_clinical"].isin([neg, pos])].copy()
    df["y"] = (df["dx_clinical"] == pos).astype(int)
    cache = ROOT / "data/cache" / f"adni_{res}mm"
    df["path"] = df["key"].map(lambda k: str(cache / f"{k}.npy"))
    df = df[df["path"].map(lambda p: Path(p).exists())]
    return {s: df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test")}


# ----------------------------------------------------------------------------- models
def build_model(name: str, dropout: float = 0.2) -> nn.Module:
    from monai.networks import nets
    if name == "resnet3d":
        body = nets.resnet18(spatial_dims=3, n_input_channels=1, num_classes=1, widen_factor=0.5)
        body.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(body.fc.in_features, 1))
        return body
    if name == "medicalnet":
        body = nets.resnet10(spatial_dims=3, n_input_channels=1, pretrained=True, feed_forward=False,
                             shortcut_type="B", bias_downsample=False)
        return nn.Sequential(body, nn.Dropout(dropout), nn.Linear(512, 1))
    raise ValueError(name)


def git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for x, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            out.append(model(x.to(device)).float().squeeze(1).cpu())
    return torch.cat(out).numpy()


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet3d", choices=["resnet3d", "medicalnet"])
    ap.add_argument("--task", default="cn_ad", choices=list(TASKS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--res", type=float, default=1.5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_id = PFX + f"{args.model}_{args.task}_s{args.seed}" + ("_smoke" if args.smoke else "")

    if args.smoke:
        n = 16
        data = {s: pd.DataFrame({"key": [f"r{i}" for i in range(n)], "path": [""] * n, "y": np.arange(n) % 2}) for s in ("train", "val", "test")}
        args.epochs, args.workers = 2, 0
    else:
        data = load_split(args.task, args.res)
    tr, va = data["train"], data["val"]
    print(f"{run_id}: train {len(tr)} (pos {tr.y.sum()}), val {len(va)} (pos {va.y.sum()})", flush=True)

    w = np.where(tr.y == 1, 0.5 / tr.y.mean(), 0.5 / (1 - tr.y.mean()))
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(tr), replacement=True)
    dl_tr = DataLoader(VolumeDS(tr.path, tr.y, augment=True), batch_size=args.bs, sampler=sampler,
                       num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    dl_va = DataLoader(VolumeDS(va.path, va.y), batch_size=args.bs, num_workers=args.workers)

    model = build_model(args.model).to(device)
    lr_body = args.lr * (0.3 if args.model == "medicalnet" else 1.0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_body, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    lossf = nn.BCEWithLogitsLoss()

    ckpt_dir = ROOT / "models/runs"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    curves_dir = ROOT / "results/training_curves"; curves_dir.mkdir(parents=True, exist_ok=True)
    best, best_ep, hist = -1.0, 0, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); tot, nb = 0.0, 0
        tr_correct, tr_n = 0, 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(dl_tr):
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                out = model(x.to(device, non_blocking=True)).squeeze(1)
                loss = lossf(out, y.to(device)) / args.accum
            tr_correct += int(((out.detach().float() >= 0).cpu() == (y >= 0.5)).sum()); tr_n += len(y)
            scaler.scale(loss).backward()
            if (i + 1) % args.accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            tot += loss.item() * args.accum; nb += 1
        sched.step()
        logit = predict(model, dl_va, device)
        vloss = float(nn.functional.binary_cross_entropy_with_logits(torch.tensor(logit), torch.tensor(va.y.values, dtype=torch.float32)))
        vauc = float(roc_auc_score(va.y, logit)) if va.y.nunique() > 1 else float("nan")
        vacc = float(((logit >= 0) == (va.y.values == 1)).mean())
        hist.append({"epoch": ep, "train_loss": tot / max(nb, 1), "val_loss": vloss, "train_acc": tr_correct / max(tr_n, 1),
                     "val_acc": vacc, "val_auc": vauc, "lr": sched.get_last_lr()[0]})
        print(f"ep {ep:3d} train {tot / max(nb, 1):.4f} val {vloss:.4f} auc {vauc:.4f} "
              f"({(time.time() - t0) / 60:.1f} min, peak {torch.cuda.max_memory_allocated() / 1e9 if device == 'cuda' else 0:.2f} GB)", flush=True)
        if vauc > best:
            best, best_ep = vauc, ep
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep, "val_auc": vauc}, ckpt_dir / f"{run_id}.pt")
        pd.DataFrame(hist).to_csv(curves_dir / f"{run_id}.csv", index=False)
        if ep - best_ep >= args.patience:
            print(f"early stop at {ep}; best epoch {best_ep}", flush=True)
            break

    # internal predictions (val + test) with the best checkpoint
    model.load_state_dict(torch.load(ckpt_dir / f"{run_id}.pt", map_location=device)["model"])
    pred_dir = ROOT / "results/predictions"; pred_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in ("val", "test"):
        d = data[s]
        lg = predict(model, DataLoader(VolumeDS(d.path, d.y), batch_size=args.bs, num_workers=0), device)
        rows.append(pd.DataFrame({"key": d.key, "split": s, "y_true": d.y, "logit": lg}))
    pd.concat(rows).to_csv(pred_dir / f"{run_id}_adni.csv", index=False)

    idx = ROOT / "results/runs_index.csv"
    new = not idx.exists()
    with idx.open("a", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        if new:
            wr.writerow(["run_id", "timestamp", "git_hash", "model", "task", "label_def", "seed", "config_path", "frozen", "status"])
        wr.writerow([run_id, time.strftime("%Y-%m-%dT%H:%M"), git_hash(), args.model, args.task, "clinical", args.seed,
                     json.dumps(vars(args)), False, f"done best_val_auc={best:.4f}@{best_ep}"])
    print(f"done {run_id}: best val AUROC {best:.4f} at epoch {best_ep}; {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
