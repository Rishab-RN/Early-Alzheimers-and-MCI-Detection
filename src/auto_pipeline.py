"""Unattended orchestrator: keeps the data pipeline moving while downloads continue.

Every few minutes:
  1. converts newly completed ADNI zips -> NIfTI (src/adni_convert.py)
  2. rebuilds the OASIS-3 manifest from labels/qc of downloaded scans
  3. if no preprocessing worker is running, preprocesses outstanding scans
     (ADNI first, then OASIS-3), one GPU worker at a time
  4. refreshes results/pipeline_status.json
Stops when every selected scan is preprocessed (or after --hours).
Run detached:  .venv-gpu\\Scripts\\python.exe src\\auto_pipeline.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
LOG = ROOT / "data" / "auto_pipeline.log"
MIN_FOR_FIRST_TRAIN = 550


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def running(script: str) -> bool:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                          f"Where-Object {{ $_.CommandLine -like '*{script}*' }} | Measure-Object | "
                          "Select-Object -ExpandProperty Count"], capture_output=True, text=True)
    try:
        return int(out.stdout.strip() or 0) > 0
    except ValueError:
        return True


def oasis_manifest() -> Path:
    """All downloaded OASIS-3 T1w scans (one per subject), labelled or not."""
    rows = []
    for nii in sorted(glob.glob(str(ROOT / "oasis3_t1w" / "*" / "*" / "*.nii.gz"))):
        p = Path(nii)
        rows.append({"key": p.parent.name, "cohort": "oasis3", "path": str(p)})
    seen, uniq = set(), []
    for r in rows:
        if r["key"] not in seen:
            seen.add(r["key"]); uniq.append(r)
    out = ROOT / "data" / "manifests" / "oasis3.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["key", "cohort", "path"]); w.writeheader(); w.writerows(uniq)
    return out


def outstanding(manifest: Path, cohort: str) -> int:
    if not manifest.exists():
        return 0
    done = set()
    for q in glob.glob(str(ROOT / "data" / "preprocessed" / cohort / "qc_*.csv")):
        with open(q, newline="", encoding="utf-8") as fh:
            done |= {r["key"] for r in csv.DictReader(fh)}
    with manifest.open(newline="", encoding="utf-8") as fh:
        keys = [r["key"] for r in csv.DictReader(fh)]
    return sum(1 for k in keys if k not in done)


def launch(manifest: Path, cohort: str) -> None:
    out = ROOT / "data" / "preprocessed" / cohort
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, nnUNet_n_proc_DA="1")
    flags = 0x00000008 | 0x00000200 if os.name == "nt" else 0   # DETACHED_PROCESS | NEW_PROCESS_GROUP
    with (out / "log.txt").open("ab") as lf:
        subprocess.Popen([PY, str(ROOT / "src" / "preprocess.py"), "--manifest", str(manifest)],
                         cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT, creationflags=flags)
    log(f"launched preprocessing for {cohort}")


RES = 2.0                      # training/eval resolution (compute budget; DECISIONS.md 2026-11-19)
QUEUE = [("resnet3d", "cn_ad", s) for s in range(5)] + [("resnet3d", "cn_mci", s) for s in range(5)] +         [("medicalnet", "cn_ad", s) for s in range(3)] + [("medicalnet", "cn_mci", s) for s in range(3)]


def done_runs() -> set:
    idx = ROOT / "results" / "runs_index.csv"
    if not idx.exists():
        return set()
    import pandas as pd
    d = pd.read_csv(idx)
    return set(d[d.status.astype(str).str.startswith("done")].run_id)


def run_experiments() -> None:
    """Train queued runs one at a time; after each config's seeds finish, run frozen external analyses."""
    if any(running(x) for x in ("train.py", "evaluate.py", "xai_reliability.py", "shift_decomposition.py")):
        return
    for c in ("adni", "oasis3"):
        subprocess.run([PY, str(ROOT / "src" / "cache_tensors.py"), "--cohort", c, "--res", str(RES)],
                       cwd=ROOT, capture_output=True, timeout=6 * 3600)
    # configs whose seeds are all trained -> frozen external analyses first (once per config)
    done = done_runs()
    for model, task in dict.fromkeys((m, t) for m, t, _ in QUEUE):
        cfg = f"{model}_{task}"
        if not all(f"{m}_{t}_s{x}" in done for m, t, x in QUEUE if (m, t) == (model, task)):
            continue
        marker = ROOT / "data" / f"external_done_{cfg}.txt"
        if marker.exists():
            continue
        marker.write_text(time.strftime("%Y-%m-%d %H:%M"))
        with (ROOT / "data" / f"external_{cfg}.log").open("ab") as lf:
            steps = [["evaluate.py", "--model", model, "--task", task, "--res", str(RES), "--external",
                      "--cohorts", "oasis3", "--tta", "none,bnadapt,tent"],
                     ["xai_reliability.py", "--model", model, "--task", task, "--seed", "0", "--res", str(RES),
                      "--external", "oasis3"],
                     ["shift_decomposition.py", "--run", f"ensemble_{cfg}", "--external", "oasis3"],
                     ["make_paper_assets.py"]]
            for st in steps:
                r = subprocess.run([PY, str(ROOT / "src" / st[0])] + st[1:], cwd=ROOT / "src",
                                   stdout=lf, stderr=subprocess.STDOUT, timeout=10 * 3600)
                log(f"{cfg}: {st[0]} exit {r.returncode}")
        log(f"external analyses finished for {cfg}")
        return
    for model, task, seed in QUEUE:
        rid = f"{model}_{task}_s{seed}"
        if rid not in done:
            lf = (ROOT / "data" / "train_queue.log").open("ab")
            subprocess.Popen([PY, str(ROOT / "src" / "train.py"), "--model", model, "--task", task, "--seed",
                              str(seed), "--res", str(RES), "--epochs", "60", "--workers", "0"], cwd=ROOT,
                             stdout=lf, stderr=subprocess.STDOUT, creationflags=0x00000008 | 0x00000200)
            log(f"launched training {rid}")
            return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--hours", type=float, default=96)
    args = ap.parse_args()
    t_end = time.time() + args.hours * 3600
    log("auto_pipeline started")
    while time.time() < t_end:
        try:
            subprocess.run([PY, str(ROOT / "src" / "adni_convert.py")], cwd=ROOT, timeout=3 * 3600)
            om = oasis_manifest()
            am = ROOT / "data" / "manifests" / "adni.csv"
            n_a, n_o = outstanding(am, "adni"), outstanding(om, "oasis3")
            training = any(running(x) for x in ("train.py", "evaluate.py", "xai_reliability.py"))
            if training and running("preprocess.py"):
                # 4 GB GPU: preprocessing + training thrash each other; pause preprocessing (resumable)
                subprocess.run(["powershell", "-NoProfile", "-Command",
                                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                                "Where-Object { $_.CommandLine -like '*preprocess.py*' } | "
                                "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"])
                log("paused preprocessing while a training run uses the GPU")
            if not training and not running("preprocess.py"):
                if n_a:
                    launch(am, "adni")
                elif n_o:
                    launch(om, "oasis3")
            # ---- stage 2: full experiment queue once every scan is preprocessed ----
            if n_a == 0 and n_o == 0 and not running("preprocess.py"):
                run_experiments()
            subprocess.run([PY, str(ROOT / "src" / "pipeline_status.py")], cwd=ROOT,
                           capture_output=True, timeout=600)
            zips_left = len(list((ROOT / "adni/raw_zips").glob("batch_*/*.part")))
            log(f"tick: adni outstanding={n_a} oasis outstanding={n_o} partial_zips={zips_left}")
        except Exception as exc:  # never die silently
            log(f"error: {type(exc).__name__}: {exc}")
        time.sleep(args.interval)
    log("auto_pipeline finished (time limit)")


if __name__ == "__main__":
    main()
