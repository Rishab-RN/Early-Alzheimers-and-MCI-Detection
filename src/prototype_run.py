"""Overnight Phase-I prototype run (results tagged proto_, separate from the final paper runs).

1. cache ADNI tensors at 2 mm (QC-gated)
2. train resnet3d CN-vs-AD and CN-vs-MCI (seed 0) on the ADNI scans preprocessed so far
3. preprocess a stratified OASIS-3 subset (all clinical AD + MCI, 150 CN, seed 0)
4. frozen evaluation on ADNI test + OASIS-3 subset (no adaptation + BN-adapt), XAI/ADRS, shift decomposition
Progress: data/prototype.log
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
LOG = ROOT / "data" / "prototype.log"
ENV = dict(os.environ, ANN_RUN_PREFIX="proto_", nnUNet_n_proc_DA="1", HF_HOME=os.environ.get("HF_HOME", str(Path(__file__).resolve().parents[1] / ".hf-cache")))


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def run(args: list[str], cwd: Path = ROOT / "src") -> int:
    log("RUN " + " ".join(args))
    with (ROOT / "data" / "prototype_steps.log").open("ab") as lf:
        r = subprocess.run([PY] + args, cwd=cwd, env=ENV, stdout=lf, stderr=subprocess.STDOUT)
    log(f"EXIT {r.returncode} {args[0]}")
    return r.returncode


def oasis_subset() -> Path:
    lab = pd.read_csv(ROOT / "data/labels/oasis3_labels.csv")
    man = pd.read_csv(ROOT / "data/manifests/oasis3.csv")
    lab = lab[lab.key.isin(man.key)]
    pos = lab[lab.dx_clinical.isin(["AD", "MCI"])]
    cn = lab[lab.dx_clinical == "CN"]
    cn = pd.concat([cn[cn.amyloid_pos.notna()].sample(n=min(100, cn.amyloid_pos.notna().sum()), random_state=0),
                    cn[cn.amyloid_pos.isna()].sample(n=50, random_state=0)])
    keys = set(pos.key) | set(cn.key)
    sub = man[man.key.isin(keys)]
    out = ROOT / "data/manifests/oasis3_proto_subset.csv"
    sub.to_csv(out, index=False)
    log(f"OASIS subset: {len(sub)} scans (AD {len(pos[pos.dx_clinical=='AD'])}, MCI {len(pos[pos.dx_clinical=='MCI'])}, CN {len(cn)})")
    return out


def main() -> None:
    log("prototype run started")
    run(["cache_tensors.py", "--cohort", "adni", "--res", "2.0"])
    for task in ("cn_ad", "cn_mci"):
        run(["train.py", "--model", "resnet3d", "--task", task, "--seed", "0", "--res", "2.0",
             "--epochs", "45", "--patience", "10", "--workers", "0"])
    run(["preprocess.py", "--manifest", str(oasis_subset())])
    for task in ("cn_ad", "cn_mci"):
        run(["evaluate.py", "--model", "resnet3d", "--task", task, "--res", "2.0", "--external",
             "--cohorts", "oasis3", "--tta", "none,bnadapt"])
        run(["xai_reliability.py", "--model", "resnet3d", "--task", task, "--seed", "0", "--res", "2.0",
             "--external", "oasis3"])
        run(["shift_decomposition.py", "--run", f"proto_ensemble_resnet3d_{task}", "--external", "oasis3",
             "--n_boot", "300"])
    run(["pipeline_status.py"])
    log("prototype run finished")


if __name__ == "__main__":
    main()
