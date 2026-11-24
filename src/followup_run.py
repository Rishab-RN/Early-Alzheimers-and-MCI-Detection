"""Runs after prototype_run.py: MLP baseline, seed-1 CNN runs (with accuracy curves), 2-seed ensemble re-evaluation."""
from __future__ import annotations

import time

from prototype_run import LOG, ROOT, log, run


def main() -> None:
    while "prototype run finished" not in LOG.read_text(encoding="utf-8", errors="ignore"):
        time.sleep(120)
    log("follow-up run started")
    run(["mlp_baseline.py"])
    for task in ("cn_ad", "cn_mci"):
        run(["train.py", "--model", "resnet3d", "--task", task, "--seed", "1", "--res", "2.0",
             "--epochs", "45", "--patience", "10", "--workers", "0"])
        run(["evaluate.py", "--model", "resnet3d", "--task", task, "--res", "2.0", "--external",
             "--cohorts", "oasis3", "--tta", "none,bnadapt"])
    run(["pipeline_status.py"])
    log("follow-up run finished")


if __name__ == "__main__":
    main()
