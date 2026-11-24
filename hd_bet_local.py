#!/usr/bin/env python3
"""Run official HD-BET weights sequentially with cache inside the project.

HD-BET 2.x normally writes weights to the Windows user profile. That is
outside our approved workspace, so point the library's model references at
models/hd-bet_params/release_2.0.0 before invoking its prediction routines.
"""

import argparse
import os
from pathlib import Path

import torch

from HD_BET import paths


MODEL_FOLDER = Path(__file__).resolve().parent / "models" / "hd-bet_params" / "release_2.0.0"
paths.folder_with_parameter_files = str(MODEL_FOLDER)

# HD-BET imports these constants by value; update both imported copies.
from HD_BET import checkpoint_download, hd_bet_prediction

checkpoint_download.folder_with_parameter_files = str(MODEL_FOLDER)
hd_bet_prediction.folder_with_parameter_files = str(MODEL_FOLDER)

def main() -> None:
    """Use nnU-Net's supported single-process inference on one 3D MRI.

    The stock HD-BET CLI starts several worker pools, which can fail on
    Windows named pipes in a restricted runtime. Its underlying predictor
    supports a sequential method with the same model and preprocessing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-device", default="cpu")
    parser.add_argument("--disable_tta", action="store_true")
    args = parser.parse_args()

    source = Path(args.input).resolve()
    target = Path(args.output).resolve()
    if not source.is_file() or not source.name.endswith(".nii.gz"):
        parser.error("input must be an existing .nii.gz")
    if not target.name.endswith(".nii.gz"):
        parser.error("output must end in .nii.gz")
    target.parent.mkdir(parents=True, exist_ok=True)
    mask = target.with_name(target.name[:-7] + "_bet.nii.gz")

    checkpoint_download.maybe_download_parameters()
    predictor = hd_bet_prediction.get_hdbet_predictor(
        use_tta=not args.disable_tta, device=torch.device(args.device))
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    print("HD-BET: sequential nnU-Net inference (no worker pool)", flush=True)
    try:
        predictor.predict_from_files_sequential(
            [[str(source)]], [str(mask)], overwrite=True,
            save_probabilities=False)
        if not mask.exists():
            raise RuntimeError("HD-BET did not produce a brain mask")
        hd_bet_prediction.apply_bet(str(source), str(mask), str(target))
    finally:
        mask.unlink(missing_ok=True)
        for filename in ("dataset.json", "plans.json", "predict_from_raw_data_args.json"):
            (target.parent / filename).unlink(missing_ok=True)
    print(f"HD-BET: wrote {target}", flush=True)


if __name__ == "__main__":
    main()
