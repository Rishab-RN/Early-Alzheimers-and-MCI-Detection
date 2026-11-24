#!/usr/bin/env python3
"""Fetch the public HD-BET weights into project/models (no patient data sent)."""

from hd_bet_local import MODEL_FOLDER, checkpoint_download


if __name__ == "__main__":
    checkpoint_download.maybe_download_parameters()
    weight = MODEL_FOLDER / "fold_all" / "checkpoint_final.pth"
    if not weight.is_file():
        raise SystemExit(f"HD-BET weights incomplete: {weight}")
    print(f"HD-BET weights ready in {MODEL_FOLDER}")
