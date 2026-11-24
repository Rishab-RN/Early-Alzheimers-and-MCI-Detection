#!/usr/bin/env python3
"""Launch the resumable full worker detached on Windows; credentials via pipe.

Only a PID and ordinary progress logs are written. No password in CLI flags,
environment variables, files or progress logs.
"""

from __future__ import annotations

import argparse
import getpass
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Download only first N subjects")
    parser.add_argument("--skip-clinical", action="store_true")
    args = parser.parse_args()
    username = input("NITRC username: ").strip()
    password = getpass.getpass("NITRC password: ")
    if not username or not password or "\n" in username + password:
        raise ValueError("Nonempty one-line credentials required")

    cmd = [sys.executable, "-u", str(Path(__file__).resolve().parent / "run_oasis_pipeline.py"),
           "--credentials-stdin"]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.skip_clinical:
        cmd.append("--skip-clinical")

    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    with (ROOT / "pipeline.log").open("ab", buffering=0) as log_file:
        process = subprocess.Popen(
            cmd, cwd=str(ROOT), stdin=subprocess.PIPE, stdout=log_file,
            stderr=subprocess.STDOUT, creationflags=flags,
            close_fds=True,
        )
        assert process.stdin is not None
        process.stdin.write((username + "\n" + password + "\n").encode("utf-8"))
        process.stdin.close()

    (ROOT / "download_worker.pid").write_text(str(process.pid), encoding="ascii")
    print(f"Detached worker PID={process.pid}. Progress: {ROOT / 'pipeline.log'}")
    print(f"Status: {ROOT / 'STATUS.md'}")


if __name__ == "__main__":
    main()
