#!/usr/bin/env python3
"""Start a detached, local-only finalizer after launching the full downloader."""

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0
with (ROOT / "finalizer.log").open("ab", buffering=0) as output:
    child = subprocess.Popen(
        [sys.executable, "-u", str(Path(__file__).resolve().parent / "finalize_oasis3.py"), "--watch"],
        cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=output,
        stderr=subprocess.STDOUT, creationflags=flags, close_fds=True,
    )
(ROOT / "finalizer.pid").write_text(str(child.pid), encoding="ascii")
print(f"Detached local finalizer PID={child.pid}; log: {ROOT / 'finalizer.log'}")
