"""Read-only local API for the project dashboard.

Run:  .venv-app\\Scripts\\python.exe -m uvicorn app.server:app --host 127.0.0.1 --port 8000
Serves the built React app (app/web/dist) and JSON views of results/ (falls back to
app/fixtures/, flagged synthetic). Binds to localhost only; no outbound calls.
Local-data mode (brain images) only when env ANN_LOCAL_DATA=1.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
DIST = Path(__file__).resolve().parent / "web" / "dist"
LOCAL_DATA = os.environ.get("ANN_LOCAL_DATA") == "1"
SAFE = re.compile(r"^[A-Za-z0-9_.\-]+$")

app = FastAPI(title="ANN AD/MCI dashboard", docs_url=None, redoc_url=None)


def _locate(rel: str) -> tuple[Path | None, bool]:
    real = RESULTS / rel
    if real.exists():
        return real, False
    if os.environ.get("ANN_FIXTURES") == "1":          # synthetic fixtures only on explicit request (dev)
        fake = FIXTURES / rel
        if fake.exists():
            return fake, True
    return None, False


def _table(rel: str, **filters) -> dict:
    path, synthetic = _locate(rel)
    if path is None:
        return {"source": "missing", "synthetic": False, "rows": []}
    df = pd.read_csv(path)
    for k, v in filters.items():
        if v is not None and k in df.columns:
            df = df[df[k].astype(str) == str(v)]
    df = df.astype(object).where(pd.notna(df), None)
    return {"source": str(path.relative_to(ROOT)).replace("\\", "/"), "synthetic": synthetic,
            "rows": df.to_dict(orient="records")}


def _git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return "unknown"


@app.get("/api/meta")
def meta():
    return {"git_hash": _git_hash(), "local_data": LOCAL_DATA,
            "title": "When Explanations Drift: Label-Aware Cross-Cohort Evaluation and "
                     "Attribution-Based Failure Detection for MRI-Based Early Alzheimer's Detection",
            "disclaimer": "Research prototype. Not a medical device. Not for clinical use."}


@app.get("/api/table/{name}")
def table(name: str, request: Request):
    if not SAFE.match(name):
        raise HTTPException(400, "bad name")
    return _table(f"{name}.csv", **dict(request.query_params))


@app.get("/api/file/{folder}/{name}")
def subtable(folder: str, name: str):
    allowed = {"curves", "confusion", "training_curves", "roi_attribution", "risk_coverage"}
    if folder not in allowed or not SAFE.match(name):
        raise HTTPException(400, "bad path")
    return _table(f"{folder}/{name}.csv")


@app.get("/api/list/{folder}")
def listing(folder: str):
    allowed = {"curves", "confusion", "training_curves", "roi_attribution", "risk_coverage"}
    if folder not in allowed:
        raise HTTPException(400, "bad folder")
    names, synthetic = set(), False
    for base, syn in ((RESULTS, False), (FIXTURES, True)):
        d = base / folder
        if d.exists():
            found = {p.stem for p in d.glob("*.csv")}
            if found and not names:
                synthetic = syn
            names |= found if not names else set()
    return {"synthetic": synthetic, "names": sorted(names)}


@app.get("/api/status")
def status():
    path, synthetic = _locate("pipeline_status.json")
    if path is None:
        return {"synthetic": False, "status": {}}
    return {"synthetic": synthetic, "status": json.loads(path.read_text(encoding="utf-8"))}


@app.post("/api/export")
async def export_figure(request: Request):
    body = await request.json()
    name = body.get("name", "figure")
    if not SAFE.match(name):
        raise HTTPException(400, "bad name")
    out = RESULTS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    data = body.get("svg")
    if not data:
        raise HTTPException(400, "no svg")
    (out / f"{name}.svg").write_text(data, encoding="utf-8")
    return {"saved": f"results/figures/{name}.svg"}


# ---------------- local-data mode only ----------------
@app.get("/api/scans")
def scans():
    if not LOCAL_DATA:
        raise HTTPException(403, "local-data mode is off")
    rows = []
    for cohort_dir in (ROOT / "data" / "preprocessed").glob("*"):
        for p in sorted(cohort_dir.glob("*.nii.gz"))[:400]:
            rows.append({"cohort": cohort_dir.name, "key": p.name[:-7]})
    return {"rows": rows}


@app.get("/api/scan/{cohort}/{key}")
def scan(cohort: str, key: str):
    if not LOCAL_DATA:
        raise HTTPException(403, "local-data mode is off")
    if not (SAFE.match(cohort) and SAFE.match(key)):
        raise HTTPException(400, "bad path")
    p = ROOT / "data" / "preprocessed" / cohort / f"{key}.nii.gz"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="application/gzip", filename=f"{key}.nii.gz")


@app.get("/api/maps")
def maps():
    d = RESULTS / "maps"
    return {"names": sorted(p.name[:-7] for p in d.glob("*.nii.gz")) if d.exists() else []}


@app.get("/api/map/{name}")
def map_file(name: str):
    if not SAFE.match(name):
        raise HTTPException(400, "bad name")
    p = RESULTS / "maps" / f"{name}.nii.gz"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="application/gzip", filename=f"{name}.nii.gz")


@app.get("/api/template")
def template():
    return FileResponse(ROOT / "references" / "tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz",
                        media_type="application/gzip", filename="mni.nii.gz")



# ---------------- scan analysis (local only) ----------------
import shutil
import uuid
from fastapi import File, UploadFile

JOBS = ROOT / "data" / "app_jobs"
GPU_PY = next((p for p in (ROOT / ".venv-gpu" / "Scripts" / "python.exe", ROOT / ".venv-gpu" / "bin" / "python") if p.exists()),
              ROOT / ".venv-gpu" / "Scripts" / "python.exe")
EXAMPLES = ROOT / "data" / "app_examples.json"


def _start_job(src: Path) -> dict:
    jid = uuid.uuid4().hex[:12]
    job = JOBS / jid
    job.mkdir(parents=True, exist_ok=True)
    (job / "status.json").write_text(json.dumps({"step": "Queued", "pct": 2}), encoding="utf-8")
    subprocess.Popen([str(GPU_PY), str(ROOT / "src" / "analyze_scan.py"), str(src), str(job)], cwd=ROOT,
                     stdout=open(job / "log.txt", "wb"), stderr=subprocess.STDOUT)
    return {"job": jid}


@app.get("/api/examples")
def examples():
    ex = json.loads(EXAMPLES.read_text()) if EXAMPLES.exists() else []
    return [{k: e[k] for k in ("id", "label", "truth", "age", "sex")} for e in ex if Path(e["path"]).exists()]


@app.post("/api/analyze/example/{eid}")
def analyze_example(eid: str):
    ex = {e["id"]: e for e in json.loads(EXAMPLES.read_text())} if EXAMPLES.exists() else {}
    if eid not in ex:
        raise HTTPException(404)
    return _start_job(Path(ex[eid]["path"]))


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    name = (file.filename or "").lower()
    if not (name.endswith(".nii") or name.endswith(".nii.gz")):
        raise HTTPException(400, "Please upload a NIfTI file (.nii or .nii.gz)")
    up = JOBS / "_uploads"
    up.mkdir(parents=True, exist_ok=True)
    dest = up / (uuid.uuid4().hex[:12] + (".nii.gz" if name.endswith(".gz") else ".nii"))
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    return _start_job(dest)


@app.get("/api/job/{jid}")
def job_status(jid: str):
    if not SAFE.match(jid):
        raise HTTPException(400)
    job = JOBS / jid
    if not job.exists():
        raise HTTPException(404)
    st = json.loads((job / "status.json").read_text(encoding="utf-8"))
    if (job / "result.json").exists():
        st["result"] = json.loads((job / "result.json").read_text(encoding="utf-8"))
    return st


@app.get("/api/job/{jid}/{name}")
def job_file(jid: str, name: str):
    if not (SAFE.match(jid) and name in {"brain", "brain_disp", "attr_cn_ad", "attr_cn_mci", "gradcam_cn_ad", "gradcam_cn_mci"}):
        raise HTTPException(400)
    p = JOBS / jid / f"{name}.nii.gz"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="application/gzip", filename=f"{name}.nii.gz")

if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = DIST / path
        if path and f.is_file():
            return FileResponse(f)
        return FileResponse(DIST / "index.html")
else:
    @app.get("/")
    def root():
        return JSONResponse({"msg": "frontend not built; run npm run build in app/web"})
