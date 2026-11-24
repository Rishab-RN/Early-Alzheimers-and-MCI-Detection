# NeuroLens dashboard

Local, offline web app: upload a T1 MRI and get CN/AD and CN/MCI risk with an explanation, plus pages for model performance, explainability, reliability and data.

## Run

```powershell
.\run_dashboard.bat
```

Opens http://127.0.0.1:8000 (localhost only, no outbound calls). The prebuilt frontend is in `app/web/dist`, so Node is only needed if you change the UI.

## Architecture

- `app/server.py`: FastAPI. Read-only over `results/` (column specs in `results/CONTRACT.md`). Scan analysis runs `src/analyze_scan.py` in `.venv-gpu` as a background job under `data/app_jobs/`.
- `app/web/`: React + Vite + TypeScript + Tailwind v4, Plotly.js charts, NiiVue MRI viewer.

## Example scans (optional)

Add `data/app_examples.json` to show one-click examples on the Analyse page. Entries whose file is missing are hidden.

```json
[{"id": "ex1", "label": "Example 1", "truth": "CN", "age": 70, "sex": "F", "path": "data/preprocessed/adni/<scan>.nii.gz"}]
```

## Rebuild the frontend

```powershell
cd app\web
npm install
npm run build
```
