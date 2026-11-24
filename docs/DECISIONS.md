# Decision log
One line per decision: date | decision | reason.

- 2026-11-18 | Cohorts: ADNI (train/val/internal test), OASIS-3 (locked external test), AIBL (second external test, access confirmed by user) | Cross-cohort claim needs >=1 independent external cohort; AIBL strengthens C1.
- 2026-11-18 | Kaggle Alzheimer's MRI set not used | 2D slices, single source, subject leakage, no MCI.
- 2026-11-18 | Affine (not SyN) registration to MNI152NLin2009cAsym for model inputs | SyN warps away atrophy signal; affine is the standard CNN setting (Wen et al. 2020).
- 2026-11-18 | Train only on ADNI clinical diagnosis; evaluate on 3 label definitions in every cohort | Avoids 3x run multiplication on a 4 GB GPU; isolates label-definition shift at evaluation.
- 2026-11-18 | No scanner-vendor filtering in ADNI selection | Scanner diversity is part of the research question.
- 2026-11-18 | ADNI download in batches: A = ADNI-2/3 3T baselines (~500, stratified) first, B = remainder | Start preprocessing/debugging while downloads continue.
- 2026-11-18 | New `.venv-gpu` with torch 2.11.0+cu128 (newest CUDA build on the cu128 index); `.venv-ml` untouched | 2.14.1 has no CUDA wheel there; keeps working env stable.
- 2026-11-18 | Dashboard: React + Vite + TS + Tailwind + Plotly.js + NiiVue, served by FastAPI on 127.0.0.1 (no Streamlit) | User found Streamlit UI poor; this stack is free, offline, professional.
- 2026-11-18 | Git tracks code only; status/handoff files with subject IDs excluded | ADNI/OASIS DUAs; no subject IDs in tracked files.
- 2026-11-18 | IDA accessed through user's Chrome (Claude in Chrome) | Cloudflare blocked the embedded browser pane.
- 2026-11-18 | AIBL: user's IDA account has no AIBL access; user is applying. Pipeline built cohort-agnostic so AIBL can be added later as 2nd external test | Not on critical path; paper stands on ADNI->OASIS-3.
- 2026-11-18 | ADNI tables taken from ADNIMERGE2 R package (read with pyreadr) | Individual table downloads failed due to ISP public-IP rotation; ADNIMERGE2 contains all needed tables.
- 2026-11-18 | ADNI images fetched with src/fetch_urls.py (HTTP range-resume + retries) from IDA pre-signed URL lists, batch A split into 10 zips | Chrome downloads failed on IP changes; resumable CLI survives them.
- 2026-11-18 | Baseline image = earliest T1 date per subject; prefer non-repeat, non-_ND series; dx from DXSUM within +/-180 d | Leakage-free one-scan-per-subject; 3,153 subjects selected (A=600, B=2,553).
- 2026-11-19 | AIBL application already submitted by user (status PENDING, PI = user, proposed analysis = external validation). If approved: AIBL Management Committee must review manuscripts (<=2 weeks) before submission; required author-line + acknowledgment text | Keeps AIBL optional; paper timeline must add 2 weeks if AIBL used.
- 2026-11-19 | Preprocessing: if affine template-correlation < 0.3, search 24 axis-aligned initial orientations at 3 mm and keep best; scans still < 0.3 are excluded by QC | 116/839 OASIS-3 'Vision' scans have sagittal data labelled axial in the header; plain affine fails on them.
- 2026-11-19 | Long jobs launched from the user's PowerShell terminal (Start-Process) instead of the Claude session | A session interruption killed all session-spawned background jobs at 00:38.
- 2026-11-19 | ROI atlas: Harvard-Oxford cortical (L/R split, 96) + 16 subcortical incl. hippocampus/amygdala (nilearn, FSL MNI152) resampled to MNI2009c 1 mm, instead of CerebrA | Freely fetchable offline-cacheable; amendment made before any attribution analysis.
- 2026-11-19 | ROI attribution = mean |IG| per voxel within ROI (density), normalised to sum 1 | Sum-per-ROI is dominated by ROI size; amendment before any trained-model attribution.
- 2026-11-19 | Frozen baselines re-evaluated on the complete OASIS-3 cohort (1,376 scans; earlier run used the 839 then downloaded). Same frozen config, no changes; this run supersedes the partial one | Test-set completeness, not tuning.
- 2026-11-19 | GPU jobs serialised: preprocessing pauses while a training run is active | Concurrent HD-BET + training on the 4 GB GPU thrashed (epoch 1.3->4.4 min, scan 40->260 s).
- 2026-11-19 | Final training/eval at 2 mm (80x96x88) instead of 1.5 mm; resnet3d 5 seeds, MedicalNet 3 seeds | 4 GB GPU: 1.5 mm would need ~3 h/run (~60 h for 20 runs); 2 mm is ~2x faster. 1.5 mm run kept as Phase I prototype (proto15_*).
- 2026-11-19 | Four CNN configs frozen in PROTOCOL.md before any external CNN evaluation; orchestrator runs training queue then frozen OASIS-3 evaluation (+BN-adapt/TENT), XAI/ADRS, shift decomposition, paper assets per config | Autonomous overnight execution with test-set lock intact.
- 2026-11-23 | Phase-I prototype (for Wednesday review): proto_-tagged runs trained on the 2,019 ADNI scans preprocessed so far; evaluated once on a stratified OASIS-3 subset. Preliminary only: no config change may be motivated by it; final paper uses full queue + full OASIS-3 | Course deadline; test-set lock respected (frozen configs, nothing tuned).
- 2026-11-24 | Demo-only: XAI runs (both tasks) with 16 IG steps / 10 MC-dropout passes (protocol: 32/20) to finish in time on a shared 4 GB GPU | Final study uses protocol settings.
