# Results contract (v1)

`src/` scripts write these files; the dashboard (`app/`) only reads them. All tables are tidy CSV.
Everything except `predictions/` is aggregate-only (no subject IDs). The app falls back to
`app/fixtures/` (watermarked SYNTHETIC) for any file that does not exist yet.

| File | Columns |
|---|---|
| `manifest.json` | schema_version, generated_at, git_hash |
| `runs_index.csv` | run_id, timestamp, git_hash, model, task, label_def, seed, config_path, frozen, status |
| `metrics.csv` | run_id, cohort, task, label_def, subset, metric, value, ci_low, ci_high, n |
| `curves/{run_id}_{cohort}_{label_def}_{roc,pr,calib}.csv` | x, y, threshold |
| `confusion/{run_id}_{cohort}_{label_def}.csv` | true, pred, count |
| `training_curves/{run_id}.csv` | epoch, train_loss, val_loss, val_auc, lr |
| `cohort_summary.csv` | cohort, label_def, variable, group, count, mean, sd |
| `label_overlap.csv` | cohort, def_a, def_b, class_a, class_b, count |
| `leakage_checks.csv` | check, passed, detail |
| `shift_decomposition.csv` | task, component, value, ci_low, ci_high |
| `roi_attribution/{run_id}_{cohort}.csv` | roi, mean, sd, n, ad_prior |
| `xai_stats.csv` | run_id, comparison, metric, value, ci_low, ci_high |
| `reliability.csv` | run_id, method, cohort, metric, value, ci_low, ci_high |
| `risk_coverage/{run_id}_{cohort}_{method}.csv` | coverage, risk |
| `pipeline_status.json` | aggregate counts of downloads / preprocessing / QC per cohort |
| `predictions/{run_id}_{cohort}.csv` | key, y_true, p_*, adrs, ens_sd — LOCAL ONLY, git-ignored |

Vocabularies: cohort ∈ {adni, oasis3, aibl}; task ∈ {cn_ad, cn_mci, smci_pmci};
label_def ∈ {clinical, cdr, amyloid}; subset ∈ {all, age_matched, 1.5T, 3T};
model ∈ {age_only, hv_age, resnet3d, medicalnet, ensemble_resnet3d, ensemble_medicalnet}.
