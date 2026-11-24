# Analysis protocol (pre-registered, 2026-11-18)

Changes after this date are allowed only with a dated entry in "Amendments" and in DECISIONS.md.
No amendment may be motivated by OASIS-3 or AIBL test results.

## Hypotheses
- **H1 (C1, shift decomposition).** The ADNI->external performance drop is partly explained by label-definition shift: the drop is smaller under clinical-dx and amyloid-confirmed labels than under CDR-proxy, and smaller on age/sex-matched and 3T subsets.
- **H2 (C2, explanation drift).** ROI-attribution profiles differ between ADNI and external cohorts (cross-cohort Spearman below internal split-half agreement); test-time adaptation improves accuracy more than it restores attribution agreement.
- **H3 (C3, ADRS).** The Mahalanobis atypicality of a scan's ROI-attribution profile (vs ADNI validation) detects misclassifications on external cohorts with AUROC above max-softmax probability.
- **H4 (C4).** The 3D CNN beats age-only and FreeSurfer hippocampal-volume+age baselines for CN vs MCI and CN vs AD, internally and externally.

## Cohorts and roles
| Cohort | Role | Rule |
|---|---|---|
| ADNI 1/GO/2/3 | train / val / internal test (70/15/15) | one baseline T1 per subject; subject-level split stratified by dx, age bin, sex, field strength |
| OASIS-3 | external test, LOCKED | never used for training, model selection, thresholds, temperature or TTA settings |
| AIBL | external test, LOCKED | same rules as OASIS-3 |

## Tasks
T1 CN vs AD; T2 CN vs MCI; stretch T3 sMCI vs pMCI (conversion within 36 months).

## Labels
- **Training label:** ADNI clinical diagnosis at the scan visit: CN; MCI = EMCI + LMCI; AD = dementia. SMC excluded.
- **Evaluation label definitions (every cohort where available):**
  1. `cdr`: global CDR 0 -> CN, 0.5 -> MCI-like, >=1 -> AD-like.
  2. `clinical`: ADNI DX; OASIS-3 UDSd1 (normal cognition / MCI / dementia with AD primary etiology); AIBL clinical diagnosis.
  3. `amyloid`: clinical label restricted to biomarker-concordant cases (CN amyloid-negative; MCI/AD amyloid-positive). Cutoffs fixed a priori: PET Centiloid >= 20 positive; CSF Abeta42 (Elecsys) < 980 pg/mL positive.
- Scan-to-assessment matching window +/-365 days (sensitivity analysis +/-180).

## Preprocessing (identical for all cohorts)
N4 -> HD-BET -> affine registration to MNI152NLin2009cAsym 1 mm -> robust intensity normalisation -> resample (1.5 or 2 mm, chosen on ADNI val) -> fixed crop. Auto-QC exclusion by brain-mask volume and registration-similarity outliers (thresholds set on ADNI only).

## Models
- M1 light 3D ResNet from scratch; M2 MedicalNet ResNet-10 fine-tuned; 5 seeds each (= deep ensemble).
- MONAI, AMP, class-balanced sampling, early stopping on ADNI val AUROC; all hyperparameters chosen on ADNI val.
- Baselines: B1 age-only logistic regression; B2 FreeSurfer hippocampal volume/ICV + age logistic regression.
- Temperature scaling and decision threshold (max balanced accuracy) fitted on ADNI val.

## Test-time adaptation (frozen)
- BN-adapt: recompute BatchNorm statistics on the full unlabelled target cohort in one pass (cumulative average).
- TENT: entropy minimisation, BN affine params only, Adam lr 1e-4, 1 epoch, batch 4.
- Sanity-checked only on an internal ADNI 1.5T -> 3T shift.

## Explainability (C2)
- Integrated Gradients (zero baseline in normalised space, 32 steps) and Grad-CAM (last conv block).
- ROI aggregation on CerebrA + Harvard-Oxford subcortical (MNI2009cAsym); mean |attribution| per ROI, normalised to sum 1.
- A-priori AD ROIs: hippocampus, amygdala, entorhinal, parahippocampal, inferior/middle temporal, fusiform.
- Metrics: cross-cohort Spearman, Jensen-Shannon divergence, AD-prior attribution mass, ROI-occlusion deletion AUC, cascading model-randomisation similarity.

## Reliability (C3)
- ADRS = Mahalanobis distance of the (log-ratio transformed) ROI-attribution profile to the ADNI validation distribution of the predicted class, Ledoit-Wolf covariance.
- Comparators: MSP, energy, MC-dropout (20 passes), ensemble disagreement, penultimate-feature Mahalanobis.
- Metrics: misclassification-detection AUROC, AURC, accuracy at 80%/90% coverage.

## Statistics
AUROC, balanced accuracy, accuracy, precision, recall, F1, ECE (15 bins); 95% CIs from 2,000 subject-level bootstraps; DeLong for AUROC comparisons. Subsets: all, age/sex-matched, 1.5T, 3T. Each external result logged once per frozen configuration with git hash.

## Frozen configurations
| config_id | model | task | frozen on | git hash |
|---|---|---|---|---|
| resnet3d_cn_ad | 3D ResNet-18 (width 0.5), scratch, 2 mm, AdamW lr 3e-4 wd 1e-4, cosine, bs 4 x accum 2, <=60 ep, patience 12, seeds 0-4 | CN vs AD | 2026-11-19 | see git log |
| resnet3d_cn_mci | same as resnet3d_cn_ad | CN vs MCI | 2026-11-19 | see git log |
| medicalnet_cn_ad | MedicalNet ResNet-10 fine-tuned (lr x0.3), otherwise as above, seeds 0-2 | CN vs AD | 2026-11-19 | see git log |
| medicalnet_cn_mci | same as medicalnet_cn_ad | CN vs MCI | 2026-11-19 | see git log |
| mlp_cn_ad | MLP (256-64) on 8 mm voxels, PCA 128, early stop on ADNI val AUROC (faculty Level 1 ANN baseline) | CN vs AD | 2026-11-23 | see git log |
| mlp_cn_mci | same as mlp_cn_ad | CN vs MCI | 2026-11-23 | see git log |

## Amendments
(none)

## Frozen baseline configurations (2026-11-19, before any external evaluation)
| config_id | model | task | frozen on |
|---|---|---|---|
| age_only_cn_ad / age_only_cn_mci | logistic regression on age (balanced, standardised) | CN vs AD / CN vs MCI | 2026-11-19 |
| hv_age_cn_ad / hv_age_cn_mci | logistic regression on (L+R hippocampus)/ICV x1000 + age | CN vs AD / CN vs MCI | 2026-11-19 |
Fit on ADNI train (clinical dx), threshold on ADNI val; evaluated once on ADNI test and OASIS-3.
Age-matched subset: 1:1 nearest-age matching within sex, caliper 2 years.

## Amendments
- 2026-11-19: ROI atlas = Harvard-Oxford cortical+subcortical (references/atlas_ho_mni2009c_1mm.nii.gz) instead of CerebrA + HO-subcortical. Made before any attribution was computed.
- 2026-11-19: Shift decomposition (C1) fixed as sequential matching on frozen-ensemble predictions: total drop D0 (clinical, all) -> D1 (amyloid labels) -> D2 (+ age/sex matching) -> D3 (+ 3T only); components label_definition=D0-D1, age_sex=D1-D2, field_strength=D2-D3, residual=D3; 1,000 joint subject bootstraps. Specified before any CNN external evaluation (src/shift_decomposition.py).
