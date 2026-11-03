# Research Journey

## Alzheimer's MRI Reliability Assessment

This document records the development of the research project, from its initial research question to implementation, experimentation, evaluation, and final findings.

It serves as a chronological research log, documenting methodological decisions, technical challenges, experimental observations, and changes made throughout the project.

The purpose is to maintain transparency and make the research process traceable and reproducible.

---

## 1. Research Motivation

### Initial Problem

Deep learning models can achieve promising results in medical image classification. However, classification performance alone does not establish whether their predictions are reliable.

This project began with an interest in investigating model reliability beyond conventional accuracy metrics, particularly under changes in imaging cohorts.

### Research Direction

The investigation focuses on three connected aspects:

- Predictive uncertainty.
- Explanation stability.
- Cross-cohort generalisation.

The central question is whether uncertainty and explanation instability provide complementary information when identifying potentially unreliable predictions.

---

## 2. Research Development

Record how the initial idea developed into the current research methodology.

| Milestone | Research development | Status |
|---|---|---|
| Initial exploration | Review of deep learning approaches for Alzheimer's MRI classification | Pending |
| Novelty investigation | Identify gaps in reliability, uncertainty and explainability research | Pending |
| Dataset selection | Evaluate ADNI and OASIS-3 for development and external evaluation | Selected; access pending |
| Methodology design | Define preprocessing, model architectures and evaluation strategy | Proposed |
| Reliability framework | Formulate uncertainty and explanation-instability comparison | Proposed |
| Implementation | Develop preprocessing and model pipelines | Pending |
| Evaluation | Conduct internal and external experiments | Pending |

Update this table using actual project progress.

---

## 3. Research Decisions

Record important decisions and their rationale.

### Decision 1 — Dataset Selection

**Selected:** ADNI and OASIS-3.

**Rationale:** ADNI will serve as the development cohort, while OASIS-3 will provide an independent external evaluation setting.

**Considerations:** Clinical label harmonisation, longitudinal scan matching, access permissions and cohort differences.

**Status:** Access applications pending.

### Decision 2 — Volumetric MRI Modelling

**Selected:** Three-dimensional CNN architectures.

**Rationale:** 3D convolutions preserve spatial information across the MRI volume rather than treating individual slices as independent observations.

### Decision 3 — Model Architectures

**Selected:**
- Custom 3D CNN.
- 3D ResNet-18.
- Compact 3D DenseNet.

**Rationale:** Compare a baseline convolutional architecture, residual learning and dense feature connectivity without unnecessarily expanding the experiment.

### Decision 4 — Explainability

**Selected:**
- Integrated Gradients or 3D Grad-CAM for spatial attribution.
- Atlas-based regional SHAP analysis for complementary feature interpretation.

**Rationale:** Investigate both spatial attribution and regional feature contributions while keeping their interpretations distinct.

### Decision 5 — Reliability Assessment

**Selected:** Compare uncertainty-only, explanation-instability-only and combined reliability signals.

**Rationale:** Determine whether explanation instability contributes information beyond predictive uncertainty.

---

## 4. Dataset and Preprocessing Progress

Record:
- Dataset application dates and approval status.
- Available imaging modalities.
- Clinical variables examined.
- Label mapping decisions.
- Participant inclusion and exclusion criteria.
- Scan-to-assessment matching rules.
- Preprocessing configurations.
- Quality-control observations.

Never include participant-identifiable information, restricted data, credentials or raw MRI files.

---

## 5. Implementation Log

Use chronological entries.

### Entry Template

**Date:**

**Research stage:**

**Objective:**

**Work completed:**

**Technical decisions:**

**Problems encountered:**

**How they were addressed:**

**Results or observations:**

**Next steps:**

**Relevant commits / experiment IDs:**

---

## 6. Experimental Record

Record every meaningful experiment.

| Experiment ID | Model | Dataset | Configuration | Outcome | Notes |
|---|---|---|---|---|---|
| EXP-001 | To be added | To be added | To be added | Pending | Pilot experiment |
| EXP-002 | To be added | To be added | To be added | Pending | Calibration |
| EXP-003 | To be added | To be added | To be added | Pending | Explainability |

Link detailed logs, plots and permitted derived results where appropriate.

Do not replace failed experiments with only successful results. Failed and inconclusive experiments are part of the research record.

---

## 7. Challenges and Lessons Learned

Document challenges such as:

- Dataset access delays.
- Clinical label inconsistencies.
- MRI preprocessing failures.
- GPU memory limitations.
- Model instability across random seeds.
- Calibration deterioration across cohorts.
- Explanation instability.
- Reproducibility issues.

For each challenge, record its cause, attempted solutions, final decision and remaining limitations.

---

## 8. Methodology Changes

Document every substantial change to the original protocol.

| Date | Original approach | Revised approach | Reason |
|---|---|---|---|
| To be updated | To be updated | To be updated | To be updated |

Distinguish planned changes from completed changes.

---

## 9. Research Milestones

- [ ] Research problem finalised.
- [ ] Literature review completed.
- [ ] Dataset applications submitted.
- [ ] Dataset access approved.
- [ ] Clinical label mapping validated.
- [ ] MRI preprocessing pipeline validated.
- [ ] Participant-level leakage checks completed.
- [ ] Baseline model trained.
- [ ] Model comparison completed.
- [ ] Calibration analysis completed.
- [ ] Predictive uncertainty analysis completed.
- [ ] Explanation stability analysis completed.
- [ ] Regional SHAP analysis completed.
- [ ] Joint reliability experiment completed.
- [ ] External evaluation completed, if access is approved.
- [ ] Statistical analysis completed.
- [ ] Research paper finalised.

---

## 10. Final Research Reflection

Complete this section after the experiments.

Discuss:
- What the experiments demonstrated.
- Which hypotheses were supported or rejected.
- Whether explanation instability added useful information.
- Whether the combined reliability score improved error detection.
- How external evaluation changed the conclusions.
- What limitations remain.
- What should be investigated next.

**Research principle:** Document what the experiments show, not what the project was expected to show.