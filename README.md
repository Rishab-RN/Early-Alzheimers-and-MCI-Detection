# Explainable Deep Learning model for early Alzheimer's detection using MRI images. 

### Cross-Cohort Generalisation, Predictive Reliability and Explanation Robustness in Deep Learning for Alzheimer's Disease and Mild Cognitive Impairment Classification from Structural MRI.

<p align="center">
  <strong>Research Project | Deep Learning | Medical Image Analysis | Explainable AI</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python 3.11">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/MONAI-Medical%20Imaging-00a98f" alt="MONAI">
  <img src="https://img.shields.io/badge/Research-In%20Progress-orange" alt="Research in progress">
</p>

---

## Overview

Deep learning has shown considerable potential in analysing structural brain MRI for Alzheimer's disease classification. However, classification performance alone does not establish whether a model can be trusted.

A model may produce a confident prediction despite being incorrect. Its performance may deteriorate when evaluated on data collected using different scanners, imaging protocols, or patient populations. Even when two models make the same prediction, they may rely on substantially different anatomical information.

This project investigates the reliability of deep learning models for Alzheimer's disease classification using three-dimensional T1-weighted brain MRI.

Rather than focusing exclusively on classification accuracy, the study examines the relationship between predictive uncertainty and explanation stability. It investigates whether combining these signals can help identify unreliable predictions, particularly when a model is evaluated across different imaging cohorts.

The research uses the Alzheimer's Disease Neuroimaging Initiative (ADNI) as the proposed development cohort and OASIS-3 as the proposed independent external validation cohort.

## Research Question

**Can predictive uncertainty and explanation instability jointly identify unreliable Alzheimer's disease MRI classifications under cross-cohort distribution shift, and does their combination improve error detection compared with either signal independently?**

## Research Objectives

1. Develop deep learning models for three-class classification of structural brain MRI into Cognitively Normal (CN), Mild Cognitive Impairment (MCI), and Alzheimer's Disease (AD) groups.
2. Evaluate model generalisation across independently collected imaging cohorts.
3. Investigate calibration and predictive uncertainty under changes in cohort distribution.
4. Measure the consistency of model explanations across training seeds, input perturbations, and architectures.
5. Examine anatomical feature contributions using atlas-based regional analysis and SHAP.
6. Evaluate whether combining uncertainty and explanation instability improves error detection and selective prediction.


---

## Dataset Selection

### 1. Alzheimer's Disease Neuroimaging Initiative (ADNI)

**Role:** Model development and internal evaluation.

ADNI is a longitudinal research initiative containing neuroimaging, clinical, cognitive, and demographic information collected across multiple study phases.

The proposed study will use eligible T1-weighted structural MRI scans and corresponding clinical information.

Official resource: https://adni.loni.usc.edu/

### 2. Open Access Series of Imaging Studies (OASIS-3)

**Role:** Independent external evaluation.

OASIS-3 provides longitudinal neuroimaging and clinical information from participants across different stages of cognitive health and decline.

Its use as an external cohort allows the study to examine whether model reliability changes when evaluated on data collected independently of the development cohort.

Official resource: https://sites.wustl.edu/oasisbrains/home/oasis-3/

### Dataset Strategy

| Dataset | Research role | Proposed use |
|---|---|---|
| ADNI | Development cohort | Training, validation and internal testing |
| OASIS-3 | External cohort | Frozen external evaluation |

**Clinical label harmonisation is a critical part of the study.** ADNI diagnostic categories and OASIS-3 clinical assessments cannot be assumed to be directly interchangeable. MRI sessions must be matched to suitable clinical assessments, and ambiguous cases must be handled using predefined eligibility criteria.

The final sample counts will be reported after data access, cohort construction, and quality control.

> **Data access status:** Applications and authorisations are pending. No claim is made that the datasets have already been acquired or processed.

### Data Governance

Both datasets are subject to their respective access agreements and research-use restrictions. Raw imaging data, participant-level information, and access credentials will not be committed to this repository.

---

## Proposed Research Methodology

The methodology is organised into eight stages.

### Stage 1 — Cohort Construction

- Identify eligible T1-weighted MRI sessions.
- Retrieve corresponding clinical assessments.
- Establish CN, MCI, and AD group definitions.
- Match imaging sessions with appropriate clinical assessments.
- Apply predefined inclusion and exclusion criteria.
- Construct participant-level development and evaluation partitions.

### Stage 2 — MRI Preprocessing

The proposed preprocessing pipeline consists of:

1. **Image conversion:** Convert DICOM images into NIfTI format when required.
2. **Orientation standardisation:** Establish a consistent image orientation.
3. **Bias-field correction:** Evaluate N4 correction to address intensity non-uniformity.
4. **Brain extraction:** Remove non-brain tissue when required by the selected pipeline.
5. **Spatial registration:** Align images to a common MNI152 reference space.
6. **Resampling:** Standardise voxel spacing, initially targeting approximately 2 mm isotropic resolution.
7. **Intensity normalisation:** Apply a consistent normalisation procedure.
8. **Quality control:** Inspect registration, image dimensions, extraction quality, and corrupted scans.
9. **Data augmentation:** Apply controlled transformations exclusively to training samples.

The proposed model input is a single-channel volumetric MRI of approximately 96 × 96 × 96 voxels. Final dimensions and preprocessing parameters will be confirmed through pilot experiments.

### Stage 3 — Deep Learning Architecture

Three model architectures are planned.

| Architecture | Purpose |
|---|---|
| Custom 3D CNN | Baseline volumetric feature learning |
| 3D ResNet-18 | Primary architecture for reliability evaluation |
| Compact 3D DenseNet | Supporting architectural comparison |

Three-dimensional convolutions preserve spatial relationships across the axial, coronal, and sagittal dimensions of MRI volumes.

The initial training configuration includes:

- AdamW optimiser.
- Cosine learning-rate scheduling.
- Class-weighted cross-entropy.
- Validation-based early stopping.
- Mixed-precision training where supported.
- Multiple independent training seeds.

### Stage 4 — Calibration and Uncertainty Estimation

Classification confidence and prediction reliability are not equivalent.

The study will investigate:

**Calibration**
- Temperature scaling.
- Expected Calibration Error (ECE).
- Brier score.
- Negative Log-Likelihood (NLL).
- Reliability diagrams.

**Predictive uncertainty**
- Maximum Softmax Probability.
- Predictive Entropy.
- Monte Carlo Dropout as an optional extension.

Temperature scaling will be fitted using ADNI validation data and frozen before external evaluation.

### Stage 5 — Explainability Analysis

Two complementary explainability approaches are proposed.

**A. Spatial attribution**

Integrated Gradients or 3D Grad-CAM will be used to investigate which spatial regions influence model predictions.

The analysis will examine:
- Attribution patterns across training seeds.
- Sensitivity to controlled input perturbations.
- Similarity between explanations produced by different architectures.
- Differences between correct and incorrect predictions.

**B. Regional SHAP analysis**

A predefined anatomical atlas will be used to derive regional MRI measurements.

The proposed analysis includes:
- Anatomical region extraction.
- Construction of regional feature representations.
- Training of a separate regional-feature classifier.
- SHAP-based feature contribution analysis.
- Comparison of influential regions across clinical groups.

SHAP explanations from the regional-feature classifier will be interpreted separately from voxel-level explanations produced by the 3D CNN. They will not be presented as explanations of the CNN's internal decisions.

### Stage 6 — Explanation Stability

Explanation stability will be studied through three comparisons:

| Analysis | Objective |
|---|---|
| Seed stability | Examine attribution consistency across independently trained models |
| Perturbation stability | Evaluate explanation changes under controlled input modifications |
| Architecture stability | Compare attribution patterns across different model architectures |

Regional SHAP consistency will be investigated as a complementary analysis.

### Stage 7 — Joint Reliability Assessment

The central experiment investigates whether predictive uncertainty and explanation instability provide complementary information.

Let:

- \(U\) represent predictive uncertainty.
- \(I\) represent explanation instability.

Three approaches will be compared:

\[
S_1 = U
\]

\[
S_2 = I
\]

\[
S_3 = \alpha U + \beta I
\]

The signals will be standardised using ADNI validation statistics. Any combination weights will be fixed or estimated using development data only.

The comparison will investigate whether the combined score identifies classification errors more effectively than either individual signal.

SHAP regional importance will remain a complementary interpretation analysis rather than being automatically incorporated into the primary reliability score.

### Stage 8 — External Evaluation

After model selection and development-stage analysis, the final configuration will be frozen.

The external evaluation will use OASIS-3 without tuning model weights, calibration parameters, thresholds, or preprocessing settings based on external test results.

This stage will assess the transferability of classification and reliability characteristics across cohorts.

---

## Evaluation Metrics

| Category | Metrics |
|---|---|
| Classification | Balanced Accuracy, Macro-F1, Macro AUROC |
| Class-wise performance | Precision, Recall, Confusion Matrix |
| Calibration | ECE, Brier Score, NLL |
| Uncertainty | Error-detection AUROC |
| Selective prediction | AURC, Risk at 80% Coverage |
| Explanation stability | Spatial similarity and attribution consistency |
| Regional explainability | SHAP feature contributions and consistency |
| Statistical analysis | Participant-level bootstrap confidence intervals |

Participant-level evaluation and statistical resampling will be used to avoid treating multiple scans or slices from the same participant as independent observations.

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python 3.11 |
| Deep learning | PyTorch |
| Medical imaging | MONAI |
| MRI file handling | nibabel |
| Image registration | ANTs / ANTsPy |
| DICOM conversion | dcm2niix |
| Model interpretation | Captum |
| SHAP analysis | SHAP |
| Neuroimaging analysis | Nilearn |
| Numerical computing | NumPy, SciPy |
| Data processing | pandas |
| Evaluation | scikit-learn |
| Visualisation | Matplotlib |
| Experiment tracking | MLflow / structured experiment logs |
| Compute environment | Google Colab GPU, subject to data-use approval |

---

## Experimental Design

The planned experiments are organised as follows:

| Experiment | Investigation |
|---|---|
| E1 | Internal classification performance on ADNI |
| E2 | External performance evaluation on OASIS-3 |
| E3 | Calibration transfer across cohorts |
| E4 | Uncertainty-based error detection |
| E5 | Explanation stability across training seeds |
| E6 | Regional SHAP attribution analysis |
| E7 | Combined uncertainty and explanation-instability analysis |
| E8 | Selective prediction under distribution shift |

The primary research comparison is the error-detection performance of the combined reliability signal against uncertainty alone on the frozen external cohort.

---

## Reproducibility and Leakage Prevention

Reliability analysis requires strict separation between model development and external evaluation.

The following controls are planned:

- Participant-level dataset splitting.
- No participant overlap between training, validation, and internal testing.
- All scans from the same participant remain in one partition.
- Preprocessing procedures are applied consistently.
- Data augmentation is restricted to training samples.
- Calibration parameters are fitted using validation data only.
- External cohort evaluation remains frozen.
- Random seeds, model configurations, and checkpoints are recorded.
- Raw restricted datasets are excluded from version control.

---

## Project Timeline

The project is planned over twelve weeks.

| Phase | Timeline | Activities |
|---|---|---|
| Research setup | Week 1 | Literature review, access applications, protocol definition |
| Cohort construction | Week 2 | Clinical label mapping and data inventory |
| Preprocessing | Weeks 3–4 | MRI preprocessing and feasibility testing |
| Model development | Weeks 5–7 | Architecture implementation and training |
| Reliability analysis | Weeks 8–9 | Uncertainty, attribution, SHAP and model freeze |
| External evaluation | Week 10 | Frozen OASIS-3 evaluation, subject to access |
| Statistical analysis | Week 11 | Joint reliability experiments |
| Research documentation | Week 12 | Paper, presentation and final documentation |

The schedule depends on timely dataset authorisation and available compute resources.

---

## Expected Research Contributions

The study aims to contribute:

1. A systematic evaluation of classification reliability beyond conventional performance metrics.
2. An investigation of calibration and predictive uncertainty under cross-cohort distribution shift.
3. A quantitative analysis of explanation stability across training seeds and architectures.
4. A complementary regional interpretation analysis using SHAP.
5. An evaluation of whether uncertainty and explanation instability jointly improve error detection.
6. A reproducible methodology for assessing reliability in volumetric medical-image classification.

These are intended contributions, not completed findings.

## Limitations

The study recognises several limitations:

- Dataset access and data-use approvals may affect the implementation timeline.
- Clinical diagnostic labels may differ between cohorts.
- MRI acquisition differences can introduce distribution shift.
- Explanation stability does not establish clinical correctness.
- SHAP regional analysis uses a separate feature-based model and may not reflect the internal reasoning of the 3D CNN.
- The combined reliability score may not outperform uncertainty alone.
- External validation on one independent cohort does not establish clinical generalisability.

The study is intended for research evaluation and is not a clinical diagnostic system.

---

## Repository Structure

```text
alzheimer-mri-reliability/
│
├── configs/
│   ├── data.yaml
│   ├── preprocessing.yaml
│   ├── model.yaml
│   └── experiment.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_mri_quality_control.ipynb
│   ├── 03_preprocessing_validation.ipynb
│   └── 04_explainability_analysis.ipynb
│
├── src/
│   ├── data/
│   │   ├── cohort_builder.py
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   │
│   ├── models/
│   │   ├── cnn3d.py
│   │   ├── resnet3d.py
│   │   └── densenet3d.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── calibration.py
│   │
│   ├── uncertainty/
│   │   └── estimation.py
│   │
│   ├── explainability/
│   │   ├── integrated_gradients.py
│   │   ├── gradcam.py
│   │   └── shap_analysis.py
│   │
│   ├── reliability/
│   │   ├── explanation_stability.py
│   │   ├── reliability_score.py
│   │   └── selective_prediction.py
│   │
│   └── evaluation/
│       ├── metrics.py
│       └── statistical_analysis.py
│
├── experiments/
│   ├── configs/
│   └── logs/
│
├── reports/
│   ├── figures/
│   └── tables/
│
├── tests/
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

This structure describes the intended organisation; directories and files will be added as implementation progresses.

---

## Getting Started

### Prerequisites

- Python 3.11
- Git
- PyTorch-compatible compute environment
- Google Colab GPU or an approved alternative
- Authorised access to the selected datasets

### Environment Setup

Clone the repository:

```bash
git clone https://github.com/Rishab-RN/Early-Alzheimers-and-MCI-Detection.git
cd alzheimer-mri-reliability
```

Create a Python environment:

```bash
conda create -n alzheimer-mri python=3.11
conda activate alzheimer-mri
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the appropriate PyTorch build for the available hardware using the official installation instructions:

https://pytorch.org/get-started/locally/

### Dataset Configuration

Place authorised data in the appropriate local or institution-approved storage location.

Update the configuration files with the relevant data paths.

Do not upload raw MRI scans, identifiable clinical information, or dataset credentials to GitHub.

### Running Experiments

Training and evaluation commands will be documented as the corresponding modules are implemented.


---

## References and Technical Resources

1. ADNI — https://adni.loni.usc.edu/
2. OASIS-3 — https://sites.wustl.edu/oasisbrains/home/oasis-3/
3. MONAI — https://monai.io/
4. PyTorch — https://pytorch.org/
5. Captum — https://captum.ai/
6. SHAP — https://shap.readthedocs.io/
7. Nilearn — https://nilearn.github.io/

Research references will be expanded as the literature review is finalised.

---

## Project Status

**Current stage: Research planning and methodology finalisation.**

Dataset applications, cohort construction, preprocessing validation, and model implementation are pending.

The repository will be updated as individual research components are implemented and experimentally validated.

---

<p align="center">
  <strong>Beyond classification performance: investigating when and why medical imaging models can be trusted.</strong>
</p>