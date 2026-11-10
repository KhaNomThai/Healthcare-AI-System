# Skin Cancer Multi-Class Classification & Screening System
## Clinical Decision Support Tool -- ISIC 2019 Dataset (8 Classes with Multi-Model Comparison)

An AI-driven **8-Class Disease Classification & Two-Level Cancer Screening System** for dermatoscopic skin lesion analysis using the ISIC 2019 dataset, featuring **Multi-Model Benchmark Comparison** across ResNet50, EfficientNet-B3, and DenseNet121.

**[Live Demo](https://kulachart2014-derma-ai.static.hf.space)** (runs entirely in your browser, images never leave your device) · **[Download Trained Models](https://github.com/KhaNomThai/Healthcare-AI-System/releases/latest)**

> [!IMPORTANT]
> This AI system is designed as an academic and research-oriented clinical decision support tool for preliminary screening only. It is **not** a substitute for professional medical diagnosis, consultation, or histopathological examination by certified dermatologists.

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Key Features](#2-key-features)
- [3. Project Structure](#3-project-structure)
- [4. Dataset: ISIC 2019](#4-dataset-isic-2019)
- [5. Disease Taxonomy & Clinical Categories](#5-disease-taxonomy--clinical-categories)
- [6. Two-Level Clinical Decision Architecture](#6-two-level-clinical-decision-architecture)
- [7. Multi-Model Architecture Comparison](#7-multi-model-architecture-comparison)
- [8. Empirical Benchmark & Validation Results](#8-empirical-benchmark--validation-results)
- [9. Technology Stack & Dependencies](#9-technology-stack--dependencies)
- [10. Installation & Usage](#10-installation--usage)
- [11. Platform & Environment Configuration](#11-platform--environment-configuration)
- [12. Web Interface](#12-web-interface)
- [13. Browser Web App (Static Deployment)](#13-browser-web-app-static-deployment)

---

## 1. Project Overview

This project implements an end-to-end **Skin Cancer Multi-class Classification & Screening System** designed for comprehensive dermatoscopic lesion analysis:

1. **Multi-class Identification (8 Disease Classes):** Detects Melanoma (MEL), Basal Cell Carcinoma (BCC), Squamous Cell Carcinoma (SCC), Actinic Keratoses (AK), Melanocytic Nevi (NV), Benign Keratosis (BKL), Vascular Lesions (VASC), and Dermatofibroma (DF).
2. **Two-Level Clinical Output:**
   - **Level 1 (Binary Screening):** Aggregated probability evaluation distinguishing `Cancer / Pre-Cancer (Malignant)` vs. `Non-Cancer (Benign)`.
   - **Level 2 (Differential Sub-Type Breakdown):** Detailed probability distribution across specific sub-types within each clinical category.
3. **Multi-Model Clinical Benchmark:** Compares three deep learning architectures (ResNet50, EfficientNet-B3, DenseNet121) under identical data partitions and class-weighting schemes.
4. **Differential Diagnosis Concordance:** Evaluates both single-label exact match (Top-1) and clinical differential diagnosis (Top-2 and Top-3), aligning with real-world dermatological triage workflows.

Developed as part of the **CPE310 -- Healthcare AI System** course.

---

## 2. Key Features

| Feature | Description |
|---|---|
| **Modular Architecture** | Clean separation of concerns: config, data, models, training, evaluation, and UI |
| **8 Disease Classes** | Full taxonomy covering both common benign lesions and aggressive carcinomas |
| **Multi-Model Support** | Standardized training and benchmark across ResNet50, EfficientNet-B3, and DenseNet121 |
| **Two-Tab Clinical UI** | Dedicated tabs for patient inference (Clinical Screening) and model analytics (Benchmark) |
| **Hot Model Switching** | Switch between trained models live in the web interface |
| **Two-Level Triage** | Binary screening decision accompanied by detailed sub-type differential probability |
| **Differential Diagnosis Metrics** | Reports Top-1, Top-2, Top-3, and AUC-ROC reflecting clinical workflow reality |
| **Apple Silicon GPU Acceleration** | Full native acceleration via PyTorch MPS (Metal Performance Shaders) |
| **Class Imbalance Mitigation** | Balanced class-weighted Cross-Entropy Loss compensating for majority classes |
| **Zero Emojis Policy** | Formal, distraction-free clinical user interface adhering to professional standards |

---

## 3. Project Structure

```
Healthcare AI System/
├── config.py                     # Central configuration, paths, taxonomy, model registry
├── data.py                       # Data loading, dataset class, transforms, dataloaders
├── models.py                     # Model architectures (ResNet50, EfficientNet-B3, DenseNet121)
├── train.py                      # Training loop, early stopping, LR scheduling
├── evaluate.py                   # Evaluation metrics, confusion matrix, comparison table
├── main.py                       # CLI entry point (train, evaluate)
├── export_onnx.py                # Export checkpoints to ONNX + assets for web/
├── web/                          # React web app (in-browser ONNX inference)
├── space/                        # Hugging Face Space README
│
├── model_resnet50.pth            # Trained ResNet50 checkpoint (98.6 MB)
├── model_efficientnet_b3.pth     # Trained EfficientNet-B3 checkpoint (46.5 MB)
├── model_densenet121.pth         # Trained DenseNet121 checkpoint (30.5 MB)
│
├── metrics_resnet50.json         # Empirical evaluation metrics for ResNet50
├── metrics_efficientnet_b3.json  # Empirical evaluation metrics for EfficientNet-B3
├── metrics_densenet121.json      # Empirical evaluation metrics for DenseNet121
│
├── cm_resnet50.png               # Confusion matrix plot for ResNet50
├── cm_efficientnet_b3.png        # Confusion matrix plot for EfficientNet-B3
├── cm_densenet121.png            # Confusion matrix plot for DenseNet121
│
├── Dataset/                      # ISIC 2019 dataset (ignored by git)
│   └── ISIC 2019/                # 25,331 training images, ground truth & metadata CSV (lesion_id)
│
├── ABCDE Rule.pdf                # Clinical reference for dermatological screening
├── requirements.txt              # Project dependencies
├── .gitignore                    # Git exclusion rules for large datasets and binaries
└── README.md                     # Comprehensive project documentation
```

---

## 4. Dataset: ISIC 2019

The **ISIC 2019** dataset (*International Skin Imaging Collaboration 2019 Challenge*) integrates images from three major clinical repositories: HAM10000, BCN_20000, and MSK.

- **Total Sample Count:** 25,331 dermoscopic images
- **Dataset Composition:** Incorporates the full HAM10000 collection (10,015 images) plus 15,316 independent cases, including the high-risk SCC category.
- **Partitioning:** Split by lesion (`lesion_id` from `ISIC_2019_Training_Metadata.csv`) and stratified by class with `StratifiedGroupKFold`: about 71% Training (18,093 images), 14% Validation (3,619) and 14% Unseen Test (3,619). All images of one lesion stay in one split, so the test set never contains a lesion seen in training.

---

## 5. Disease Taxonomy & Clinical Categories

| Index | Code | English Name | Thai Name | Clinical Group | Risk Level |
|:---:|:---:|---|---|:---:|:---:|
| 0 | **MEL** | Melanoma | มะเร็งผิวหนังเมลาโนมา | Cancer | Critical / High |
| 1 | **BCC** | Basal Cell Carcinoma | มะเร็งเซลล์ฐาน | Cancer | High |
| 2 | **SCC** | Squamous Cell Carcinoma | มะเร็งเซลล์สความัส | Cancer | High |
| 3 | **AK** | Actinic Keratoses / Bowen's | ผิวหนังก่อนมะเร็ง | Pre-Cancer | Moderate / Warning |
| 4 | **NV** | Melanocytic Nevi | ไฝและขี้แมลงวัน | Benign | Low |
| 5 | **BKL** | Benign Keratosis-like Lesions | รอยโรคสะเก็ดไม่ร้ายแรง | Benign | Low |
| 6 | **VASC** | Vascular Lesions | รอยโรคหลอดเลือด | Benign | Low |
| 7 | **DF** | Dermatofibroma | เนื้องอกเส้นใยผิวหนัง | Benign | Low |

---

## 6. Two-Level Clinical Decision Architecture

```
[ Input Dermoscopic Image (224x224) ]
                 │
                 ▼
[ Active Model: ResNet50 / EfficientNet-B3 / DenseNet121 ]
                 │
                 ▼ (Softmax)
[ 8 Class Probabilities: P(MEL), P(BCC), P(SCC), P(AK), P(NV), P(BKL), P(VASC), P(DF) ]
                 │
                 ├──────────────────────────────────────┐
                 ▼                                      ▼
[ Level 1: Binary Clinical Screening ]   [ Level 2: Sub-Type Differential Breakdown ]
  P(Cancer) = Σ P(MEL, BCC, SCC, AK)       Cancer Sub-Types:
  P(Benign) = Σ P(NV, BKL, VASC, DF)         P(MEL), P(BCC), P(SCC), P(AK)
                                           Benign Sub-Types:
  Decision Logic:                            P(NV), P(BKL), P(VASC), P(DF)
  - HIGH RISK (P(Cancer) ≥ threshold)
  - LOW RISK  (P(Cancer) < threshold)    Differential Clinical Ranking (Top 1-3)
```

The threshold is chosen per model on the **validation** set as the highest value that still flags at least
90% of cancers (`TARGET_SENSITIVITY` in `config.py`), capped at 0.50. It is saved as `cancer_threshold` in
`metrics_*.json` and used by both the test-set report and the web app. Models exported before this change fall
back to 0.50.

---

## 7. Multi-Model Architecture Comparison

All architectures leverage transfer learning from ImageNet pre-trained weights with customized classification heads:

```
Linear(in_features, 512) -> ReLU -> BatchNorm1d(512) -> Dropout(0.3) -> Linear(512, 8)
```

| Architecture | Total Parameters | Trainable Parameters | Fine-Tuning Backbone Depth |
|---|---:|---:|---|
| **ResNet50** | 24,562,248 | 23,117,320 | Layer 3 + Layer 4 + Classifier Head |
| **EfficientNet-B3** | 11,488,304 | 9,298,150 | Stages 6–8 + Head |
| **DenseNet121** | 7,483,784 | 6,054,152 | DenseBlock 3–4 + Transitions + Head |

---

## 8. Empirical Benchmark & Validation Results

Values below are copied from `metrics_*.json` (also shown in the web app's Model Performance tab).

All numbers are on a **lesion-grouped** test split: every image of a lesion is in exactly one of
train / validation / test, so the model is never tested on a lesion it saw during training. This is
stricter (and lower) than the image-level random splits common in public notebooks.

| Performance Metric | ResNet50 | DenseNet121 | EfficientNet-B3 | Clinical Benchmark Target |
|---|:---:|:---:|:---:|:---:|
| **Top-2 Differential Diagnosis** | **88.70% (Best)** | 86.96% | 84.80% | Primary Clinical Target (≥ 85%) |
| **Top-3 Differential Diagnosis** | **94.89% (Best)** | 93.45% | 91.66% | Secondary Triage Target (≥ 90%) |
| **Cancer Screening AUC-ROC** | **0.8854 (Best)** | 0.8852 | 0.8789 | Diagnostic Gold Standard (≥ 0.90) |
| **Binary Malignancy Screening** | **75.38% (Best)** | 73.61% | 73.53% | Malignant vs Benign Triage |
| **Exact 8-Class Match (Top-1)** | **71.73% (Best)** | 68.36% | 67.56% | Single Fine-Grained Label Match |
| **Macro F1-Score** | **55.17% (Best)** | 50.63% | 49.42% | Cross-Class Balanced Harmonic Mean |
| **Weighted F1-Score** | **72.12% (Best)** | 69.32% | 69.07% | Frequency-Weighted F1 |
| **Cancer Sensitivity (Recall)** | 88.08% | **90.93% (Best)** | 89.88% | Malignancy Detection Rate (target 90% on validation) |
| **Cancer Specificity (Benign)** | **67.96% (Best)** | 63.50% | 63.98% | Non-Cancer Confirmation Rate |
| **False Negative Rate (FNR)** | 11.92% | **9.07% (Best)** | 10.12% | Critical Missed Malignancy Rate |
| **Cancer Threshold** | 0.1849 | 0.2231 | 0.2707 | Tuned on validation set |
| **Training Duration** | 228.3 min | 180.2 min | 135.2 min | Wall-clock training time |
| **Total Model Parameters** | 24.6M | 7.5M | 11.5M | Model Footprint |

---

## 9. Technology Stack & Dependencies

- **Language:** Python 3.10+
- **Deep Learning:** PyTorch & Torchvision (MPS Apple Silicon GPU Accelerated)
- **Machine Learning & Evaluation:** Scikit-learn, NumPy, Pandas
- **Image Processing:** Pillow (PIL)
- **Visualization:** Matplotlib (Agg headless backend)
- **Web App:** React, Vite, Tailwind CSS, ONNX Runtime Web (`onnx` / `onnxruntime` for export)
- **User Interface:** React + Vite + Tailwind, ONNX Runtime Web (see Section 13)

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 10. Installation & Usage

### 1. Environment Setup
```bash
cd "Healthcare AI System"
source venv/bin/activate
pip install -r requirements.txt
```

#### Download Trained Models
Model weights are not stored in git. Download them from the
[latest release](https://github.com/KhaNomThai/Healthcare-AI-System/releases/latest) into the project root:
```bash
gh release download --repo KhaNomThai/Healthcare-AI-System --pattern "*.pth"
```
Training from scratch instead requires the ISIC 2019 dataset under `Dataset/` (see Section 4).

### 2. Execution Modes

#### Launch Web Demo (Recommended)
See [Section 13](#13-browser-web-app-static-deployment): `python export_onnx.py`, then `cd web && npm install && npm run dev`.

#### Full Multi-Model Training & Benchmark
```bash
python main.py
```
Sequentially trains ResNet50, EfficientNet-B3, and DenseNet121, picks each model's cancer threshold on the
validation split, evaluates on the test split, and generates comparison tables.
`python test_evaluate.py` self-checks the threshold picker.

#### Train a Specific Architecture
```bash
python main.py --model ResNet50
python main.py --model DenseNet121
python main.py --model EfficientNet-B3
```

#### Skip Training, Run Evaluation & Demo
```bash
python main.py --skip-train
```
Loads existing model checkpoints, executes test evaluation, and updates metrics. Models without a checkpoint are skipped.

---

## 11. Platform & Environment Configuration

- **macOS Apple Silicon:** Automatic selection of `torch.device("mps")`.
- **SSL Certificates (macOS):** If downloading pretrained weights fails with a certificate error, run `Install Certificates.command` from your Python install folder once. Certificate checks stay on.
- **DataLoader Configuration:** `num_workers=2` for all loaders (limited for macOS fork safety).
- **Headless Visualization:** `MPLCONFIGDIR` set to temporary storage with `matplotlib.use("Agg")` to prevent macOS permission locks.

---

## 12. Web Interface

The web app (`web/`) has two tabs:

### Tab 1: Clinical Screening
- **Model Selector:** Cards for ResNet50 / EfficientNet-B3 / DenseNet121 with Top-1 accuracy and download size.
- **Image Input:** Click or drag-and-drop upload, plus 8 sample images from the held-out test set (with ground-truth badge).
- **Two-Level Result:** High/Low risk banner, cancer vs benign bars, Top-1 diagnosis in English and Thai,
  sub-type probabilities for both groups, and a triage recommendation with disclaimer.

### Tab 2: Model Performance
- **Summary Cards:** Top-1 accuracy, AUC-ROC, sensitivity, parameters and training time per model.
- **Metric Comparison Table:** 10 metrics side by side, best value per row highlighted.
- **Per-Class F1:** Bars for each of the 8 classes per model.
- **Confusion Matrices:** Test-set confusion matrix for each model.

---

## 13. Browser Web App (Static Deployment)

`web/` is a React (Vite + Tailwind) version of the demo that runs the models **entirely in the browser**
with ONNX Runtime Web, so it can be hosted as static files (no Python server, images never leave the device).

```bash
python export_onnx.py                     # .pth -> web/public/models/*.onnx (+ metrics, samples); re-run after retraining

cd web
npm install
npm run dev                               # local preview at http://localhost:5173
npm run build                             # static site in web/dist/
npm test                                  # checks browser preprocessing == torchvision
```

If you change preprocessing, regenerate the test fixtures with `python web/test/make_fixtures.py`.

Deploy `web/dist/` to any static host. For a Hugging Face **static** Space (free):

```bash
hf repos create <user>/derma-ai --type space --sdk static --public   # first time only
hf upload <user>/derma-ai web/dist . --type space
hf upload <user>/derma-ai space/README.md README.md --type space
```

After that, pushes to `main` that touch `web/` redeploy the app code automatically
(`.github/workflows/deploy-space.yml`, needs a `HF_TOKEN` repo secret with write access to the Space).
Models, `data.json` and sample images are not in git, so after re-running `export_onnx.py` upload them by hand
with the command above.
