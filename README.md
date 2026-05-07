# RASTA Retinal Disease Classification — Inference Guide

This tool classifies retinal OCTA images into disease severity classes using a pre-trained deep learning model. Follow the steps below to set it up and run it.

---

## 1. System Requirements

| Component | Minimum |
|-----------|---------|
| Python | 3.9 or later |
| RAM | 8 GB |
| GPU | Optional (CPU works, but slower) |
| OS | Windows / Linux / macOS |

---

## 2. Installation

### Step 1 — Install Python

Download Python 3.10 from [python.org](https://www.python.org/downloads/) and install it.
On Windows, check **"Add Python to PATH"** during installation.

```bash
python --version   # verify
```

### Step 2 — Install Git

Download and install Git from [git-scm.com](https://git-scm.com/downloads).

```bash
git --version   # verify
```

### Step 3 — Clone the repository

```bash
git clone https://github.com/mhmd-zein/CHA2DS2VASC-Prediction.git
cd CHA2DS2VASC-Prediction
```

### Step 4 — Create a virtual environment

```bash
python -m venv rasta_env

# Activate — Windows:
rasta_env\Scripts\activate

# Activate — Linux / macOS:
source rasta_env/bin/activate
```

### Step 5 — Install dependencies

```bash
pip install -r requirements_inference.txt
```

> **GPU (optional):** For faster inference on an NVIDIA GPU, install PyTorch with CUDA support first (see [pytorch.org](https://pytorch.org/get-started/locally/)), then run the command above.

---

## 3. Data Organization

Place your images inside the `data/` folder at the root of the repository.
The `data/` folder must contain one subfolder per imaging modality:

```
data/
├── sup/        ← superficial retinal layer images
├── deep/       ← deep retinal layer images
└── cc/         ← choriocapillaris images
```

### Image requirements

| Property | Value |
|----------|-------|
| Format | BMP (`.bmp`) — PNG, JPG, and TIFF are also accepted |
| Color | Grayscale |
| Resolution | Any — images are automatically resized to 256 × 256 |

### Patient ID convention

The **patient ID** is extracted from the filename as everything before the first underscore `_`.

```
PAT001_sup_OD.bmp   →  patient ID: PAT001
PAT001_deep_OD.bmp  →  patient ID: PAT001   ← same ID links the modalities
PAT001_cc_OD.bmp    →  patient ID: PAT001
```

The patient ID **must be identical** across all three modality folders for the same patient. If the IDs don't match, that patient is skipped.

### Option A — Prediction only (no ground truth)

Place all images directly inside each modality folder:

```
data/
├── sup/
│   ├── PAT001_sup_OD.bmp
│   ├── PAT002_sup_OD.bmp
│   └── ...
├── deep/
│   ├── PAT001_deep_OD.bmp
│   ├── PAT002_deep_OD.bmp
│   └── ...
└── cc/
    ├── PAT001_cc_OD.bmp
    ├── PAT002_cc_OD.bmp
    └── ...
```

### Option B — With ground truth labels (to evaluate model accuracy)

Create numbered subfolders (`0/`, `1/`, `2/`) inside each modality folder and place images by class:

```
data/
├── sup/
│   ├── 0/
│   │   └── PAT001_sup_OD.bmp
│   ├── 1/
│   │   └── PAT002_sup_OD.bmp
│   └── 2/
│       └── PAT003_sup_OD.bmp
├── deep/
│   ├── 0/
│   │   └── PAT001_deep_OD.bmp
│   ├── 1/
│   │   └── PAT002_deep_OD.bmp
│   └── 2/
│       └── PAT003_deep_OD.bmp
└── cc/
    ├── 0/
    │   └── PAT001_cc_OD.bmp
    ├── 1/
    │   └── PAT002_cc_OD.bmp
    └── 2/
        └── PAT003_cc_OD.bmp
```

The script automatically detects which option you are using based on the folder structure.

---

## 4. Run Inference

Specify whether the model should distinguish **2 classes** (binary) or **3 classes** (multi-class):

```bash
# Binary classification (classes 0 and 1 — original class 2 is merged into class 1):
python inference.py --num_classes 2

# Multi-class classification (classes 0, 1, and 2):
python inference.py --num_classes 3
```

Use whichever matches the model version provided to you.

---

## 5. Output

Results are saved to `predictions/predictions.csv`.

| Column | Description |
|--------|-------------|
| `patient_id` | Patient identifier (from filename) |
| `predicted_class` | Predicted class label (0 or 1 for 2-class; 0, 1, or 2 for 3-class) |
| `prob_class_0` | Model confidence for class 0 (0.0 – 1.0) |
| `prob_class_1` | Model confidence for class 1 (0.0 – 1.0) |
| `prob_class_2` | Model confidence for class 2 — **only present in 3-class mode** |
| `true_label` | Ground truth label — only present when using Option B |

When using Option B, accuracy, AUROC, F1, MCC, and Cohen's Kappa are also printed to the console.

---

## 6. Troubleshooting

**`ModuleNotFoundError: No module named 'torch'`**
The virtual environment is not active. Run `rasta_env\Scripts\activate` (Windows) or `source rasta_env/bin/activate` (Linux/macOS), then retry.

**`No samples found`**
- Confirm the `data/` folder exists at the root of the repository.
- Confirm the subfolders are named exactly `sup`, `deep`, and `cc`.
- Confirm the patient ID (the part before `_`) is the same across all three modality folders for each patient.

**`Model file not found`**
The repository was not cloned correctly or a file is missing. Re-run `git clone` and ensure all files downloaded.

---

## 7. Class Labels

| Class | Clinical meaning | Present in 2-class mode | Present in 3-class mode |
|-------|-----------------|------------------------|------------------------|
| 0 | *(fill in)* | Yes | Yes |
| 1 | *(fill in — includes original class 2 in 2-class mode)* | Yes | Yes |
| 2 | *(fill in)* | No (merged into class 1) | Yes |
