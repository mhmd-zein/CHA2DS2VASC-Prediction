# RASTA Retinal Disease Classification — Inference Guide

This system classifies retinal OCT/OCTA images into disease severity classes using a pre-trained EfficientNet-based deep learning model. This guide covers everything needed to run inference at your site.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Installation](#2-installation)
3. [What You Need Before Starting](#3-what-you-need-before-starting)
4. [Data Format](#4-data-format)
5. [Running Inference](#5-running-inference)
6. [Understanding the Output](#6-understanding-the-output)
7. [All Command-Line Options](#7-all-command-line-options)
8. [Common Issues & Troubleshooting](#8-common-issues--troubleshooting)
9. [Notes on Classes and Label Mapping](#9-notes-on-classes-and-label-mapping)

---

## 1. System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.9 | 3.10 or 3.11 |
| RAM | 8 GB | 16 GB |
| GPU | None (CPU works) | NVIDIA GPU with CUDA |
| Disk space | ~2 GB (model + env) | — |
| OS | Windows / Linux / macOS | Linux |

> **No GPU required.** The script auto-detects a CUDA GPU and uses it if available; otherwise it runs on CPU (slower but correct).

---

## 2. Installation

### Step 1 — Install Python

Download Python 3.10 from [python.org](https://www.python.org/downloads/) and install it. Make sure to check **"Add Python to PATH"** during installation.

Verify the installation:

```bash
python --version
```

### Step 2 — Install Git

Download and install Git from [git-scm.com](https://git-scm.com/downloads). This is required to clone the repository.

Verify the installation:

```bash
git --version
```

### Step 3 — Clone the repository

```bash
git clone https://github.com/mhmd-zein/CHA2DS2VASC-Prediction.git
cd CHA2DS2VASC-Prediction
```

### Step 4 — Create a virtual environment (recommended)

```bash
# Create the environment
python -m venv rasta_env

# Activate it
# On Windows:
rasta_env\Scripts\activate
# On Linux / macOS:
source rasta_env/bin/activate
```

### Step 5 — Install dependencies

```bash
pip install -r requirements_inference.txt
```

> **GPU acceleration (optional):** If you have a CUDA-capable NVIDIA GPU, install the CUDA-enabled PyTorch build **before** running the command above. Visit [pytorch.org](https://pytorch.org/get-started/locally/) and select your CUDA version. Example for CUDA 11.8:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> pip install -r requirements_inference.txt
> ```

---

## 3. What You Need Before Starting

You need **two things**:

| Item | Description |
|------|-------------|
| **Run command** | The exact `python inference.py ...` command provided by the research team, which specifies the correct model path and parameters for your dataset |
| **Patient images** | Your OCTA images organized in the folder layout described in Section 4 |

The research team will tell you the exact command to run. It will look like this (values filled in for you):

```bash
python inference.py \
    --model_path  Results/my_experiment/final_model.pth \
    --modalities  sup deep cc \
    --input_channels 3 \
    --data_dir    /path/to/your/images
```

The only thing **you** need to fill in is `--data_dir`, pointing to your organized image folder.

---

## 4. Data Format

### Image specifications

| Property | Value |
|----------|-------|
| Format | BMP (`.bmp`) — also accepts PNG, JPG, TIFF |
| Color | Grayscale |
| Resolution | Any — images are automatically resized to **256 × 256** during loading |
| Bit depth | 8-bit |

### Patient ID convention

The **patient ID** is extracted from the filename as **everything before the first underscore `_`**.

```
Example filenames:
  1ALVA91_sup_OD.bmp    → patient ID: 1ALVA91
  PAT042_deep_OS.bmp    → patient ID: PAT042
  HOSP001_cc.bmp        → patient ID: HOSP001
```

> **Important:** The patient ID must be **identical** across all modality folders for the same patient. If the IDs don't match, that patient is silently skipped.

### Supported modalities

The model may use any combination of these modalities (as specified by the research team):

| Folder name | Description |
|-------------|-------------|
| `sup` | Superficial retinal layer |
| `deep` | Deep retinal layer |
| `cc` | Choriocapillaris |
| `perf_maps` | Perfusion maps |
| `Vessels` | Blood vessel segmentation |
| `Capillaries` | Capillary segmentation |

---

### Layout A — Prediction only (no ground truth labels)

Use this when you want to classify patients and **do not have ground truth diagnoses**.

```
data_dir/
├── sup/
│   ├── PAT001_sup_OD.bmp
│   ├── PAT002_sup_OD.bmp
│   └── PAT003_sup_OD.bmp
├── deep/                        ← only needed if model uses 'deep'
│   ├── PAT001_deep_OD.bmp
│   ├── PAT002_deep_OD.bmp
│   └── PAT003_deep_OD.bmp
└── cc/                          ← only needed if model uses 'cc'
    ├── PAT001_cc_OD.bmp
    ├── PAT002_cc_OD.bmp
    └── PAT003_cc_OD.bmp
```

- One subfolder per modality, named exactly as listed in the table above.
- All images for a given modality go directly inside that subfolder (no subfolders inside).
- Only include the modality folders that the model was trained on.

---

### Layout B — With ground truth labels (for evaluation)

Use this when you have diagnoses and want to **evaluate model performance** (accuracy, AUROC, F1, etc.).

```
data_dir/
├── sup/
│   ├── 0/                       ← class 0 images
│   │   ├── PAT001_sup_OD.bmp
│   │   └── PAT004_sup_OD.bmp
│   ├── 1/                       ← class 1 images
│   │   ├── PAT002_sup_OD.bmp
│   └── 2/                       ← class 2 images (see Section 9)
│       └── PAT003_sup_OD.bmp
├── deep/
│   ├── 0/
│   │   ├── PAT001_deep_OD.bmp
│   │   └── PAT004_deep_OD.bmp
│   ├── 1/
│   │   └── PAT002_deep_OD.bmp
│   └── 2/
│       └── PAT003_deep_OD.bmp
└── cc/
    ├── 0/
    ├── 1/
    └── 2/
```

- Each modality folder contains three subfolders: `0/`, `1/`, `2/` (you may omit `2/` if not applicable).
- A patient's image must be in **the same class subfolder** across all modality folders.

---

## 5. Running Inference

Open a terminal, activate your virtual environment (see Step 4), and navigate to the cloned repository folder:

```bash
cd <your-repo>
```

---

### Scenario A — Prediction only (Layout A data, no labels)

```bash
python inference.py \
    --model_path  Results/my_experiment/final_model.pth \
    --data_dir    /path/to/your/images \
    --modalities  sup deep cc \
    --input_channels 3
```

The only value you change is `--data_dir` — point it to your data folder (the one containing `sup/`, `deep/`, etc.). All other arguments are provided to you by the research team.

**Single-modality example:**
```bash
python inference.py \
    --model_path  Results/sup_model/final_model.pth \
    --data_dir    /path/to/your/images \
    --modalities  sup \
    --input_channels 1
```

---

### Scenario B — With ground truth labels (Layout B data, evaluation mode)

Add the `--has_labels` flag:

```bash
python inference.py \
    --model_path  Results/my_experiment/final_model.pth \
    --data_dir    /path/to/your/images \
    --modalities  sup deep cc \
    --input_channels 3 \
    --has_labels
```

This will print accuracy, AUROC, F1, MCC, and Cohen's Kappa in addition to saving the predictions.

---

### On Windows (PowerShell)

Activate the virtual environment first (`rasta_env\Scripts\activate`), then replace backslash line continuations `\` with a backtick `` ` ``:

```powershell
python inference.py `
    --model_path  Results\my_experiment\final_model.pth `
    --data_dir    C:\path\to\your\images `
    --modalities  sup deep cc `
    --input_channels 3
```

---

### Specifying the output directory

By default results are saved to `./predictions/predictions.csv`. To change this:

```bash
python inference.py ... --output_dir /path/to/output/folder
```

---

## 6. Understanding the Output

The script saves a file called `predictions.csv` in the output directory.

### Columns

| Column | Description |
|--------|-------------|
| `patient_id` | Patient identifier (extracted from the filename before the first `_`) |
| `predicted_class` | The predicted class label (integer: 0, 1, or 2) |
| `prob_class_0` | Model confidence for class 0 (0.0 – 1.0) |
| `prob_class_1` | Model confidence for class 1 (0.0 – 1.0) |
| `prob_class_2` | Model confidence for class 2 (0.0 – 1.0) |
| `true_label` | Ground truth label — **only present** when `--has_labels` is used |

### Example output

```csv
patient_id,predicted_class,prob_class_0,prob_class_1,prob_class_2,true_label
PAT001,0,0.8921,0.0901,0.0178,0
PAT002,1,0.1234,0.7812,0.0954,1
PAT003,1,0.0543,0.8901,0.0556,2
```

### Console output (example)

```
Device: cpu
Loading model from: Results/my_experiment/final_model.pth
Model loaded successfully.
Found 120 patient(s). Running inference...

=== Evaluation Metrics ===
  Accuracy:        0.8333
  AUROC:           0.9102
  Avg Precision:   0.8754
  F1 (weighted):   0.8275
  MCC:             0.7412
  Cohen's Kappa:   0.7108

=== Prediction Summary ===
  Class 0: 48 patient(s) (40.0%)
  Class 1: 72 patient(s) (60.0%)
  Class 2: 0 patient(s) (0.0%)

Predictions saved to: predictions/predictions.csv
```

---

## 7. All Command-Line Options

Run `python inference.py --help` at any time to see this table.

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model_path` | Yes | — | Path to `.pth` checkpoint file |
| `--data_dir` | Yes | — | Root data directory (contains modality subfolders) |
| `--modalities` | Yes | — | Modality names, e.g. `sup deep cc` |
| `--input_channels` | Yes | — | Number of input channels (must match training) |
| `--num_classes` | No | `3` | Number of output classes |
| `--model_version` | No | `b0` | EfficientNet version: `b0`, `b4`, or `v2s` |
| `--dropout` | No | `0.0` | Dropout rate used during training |
| `--batch_size` | No | `4` | Images processed per batch |
| `--num_workers` | No | `0` | DataLoader workers (keep `0` on Windows) |
| `--has_labels` | No | False | Enable evaluation metrics (data must be in Layout B) |
| `--output_dir` | No | `predictions` | Folder where `predictions.csv` is saved |
| `--device` | No | auto | `cuda` or `cpu` (auto-detected if omitted) |
| `--seed` | No | `42` | Random seed for reproducibility |

---

## 8. Common Issues & Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
The virtual environment is not activated, or dependencies were not installed.
```bash
# Activate the environment first (see Step 4), then:
pip install -r requirements_inference.txt
```

### "FileNotFoundError: Model checkpoint not found"
The path passed to `--model_path` does not exist. Use the full absolute path or check for typos.

### "No samples found"
Possible causes:
- The folder names in `--data_dir` don't match the modality names in `--modalities`.
  Check that the subfolder is named exactly `sup`, `deep`, `cc`, etc.
- The patient IDs in the filenames don't match across modality folders.
  Verify that the part before the first `_` is identical in every modality folder.
- The images have an unexpected file extension. Supported: `.bmp`, `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`.

### "RuntimeError: Error(s) in loading state_dict"
The `--num_classes`, `--input_channels`, or `--model_version` in your command does not match the model checkpoint. Use the exact command provided by the research team without modifying those arguments.

### Prediction takes very long on CPU
Normal for a CPU-only machine. With a batch of 4 and ~100 patients, expect 5–15 minutes depending on hardware. Use a GPU (`--device cuda`) to reduce this to under a minute.

### All predictions are the same class
This can happen when the wrong model parameters are passed (e.g., wrong number of classes or channels). Double-check `--num_classes`, `--input_channels`, and `--model_version` with the research team.

### Permission denied errors on Windows
Run the terminal as Administrator, or move the project folder to a location you own (e.g., `C:\Users\YourName\Documents\`).

---

## 9. Notes on Classes and Label Mapping

### Class definitions

| Class label | Clinical meaning |
|-------------|-----------------|
| 0 | *(fill in clinical interpretation)* |
| 1 | *(fill in clinical interpretation)* |
| 2 | *(fill in clinical interpretation)* |

> Contact the research team for the clinical interpretation of each class label.

### Label remapping

During training, **class 2 was remapped to class 1**. This means the model effectively distinguishes between class 0 and class 1 (with classes 1 and 2 merged). In practice, the model will rarely or never predict class 2.

When using `--has_labels` with Layout B data, images placed in the `2/` subfolder are automatically remapped to label `1` before metric computation, consistent with how the model was trained.

### Confidence scores

The `prob_class_0`, `prob_class_1`, `prob_class_2` columns are **softmax probabilities** and always sum to 1.0 per patient. They can be interpreted as the model's confidence for each class. The `predicted_class` is whichever class has the highest probability.

---

## Project Structure (reference)

```
RASTA_Classification/
├── inference.py              ← main entry point for inference (this script)
├── README.md                 ← this file
├── requirements_inference.txt← dependencies for inference
├── utils.py                  ← evaluation metrics
├── Training/
│   ├── Models.py             ← EfficientNet model definition
│   └── Engine.py             ← training engine (not used at inference time)
├── Data/
│   ├── Data_Loader.py        ← dataset loading (includes inference loaders)
│   └── transforms.py         ← image preprocessing pipeline
└── Results/                  ← trained model checkpoints (.pth files)
```
