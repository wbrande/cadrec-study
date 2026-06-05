# MRI De-identification QC with 3D ResNet (Face Removal + Brain Tissue Preservation)

This project builds **two 3D deep learning classifiers** to support quality control (QC) for MRI de-identification workflows. When brain MRI scans are prepared for public release, identifiable facial features must be removed — but overly aggressive defacing can also remove **brain tissue** and reduce research utility.

To support automated QC, this repo includes training/evaluation code for:

1. A model to detect whether **recognizable facial tissue/bone remains**, and
2. A model to detect whether **brain tissue was inadvertently removed**.

**Reported performance (course project):**
- Facial feature detection model: **89.09% accuracy**
- Brain tissue loss detection model: **97.37% accuracy**

> **Note:** The dataset is not included in this repository.

---

## Repository Structure

```
├── README.md
├── .gitignore
├── labels.csv                  # Labels for MRI scans (not included)
├── src/
│   ├── model.py                # 3D ResNet-18 architecture (primary model)
│   └── inception_classifier.py # Earlier Inception-style 3D CNN
├── notebooks/
│   ├── training.ipynb          # End-to-end training workflow
│   └── experiments.ipynb       # Evaluation, Grad-CAM, and histograms
├── paper/
│   ├── 8650 Paper.pdf
│   └── 8650 Cover Page.pdf
├── results/
│   ├── confusion_matrices/
│   ├── gradcam/
│   ├── f1_score_plot.png
│   ├── histogram_face_and_tissue.png
│   └── resnet3d.png
├── models/                     # Place model weights (*.pt) here
└── legacy/                     # Earlier exploration code
    └── experiments_inference.ipynb
```

---

## Setup

### Option A: pip (recommended)

Create a virtual environment, then install dependencies:

```bash
pip install -U pip
pip install torch numpy pandas scikit-learn nibabel matplotlib scipy jupyter
```

Optional (used in `experiments.ipynb`):

```bash
pip install torchsummary torchviz
```

PyTorch installation can vary by OS/CUDA. If you need GPU support, install the correct PyTorch build for your machine.

---

## Data Expectations (not included)

The notebooks expect:

- A folder of NIfTI volumes: `dataset/files/*.nii.gz`
- A labels CSV (`labels.csv`) with at least two target columns:
  - `Recognizable-Facial-Feature`
  - `Brain-Feature-Loss`

### Path assumptions (relative to notebooks/)

Both notebooks use paths relative to the `notebooks/` directory:

```python
image_dir = "../dataset/files"
label_csv = "../labels.csv"
```

Expected layout:

```text
repo-root/
  labels.csv
  dataset/
    files/
      <volumes>.nii.gz
  notebooks/
    training.ipynb
    experiments.ipynb
```

---

## Model Weights

Trained model weights (`.pt` files) are not included in the repository. Place them in the `models/` directory:

- `models/resnet3d_face_features_model.pt` — facial feature detection
- `models/resnet3d_brain_tissue_model.pt` — brain tissue loss detection

---

## Running

### 1) Train the models

Open and run:

- `notebooks/training.ipynb`

You will train and save weights for:

- Face-remnant detector
- Brain-tissue-loss detector

### 2) Evaluate + visualize

Open and run:

- `notebooks/experiments.ipynb`

This notebook loads the trained weights, computes evaluation metrics, and includes Grad-CAM visualization code to inspect what regions influence predictions.

---

## Notes / Limitations

This is a course project (CPSC 8650) and is intended as a reproducible demonstration of:

- Unstructured data handling (3D medical images)
- Binary classification
- Evaluation and interpretability tooling
- Documentation of tradeoffs (privacy vs. utility)

The dataset is not redistributed here.
