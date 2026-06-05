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

## Repository Contents

- `training.ipynb`  
  End-to-end training workflow (data loading, split, training loop, saving weights).

- `experiments.ipynb`  
  Evaluation + analysis workflow (metrics, inspection utilities, Grad-CAM-style visualization).

- `My_3DResNet_Model.py`  
  PyTorch implementation of the 3D ResNet model used for both tasks.

- `8650 Paper.pdf`  
  Project report with background, methodology, and results.


---

## Setup

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

- A folder of NIfTI volumes: `*.nii.gz`
- A labels CSV with at least two target columns:
  - `Recognizable-Facial-Feature`
  - `Brain-Feature-Loss`

### Current path assumptions (relative paths)

Both notebooks currently use:

```python
image_dir = "../dataset/files"
label_csv = "../labels.csv"
```

That means if your notebooks live in a subfolder (e.g., `notebooks/`), your repo could be laid out like:

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

If you keep the notebooks in the repo root, change those paths to:

```python
image_dir = "./dataset/files"
label_csv = "./labels.csv"
```

---

## Running

### 1) Train the models

Open and run:

- `training.ipynb`

You will train and save weights for:

- Face-remnant detector
- Brain-tissue-loss detector

### 2) Evaluate + visualize

Open and run:

- `experiments.ipynb`

This notebook loads the trained weights, computes evaluation metrics, and includes Grad-CAM-style visualization code to inspect what regions influence predictions.

---

## Notes / Limitations

This is a course project and is intended as a reproducible demonstration of:

- Unstructured data handling (3D medical images)
- Binary classification
- Evaluation and interpretability tooling
- Documentation of tradeoffs (privacy vs. utility)

The dataset is not redistributed here.

---

## Contact

Keller Brandenburg
kellerbrandenburg@gmail.com
