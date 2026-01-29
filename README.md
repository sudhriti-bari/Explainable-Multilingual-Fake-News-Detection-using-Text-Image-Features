# Explainable Multilingual Fake News Detection using Text and Image Features

This repository contains the complete implementation of our **final-year / research project** on fake news detection using **multilingual text**, **images**, and **explainable AI (XAI)** techniques.

The project is designed as a **multi-pipeline system**, where text-only, image-only, and hybrid (text + image) models are trained independently and then evaluated with **decision fusion** and **human-interpretable explanations**.

---

## 📌 Project Highlights

* ✅ Multilingual fake news detection (XLM-RoBERTa)
* ✅ Image-based fake news detection (ResNet-50)
* ✅ Hybrid multimodal model using **decision-level fusion**
* ✅ Explainability for:

  * Text (token-level importance)
  * Images (Grad-CAM)
  * Final decision (confidence-based fusion)
* ✅ Research-paper-ready structure and metrics

---

## 🧠 System Architecture

```
                ┌──────────────────┐
                │ Multilingual Text │───► Text Model (XLM-R)
                └──────────────────┘
                          │
                          │ logits
                          ▼
                    Decision Fusion ───► Final Prediction
                          ▲
                          │ logits
                ┌──────────────────┐
                │      Image        │───► Image Model (ResNet-50)
                └──────────────────┘

Explainability:
- Text → Token importance (gradients / attention)
- Image → Grad-CAM heatmaps
```

---

## 📂 Repository Structure

```
├── code/
│   ├── hybrid/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── explainability.py
│   │
│   ├── image_only/
│   │   ├── train.py
│   │   ├── test.py
│   │   ├── evaluate.py
│   │   └── explain.py
│   │
│   ├── text_multilingual/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── explain.py
│   │
│   └── text_english_only/
│       ├── train.py
│       ├── evaluate.py
│       └── explain.py
│
├── checkpoints/          # Saved trained models (not uploaded)
├── dataset/              # Dataset structure (TSV only)
├── README.md
└── requirements.txt
```

---

## 📊 Dataset Information

* **Base Dataset**: Fakeddit (multimodal fake news dataset)
* **Labels**:

  * `0` → REAL
  * `1` → FAKE
* **Modalities**:

  * Text (English + Multilingual)
  * Images (news-related images)

### Dataset Files (TSV)

Each TSV file contains:

| Column Name         | Description                            |
| ------------------- | -------------------------------------- |
| `text_multilingual` | News text (translated / original)      |
| `clean_title`       | Cleaned English title                  |
| `image_path`        | Local path to image                    |
| `2_way_label`       | Binary label (REAL / FAKE)             |
| `language`          | Language code (for multilingual setup) |

---

## 🚫 Why the Dataset Is NOT Uploaded to GitHub

The dataset is **too large** (images + TSVs exceed GitHub’s file size limits).

### ✅ Recommended Solution (Best Practice)

1. **Do NOT upload images to GitHub**
2. Upload only:

   * TSV structure
   * Dataset preparation scripts
3. Add this to `.gitignore`:

```
dataset/
images_train/
images_val/
images_test/
*.jpg
*.png
```

### 📥 How Users Can Get the Dataset

Mention this in your README (already done here):

* Download Fakeddit dataset from official source
* Run dataset preparation scripts to recreate TSVs
* Update `image_path` fields to local image folders

This keeps your repository **clean, professional, and publishable**.

---

## 🧪 Pipelines Explained

### 1️⃣ Multilingual Text Pipeline

* Model: **XLM-RoBERTa (base)**
* Pooling: CLS + Mean Pooling
* Classifier: MLP with dropout
* Explainability: Token importance using attention / gradients

**Scripts**:

* `train.py`
* `evaluate.py`
* `explain.py` (HTML-based explanations)

---

### 2️⃣ Image-Only Pipeline

* Model: **ResNet-50**
* Training Strategy:

  * Phase 1: Feature extraction
  * Phase 2: Light fine-tuning (last block)
* Explainability: **Grad-CAM**

**Scripts**:

* `train.py`
* `test.py`
* `evaluate.py`
* `explain.py`

---

### 3️⃣ Hybrid Multimodal Pipeline (Core Contribution)

* Text Encoder: XLM-RoBERTa
* Image Encoder: ResNet-50
* Fusion Type: **Decision-level fusion**

```
final_logits = (text_logits + image_logits) / 2
```

* Explainability:

  * Text → Important tokens
  * Image → Grad-CAM
  * Final decision → Confidence comparison

**Scripts**:

* `train.py`
* `evaluate.py`
* `explainability.py` (HTML report)

---

## 📈 Evaluation Metrics

All pipelines report:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Balanced Accuracy (hybrid)

Metrics are **paper-consistent and reproducible**.

---

## 🌐 Explainability Outputs

Explainability scripts generate **HTML reports** showing:

* Highlighted text tokens
* Original images + Grad-CAM heatmaps
* Model confidence scores
* Correct vs incorrect predictions

These files are **not uploaded** but generated locally:

```
*.html
```

---

## ⚙️ Installation & Setup

```bash
pip install -r requirements.txt
```

Recommended:

* Python 3.9+
* PyTorch with CUDA (optional but faster)

---

## ▶️ How to Run (Example)

### Train Hybrid Model

```bash
python code/hybrid/train.py
```

### Evaluate

```bash
python code/hybrid/evaluate.py
```

### Generate Explainability Report

```bash
python code/hybrid/explainability.py
```

---

## 📌 Research & Academic Use

This repository is suitable for:

* Final-year engineering projects
* Research paper submission
* Multimodal ML demonstrations
* Explainable AI case studies

If you use this work, please **cite the Fakeddit dataset** and relevant transformer / CNN models.

---

## ⭐ Final Notes

* Dataset intentionally excluded (best practice)
* Code is modular, clean, and research-ready
* Explainability is a **first-class component**, not an afterthought

If you found this project helpful, consider starring ⭐ the repository.
