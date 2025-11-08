# Explainable Multilingual Fake News Detection using Text and Image Features
### Major Project - Phase 1

This project focuses on detecting fake news using machine learning techniques and providing **explainable predictions**. In this phase, we have implemented a **text-based fake news classifier** with explainability. Future phases will extend to **multilingual news** and **image-based verification**.

---

##  What is Done in Phase 1
- Collected and combined **Real** and **Fake** news datasets from Kaggle.
- Preprocessed text (cleaning, stopword removal, formatting).
- Converted text into features using **TF-IDF Vectorization**.
- Trained a **Logistic Regression** model for classification.
- Integrated **LIME** to explain why the model predicted a news article as Real or Fake.
- Evaluated model performance using Accuracy, Precision, Recall, and Confusion Matrix.

---

##  Model Accuracy (Phase 1 Result)
**Accuracy:** ~ **99%**  
The model performs well and is interpretable.

---

##  Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn (Logistic Regression)
- TF-IDF Vectorizer
- LIME (Explainability)
- Matplotlib (Graphs)

---

##  Dataset Used
Kaggle Dataset: *Fake and Real News*  
- `True.csv` → Real News  
- `Fake.csv` → Fake News  

> Currently supports **English news only**.  
> Multilingual dataset will be added in Phase 2.

---

## Limitations (Phase 1)
- Only text analysis is done (No image processing yet)
- Only English language supported
- Logistic Regression may not capture deep context like neural models

---

## What Will Be Done in Next Phases
| Phase | Planned Work |
|------|--------------|
| Phase 2 | Add Multilingual News Support + Use BERT / LSTM |
| Phase 3 | Add Image Verification + Real-time Detection Dashboard |

---

## Team Members
- Devanshee Parmar
- Janvi
- Sudhriti Bari

Guide: **Prof. Monisha Mohan**

---

## Project Status
**Phase 1 Completed** (Text-based Explainable Fake News Detection)  
Further development in progress.
