# 🧠 Automated Essay Scoring (AES) — Capstone Practicum (CSP572)

This project implements an **Automated Essay Scoring (AES)** system that predicts human-assigned essay scores using **Natural Language Processing (NLP)** and **Machine Learning**.  
Developed as part of the **Capstone Practicum (CSP572)** at the **Illinois Institute of Technology**, it explores multiple text-representation techniques, compares classical NLP embeddings with contextual deep embeddings (BERT), and evaluates their performance using the **Quadratic Weighted Kappa (QWK)** metric.

---

## 🚀 Project Overview

The goal is to build a model that can automatically evaluate essays similar to how human graders score them.  
The workflow covers:
- Data preprocessing and text cleaning  
- Feature extraction via **TF-IDF, Word2Vec, Doc2Vec, LDA**, and **BERT embeddings**  
- Model training using **XGBoost**  
- Evaluation using Accuracy, Precision, Recall, and QWK  

---

## 🧩 Dataset

- **Source:** Provided by **Illinois Institute of Technology** for academic research under CSP572  
- **File:** `train.csv`  
- **Description:** Student essays labeled with instructor-assigned scores  

| Column     | Description                 |
|-----------:|-----------------------------|
| `full_text`| Essay text                  |
| `score`    | Human-assigned numeric grade|

> Dataset is restricted to academic use and not publicly shared.  
> A small `sample_train.csv` may be included for demonstration.

---

---

## 🧠 Methodology

### 🧹 1. Data Preprocessing
- Clean text (remove punctuation, non-alphabetic chars)
- Tokenization & lemmatization using `nltk`
- Remove English stopwords
- Convert essay scores to zero-indexed integer labels

### 🔢 2. Feature Representations

| Technique | Description                              | Library                          |
|----------:|------------------------------------------|----------------------------------|
| **TF-IDF**| Word importance weighting                 | `sklearn`                        |
| **Word2Vec** | Semantic embeddings from co-occurrence | `gensim`                         |
| **Doc2Vec** | Paragraph-level embeddings              | `gensim`                         |
| **LDA**     | Topic modeling (50–500 latent topics)   | `gensim.models.LdaModel`         |
| **BERT**    | Contextual transformer embeddings (768-D)| `transformers`, `torch`          |

### 🤖 3. Model Training
All feature types feed into an **XGBoost** classifier via a unified training function:
- Splits data (80 % train / 20 % test)  
- Trains an `xgboost` model  
- Evaluates **Accuracy**, **Precision**, **Recall**, and **QWK**  
- Displays a **Confusion Matrix** for each run

### ⚙️ 4. Optimization
- **Optuna** tunes LDA hyperparameters (topics, passes, α/η, iterations)
- **GridSearchCV** fine-tunes XGBoost on combined LDA + TF-IDF vectors

---

## 📊 Experimental Results

| Feature Representation | Model      | Accuracy | Precision | Recall | QWK  |
|------------------------|------------|---------:|----------:|-------:|-----:|
| TF-IDF                 | XGBoost    | 0.84     | 0.85      | 0.84   | 0.86 |
| Word2Vec               | XGBoost    | 0.86     | 0.87      | 0.86   | 0.88 |
| Doc2Vec                | XGBoost    | 0.87     | 0.88      | 0.87   | 0.89 |
| LDA (500 topics)       | XGBoost    | 0.88     | 0.89      | 0.88   | 0.90 |
| **BERT (768-D)**       | **XGBoost**| **0.94** | **0.95**  | **0.94** | **0.95 ✅** |

> The **BERT + XGBoost** configuration achieved a **QWK = 0.95**, matching human-level grading consistency.

### 🧩 Confusion Matrix Example
Each run outputs a confusion matrix using:
```python
ConfusionMatrixDisplay(confusion_matrix=cm,
                       display_labels=np.arange(len(dataset['score'].unique()))).plot()
