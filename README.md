# automated-essay-scoring-capstone







---



\## 🧠 Methodology



\### 🧹 1. Data Preprocessing

\- Clean text (remove punctuation, non-alphabetic chars)

\- Tokenization \& lemmatization using `nltk`

\- Remove English stopwords

\- Convert essay scores to zero-indexed integer labels



\### 🧮 2. Feature Representations

| Technique | Description | Library |

|------------|--------------|----------|

| \*\*TF-IDF\*\* | Word importance weighting | `sklearn` |

| \*\*Word2Vec\*\* | Semantic embeddings from word co-occurrence | `gensim` |

| \*\*Doc2Vec\*\* | Paragraph-level embeddings for overall essay meaning | `gensim` |

| \*\*LDA\*\* | Topic modeling (50–500 latent topics) | `gensim.models.LdaModel` |

| \*\*BERT\*\* | Contextual transformer embeddings (768-D) | `transformers`, `torch` |



\### 🤖 3. Model Training

All feature types feed into an \*\*XGBoost classifier\*\* via a shared training function that:

\- Splits data (80 % train / 20 % test)

\- Trains an `xgboost` model

\- Evaluates \*\*Accuracy\*\*, \*\*Precision\*\*, \*\*Recall\*\*, and \*\*QWK\*\*

\- Displays a \*\*Confusion Matrix\*\*



\### ⚙️ 4. Optimization

\- \*\*Optuna\*\* used for LDA hyperparameter tuning (topics, passes, alpha, eta, etc.)

\- \*\*GridSearchCV\*\* fine-tuned XGBoost on combined LDA + TF-IDF features



---



\## 📊 Experimental Results



| Feature Representation | Model | Accuracy | Precision | Recall | QWK |

|-------------------------|--------|-----------|------------|---------|------|

| TF-IDF | XGBoost | 0.84 | 0.85 | 0.84 | 0.86 |

| Word2Vec | XGBoost | 0.86 | 0.87 | 0.86 | 0.88 |

| Doc2Vec | XGBoost | 0.87 | 0.88 | 0.87 | 0.89 |

| LDA (500 topics) | XGBoost | 0.88 | 0.89 | 0.88 | 0.90 |

| \*\*BERT (768-D embeddings)\*\* | XGBoost | \*\*0.94\*\* | \*\*0.95\*\* | \*\*0.94\*\* | \*\*0.95 ✅\*\* |



> The \*\*BERT + XGBoost\*\* configuration achieved the highest Quadratic Weighted Kappa (\*\*0.95\*\*), matching human-level grading consistency.



\### 🧩 Confusion Matrix Example

Each experiment outputs a visual confusion matrix:

```python

ConfusionMatrixDisplay(confusion\_matrix=cm,

&nbsp;                      display\_labels=np.arange(len(dataset\['score'].unique()))).plot()



