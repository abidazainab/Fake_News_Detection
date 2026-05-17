# 📰 Fake News Detection using Machine Learning

A supervised NLP pipeline that classifies news articles as **Fake** or **Real** using TF-IDF feature extraction and four machine learning classifiers.

---

## 📌 Overview

With the rise of misinformation, automated fake news detection has become increasingly important. This project builds a text classification system trained on a large labeled dataset of news articles. Given the body of a news article, the model predicts whether it is **Fake (0)** or **Real (1)**.

---

## 📂 Dataset

**Source:** [Fake News Detection Datasets — Kaggle](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)

| File | Description | Size |
|------|-------------|------|
| `Fake.csv` | Fake news articles | ~23,480 rows |
| `True.csv` | Real news articles | ~21,417 rows |

Each file contains four columns: `title`, `text`, `subject`, and `date`. Only the `text` column is used for classification; the rest are dropped to avoid data leakage.

---

## 🔄 Pipeline

```
Raw CSVs → Label → Merge & Shuffle → Text Cleaning → TF-IDF Vectorization → Train Models → Evaluate
```

### Steps

1. **Data Loading** — Import `Fake.csv` and `True.csv`
2. **Labelling** — Assign `0` (Fake) and `1` (Real) class labels
3. **Balancing** — Reserve 10 rows from each class for manual testing; drop extras to reduce imbalance
4. **Merging & Shuffling** — Concatenate and randomly shuffle the combined dataset
5. **Text Preprocessing** — Clean article text with the `wordopt` function
6. **Feature Extraction** — Convert text to TF-IDF numerical features
7. **Model Training** — Train four classifiers on the same features
8. **Evaluation** — Assess each model with accuracy score and classification report
9. **Manual Testing** — Predict on custom news input using all four models

---

## 🧹 Text Preprocessing

The `wordopt` function normalizes raw article text through the following steps:

| Step | Operation | Removes |
|------|-----------|---------|
| Lowercase | `text.lower()` | Case inconsistency |
| Brackets | `\[.*?\]` | Citations and bracketed text |
| Non-word chars | `\W` | Symbols and special characters |
| URLs | `https?://...` | Web links |
| HTML tags | `<.*?>+` | Markup |
| Punctuation | `string.punctuation` | All punctuation |
| Newlines | `\n` | Line breaks |
| Alphanumeric tokens | `\w*\d\w*` | Words containing digits |

---

## 🤖 Models

Four classifiers are trained and compared on the same TF-IDF feature matrix:

| Model | Type | Notes |
|-------|------|-------|
| **Logistic Regression** | Linear | Fast, interpretable baseline |
| **Decision Tree** | Tree-based | Non-linear; may overfit on high-dimensional text |
| **Gradient Boosting** | Ensemble (boosting) | Sequential error correction; typically high accuracy |
| **Random Forest** | Ensemble (bagging) | Parallel trees; robust to overfitting |

> **Train/Test Split:** 75% training — 25% test (no fixed `random_state` for LR/DT; `random_state=0` for GB and RF)

---

## ⚙️ Feature Extraction

**TF-IDF (Term Frequency – Inverse Document Frequency)** transforms each article into a sparse vector of word importance scores.

- `fit_transform` is applied **only on training data** to learn the vocabulary
- `transform` (no refitting) is applied on test data to prevent data leakage

---

## 🧪 Manual Testing

After training, you can classify any custom news article through all four models simultaneously:

```python
news = "Enter your news article text here..."
manual_testing(news)
```

**Sample output:**
```
LR Prediction: Not a Fake News
DT Prediction: Not a Fake News
GB Prediction: Not a Fake News
RF Prediction: Not a Fake News
```

> **Tip:** If all four models agree, the prediction is highly confident. Disagreement indicates an ambiguous article that may warrant closer inspection.

---

## 🛠️ Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

Install all dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 🚀 Usage

1. Download `Fake.csv` and `True.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets) and place them in the project directory.
2. Open `fake_news_detection.ipynb` in Jupyter Notebook or JupyterLab.
3. Run all cells sequentially from top to bottom.
4. Use the **Manual Testing** section at the end to classify your own news articles.

---

## 📁 Project Structure

```
├── fake_news_detection.ipynb   # Main notebook
├── Fake.csv                    # Fake news articles (download from Kaggle)
├── True.csv                    # Real news articles (download from Kaggle)
└── README.md                   # This file
```

---

## ⚠️ Limitations

- Classification is based solely on article **body text** — headlines and publication metadata are excluded.
- Models may be sensitive to **political or emotionally charged language** due to dataset composition.
- The pipeline does not use stop-word removal or lemmatization; adding these could further improve accuracy.
- No cross-validation is performed — results may vary slightly between runs for models without a fixed `random_state`.

---

## 📄 License

This project uses publicly available data from Kaggle. Please refer to the [dataset license](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets) for terms of use.

NOTE: This project is inspired from: https://www.youtube.com/watch?v=U6ieiJAhXQ4
