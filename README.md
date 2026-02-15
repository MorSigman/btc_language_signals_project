# Bitcoin Sentiment Analysis: Can Tweets Predict Price Direction?

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-yellowgreen)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

A data science project investigating whether sentiment extracted from Bitcoin-related tweets can predict future Bitcoin price direction (30-day returns). The study employs both classic NLP (TF-IDF + Logistic Regression) and deep learning (Bidirectional LSTM) under a rigorous hypothesis-driven evaluation framework.

---

## Research Question

> **Can tweet sentiment help distinguish between future 30-day positive returns ("Pump") versus negative/flat returns ("Risk")?**

---

## Dataset

| Property | Value |
| :--- | :--- |
| **Source** | Bitcoin-related tweets |
| **Total size (after cleaning)** | ~66,000 tweets |
| **Language** | English |
| **Time span** | February 2021 |
| **Train set** | 155 days / 46,062 tweets |
| **Validation set** | ~33 days / ~9,945 tweets |
| **Test set** | ~35 days / ~10,179 tweets |

**Label definition:**
- `1` = **Pump** — 30-day return (r30) > 0
- `0` = **Risk** — 30-day return (r30) ≤ 0

The split is **temporal** (not random) so that all tweets from the same day stay in the same partition, preventing any temporal contamination.

---

## Pipeline

### 1. Text Preprocessing
- Lowercase normalization
- URL removal
- Mentions replaced with `MENTION`
- Price references replaced with `TICKER`
- Special characters and emojis stripped
- Whitespace normalization
- Duplicates and tweets with fewer than 3 tokens removed

### 2. Feature Extraction
| Model | Method |
| :--- | :--- |
| Baseline | TF-IDF vectorizer (5,000 features, fit on train only) |
| BiLSTM | Keras Tokenizer (10,000-word vocab), `pad_sequences` at 95th-percentile length |

### 3. Models

#### Baseline — TF-IDF + Logistic Regression
- `class_weight='balanced'` to handle class imbalance
- L2 regularization, `max_iter=1000`

#### Deep Learning — Bidirectional LSTM
```
Embedding (128 dim, trainable)
    → Bidirectional LSTM (64 units)
    → Dropout (0.5)
    → Dense (1, sigmoid)
```
- Optimizer: Adam
- Loss: Binary Crossentropy
- Training: up to 20 epochs with Early Stopping (`monitor=val_auc`, `patience=3`)
- Batch size: 32
- Total parameters: >1 million

---

## Results

### Tweet-Level Classification Metrics

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + LogReg (Baseline)** | 0.5486 | 0.5305 | 0.3563 | 0.4263 | **0.5599** |
| **BiLSTM** | 0.5419 | 0.5173 | 0.3989 | 0.4504 | 0.5426 |

Both models exceed random chance (AUC > 0.50). The Baseline achieves slightly higher precision and AUC; BiLSTM achieves slightly higher recall and F1.

### Day-Level Event Study

Predicted probabilities were aggregated per day and split by quantile:

- **Baseline:** Statistically significant spread between Pump and Risk days (p < 0.05) ✅
- **BiLSTM:** Positive spread but 95% Bootstrap CI crosses zero — not statistically significant ❌

The Baseline model captures a **weak but detectable** predictive signal at day-level aggregation.

### Key Observations
1. **Linear signal dominates** — TF-IDF + Logistic Regression matches or beats BiLSTM, suggesting the predictive signal is keyword-based rather than sequential/contextual.
2. **Noise at tweet level** — Day-level aggregation reduces noise and makes the latent signal clearer.
3. **Overfitting in BiLSTM** — Training AUC reached 0.93 while validation plateaued at ~0.51, indicating the 1M+ parameter model requires more data.
4. **Simple models win on noisy short text** — Interpretable baselines are preferable when data is limited and noisy.

---

## Visualizations

- ROC curves (Baseline vs. BiLSTM)
- BiLSTM training history (Loss & AUC per epoch)
- Day-level event study distributions
- Confusion matrices

---

## Tech Stack

| Category | Libraries |
| :--- | :--- |
| Data processing | `pandas`, `numpy` |
| Machine learning | `scikit-learn` |
| Deep learning | `TensorFlow / Keras` |
| Statistics | `scipy` (Welch t-test, Bootstrap CI) |
| Visualization | `matplotlib` |

---

## Project Structure

```text
bitcoin_sentiment_analysis_66K/
├── bitcoin_sentiment_analysis_66K.ipynb   # Main notebook (66K tweets)
├── bitcoin_sentiment_analysis_330K.ipynb  # Extended notebook (330K tweets)
├── DATA-FINAL.CSV                         # Cleaned and labeled tweet dataset
└── README.md                              # Project documentation
```

---

## Conclusions

Twitter sentiment about Bitcoin contains a **weak but statistically detectable signal** for predicting future price direction. The signal is primarily linear and keyword-based:

- Simple, interpretable models (TF-IDF + Logistic Regression) outperform deep sequence models on this dataset size and noise level.
- Larger and more diverse datasets, or richer feature engineering (sentiment lexicons, topic modeling, transformer embeddings), may be required to improve predictive power.
- Day-level aggregation is essential: tweet-level noise is too high for reliable individual predictions.

---

## Authors

*Deep Learning / NLP course project, 2026*
