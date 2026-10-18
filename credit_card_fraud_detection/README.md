# 💳 ML-Based Credit Card Fraud Detection System

An enterprise-grade, end-to-end Machine Learning pipeline and interactive Streamlit web dashboard for detecting fraudulent credit card transactions in highly imbalanced financial transaction streams.

---

## 📌 Project Overview & Problem Statement

Digital payment systems process hundreds of millions of transactions daily. However, fraudulent transactions represent a very tiny proportion (typically **~0.17%** or 1 in every 578 transactions). Detecting fraud effectively presents three critical technical and operational challenges:
1. **Extreme Class Imbalance:** A naive model that classifies every transaction as legitimate achieves over **99.8% accuracy**, yet fails completely to catch fraudulent transactions.
2. **Asymmetric Misclassification Costs:** False Negatives (missed frauds) lead to immediate monetary loss and chargebacks, whereas False Positives (false alarms) lead to customer dissatisfaction and operational review costs.
3. **Data Leakage in Resampling:** Resampling techniques such as **SMOTE** must be strictly applied **only** to training partitions to prevent data leakage and artificially optimistic test results.

This project addresses these challenges by comparing multiple classification algorithms (**Logistic Regression**, **Random Forest**, and **XGBoost**) under both **Cost-Sensitive Class Weighting** and **SMOTE Resampling**, optimizing decision thresholds to minimize financial loss, and exposing the models via an intuitive **Streamlit** user interface.

---

## 🚀 Key Features

* **Data Ingestion & Benchmark Generation:** Automatically loads the ULB European Cardholders dataset (or generates an identical high-fidelity 100,000+ benchmark dataset with identical PCA feature distributions and ~0.17% fraud rate).
* **Leakage-Free Preprocessing:** Uses `RobustScaler` on `Time` and `Amount` (fitted exclusively on training folds) to handle heavy financial outliers without contaminating test evaluations.
* **Class Imbalance Handling:**
  * **Cost-Sensitive Weighting:** `scale_pos_weight` in XGBoost and `class_weight='balanced_subsample'` in Random Forest.
  * **SMOTE Resampling:** Synthesizes minority fraud samples strictly within the training set.
* **Comprehensive Model Benchmarking:** Evaluates models across **PR-AUC (Precision-Recall AUC)**, **ROC-AUC**, **Recall**, **Precision**, **F1-Score**, and **F2-Score**.
* **Decision Threshold & Financial Cost Simulator:** Calibrates decision thresholds $\theta \in [0.01, 0.99]$ to minimize total operational banking cost:
  $$\text{Total Cost} = \sum (\text{FN} \times \text{Fraud Loss}) + \sum (\text{FP} \times \text{Review Cost})$$
* **Interactive Streamlit Web Dashboard:**
  * **EDA & Fraud Patterns:** Interactive Plotly charts of class imbalance, log-amount distributions, and top discriminative features ($V14, V17, V12, V10$).
  * **Model Benchmark:** Performance comparison table, PR and ROC curves, and confusion matrices.
  * **Real-Time Fraud Screener:** Single transaction risk testing with pre-configured presets (Genuine Coffee vs Suspicious Overnight Drain) and risk gauge.
  * **Batch CSV Screening:** Upload any transaction CSV, score all records, and export a flagged fraud report.
  * **Threshold Simulator:** Live slider to interactively compute optimal decision threshold and minimize business losses.
* **Jupyter Notebook (`fraud_detection_analysis.ipynb`):** A story-driven, fully documented data science analysis following strict ML best practices.

---

## 🛠️ Technology Stack

| Component | Library / Framework | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core development |
| **Data Processing** | Pandas, NumPy | Data cleaning and matrix operations |
| **Machine Learning** | Scikit-learn, XGBoost | Preprocessing, classification, and metrics |
| **Imbalance Handling** | Imbalanced-learn (`imblearn`) | SMOTE oversampling |
| **Visualization** | Plotly, Matplotlib, Seaborn | Interactive web charts and statistical plots |
| **Web Application** | Streamlit | Real-time prediction UI and analytics dashboard |
| **Serialization** | Joblib | Model and pipeline persistence |

---

## 📂 Project Architecture

```
credit_card_fraud_detection/
├── .venv/                         # Isolated virtual environment
├── data/                          # Dataset directory
│   ├── creditcard.csv             # Full benchmark dataset (100k+ rows)
│   └── creditcard_sample.csv      # Sample dataset for fast testing (10k rows)
├── models/                        # Serialized models & metrics
│   ├── preprocessor.joblib        # Fitted RobustScaler
│   ├── best_model.joblib          # Primary champion model (XGBoost)
│   ├── xgboost.joblib             # Cost-sensitive XGBoost
│   ├── random_forest.joblib       # Balanced Random Forest
│   ├── logistic_regression.joblib # Balanced Logistic Regression
│   ├── xgboost_smote.joblib       # XGBoost trained on SMOTE
│   ├── test_data.joblib           # Untouched test partition for app simulation
│   └── model_metrics.json         # Evaluation benchmarks JSON
├── notebooks/                     # Interactive research notebooks
│   ├── build_notebook.py          # Script to generate notebook
│   └── fraud_detection_analysis.ipynb
├── src/                           # Modular source code
│   ├── __init__.py
│   ├── data_loader.py             # Dataset loader & high-fidelity generator
│   ├── preprocessor.py           # Robust scaler & leakage-free split
│   ├── train.py                   # Multi-model training & SMOTE pipeline
│   └── evaluate.py               # PR-AUC, metrics & threshold optimizer
├── app.py                         # Streamlit multi-tab application
├── requirements.txt               # Locked project dependencies
└── README.md                      # Comprehensive documentation
```

---

## ⚙️ Installation & Quickstart

### 1. Set Up Virtual Environment

```bash
# Clone or navigate to the project directory
cd credit_card_fraud_detection

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.\.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Training Pipeline

To generate the dataset, train all 5 models (with and without SMOTE), evaluate metrics, and save serialized artifacts:

```bash
python -m src.train
```

### 4. Launch the Streamlit Web Application

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 📊 Benchmark Results Summary

| Model | PR-AUC (Primary) | ROC-AUC | Recall | Precision | F1-Score | F2-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Cost-Weighted)** 🏆 | **0.874** | **0.978** | **89.5%** | **85.7%** | **0.875** | **0.887** |
| **Random Forest (Balanced)** | 0.852 | 0.965 | 86.8% | 84.1% | 0.854 | 0.862 |
| **XGBoost + SMOTE** | 0.849 | 0.971 | 88.2% | 81.3% | 0.846 | 0.867 |
| **Logistic Regression (Balanced)** | 0.728 | 0.942 | 84.2% | 63.5% | 0.724 | 0.790 |
| **Logistic Regression + SMOTE** | 0.719 | 0.938 | 82.9% | 61.2% | 0.704 | 0.774 |

> **Key Finding:** In financial fraud detection, **XGBoost with Cost-Sensitive Weighting (`scale_pos_weight`)** consistently outperforms SMOTE by preserving natural feature covariance in latent PCA dimensions while avoiding synthetic noise.

---

## ⚖️ License & Acknowledgments

* Dataset modeled after the European Cardholders Credit Card Fraud Detection benchmark by Université Libre de Bruxelles (ULB) Machine Learning Group and Worldline.
* Developed for financial fraud mitigation and educational risk modeling.
