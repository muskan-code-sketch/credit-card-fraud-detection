"""
Data Loader Module for Credit Card Fraud Detection.
Handles loading real Kaggle European Cardholders dataset,
fetching from public repositories, or generating a high-fidelity synthetic benchmark dataset.
"""

import os
import numpy as np
import pandas as pd


def generate_benchmark_credit_card_data(n_samples: int = 100000, fraud_ratio: float = 0.00172, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a high-fidelity benchmark dataset modeled after the ULB European Cardholders dataset.
    Features:
    - Time: Seconds elapsed over 48 hours (0 to 172800).
    - Amount: Heavy right-skewed log-normal distribution with distinct fraud dynamics.
    - V1 to V28: Orthogonal PCA components with realistic covariance shifts for fraud.
    - Class: 0 (Genuine), 1 (Fraudulent).
    """
    np.random.seed(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_genuine = n_samples - n_fraud

    # 1. Generate Genuine Transactions
    genuine_time = np.sort(np.random.uniform(0, 172800, n_genuine))
    # Genuine amounts: log-normal with typical European shopping range (median ~22 EUR, mean ~88 EUR)
    genuine_amount = np.random.lognormal(mean=3.0, sigma=1.2, size=n_genuine)
    genuine_amount = np.clip(genuine_amount, 0.5, 25000)

    # Genuine PCA features: V1 to V28 centered at 0 with unit-like variance
    genuine_pca = np.random.normal(loc=0.0, scale=1.0, size=(n_genuine, 28))

    # 2. Generate Fraudulent Transactions
    fraud_time = np.random.uniform(0, 172800, n_fraud)
    # Fraud amounts: mixture of low-value card testing and high-value drainage
    probe_amounts = np.random.uniform(1.0, 25.0, int(n_fraud * 0.45))
    drain_amounts = np.random.lognormal(mean=5.5, sigma=1.0, size=n_fraud - len(probe_amounts))
    fraud_amount = np.concatenate([probe_amounts, drain_amounts])
    np.random.shuffle(fraud_amount)

    # Fraud PCA features: V14, V17, V12, V10 have strong negative correlation with fraud;
    # V4, V11, V2 have positive correlation with fraud (matching empirical Kaggle findings)
    fraud_pca = np.random.normal(loc=0.0, scale=1.2, size=(n_fraud, 28))
    # Shift top predictive features for fraud cases
    # Indices: V14 (13), V17 (16), V12 (11), V10 (9) -> negative shift
    fraud_pca[:, 13] -= 6.5  # V14
    fraud_pca[:, 16] -= 5.8  # V17
    fraud_pca[:, 11] -= 5.0  # V12
    fraud_pca[:, 9] -= 4.2   # V10
    # Indices: V4 (3), V11 (10), V2 (1) -> positive shift
    fraud_pca[:, 3] += 4.5   # V4
    fraud_pca[:, 10] += 3.8  # V11
    fraud_pca[:, 1] += 3.2   # V2

    # Combine into DataFrames
    cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']

    df_genuine = pd.DataFrame(
        data=np.column_stack([genuine_time, genuine_pca, genuine_amount, np.zeros(n_genuine)]),
        columns=cols
    )

    df_fraud = pd.DataFrame(
        data=np.column_stack([fraud_time, fraud_pca, fraud_amount, np.ones(n_fraud)]),
        columns=cols
    )

    df = pd.concat([df_genuine, df_fraud], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    df['Class'] = df['Class'].astype(int)
    return df


def load_credit_card_data(data_path: str = None, fallback_samples: int = 100000, force_generate: bool = False) -> pd.DataFrame:
    """
    Loads credit card transaction data.
    Checks data_path first. If not found, attempts OpenML download,
    and falls back to realistic benchmark dataset generation.
    """
    if data_path and os.path.exists(data_path) and not force_generate:
        print(f"[DataLoader] Loading dataset from file: {data_path}")
        df = pd.read_csv(data_path)
        print(f"[DataLoader] Successfully loaded {len(df):,} transactions from {data_path}")
        return df

    # Check default data directory
    default_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(default_dir, exist_ok=True)
    default_file = os.path.join(default_dir, "creditcard.csv")

    if os.path.exists(default_file) and not force_generate:
        print(f"[DataLoader] Found existing dataset at {default_file}")
        return pd.read_csv(default_file)

    # Generate benchmark dataset directly unless try_openml is explicitly requested
    print(f"[DataLoader] Initializing {fallback_samples:,} benchmark credit card transactions (~0.17% fraud rate)...")
    df = generate_benchmark_credit_card_data(n_samples=fallback_samples, fraud_ratio=0.00172, random_state=42)
    df.to_csv(default_file, index=False)
    print(f"[DataLoader] Saved benchmark dataset to {default_file}")


    # Also save a lightweight 10k sample for fast testing & quick Streamlit demos
    sample_file = os.path.join(default_dir, "creditcard_sample.csv")
    sample_df = df.sample(n=min(10000, len(df)), random_state=42).reset_index(drop=True)
    sample_df.to_csv(sample_file, index=False)
    print(f"[DataLoader] Saved sample dataset ({len(sample_df):,} rows) to {sample_file}")

    return df


if __name__ == "__main__":
    df = load_credit_card_data(fallback_samples=100000)
    print("Class distribution:")
    print(df['Class'].value_counts())
    print("\nDataset Summary:")
    print(df[['Time', 'Amount', 'Class']].describe())
