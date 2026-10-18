"""
Preprocessor Module for Credit Card Fraud Detection.
Handles data cleaning, leakage-free scaling, feature engineering,
and stratified train-test splitting.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


class FraudDataPreprocessor:
    """
    Robust Preprocessing Pipeline for Credit Card Fraud Data.
    - Scales Time and Amount using RobustScaler (resistant to financial outliers).
    - Preserves pre-computed PCA components V1-V28 without double-scaling.
    - Fits scalers strictly on training data only to avoid data leakage.
    """

    def __init__(self, time_scaler=None, amount_scaler=None):
        self.time_scaler = time_scaler or RobustScaler()
        self.amount_scaler = amount_scaler or RobustScaler()
        self.is_fitted = False
        self.feature_columns = None

    def fit(self, X: pd.DataFrame):
        """Fit scalers strictly on training features."""
        X_df = X.copy()
        if 'Time' in X_df.columns:
            self.time_scaler.fit(X_df[['Time']])
        if 'Amount' in X_df.columns:
            self.amount_scaler.fit(X_df[['Amount']])

        # Record expected feature column order (excluding target)
        self.feature_columns = [col for col in X_df.columns if col != 'Class']
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scalers."""
        if not self.is_fitted:
            raise ValueError("FraudDataPreprocessor must be fitted before calling transform!")

        X_df = X.copy()
        if 'Time' in X_df.columns:
            X_df['Time'] = self.time_scaler.transform(X_df[['Time']])
        if 'Amount' in X_df.columns:
            X_df['Amount'] = self.amount_scaler.transform(X_df[['Amount']])

        # Ensure correct column ordering
        if self.feature_columns:
            cols_to_use = [col for col in self.feature_columns if col in X_df.columns]
            return X_df[cols_to_use]
        return X_df

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit on X and transform X."""
        return self.fit(X).transform(X)

    def save(self, filepath: str):
        """Persist fitted preprocessor."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[Preprocessor] Saved preprocessor state to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Load pre-trained preprocessor."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No preprocessor file found at {filepath}")
        return joblib.load(filepath)


def prepare_train_test_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Validates, splits, and preprocesses credit card dataset without leakage.
    - Validates no NaN/Null values.
    - Performs stratified train-test split to preserve fraud ratio in both sets.
    - Fits RobustScaler ONLY on X_train.
    - Transforms X_train and X_test independently.

    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    # 1. Null check and data integrity validation
    null_counts = df.isnull().sum().sum()
    if null_counts > 0:
        print(f"[Preprocessor] Warning: {null_counts} null values detected. Imputing with median.")
        df = df.fillna(df.median(numeric_only=True))

    X = df.drop(columns=['Class'])
    y = df['Class'].astype(int)

    # 2. Stratified train-test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # 3. Fit preprocessor strictly on training data
    preprocessor = FraudDataPreprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    print(f"[Preprocessor] Train set: {len(X_train):,} samples (Frauds: {y_train.sum():,}, {y_train.mean():.4%})")
    print(f"[Preprocessor] Test set:  {len(X_test):,} samples (Frauds: {y_test.sum():,}, {y_test.mean():.4%})")

    return X_train, X_test, y_train, y_test, preprocessor
