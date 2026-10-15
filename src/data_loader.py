"""
data_loader.py
Loads, cleans, validates, and preprocesses the banking transaction dataset.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_banking_data(data_path: str = None) -> pd.DataFrame:
    if data_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, "data", "banking_transaction_dataset.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Banking dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    return df


def preprocess_banking_data(df: pd.DataFrame):
    """
    Extracts features:
    - transaction_type (one-hot encoded)
    - amount (log transformed / scaled)
    - old_balance, new_balance
    - balance_error = (old_balance - amount) - new_balance
    - hour extracted from date_time
    """
    df = df.copy()

    # Extract hour of day
    df["date_time"] = pd.to_datetime(df["date_time"])
    df["hour"] = df["date_time"].dt.hour

    # Domain feature engineering: balance error tracking
    df["balance_error_orig"] = (df["old_balance"] - df["amount"]) - df["new_balance"]

    # One-hot encode transaction_type
    type_dummies = pd.get_dummies(df["transaction_type"], prefix="type", drop_first=False)
    # Ensure standard expected types
    expected_types = ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]
    for col in expected_types:
        if col not in type_dummies.columns:
            type_dummies[col] = 0

    feature_cols = ["amount", "old_balance", "new_balance", "balance_error_orig", "hour"]
    X = pd.concat([df[feature_cols], type_dummies[expected_types]], axis=1)
    y = df["is_fraud"]

    return X, y


def get_train_test_splits(test_size: float = 0.20, random_state: int = 42):
    df = load_banking_data()
    X, y = preprocess_banking_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
