"""
test_pipeline.py
Unit tests for data loader, feature engineering, and inference accuracy.
"""

import os
import sys
import pytest

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_loader import load_banking_data, preprocess_banking_data, get_train_test_splits
from predict import FraudPredictionService


def test_data_loading():
    df = load_banking_data()
    assert df is not None
    assert len(df) >= 1000
    assert "is_fraud" in df.columns
    assert "transaction_type" in df.columns


def test_preprocessing_shapes():
    df = load_banking_data()
    X, y = preprocess_banking_data(df)
    assert len(X) == len(y)
    assert "amount" in X.columns
    assert "balance_error_orig" in X.columns
    assert "type_TRANSFER" in X.columns


def test_train_test_split_stratification():
    X_train, X_test, y_train, y_test = get_train_test_splits(test_size=0.2, random_state=42)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert y_train.sum() > 0
    assert y_test.sum() > 0


def test_inference_service():
    service = FraudPredictionService()
    legit_tx = {
        "transaction_type": "PAYMENT",
        "amount": 45.0,
        "old_balance": 1500.0,
        "new_balance": 1455.0,
        "hour": 14
    }
    result = service.predict_transaction(legit_tx)
    assert "fraud_probability" in result
    assert "risk_level" in result
    assert result["integrity_verified"] is True
    assert 0.0 <= result["fraud_probability"] <= 1.0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
