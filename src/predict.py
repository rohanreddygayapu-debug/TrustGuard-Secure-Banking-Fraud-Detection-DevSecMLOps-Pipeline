"""
predict.py
Secure Model Inference Engine:
- Verifies model integrity before loading
- Sanitizes incoming payload
- Calculates fraud probability & risk category
- Logs audit entry
"""

import sys
import os
import pandas as pd
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from security_scanner import safe_load_model, sanitize_transaction_input


class FraudPredictionService:
    def __init__(self, model_path: str = None):
        base_dir = os.path.dirname(CURRENT_DIR)
        if model_path is None:
            model_path = os.path.join(base_dir, "models", "fraud_model.joblib")
        self.model_path = model_path
        # Safe load verifies SHA-256 digest before loading into memory
        self.model = safe_load_model(self.model_path)
        self.audit_log = []

    def predict_transaction(self, raw_data: dict) -> dict:
        """
        Runs sanitized prediction with audit logging.
        """
        # 1. DevSec: Sanitize and validate inputs
        sanitized = sanitize_transaction_input(raw_data)

        # 2. Extract engineered features
        amount = sanitized["amount"]
        old_bal = sanitized["old_balance"]
        new_bal = sanitized["new_balance"]
        bal_err = (old_bal - amount) - new_bal
        hour = sanitized["hour"]
        t_type = sanitized["transaction_type"]

        # Feature order must match training data:
        # ['amount', 'old_balance', 'new_balance', 'balance_error_orig', 'hour',
        #  'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
        feature_dict = {
            "amount": amount,
            "old_balance": old_bal,
            "new_balance": new_bal,
            "balance_error_orig": bal_err,
            "hour": hour,
            "type_CASH_IN": 1 if t_type == "CASH_IN" else 0,
            "type_CASH_OUT": 1 if t_type == "CASH_OUT" else 0,
            "type_DEBIT": 1 if t_type == "DEBIT" else 0,
            "type_PAYMENT": 1 if t_type == "PAYMENT" else 0,
            "type_TRANSFER": 1 if t_type == "TRANSFER" else 0
        }

        X_input = pd.DataFrame([feature_dict])

        # 3. Model Inference
        is_fraud_pred = int(self.model.predict(X_input)[0])
        fraud_proba = float(self.model.predict_proba(X_input)[0][1])

        # Determine Risk Level
        if fraud_proba >= 0.70:
            risk_level = "CRITICAL / FRAUD"
            recommendation = "BLOCK_TRANSACTION"
        elif fraud_proba >= 0.40:
            risk_level = "MEDIUM / SUSPICIOUS"
            recommendation = "REQUIRE_2FA_CHALLENGE"
        else:
            risk_level = "LOW / SAFE"
            recommendation = "APPROVE"

        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "transaction_type": t_type,
            "amount": amount,
            "fraud_probability": round(fraud_proba, 4),
            "is_fraud_prediction": is_fraud_pred,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "integrity_verified": True
        }

        self.audit_log.append(result)
        return result


if __name__ == "__main__":
    service = FraudPredictionService()
    sample = {
        "transaction_type": "TRANSFER",
        "amount": 25000.0,
        "old_balance": 25000.0,
        "new_balance": 0.0,
        "hour": 3
    }
    pred = service.predict_transaction(sample)
    print("Sample Inference Result:")
    print(pred)
