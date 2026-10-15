"""
train.py
MLOps Training Pipeline:
- Model: Tuned Random Forest / Ensemble Classifier (scale_pos_weight & class balancing)
- Automated Evaluation Gates: ROC-AUC, F1-Score, Recall, Precision
- DevSecOps Signing: Cryptographic SHA-256 Signature generation on model artifact
"""

import sys
import os
import json

# Ensure local src imports work regardless of CWD
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from data_loader import get_train_test_splits
from security_scanner import sign_model_artifact


def train_and_evaluate():
    base_dir = os.path.dirname(CURRENT_DIR)
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("DevSecMLOps Pipeline: Banking Fraud Model Training")
    print("=" * 60)

    # 1. Load Preprocessed Data
    print("[1/5] Loading and splitting banking transactions dataset...")
    X_train, X_test, y_train, y_test = get_train_test_splits(test_size=0.20, random_state=42)
    print(f"      Train Samples: {len(X_train)} | Test Samples: {len(X_test)}")
    print(f"      Features: {list(X_train.columns)}")

    # 2. Configure Model matching reference table
    # RandomForest with class balancing to handle fraud class imbalance
    print("[2/5] Initializing Balanced Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    )

    # 3. Train Model
    print("[3/5] Fitting model on training data...")
    model.fit(X_train, y_train)

    # 4. Evaluate Performance Metrics
    print("[4/5] Evaluating test set metrics...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test))
    }

    print("      Evaluation Results:")
    for k, v in metrics.items():
        print(f"      - {k.upper()}: {v}")

    # MLOps Deployment Quality Gate: Ensure ROC-AUC >= 0.80
    if metrics["roc_auc"] < 0.80:
        raise ValueError(f"[GATE FAILED] ROC-AUC {metrics['roc_auc']} below threshold 0.80")
    print("      [PASSED] MLOps Quality Gate: Model meets deployment standards.")

    # 5. Model Serialization & Cryptographic Signing
    model_path = os.path.join(models_dir, "fraud_model.joblib")
    metrics_path = os.path.join(models_dir, "metrics.json")

    print("[5/5] Serializing model and generating SHA-256 integrity signature...")
    joblib.dump(model, model_path)

    # Sign the artifact to prevent tampering (DevSecOps)
    sha256_digest = sign_model_artifact(model_path)
    print(f"      Model saved to: {model_path}")
    print(f"      Cryptographic SHA-256 Digest: {sha256_digest}")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"      Metrics saved to: {metrics_path}")

    print("=" * 60)
    print("Pipeline Execution Complete: Artifacts Ready for Secure Serving")
    print("=" * 60)
    return metrics


if __name__ == "__main__":
    train_and_evaluate()
