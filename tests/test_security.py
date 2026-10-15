"""
test_security.py
DevSecMLOps Security Tests:
1. Model integrity verification (SHA-256)
2. Tamper resistance (rejects altered model binaries)
3. Input sanitization (rejects malicious / invalid payloads)
"""

import os
import sys
import tempfile
import pytest

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from security_scanner import (
    compute_file_sha256,
    sign_model_artifact,
    verify_model_integrity,
    safe_load_model,
    sanitize_transaction_input
)


def test_sha256_model_integrity_valid():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "fraud_model.joblib")
    checksum_path = model_path + ".sha256"

    assert os.path.exists(model_path), "Model binary must exist"
    assert os.path.exists(checksum_path), "Model checksum must exist"

    # Integrity verification must succeed for authentic model
    is_valid = verify_model_integrity(model_path, checksum_path)
    assert is_valid is True


def test_model_tampering_detection():
    """Simulates an attacker injecting malicious bytes or backdooring model weights."""
    with tempfile.NamedTemporaryFile(delete=False) as f_model:
        f_model.write(b"ORIGINAL_TRAINED_MODEL_BYTES_XYZ123")
        temp_model_path = f_model.name

    temp_sig_path = temp_model_path + ".sha256"
    sign_model_artifact(temp_model_path, temp_sig_path)

    # Verify initially valid
    assert verify_model_integrity(temp_model_path, temp_sig_path) is True

    # Attack simulation: Tamper with 1 byte of the model artifact
    with open(temp_model_path, "wb") as f_model:
        f_model.write(b"POISONED_BACKDOORED_MODEL_BYTES_ATTACK")

    # Security check MUST raise PermissionError when loaded
    with pytest.raises(PermissionError) as exc_info:
        safe_load_model(temp_model_path, temp_sig_path)

    assert "Possible model poisoning detected" in str(exc_info.value)

    # Cleanup
    os.remove(temp_model_path)
    os.remove(temp_sig_path)


def test_input_sanitization_negative_amount():
    with pytest.raises(ValueError):
        sanitize_transaction_input({"amount": -500.0, "transaction_type": "PAYMENT"})


def test_input_sanitization_invalid_type():
    with pytest.raises(ValueError):
        sanitize_transaction_input({"amount": 100.0, "transaction_type": "HACK_ATTACK"})


def test_input_sanitization_valid():
    cleaned = sanitize_transaction_input({
        "amount": "250.75",
        "old_balance": "1000",
        "new_balance": "749.25",
        "transaction_type": "cash_out",
        "hour": "15"
    })
    assert cleaned["amount"] == 250.75
    assert cleaned["transaction_type"] == "CASH_OUT"
    assert cleaned["hour"] == 15


if __name__ == "__main__":
    pytest.main(["-v", __file__])
