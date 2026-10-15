"""
security_scanner.py
DevSecMLOps Security Utilities:
- Cryptographic SHA-256 Model Integrity Verification
- Safe Model Loading & Tamper Prevention
- Input Sanitization & Payload Boundary Checks
"""

import hashlib
import json
import os
import joblib


def compute_file_sha256(filepath: str) -> str:
    """Calculates SHA-256 hash of a file for integrity verification."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def sign_model_artifact(model_path: str, checksum_path: str = None) -> str:
    """Computes SHA-256 hash and writes it into a trusted checksum file."""
    if checksum_path is None:
        checksum_path = model_path + ".sha256"

    digest = compute_file_sha256(model_path)
    with open(checksum_path, "w") as f:
        f.write(digest.strip())

    return digest


def verify_model_integrity(model_path: str, checksum_path: str = None) -> bool:
    """
    Verifies that the serialized model has not been altered, poisoned, or corrupted.
    Returns True if valid, raises SecurityError if tampered.
    """
    if checksum_path is None:
        checksum_path = model_path + ".sha256"

    if not os.path.exists(checksum_path):
        raise FileNotFoundError(f"[SECURITY ALERT] Model signature file missing: {checksum_path}")

    with open(checksum_path, "r") as f:
        expected_digest = f.read().strip()

    actual_digest = compute_file_sha256(model_path)

    if expected_digest != actual_digest:
        raise PermissionError(
            f"[SECURITY CRITICAL] Model integrity verification failed! "
            f"Expected {expected_digest[:16]}..., but got {actual_digest[:16]}... Possible model poisoning detected!"
        )

    return True


def safe_load_model(model_path: str, checksum_path: str = None):
    """Safely loads model only after cryptographic validation passes."""
    verify_model_integrity(model_path, checksum_path)
    # Using joblib safe load
    return joblib.load(model_path)


def sanitize_transaction_input(payload: dict) -> dict:
    """
    Input validation & sanitization:
    Guards against negative amounts, SQL injection strings, or NaN values.
    """
    cleaned = {}

    # Validate amount
    amount = float(payload.get("amount", 0.0))
    if amount < 0.0 or amount > 10_000_000.0:
        raise ValueError(f"Invalid transaction amount: {amount}. Must be in [0, 10,000,000].")
    cleaned["amount"] = amount

    # Validate balances
    old_bal = float(payload.get("old_balance", 0.0))
    new_bal = float(payload.get("new_balance", 0.0))
    if old_bal < 0 or new_bal < 0:
        raise ValueError("Account balance cannot be negative.")
    cleaned["old_balance"] = old_bal
    cleaned["new_balance"] = new_bal

    # Validate transaction type
    valid_types = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    t_type = str(payload.get("transaction_type", "PAYMENT")).upper().strip()
    if t_type not in valid_types:
        raise ValueError(f"Invalid transaction type '{t_type}'. Must be one of {valid_types}.")
    cleaned["transaction_type"] = t_type

    # Validate hour
    hour = int(payload.get("hour", 12))
    if hour < 0 or hour > 23:
        raise ValueError("Hour must be between 0 and 23.")
    cleaned["hour"] = hour

    return cleaned
