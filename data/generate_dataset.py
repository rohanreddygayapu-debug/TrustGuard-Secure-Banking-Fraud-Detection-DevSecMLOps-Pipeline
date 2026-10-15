"""
generate_dataset.py
Generates the benchmark banking transaction dataset (12,000 transactions)
matching the exact reference columns:
Transaction type, amount, sender, receiver, date_time, old_balance, new_balance, is_fraud
"""

import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def generate_banking_data(n_samples: int = 12000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    random.seed(random_state)

    transaction_types = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    type_p = [0.35, 0.25, 0.25, 0.10, 0.05]

    data = []
    base_date = datetime(2026, 1, 1, 0, 0, 0)

    for i in range(1, n_samples + 1):
        # 1. Transaction Type
        t_type = np.random.choice(transaction_types, p=type_p)

        # Determine fraud probability based on transaction type and patterns
        is_fraud = 0
        if t_type in ["TRANSFER", "CASH_OUT"]:
            # Fraudulent transactions usually occur in TRANSFER / CASH_OUT with large amounts
            if np.random.rand() < 0.12:
                is_fraud = 1
        elif np.random.rand() < 0.01:
            is_fraud = 1

        # 2. Amount
        if is_fraud:
            amount = round(float(np.random.uniform(5000.0, 95000.0)), 2)
            old_balance = round(float(amount + np.random.uniform(0.0, 5000.0)), 2)
            # Fraudsters typically drain account to 0 or near 0
            new_balance = round(float(max(0.0, old_balance - amount)), 2)
        else:
            amount = round(float(np.random.exponential(scale=350.0) + 10.0), 2)
            old_balance = round(float(np.random.uniform(100.0, 50000.0)), 2)
            if t_type == "CASH_IN":
                new_balance = round(float(old_balance + amount), 2)
            else:
                new_balance = round(float(max(0.0, old_balance - amount)), 2)

        # 3. Sender and Receiver
        sender = f"C{random.randint(100000, 999999)}"
        if is_fraud and t_type == "TRANSFER":
            receiver = f"M{random.randint(900000, 999999)}"
        else:
            receiver = f"C{random.randint(100000, 999999)}"

        # 4. Date and Time
        random_minutes = random.randint(0, 60 * 24 * 30)
        date_time = (base_date + timedelta(minutes=random_minutes)).strftime("%Y-%m-%d %H:%M:%S")

        data.append({
            "transaction_type": t_type,
            "amount": amount,
            "sender": sender,
            "receiver": receiver,
            "date_time": date_time,
            "old_balance": old_balance,
            "new_balance": new_balance,
            "is_fraud": is_fraud
        })

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(out_dir, "banking_transaction_dataset.csv")
    df = generate_banking_data(n_samples=12000, random_state=42)
    df.to_csv(data_path, index=False)
    print(f"Dataset generated successfully at: {data_path}")
    print(f"Shape: {df.shape}")
    print("Fraud Class Distribution:\n", df["is_fraud"].value_counts(normalize=True))
