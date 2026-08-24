"""Train a logistic regression model for invoice recovery risk."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
FEATURE_COLUMNS = [
    "invoice_amount",
    "days_overdue",
    "previous_late_payments",
    "customer_payment_score",
]


def generate_training_frame(n: int = 1600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    invoice_amount = rng.uniform(10_000, 500_000, n)
    days_overdue = rng.integers(0, 120, n).astype(float)
    previous_late_payments = rng.integers(0, 10, n).astype(float)
    customer_payment_score = rng.uniform(15, 98, n)

    # Latent risk: overdue + late history + weak payment score. Amount has a mild effect.
    logit = (
        -2.1
        + 0.028 * days_overdue
        + 0.42 * previous_late_payments
        + 0.0000012 * invoice_amount
        - 0.038 * customer_payment_score
    )
    probability = 1 / (1 + np.exp(-logit))
    noise = rng.normal(0, 0.08, n)
    high_risk = (probability + noise >= 0.50).astype(int)

    return pd.DataFrame(
        {
            "invoice_amount": invoice_amount,
            "days_overdue": days_overdue,
            "previous_late_payments": previous_late_payments,
            "customer_payment_score": customer_payment_score,
            "high_risk": high_risk,
        }
    )


def train_and_save(n: int = 1600) -> dict:
    frame = generate_training_frame(n=n)
    x = frame[FEATURE_COLUMNS]
    y = frame["high_risk"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_train_scaled, y_train)

    y_pred = model.predict(x_test_scaled)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "positive_rate": float(y.mean()),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "risk_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    return metrics


def main() -> None:
    metrics = train_and_save()
    print("RecoverAI risk model trained")
    print(f"Train size: {metrics['train_size']}  Test size: {metrics['test_size']}")
    print(f"Positive class rate: {metrics['positive_rate']:.3f}")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")
    print(f"Saved {MODELS_DIR / 'risk_model.pkl'}")
    print(f"Saved {MODELS_DIR / 'scaler.pkl'}")


if __name__ == "__main__":
    main()
