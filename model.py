"""Load the trained risk model and produce recovery insights."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "risk_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

_model = None
_scaler = None


class ModelNotReadyError(RuntimeError):
    pass


def model_files_exist() -> bool:
    return MODEL_PATH.exists() and SCALER_PATH.exists()


def load_artifacts():
    global _model, _scaler
    if not model_files_exist():
        raise ModelNotReadyError(
            "The risk model has not been trained yet. Run: python train_model.py"
        )
    if _model is None or _scaler is None:
        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
    return _model, _scaler


def _inr(amount: float) -> str:
    return f"₹{amount:,.0f}"


def _risk_level(risk_score: float) -> str:
    if risk_score >= 70:
        return "HIGH"
    if risk_score >= 40:
        return "MEDIUM"
    return "LOW"


def _recommended_action(level: str) -> str:
    if level == "HIGH":
        return "Send urgent payment reminder and prioritize collection follow-up."
    if level == "MEDIUM":
        return "Send personalized payment reminder and monitor closely."
    return "Send standard payment reminder."


def _priority_score(risk_score: float, invoice_amount: float, days_overdue: int) -> float:
    """0–100 collection priority. High value + high risk + long overdue ranks first."""
    amount_norm = min(invoice_amount / 500_000, 1.0) * 100
    overdue_norm = min(max(days_overdue, 0) / 90, 1.0) * 100
    score = 0.45 * risk_score + 0.35 * amount_norm + 0.20 * overdue_norm
    return round(float(score), 1)


def _risk_explanation(
    days_overdue: int,
    previous_late_payments: int,
    customer_payment_score: float,
    risk_level: str,
    invoice_amount: float,
) -> str:
    overdue_text = (
        f"the invoice is {days_overdue} days overdue"
        if days_overdue > 0
        else "the invoice is not overdue"
    )
    level_word = {"HIGH": "High", "MEDIUM": "Medium", "LOW": "Low"}[risk_level]
    amount_hint = ""
    if invoice_amount >= 250_000:
        amount_hint = f" The invoice value of {_inr(invoice_amount)} also increases collection priority."
    return (
        f"{level_word} risk because {overdue_text}, the customer has "
        f"{previous_late_payments} previous late payment"
        f"{'' if previous_late_payments == 1 else 's'}, and the customer payment score is "
        f"{customer_payment_score:.0f}/100.{amount_hint}"
    )


def _payment_reminder(
    invoice_number: str,
    customer_name: str,
    invoice_amount: float,
    days_overdue: int,
    due_date: str | None,
    payment_status: str | None,
) -> str:
    if payment_status == "Paid":
        body = (
            f"Dear {customer_name},\n\n"
            f"Thank you. Invoice {invoice_number} for {_inr(invoice_amount)} is marked as paid "
            f"in our records. No further action is required.\n\n"
            "Thank you,\nAccounts Receivable Team"
        )
        return f"Subject: Payment Confirmation – Invoice {invoice_number}\n\n{body}"

    if days_overdue > 0:
        status_line = (
            f"Our records indicate that invoice {invoice_number} for {_inr(invoice_amount)} "
            f"is currently overdue by {days_overdue} days."
        )
        subject = f"Payment Reminder – Invoice {invoice_number}"
    else:
        due_text = f" (due {due_date})" if due_date else ""
        status_line = (
            f"This is a courtesy reminder that invoice {invoice_number} for "
            f"{_inr(invoice_amount)} remains outstanding{due_text}."
        )
        subject = f"Upcoming Payment – Invoice {invoice_number}"

    body = (
        f"Dear {customer_name},\n\n"
        f"{status_line}\n\n"
        "We kindly request that you arrange payment at the earliest convenience.\n\n"
        "If payment has already been processed, please disregard this message.\n\n"
        "Thank you,\nAccounts Receivable Team"
    )
    return f"Subject: {subject}\n\n{body}"


def predict_invoice_risk(
    invoice_amount: float,
    days_overdue: int,
    previous_late_payments: int,
    customer_payment_score: float,
    invoice_number: str = "INV-0000",
    customer_name: str = "Customer",
    due_date: str | None = None,
    payment_status: str | None = None,
) -> dict:
    model, scaler = load_artifacts()
    features = pd.DataFrame(
        [[invoice_amount, days_overdue, previous_late_payments, customer_payment_score]],
        columns=[
            "invoice_amount",
            "days_overdue",
            "previous_late_payments",
            "customer_payment_score",
        ],
        dtype=float,
    )
    scaled = scaler.transform(features)
    risk_probability = float(model.predict_proba(scaled)[0][1])
    risk_score = round(risk_probability * 100, 1)
    risk_level = _risk_level(risk_score)
    recovery_probability = round(1 - risk_probability, 4)
    expected_recovery = round(float(invoice_amount) * recovery_probability, 2)
    priority_score = _priority_score(risk_score, float(invoice_amount), int(days_overdue))

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "recovery_probability": recovery_probability,
        "expected_recovery": expected_recovery,
        "priority_score": priority_score,
        "recommended_action": _recommended_action(risk_level),
        "risk_explanation": _risk_explanation(
            int(days_overdue),
            int(previous_late_payments),
            float(customer_payment_score),
            risk_level,
            float(invoice_amount),
        ),
        "payment_reminder": _payment_reminder(
            invoice_number,
            customer_name,
            float(invoice_amount),
            int(days_overdue),
            due_date,
            payment_status,
        ),
    }
