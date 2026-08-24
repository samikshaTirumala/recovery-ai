"""Explainable intervention and stopping rules for the demo workflow."""

from __future__ import annotations


MAX_RECOVERY_ATTEMPTS = 3


def diagnose_root_cause(invoice: dict) -> str:
    causes = []
    if (invoice.get("days_overdue") or 0) > 0:
        causes.append(f"Invoice is {invoice['days_overdue']} days overdue.")
    if (invoice.get("previous_late_payments") or 0) > 0:
        causes.append(f"Customer has {invoice['previous_late_payments']} previous late payments.")
    score = invoice.get("customer_payment_score")
    if score is not None and score < 50:
        causes.append(f"Customer payment score is {score:.0f}/100.")
    if (invoice.get("invoice_amount") or 0) >= 250000:
        causes.append("High invoice value increases potential revenue exposure.")
    return " ".join(causes) or "No significant risk driver detected in the available invoice data."


def choose_intervention(invoice: dict) -> dict:
    attempts = int(invoice.get("recovery_attempts") or 0)
    level = invoice.get("risk_level") or "MEDIUM"
    if attempts >= MAX_RECOVERY_ATTEMPTS:
        return {"action": "Collections escalation", "reason": "Three unsuccessful recovery attempts reached the stopping rule.", "escalation_level": 3, "next_action": "Manual collections review required", "stopping_rule": "No automatic recovery after 3 attempts."}
    if level == "HIGH":
        action, reason, next_action, escalation = "Urgent payment reminder", "High risk and revenue exposure require immediate contact.", "Follow up within 24 hours", 1
    elif level == "MEDIUM":
        action, reason, next_action, escalation = "Personalized reminder + follow-up", "Medium risk merits a reminder and a scheduled check-in.", "Follow up in 2 days", 2
    else:
        action, reason, next_action, escalation = "Standard payment reminder", "Low risk can use the standard reminder cadence.", "Review in 5 days", 1
    return {"action": action, "reason": reason, "escalation_level": escalation, "next_action": next_action, "stopping_rule": "Maximum 3 recovery attempts or stop immediately after payment."}