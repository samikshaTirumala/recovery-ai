"""RecoverAI Flask application."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone

from flask import Flask, flash, jsonify, redirect, render_template, request, url_for

import database
from generate_data import CSV_PATH, generate_invoices, write_csv
from model import ModelNotReadyError, model_files_exist, predict_invoice_risk
from recovery_engine import MAX_RECOVERY_ATTEMPTS, choose_intervention, diagnose_root_cause

app = Flask(__name__)
app.secret_key = "recoverai-hackathon-local-demo"


def format_inr(value) -> str:
    try:
        return f"₹{float(value):,.0f}"
    except (TypeError, ValueError):
        return "—"


app.jinja_env.filters["inr"] = format_inr


def seed_database() -> None:
    if database.invoice_count() > 0:
        return
    if not CSV_PATH.exists():
        write_csv(generate_invoices())
    with CSV_PATH.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            invoice = {
                "invoice_number": row["invoice_number"],
                "customer_name": row["customer_name"],
                "invoice_amount": float(row["invoice_amount"]),
                "invoice_date": row["invoice_date"],
                "due_date": row["due_date"],
                "payment_status": row["payment_status"],
                "days_overdue": int(float(row["days_overdue"])),
                "previous_late_payments": int(float(row["previous_late_payments"])),
                "customer_payment_score": float(row["customer_payment_score"]),
            }
            invoice_id = database.insert_invoice(invoice)
            if model_files_exist():
                try:
                    prediction = predict_invoice_risk(
                        invoice_amount=invoice["invoice_amount"],
                        days_overdue=invoice["days_overdue"],
                        previous_late_payments=invoice["previous_late_payments"],
                        customer_payment_score=invoice["customer_payment_score"],
                        invoice_number=invoice["invoice_number"],
                        customer_name=invoice["customer_name"],
                        due_date=invoice["due_date"],
                        payment_status=invoice["payment_status"],
                    )
                    database.update_prediction(invoice_id, prediction)
                except ModelNotReadyError:
                    pass


def bootstrap() -> None:
    database.init_db()
    seed_database()


bootstrap()


@app.context_processor
def inject_globals():
    return {
        "model_ready": model_files_exist(),
        "active_page": request.endpoint,
    }


@app.errorhandler(404)
def not_found(_error):
    return (
        render_template(
            "error.html",
            title="Page not found",
            message="The page you requested does not exist.",
        ),
        404,
    )


@app.errorhandler(500)
def server_error(_error):
    return (
        render_template(
            "error.html",
            title="Something went wrong",
            message="RecoverAI could not complete that request. Please try again.",
        ),
        500,
    )


@app.route("/")
def index():
    stats = database.get_dashboard_statistics()
    invoices = database.get_all_invoices()
    unpaid = [inv for inv in invoices if inv.get("payment_status") != "Paid"]
    high_risk = [inv for inv in unpaid if inv.get("risk_level") == "HIGH"]
    priorities = sorted(
        unpaid,
        key=lambda inv: (inv.get("priority_score") or 0),
        reverse=True,
    )[:8]
    return render_template(
        "index.html",
        stats=stats,
        high_risk=high_risk[:8],
        priorities=priorities,
    )


@app.route("/invoices")
def invoices():
    return render_template("invoices.html", invoices=database.get_all_invoices())


@app.route("/invoice/<int:invoice_id>")
def invoice_detail(invoice_id: int):
    invoice = database.get_invoice_by_id(invoice_id)
    if not invoice:
        return (
            render_template(
                "error.html",
                title="Invoice not found",
                message="That invoice is not in the RecoverAI database.",
            ),
            404,
        )
    return render_template(
        "invoice_detail.html",
        invoice=invoice,
        intervention=choose_intervention(invoice),
        root_cause=diagnose_root_cause(invoice),
        recovery_events=database.get_recovery_events(invoice_id),
        max_recovery_attempts=MAX_RECOVERY_ATTEMPTS,
    )


@app.route("/predict/<int:invoice_id>", methods=["POST"])
def predict(invoice_id: int):
    invoice = database.get_invoice_by_id(invoice_id)
    if not invoice:
        flash("Invoice not found.", "error")
        return redirect(url_for("invoices"))
    try:
        prediction = predict_invoice_risk(
            invoice_amount=invoice["invoice_amount"],
            days_overdue=invoice["days_overdue"] or 0,
            previous_late_payments=invoice["previous_late_payments"] or 0,
            customer_payment_score=invoice["customer_payment_score"] or 0,
            invoice_number=invoice["invoice_number"],
            customer_name=invoice["customer_name"],
            due_date=invoice["due_date"],
            payment_status=invoice["payment_status"],
        )
        database.update_prediction(invoice_id, prediction)
        database.log_recovery_event(invoice_id, "AI Risk Analysis", invoice.get("recovery_status") or "NOT_STARTED", prediction["risk_explanation"], actor="AI risk model")
        flash("AI risk analysis updated for this invoice.", "success")
    except ModelNotReadyError as exc:
        flash(str(exc), "error")
    except Exception:
        flash("Prediction failed. Check that the model files exist and try again.", "error")
    return redirect(url_for("invoice_detail", invoice_id=invoice_id))


def _redirect_with_message(invoice_id: int, message: str, category: str = "success"):
    flash(message, category)
    return redirect(url_for("invoice_detail", invoice_id=invoice_id))


@app.route("/recover/<int:invoice_id>", methods=["POST"])
def recover(invoice_id: int):
    invoice = database.get_invoice_by_id(invoice_id)
    if not invoice:
        return _redirect_with_message(invoice_id, "Invoice not found.", "error")
    status = invoice.get("recovery_status") or "NOT_STARTED"
    attempts = int(invoice.get("recovery_attempts") or 0)
    if invoice.get("payment_status") == "Paid" or status == "PAYMENT_RECEIVED":
        return _redirect_with_message(invoice_id, "Recovery stopped because payment is already received.", "error")
    if status in {"ESCALATED", "STOPPED"} or attempts >= MAX_RECOVERY_ATTEMPTS:
        return _redirect_with_message(invoice_id, "Automatic recovery is stopped; manual collections review is required.", "error")
    decision = choose_intervention(invoice)
    attempt = attempts + 1
    now = datetime.now(timezone.utc)
    if attempt >= MAX_RECOVERY_ATTEMPTS:
        next_status = "ESCALATED"
        next_action = "Manual collections review required"
    else:
        next_status = "REMINDER_SENT" if attempt == 1 else "FOLLOW_UP"
        next_action = decision["next_action"]
    next_date = (now + timedelta(days=1 if attempt == 1 else 2)).date().isoformat()
    database.update_recovery(
        invoice_id,
        recovery_status=next_status,
        recovery_attempts=attempt,
        last_recovery_action=decision["action"],
        next_action_date=next_date,
        recovery_started_at=invoice.get("recovery_started_at") or now.isoformat(timespec="seconds"),
    )
    database.log_recovery_event(invoice_id, "Recovery action executed", next_status, decision["reason"], invoice["invoice_amount"], attempt, notes=f"{decision['action']}; next action: {next_action}")
    if next_status == "ESCALATED":
        database.log_recovery_event(invoice_id, "Invoice escalated", next_status, "Maximum 3 recovery attempts reached.", invoice["invoice_amount"], attempt, actor="Recovery stopping rule")
        return _redirect_with_message(invoice_id, "Maximum attempts reached — manual collections review required.")
    return _redirect_with_message(invoice_id, "Recovery action executed successfully — demo reminder generated.")


@app.route("/simulate-payment/<int:invoice_id>", methods=["POST"])
def simulate_payment(invoice_id: int):
    invoice = database.get_invoice_by_id(invoice_id)
    if not invoice:
        return _redirect_with_message(invoice_id, "Invoice not found.", "error")
    if invoice.get("payment_status") == "Paid" or invoice.get("recovery_status") == "PAYMENT_RECEIVED":
        return _redirect_with_message(invoice_id, "Payment is already recorded for this invoice.", "error")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    database.update_recovery(invoice_id, recovery_status="PAYMENT_RECEIVED", amount_recovered=invoice["invoice_amount"], recovery_completed_at=now, last_recovery_action="Payment received", next_action_date=None)
    with database.get_connection() as conn:
        conn.execute("UPDATE invoices SET payment_status = 'Paid' WHERE id = ?", (invoice_id,))
        conn.commit()
    database.log_recovery_event(invoice_id, "Payment received", "PAYMENT_RECEIVED", "Synthetic demo payment recorded.", invoice["invoice_amount"], invoice.get("recovery_attempts") or 0, actor="Demo simulation", notes="No real payment occurred.")
    database.log_recovery_event(invoice_id, "Recovery stopped", "STOPPED", "Payment received; future automatic actions stopped.", invoice["invoice_amount"], invoice.get("recovery_attempts") or 0, actor="Recovery stopping rule")
    return _redirect_with_message(invoice_id, f"{format_inr(invoice['invoice_amount'])} recovered successfully — Demo Simulation")


@app.route("/escalate/<int:invoice_id>", methods=["POST"])
def escalate(invoice_id: int):
    invoice = database.get_invoice_by_id(invoice_id)
    if not invoice:
        return _redirect_with_message(invoice_id, "Invoice not found.", "error")
    if invoice.get("payment_status") == "Paid":
        return _redirect_with_message(invoice_id, "Paid invoices cannot be escalated.", "error")
    database.update_recovery(invoice_id, recovery_status="ESCALATED", last_recovery_action="Collections escalation", next_action_date=None)
    database.log_recovery_event(invoice_id, "Invoice escalated", "ESCALATED", "Manual collections review requested.", invoice["invoice_amount"], invoice.get("recovery_attempts") or 0, actor="Collections team")
    return _redirect_with_message(invoice_id, "Invoice escalated — manual collections review required.")


@app.route("/api/dashboard")
def api_dashboard():
    try:
        return jsonify(database.get_dashboard_statistics())
    except Exception:
        return jsonify({"error": "Unable to load dashboard statistics."}), 500


@app.route("/api/recovery-statistics")
def api_recovery_statistics():
    return jsonify(database.get_dashboard_statistics())


@app.route("/api/invoices")
def api_invoices():
    try:
        return jsonify(database.get_all_invoices())
    except Exception:
        return jsonify({"error": "Unable to load invoices."}), 500


@app.route("/api/risk-distribution")
def api_risk_distribution():
    try:
        stats = database.get_dashboard_statistics()
        return jsonify(
            {
                "high": stats["high_risk_count"],
                "medium": stats["medium_risk_count"],
                "low": stats["low_risk_count"],
            }
        )
    except Exception:
        return jsonify({"error": "Unable to load risk distribution."}), 500


@app.route("/api/recovery-by-risk")
def api_recovery_by_risk():
    try:
        stats = database.get_dashboard_statistics()
        grouped = stats.get("recovery_by_risk", {})
        return jsonify(
            {
                "high": grouped.get("HIGH", 0),
                "medium": grouped.get("MEDIUM", 0),
                "low": grouped.get("LOW", 0),
                "unscored": grouped.get("UNSCORED", 0),
            }
        )
    except Exception:
        return jsonify({"error": "Unable to load recovery by risk."}), 500


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
