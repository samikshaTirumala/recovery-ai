"""SQLite helpers for RecoverAI invoices."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "database"
DB_PATH = DB_DIR / "recoverai.db"

INVOICE_COLUMNS = [
    "id",
    "invoice_number",
    "customer_name",
    "invoice_amount",
    "invoice_date",
    "due_date",
    "payment_status",
    "days_overdue",
    "previous_late_payments",
    "customer_payment_score",
    "risk_score",
    "risk_level",
    "recovery_probability",
    "expected_recovery",
    "priority_score",
    "recommended_action",
    "risk_explanation",
    "payment_reminder",
    "recovery_status",
    "recovery_attempts",
    "amount_recovered",
    "last_recovery_action",
    "next_action_date",
    "recovery_started_at",
    "recovery_completed_at",
]

RECOVERY_COLUMNS = {
    "recovery_status": "TEXT DEFAULT 'NOT_STARTED'",
    "recovery_attempts": "INTEGER DEFAULT 0",
    "amount_recovered": "REAL DEFAULT 0",
    "last_recovery_action": "TEXT",
    "next_action_date": "TEXT",
    "recovery_started_at": "TEXT",
    "recovery_completed_at": "TEXT",
}


def get_connection() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS invoices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                invoice_number TEXT UNIQUE NOT NULL,
                customer_name TEXT NOT NULL,
                invoice_amount REAL NOT NULL,
                invoice_date TEXT,
                due_date TEXT,
                payment_status TEXT,
                days_overdue INTEGER,
                previous_late_payments INTEGER,
                customer_payment_score REAL,
                risk_score REAL,
                risk_level TEXT,
                recovery_probability REAL,
                expected_recovery REAL,
                priority_score REAL,
                recommended_action TEXT,
                risk_explanation TEXT,
                payment_reminder TEXT,
                recovery_status TEXT DEFAULT 'NOT_STARTED',
                recovery_attempts INTEGER DEFAULT 0,
                amount_recovered REAL DEFAULT 0,
                last_recovery_action TEXT,
                next_action_date TEXT,
                recovery_started_at TEXT,
                recovery_completed_at TEXT
            )
            """
        )
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(invoices)")}
        for column, definition in RECOVERY_COLUMNS.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE invoices ADD COLUMN {column} {definition}")
        conn.execute("UPDATE invoices SET recovery_status = 'PAYMENT_RECEIVED' WHERE payment_status = 'Paid'")
        conn.execute("UPDATE invoices SET recovery_status = 'NOT_STARTED' WHERE recovery_status IS NULL OR recovery_status = ''")
        conn.execute("UPDATE invoices SET amount_recovered = invoice_amount WHERE payment_status = 'Paid' AND COALESCE(amount_recovered, 0) = 0")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recovery_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                invoice_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT,
                amount REAL DEFAULT 0,
                attempt_number INTEGER DEFAULT 0,
                actor TEXT NOT NULL,
                notes TEXT,
                FOREIGN KEY (invoice_id) REFERENCES invoices(id)
            )
            """
        )
        conn.commit()


def insert_invoice(invoice: dict[str, Any]) -> int:
    sql = """
        INSERT INTO invoices (
            invoice_number, customer_name, invoice_amount, invoice_date, due_date,
            payment_status, days_overdue, previous_late_payments, customer_payment_score,
            risk_score, risk_level, recovery_probability, expected_recovery,
            priority_score, recommended_action, risk_explanation, payment_reminder,
            recovery_status, recovery_attempts, amount_recovered
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    values = (
        invoice["invoice_number"],
        invoice["customer_name"],
        invoice["invoice_amount"],
        invoice.get("invoice_date"),
        invoice.get("due_date"),
        invoice.get("payment_status"),
        invoice.get("days_overdue", 0),
        invoice.get("previous_late_payments", 0),
        invoice.get("customer_payment_score"),
        invoice.get("risk_score"),
        invoice.get("risk_level"),
        invoice.get("recovery_probability"),
        invoice.get("expected_recovery"),
        invoice.get("priority_score"),
        invoice.get("recommended_action"),
        invoice.get("risk_explanation"),
        invoice.get("payment_reminder"),
        "PAYMENT_RECEIVED" if invoice.get("payment_status") == "Paid" else "NOT_STARTED",
        0,
        invoice.get("invoice_amount", 0) if invoice.get("payment_status") == "Paid" else 0,
    )
    with get_connection() as conn:
        cursor = conn.execute(sql, values)
        conn.commit()
        return int(cursor.lastrowid)


def get_all_invoices() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM invoices ORDER BY COALESCE(priority_score, 0) DESC, invoice_amount DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_invoice_by_id(invoice_id: int) -> Optional[dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM invoices WHERE id = ?", (invoice_id,)
        ).fetchone()
    return dict(row) if row else None


def invoice_count() -> int:
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM invoices").fetchone()
    return int(row["n"]) if row else 0


def update_prediction(invoice_id: int, prediction: dict[str, Any]) -> None:
    sql = """
        UPDATE invoices
        SET risk_score = ?,
            risk_level = ?,
            recovery_probability = ?,
            expected_recovery = ?,
            priority_score = ?,
            recommended_action = ?,
            risk_explanation = ?,
            payment_reminder = ?
        WHERE id = ?
    """
    values = (
        prediction["risk_score"],
        prediction["risk_level"],
        prediction["recovery_probability"],
        prediction["expected_recovery"],
        prediction["priority_score"],
        prediction["recommended_action"],
        prediction["risk_explanation"],
        prediction["payment_reminder"],
        invoice_id,
    )
    with get_connection() as conn:
        conn.execute(sql, values)
        conn.commit()


def update_recovery(invoice_id: int, **fields: Any) -> None:
    updates = [(key, value) for key, value in fields.items() if key in RECOVERY_COLUMNS]
    if not updates:
        return
    sql = "UPDATE invoices SET " + ", ".join(f"{key} = ?" for key, _ in updates) + " WHERE id = ?"
    with get_connection() as conn:
        conn.execute(sql, [value for _, value in updates] + [invoice_id])
        conn.commit()


def log_recovery_event(
    invoice_id: int,
    action: str,
    status: str,
    reason: str = "",
    amount: float = 0,
    attempt_number: int = 0,
    actor: str = "RecoverAI demo",
    notes: str = "",
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO recovery_events
                (invoice_id, timestamp, action, status, reason, amount, attempt_number, actor, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (invoice_id, datetime.now(timezone.utc).isoformat(timespec="seconds"), action,
             status, reason, amount, attempt_number, actor, notes),
        )
        conn.commit()


def get_recovery_events(invoice_id: int) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM recovery_events WHERE invoice_id = ? ORDER BY timestamp, id",
            (invoice_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_dashboard_statistics() -> dict[str, Any]:
    with get_connection() as conn:
        total_invoices = conn.execute("SELECT COUNT(*) AS n FROM invoices").fetchone()["n"]
        outstanding = conn.execute(
            """
            SELECT COALESCE(SUM(invoice_amount), 0) AS total
            FROM invoices
            WHERE payment_status != 'Paid'
            """
        ).fetchone()["total"]
        expected = conn.execute(
            """
            SELECT COALESCE(SUM(expected_recovery), 0) AS total
            FROM invoices
            WHERE payment_status != 'Paid' AND expected_recovery IS NOT NULL
            """
        ).fetchone()["total"]
        recovered = conn.execute(
            "SELECT COALESCE(SUM(amount_recovered), 0) AS total FROM invoices"
        ).fetchone()["total"]
        recovery_counts = conn.execute(
            """
            SELECT
                SUM(CASE WHEN amount_recovered > 0 THEN 1 ELSE 0 END) AS successful,
                COALESCE(SUM(recovery_attempts), 0) AS attempts,
                SUM(CASE WHEN recovery_status = 'ESCALATED' THEN 1 ELSE 0 END) AS escalated
            FROM invoices
            """
        ).fetchone()
        risk_counts = conn.execute(
            """
            SELECT
                SUM(CASE WHEN risk_level = 'HIGH' THEN 1 ELSE 0 END) AS high_risk_count,
                SUM(CASE WHEN risk_level = 'MEDIUM' THEN 1 ELSE 0 END) AS medium_risk_count,
                SUM(CASE WHEN risk_level = 'LOW' THEN 1 ELSE 0 END) AS low_risk_count
            FROM invoices
            WHERE payment_status != 'Paid'
            """
        ).fetchone()
        recovery_by_risk = conn.execute(
            """
            SELECT
                COALESCE(risk_level, 'UNSCORED') AS risk_level,
                COALESCE(SUM(invoice_amount), 0) AS outstanding
            FROM invoices
            WHERE payment_status != 'Paid'
            GROUP BY COALESCE(risk_level, 'UNSCORED')
            """
        ).fetchall()
        status_distribution = {
            row["recovery_status"]: row["count"]
            for row in conn.execute(
                "SELECT COALESCE(recovery_status, 'NOT_STARTED') AS recovery_status, COUNT(*) AS count FROM invoices GROUP BY COALESCE(recovery_status, 'NOT_STARTED')"
            ).fetchall()
        }

    expected_recovery = float(expected or 0)
    total_outstanding = float(outstanding or 0)
    actual_recovered = float(recovered or 0)
    recovery_rate = (
        (expected_recovery / total_outstanding) * 100 if total_outstanding else 0.0
    )
    return {
        "total_invoices": int(total_invoices or 0),
        "total_outstanding": round(total_outstanding, 2),
        "high_risk_count": int(risk_counts["high_risk_count"] or 0),
        "medium_risk_count": int(risk_counts["medium_risk_count"] or 0),
        "low_risk_count": int(risk_counts["low_risk_count"] or 0),
        "expected_recovery": round(expected_recovery, 2),
        "revenue_at_risk": round(total_outstanding, 2),
        "actual_recovered": round(actual_recovered, 2),
        "successful_recoveries": int(recovery_counts["successful"] or 0),
        "recovery_attempts": int(recovery_counts["attempts"] or 0),
        "escalated_invoices": int(recovery_counts["escalated"] or 0),
        "recovery_success_rate": round((actual_recovered / total_outstanding) * 100, 1) if total_outstanding else 0.0,
        "status_distribution": status_distribution,
        "recovery_rate": round(recovery_rate, 1),
        "recovery_by_risk": {
            row["risk_level"]: float(row["outstanding"]) for row in recovery_by_risk
        },
    }
