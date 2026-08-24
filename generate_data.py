"""Generate synthetic invoice CSV for RecoverAI demos (no real customer data)."""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "invoices.csv"

COMPANIES = [
    "ABC Technologies",
    "TechNova",
    "Global Systems",
    "DataWorks",
    "Alpha Corp",
    "Bright Systems",
    "NextGen Ltd",
    "SmartSoft",
    "FutureTech",
    "CloudWorks",
    "InnovateX",
    "DigitalEdge",
    "Nova Solutions",
    "Vertex Systems",
    "Quantum Labs",
]

# Extra fictional names so 100+ invoices do not look copy-pasted.
COMPANY_SUFFIXES = [
    "",
    " Pvt Ltd",
    " India",
    " Group",
    " Holdings",
]


def generate_invoices(n: int = 120, seed: int = 21) -> list[dict]:
    rng = random.Random(seed)
    today = date(2026, 8, 21)
    rows: list[dict] = []

    for i in range(n):
        company = rng.choice(COMPANIES)
        suffix = rng.choice(COMPANY_SUFFIXES)
        customer_name = f"{company}{suffix}".strip()

        invoice_amount = float(rng.randint(100, 5000) * 100)
        invoice_date = today - timedelta(days=rng.randint(15, 180))

        previous_late_payments = rng.choices(
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            weights=[18, 16, 14, 12, 10, 10, 8, 7, 5],
            k=1,
        )[0]
        customer_payment_score = max(
            12.0,
            min(98.0, 88 - previous_late_payments * 8 + rng.uniform(-10, 10)),
        )
        customer_payment_score = round(customer_payment_score, 1)

        if customer_payment_score >= 75 and previous_late_payments <= 1:
            status_weights = (0.42, 0.38, 0.20)
        elif customer_payment_score <= 40 or previous_late_payments >= 5:
            status_weights = (0.10, 0.22, 0.68)
        else:
            status_weights = (0.22, 0.33, 0.45)

        payment_status = rng.choices(
            ["Paid", "Pending", "Overdue"],
            weights=status_weights,
            k=1,
        )[0]

        if payment_status == "Paid":
            due_date = invoice_date + timedelta(days=rng.randint(15, 45))
            days_overdue = 0
        elif payment_status == "Pending":
            due_date = today + timedelta(days=rng.randint(0, 25))
            days_overdue = 0
        else:
            days_overdue = rng.randint(3, 90)
            if previous_late_payments >= 5:
                days_overdue = rng.randint(18, 95)
            due_date = today - timedelta(days=days_overdue)

        rows.append(
            {
                "invoice_number": f"INV-{1001 + i}",
                "customer_name": customer_name,
                "invoice_amount": invoice_amount,
                "invoice_date": invoice_date.isoformat(),
                "due_date": due_date.isoformat(),
                "payment_status": payment_status,
                "days_overdue": days_overdue,
                "previous_late_payments": previous_late_payments,
                "customer_payment_score": customer_payment_score,
            }
        )

    return rows


def write_csv(rows: list[dict], path: Path = CSV_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "invoice_number",
        "customer_name",
        "invoice_amount",
        "invoice_date",
        "due_date",
        "payment_status",
        "days_overdue",
        "previous_late_payments",
        "customer_payment_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    rows = generate_invoices()
    path = write_csv(rows)
    paid = sum(1 for r in rows if r["payment_status"] == "Paid")
    pending = sum(1 for r in rows if r["payment_status"] == "Pending")
    overdue = sum(1 for r in rows if r["payment_status"] == "Overdue")
    print(f"Wrote {len(rows)} invoices to {path}")
    print(f"Paid={paid} Pending={pending} Overdue={overdue}")


if __name__ == "__main__":
    main()
