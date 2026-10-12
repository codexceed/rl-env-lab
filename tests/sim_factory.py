"""Hand-built simulator states shared by the sim tests."""

from envlab.sim.entities import (
    Case,
    CaseStatus,
    ContractBasis,
    Customer,
    Invoice,
    InvoiceStatus,
    LineItem,
    Payment,
    PaymentStatus,
    State,
)
from envlab.sim.ledger import build_ledger


def customers() -> tuple[Customer, ...]:
    return (
        Customer(
            "C0001",
            "Alice Smith",
            "alice@example.com",
            "+1-415-555-0101",
            "1 Main St",
            ContractBasis.FLAT,
            4999,
            0,
        ),
        Customer(
            "C0002",
            "Bob Jones",
            "bob@example.net",
            "+1-212-555-0102",
            "2 Oak Ave",
            ContractBasis.PER_SEAT,
            1000,
            500,
        ),
    )


def invoices() -> tuple[Invoice, ...]:
    plan = LineItem("Plan: Pro", 1, 5999)
    return (
        # Overbilled, open, no payments.
        Invoice("INV-00001", "C0001", "2026-08", (plan,), 5999, 5999, InvoiceStatus.OPEN),
        # Paid twice (duplicate), overpaid.
        Invoice(
            "INV-00002",
            "C0001",
            "2026-07",
            (LineItem("Plan: Pro", 1, 4999),),
            4999,
            -4999,
            InvoiceStatus.PAID,
        ),
        # Open, two lines, with an open dispute.
        Invoice(
            "INV-00003",
            "C0002",
            "2026-08",
            (LineItem("Plan: Team seats", 5, 1000), LineItem("Add-on: Backups", 1, 1500)),
            6500,
            6500,
            InvoiceStatus.OPEN,
        ),
        # Paid once after a failed attempt.
        Invoice(
            "INV-00004",
            "C0002",
            "2026-06",
            (LineItem("Plan: Team seats", 3, 1000),),
            3000,
            0,
            InvoiceStatus.PAID,
        ),
        # Duplicate already refunded.
        Invoice(
            "INV-00005",
            "C0002",
            "2026-05",
            (LineItem("Plan: Team seats", 2, 1000),),
            2000,
            0,
            InvoiceStatus.PAID,
        ),
        # Partly paid, still open.
        Invoice(
            "INV-00006",
            "C0001",
            "2026-09",
            (LineItem("Plan: Pro", 1, 4999), LineItem("Add-on: SSO", 1, 5001)),
            10000,
            6000,
            InvoiceStatus.OPEN,
        ),
        # Open with only a failed payment.
        Invoice(
            "INV-00007",
            "C0002",
            "2026-09",
            (LineItem("Plan: Team seats", 1, 1000),),
            1000,
            1000,
            InvoiceStatus.OPEN,
        ),
    )


def payments() -> tuple[Payment, ...]:
    # Deliberately not in created_at order, to exercise build_ledger's sort.
    S, R, F = PaymentStatus.SUCCEEDED, PaymentStatus.REFUNDED, PaymentStatus.FAILED
    return (
        Payment("PAY-00002", "INV-00002", "C0001", 4999, "card", "2026-08-02T10:05", S),
        Payment("PAY-00001", "INV-00002", "C0001", 4999, "card", "2026-08-02T10:00", S),
        Payment("PAY-00003", "INV-00004", "C0002", 3000, "ach", "2026-07-01T09:00", F),
        Payment("PAY-00004", "INV-00004", "C0002", 3000, "ach", "2026-07-02T09:00", S),
        Payment("PAY-00005", "INV-00005", "C0002", 2000, "card", "2026-06-03T12:00", S),
        Payment(
            "PAY-00006",
            "INV-00005",
            "C0002",
            2000,
            "card",
            "2026-06-03T12:30",
            R,
            "Refund duplicate payment",
        ),
        Payment("PAY-00007", "INV-00006", "C0001", 4000, "card", "2026-10-01T08:00", S),
        Payment("PAY-00008", "INV-00007", "C0002", 1000, "card", "2026-10-02T08:00", F),
    )


def cases() -> tuple[Case, ...]:
    return (
        Case(
            "CS-0001",
            "C0002",
            "INV-00003",
            "Customer disputes the 'Add-on: Backups' charge.",
            1500,
            CaseStatus.OPEN,
        ),
        Case(
            "CS-0002",
            "C0002",
            "INV-00004",
            "Customer disputes the plan charge.",
            3000,
            CaseStatus.RESOLVED,
            0,
            "Dispute rejected: charge is valid",
        ),
        # Open case on a paid invoice.
        Case(
            "CS-0003",
            "C0002",
            "INV-00004",
            "Customer disputes a seat.",
            1000,
            CaseStatus.OPEN,
        ),
    )


def make_state() -> State:
    cs, inv, pay = customers(), invoices(), payments()
    return State(
        customers=cs, invoices=inv, payments=pay, cases=cases(), ledger=build_ledger(cs, inv, pay)
    )
