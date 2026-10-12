"""Simulator records and State (contract file, SPEC.md Appendix A)."""

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class InvoiceStatus(StrEnum):
    OPEN = "open"
    PAID = "paid"
    VOID = "void"


class PaymentStatus(StrEnum):
    SUCCEEDED = "succeeded"
    REFUNDED = "refunded"
    FAILED = "failed"


class CaseStatus(StrEnum):
    OPEN = "open"
    RESOLVED = "resolved"


class ContractBasis(StrEnum):
    FLAT = "flat"
    PER_SEAT = "per_seat"


class LedgerKind(StrEnum):
    CHARGE = "charge"
    PAYMENT = "payment"
    REFUND = "refund"
    CREDIT = "credit"
    DISPUTE_CREDIT = "dispute_credit"
    ADJUSTMENT = "adjustment"
    VOID = "void"
    ACCOUNT_CREDIT = "account_credit"


@dataclass(frozen=True, slots=True)
class LineItem:
    description: str
    quantity: int
    unit_price: int

    @property
    def total(self) -> int:
        return self.quantity * self.unit_price


@dataclass(frozen=True, slots=True)
class Customer:
    id: str
    name: str
    email: str
    phone: str
    address: str
    contract_basis: ContractBasis
    contract_rate: int
    credit_balance: int


@dataclass(frozen=True, slots=True)
class Invoice:
    id: str
    customer_id: str
    period: str
    line_items: tuple[LineItem, ...]
    amount_billed: int
    amount_due: int
    status: InvoiceStatus
    note: str = ""


@dataclass(frozen=True, slots=True)
class Payment:
    id: str
    invoice_id: str
    customer_id: str
    amount: int
    method: str
    created_at: str
    status: PaymentStatus
    note: str = ""


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    customer_id: str
    invoice_id: str
    summary: str
    disputed_amount: int
    status: CaseStatus
    resolution: int | None = None
    note: str = ""


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    id: str
    customer_id: str
    invoice_id: str | None
    payment_id: str | None
    kind: LedgerKind
    amount: int
    memo: str


@dataclass(frozen=True, slots=True)
class State:
    customers: tuple[Customer, ...] = ()
    invoices: tuple[Invoice, ...] = ()
    payments: tuple[Payment, ...] = ()
    cases: tuple[Case, ...] = ()
    ledger: tuple[LedgerEntry, ...] = ()

    def customer(self, id: str) -> Customer | None:
        return next((r for r in self.customers if r.id == id), None)

    def invoice(self, id: str) -> Invoice | None:
        return next((r for r in self.invoices if r.id == id), None)

    def payment(self, id: str) -> Payment | None:
        return next((r for r in self.payments if r.id == id), None)

    def case(self, id: str) -> Case | None:
        return next((r for r in self.cases if r.id == id), None)

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON types only (lists, str, int, None)."""
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> State:
        return cls(
            customers=tuple(
                Customer(**{**c, "contract_basis": ContractBasis(c["contract_basis"])}) for c in d["customers"]
            ),
            invoices=tuple(
                Invoice(
                    **{
                        **i,
                        "line_items": tuple(LineItem(**li) for li in i["line_items"]),
                        "status": InvoiceStatus(i["status"]),
                    }
                )
                for i in d["invoices"]
            ),
            payments=tuple(Payment(**{**p, "status": PaymentStatus(p["status"])}) for p in d["payments"]),
            cases=tuple(Case(**{**k, "status": CaseStatus(k["status"])}) for k in d["cases"]),
            ledger=tuple(LedgerEntry(**{**e, "kind": LedgerKind(e["kind"])}) for e in d["ledger"]),
        )
