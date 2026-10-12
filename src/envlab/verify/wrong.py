"""Scripted wrong-value call for gate G3 (SPEC §7 table).

Derived only from `oracle_call` and `initial_state`. Free-text args copy the oracle's.
Raises ValueError when the table's rule has nothing to act on; G3 reports that as a failure.
"""

from envlab.calls import ToolCall
from envlab.sim.entities import PaymentStatus, State
from envlab.tasks.instance import TaskInstance


def _int(call: ToolCall, key: str) -> int:
    value = call.arguments[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"oracle argument {key!r} is not an int")
    return value


def _str(call: ToolCall, key: str) -> str:
    value = call.arguments[key]
    if not isinstance(value, str):
        raise ValueError(f"oracle argument {key!r} is not a str")
    return value


def _f1(oracle: ToolCall, state: State) -> dict[str, str | int]:
    return {"amount": _int(oracle, "amount") - 1}


def _f2(oracle: ToolCall, state: State) -> dict[str, str | int]:
    """Refund the original: earliest succeeded payment on T with the duplicate's amount."""
    dup = state.payment(_str(oracle, "payment_id"))
    if dup is None:
        raise ValueError("duplicate payment not in initial_state")
    originals = [
        p
        for p in state.payments
        if p.invoice_id == dup.invoice_id
        and p.status == PaymentStatus.SUCCEEDED
        and p.amount == dup.amount
        and p.id != dup.id
    ]
    if not originals:
        raise ValueError("no original payment to refund instead")
    # `min` keeps the first in state order on equal created_at.
    return {"payment_id": min(originals, key=lambda p: p.created_at).id}


def _f3(oracle: ToolCall, state: State) -> dict[str, str | int]:
    case = state.case(_str(oracle, "case_id"))
    invoice = None if case is None else state.invoice(case.invoice_id)
    if invoice is None:
        raise ValueError("case or its invoice not in initial_state")
    a = _int(oracle, "adjustment")
    return {"adjustment": a + 1 if a + 1 <= invoice.amount_due else a - 1}


def _f4(oracle: ToolCall, state: State) -> dict[str, str | int]:
    invoice = state.invoice(_str(oracle, "invoice_id"))
    if invoice is None:
        raise ValueError("invoice not in initial_state")
    s = _int(oracle, "new_amount")  # the oracle's new_amount is S
    return {"new_amount": s + 2 if s + 1 == invoice.amount_billed else s + 1}


def _bump_last_digit(phone: str) -> str:
    for i in range(len(phone) - 1, -1, -1):
        if phone[i].isdigit():
            return phone[:i] + str((int(phone[i]) + 1) % 10) + phone[i + 1 :]
    raise ValueError(f"phone {phone!r} has no digit")


def _f5(oracle: ToolCall, state: State) -> dict[str, str | int]:
    field, value = _str(oracle, "field"), _str(oracle, "value")
    if field == "email":
        return {"value": "x" + value}
    if field == "address":
        return {"value": value + " Apt 2"}
    if field == "phone":
        return {"value": _bump_last_digit(value)}
    raise ValueError(f"no wrong-value rule for field {field!r}")


_RULES = {
    "apply_credit": _f1,
    "refund_payment": _f2,
    "resolve_case": _f3,
    "adjust_invoice": _f4,
    "update_customer": _f5,
}


def wrong_value_call(instance: TaskInstance) -> ToolCall:
    """The oracle call with its value perturbed per SPEC §7; all other args are the oracle's."""
    oracle = instance.oracle_call
    if oracle.name not in _RULES:
        raise ValueError(f"no wrong-value rule for tool {oracle.name!r}")
    changed = _RULES[oracle.name](oracle, instance.initial_state)
    return ToolCall(oracle.name, {**oracle.arguments, **changed})
