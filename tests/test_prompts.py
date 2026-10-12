"""Prompt layout, tool-call rendering and the token estimate (SPEC §10, §11.8)."""

import pytest
from tests.tasks_fixtures import hand_built_instance

from envlab import prompts
from envlab.calls import ToolCall
from envlab.sim.schemas import TOOL_SCHEMAS

EXPECTED_USER = """Request: Resolve case CS-2001. Approve the dispute in full.

Records:
customers:
{"id":"C1234","name":"Ada Chen","email":"ada.chen@example.com","phone":"+1-415-555-0101","address":"12 Oak St, Salem OR","contract_basis":"flat","contract_rate":4999}
invoices:
{"id":"INV-10001","customer_id":"C1234","period":"2026-08","line_items":[{"description":"Plan: Pro","quantity":1,"unit_price":4999},{"description":"Add-on: SSO","quantity":2,"unit_price":1500}],"amount_billed":7999,"amount_due":7999,"status":"open"}
cases:
{"id":"CS-2001","customer_id":"C1234","invoice_id":"INV-10001","summary":"Customer disputes the 'Add-on: SSO' charge.","disputed_amount":3000,"status":"open"}"""  # noqa: E501

EXPECTED_CALL = """<tool_call>
<function=apply_credit>
<parameter=invoice_id>
INV-48213
</parameter>
<parameter=amount>
1250
</parameter>
<parameter=reason>
Credit overcharge versus contract rate
</parameter>
</function>
</tool_call>"""


def test_system_prompt_is_exact():
    assert prompts.SYSTEM_PROMPT == (
        "You are a billing back-office agent. Resolve the request with exactly one tool call. All money values "
        "are integer cents. Use only ids that appear in the records. Reply with the tool call only."
    )


def test_tools_are_the_contract_schemas():
    assert list(TOOL_SCHEMAS) == prompts.TOOLS


def test_user_message_layout_is_stable():
    # golden: omits note "", resolution None, credit_balance 0, the empty payments group and the ledger
    assert prompts.user_message(hand_built_instance()) == EXPECTED_USER


def test_user_message_keeps_non_default_fields():
    inst = hand_built_instance()
    msg = prompts.format_user_message(inst.request, inst.oracle_final_state)
    assert '"resolution":3000,"note":"ok"' in msg
    assert "L0002" not in msg  # ledger never shown


def test_build_messages():
    inst = hand_built_instance()
    assert prompts.build_messages(inst) == [
        {"role": "system", "content": prompts.SYSTEM_PROMPT},
        {"role": "user", "content": EXPECTED_USER},
    ]


class _FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "rendered"


def test_render_prompt_disables_thinking():
    tok = _FakeTokenizer()
    inst = hand_built_instance()
    assert prompts.render_prompt(tok, inst) == "rendered"
    ((messages, kwargs),) = tok.calls
    assert messages == prompts.build_messages(inst)
    assert kwargs == {
        "tools": prompts.TOOLS,
        "add_generation_prompt": True,
        "enable_thinking": False,
        "tokenize": False,
    }


def test_render_call_native_form():
    call = ToolCall(
        "apply_credit", {"reason": "Credit overcharge versus contract rate", "amount": 1250, "invoice_id": "INV-48213"}
    )
    assert prompts.render_call(call) == EXPECTED_CALL  # args follow PARAM_TYPES order, not dict order


def test_render_call_round_trip():
    parse = pytest.importorskip("envlab.parse")
    calls = [
        ToolCall("apply_credit", {"invoice_id": "INV-48213", "amount": 1250, "reason": "x"}),
        ToolCall("refund_payment", {"payment_id": "PAY-10001", "reason": "Refund duplicate payment"}),
        ToolCall("adjust_invoice", {"invoice_id": "INV-1", "new_amount": 0, "reason": "r"}),
        ToolCall("update_customer", {"customer_id": "C1234", "field": "address", "value": "12 Oak St, Salem OR"}),
        ToolCall("resolve_case", {"case_id": "CS-1", "adjustment": 7, "note": "n"}),
        ToolCall("void_invoice", {"invoice_id": "INV-1", "reason": "r"}),
    ]
    for call in calls:
        assert parse.parse_tool_call(prompts.render_call(call)) == call


def test_estimate_formula():
    inst = hand_built_instance()
    n = len(EXPECTED_USER)
    assert prompts.estimate_prompt_tokens(inst) == 850 + -(-n * 2 // 5)  # 850 + ceil(n / 2.5)
    assert prompts.ESTIMATE_BUDGET == 1500
    assert prompts.PROMPT_TOKEN_LIMIT == 1536
