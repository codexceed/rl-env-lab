# SPEC v3 — harder, larger task set (project version v3)

**Version:** v3.0 · **Status:** draft, implements [ADR 0007](docs/adr/0007-v3-task-set.md) (Proposed) · **Label:** v3, exploratory

v3 changes **only the task set**. Everything not stated here is exactly as in [SPEC.md](SPEC.md): entities and status rules (§1), ledger (§2), tools (§3), graders (§6), gate definitions (§7), prompt format (§10), Python contract files (§11, Appendix A), eval record (§12). If code and this file disagree, this file wins for v3 data; SPEC.md wins for everything else.

Contents: 1 What changes · 2 Common rules · 3 Notation · 4 Templates (F1–F5) · 5 Difficulty knobs · 6 Graders · 7 Gates · 8 Broken variants · 9 Data build and splits · 10 Prompt budget · 11 Python interfaces · 12 Design notes

---

## 1. What changes from v1

| | v1 (SPEC.md) | v3 (this file) |
|---|---|---|
| Templates | 11 train + 5 held-out (T1, T2, T3, F1T5) | **15 train** (A, B, C per family) + **5 near** (N) + **5 far** (X) held-out |
| Held-out meaning | T3 needs a formula never trained | **N** only recombines trained steps (hacking vs carelessness); **X** needs one rule never trained (generalization) |
| v1 held-out formulas | per-seat, unused units, discount lines held out | **trained**: F1A per-seat, F3B unused units, F4A discount lines |
| Distractor records k | base {2, 3, 4} | **{4, 5, 6, 7}**, never trimmed below 4 |
| Customer finding | ids in most requests | 9 of 15 train templates find the customer by name, email or phone among **same-name customers** |
| Arithmetic | one step, small numbers | multi-step, stated rounding (always **rounded down to the cent**), larger quantities, more lines |
| Prompt budget | 1,536 tokens | **2,048 tokens** (estimate ≤ 2,000) |
| Data | `data/` | **`data/v3/`** with its own `manifest.json` (`spec_version` `"v3.0"`) |
| Sizes | 2,000 train, 250 held-out | **6,000 train** (15 × 400), **1,000 held-out** (10 × 100), 500 same-type, 100 broken |

Unchanged: one task = one tool call; the simulator (`envlab.sim`), the six tools, robust and leaky grading, G1–G3 definitions, the §10 prompt, integer cents. No entity fields are added: information a rule needs that no field holds (a tax rate, a commitment, price tiers, an allowance, a downgrade date) is stated in the request, or in an existing free-text field (case summary).

## 2. Common rules

- Everything in SPEC.md §4 "Common rules" holds, except where this section says otherwise.
- Template ids: `F1A F1B F1C F1N F1X … F5A F5B F5C F5N F5X` (registry `envlab.tasks.v3`). A, B, C train; N near held-out; X far held-out. Held-out templates are never in `train`, `sametype` or `broken`.
- Every instance has `variant = "base"` (v3 has no hard variant) and `meta = {"n_distractors", "target_ids", "spec_version": "v3.0"}`.
- **Distractors:** k drawn uniformly from `K_CHOICES = {4, 5, 6, 7}`. Units as in SPEC.md §4: required units first (smallest form), then random allowed units that fit, then `z` to reach k. A required unit may push the count above k. Trimming to the prompt budget (§10) drops the last-added non-required units, never below `max(4, records in required units)`; a draw that still does not fit is redrawn from `derive_seed(seed, "retry", attempt)` (as v1).
- A distractor must never satisfy the template's rule under the instance's `params` (G2 checks).
- **Money in requests** is written as dollars, `$D.CC` (`dollars()`), and only where a rule needs a number that no record holds (F1X tier prices, F3A cap, F4X commitment). Percentages are integers, except tax rates, which are written from basis points (`825` → `8.25%`, `650` → `6.5%`, `600` → `6%`). Months render as `August 2026`; dates as `August 19, 2026`.
- **Rounding:** every division in an oracle rule is integer floor division on non-negative integers ("rounded down to the cent"), and the request says so wherever rounding can occur.
- **Free-text argument** per family, as v1: F1 `"Credit overcharge versus contract rate"` · F2 `"Refund duplicate payment"` · F3 `"Dispute resolved per policy"` · F4 `"Correct total to sum of line items"`.
- `params` hold exactly the keys listed per template. F1–F4 params carry `rule`, which names the oracle rule (§4). The independent solver (G2) uses only `params` + `initial_state`.
- Generator conventions (SPEC.md §1) hold: one `Plan: …` line, first; non-target invoices sum correctly (exception: F4X unit `m`, billed per the commitment rule); `amount_due = amount_billed − Σ succeeded payments`; background resolved cases `resolution = 0` with the rejection note; background refunded payments note `"Refund duplicate payment"`. Shared ranges as SPEC.md §4 unless a template overrides; add-on unit price ∈ [500, 5000].
- **Same-name customer ("twin"):** a different customer with C's exact `name`; its email shares C's local part on another domain when one is free, else gets a numbered local part. A **look-alike phone** keeps C's area code and changes the last four digits by one adjacent swap or one digit.

## 3. Notation

C target customer, T target invoice, K target case, P T's plan line (qty N, unit U), R = `C.contract_rate`, S = Σ T's line totals, B = `T.amount_billed`, M = days in T's month, `⌊x⌋` floor. `{name}`, `{customer_id}`, `{invoice_id}`, `{case_id}` are C's / T's / K's.

## 4. Task families and templates

### F1 Overcharge → `apply_credit`

**Oracle (all F1):** T open, unpaid (`amount_due == amount_billed`). Credit `amount` by `params.rule` (below); `2 ≤ amount < T.amount_due`. Final: T `amount_due −= amount`, stays open, `note` set, one `credit` entry.
**Core:** C, T. T = P + 1–3 add-ons (qty 1).
**Units:** `a` another invoice of C, other month, at the contract rate (F1X: seats ≤ n), open (1 rec) or paid once (2) · `p` another open invoice of C, other month, plan overcharged the same way (1; F1A, F1N) · `s` a twin with C's contract basis and a correctly priced invoice in T's month (2) · `b` another customer with a correct invoice (2) · `f` a customer sharing C's first name (1) · `z`.
**Solver:** customers matching `customer_id` / `customer_name`; their **open** invoices filtered by `invoice_id` / `period`; amount by `rule` from P (first `Plan: ` line) and the customer's R; keep `amount > 0`; one call each.

| Template | Scenario and generation | Request | `params` | `rule`: amount | Required |
|---|---|---|---|---|---|
| **F1A** | Per-seat C, R ∈ [800, 9000]. P = `Plan: Team seats`, N ∈ [12, 180], U = R + d, d ∈ [25, 900]. | `{name} is on a per-seat contract. Their {Month YYYY} invoice charged more per seat than their contract rate. Credit the overcharge.` | `customer_name`, `period`, `rule` | `per_seat`: N × (U − R) | `s`, `p` |
| **F1B** | C flat (P qty 1, U = R) or, with share 0.5, per-seat (N ∈ [12, 180], U = R ∈ [800, 9000]). The plan is billed at the full contract price; p ∈ {6, 8, 12, 15, 18, 22, 24, 27, 35}. | `{name} ({customer_id}) was promised {p}% off their plan, but their {Month YYYY} invoice billed the plan at the full price. Credit {p}% of that invoice's plan line total, rounded down to the cent.` | `customer_id`, `period`, `rule`, `percent` | `percent_off`: ⌊N × U × p / 100⌋ | `a` |
| **F1C** | Flat C (new plan, price R). P = another flat plan name (the old plan), qty 1, U = R + g, g ∈ [500, 15000]. Downgrade day D ∈ [2, M − 1]. `{old}`/`{new}` = plan names without `Plan: `. | `{name} ({customer_id}) switched from the {old} plan to the cheaper {new} plan on {Month D, YYYY}; their contract rate is now the new plan's monthly price. Invoice {invoice_id} still charged the old plan's price for the whole month. Credit (old price - new price) x days on the new plan / days in the month, rounded down to the cent. Days on the new plan run from the switch date through the last day of the month.` | `customer_id`, `invoice_id`, `rule`, `downgrade_date` (`YYYY-MM-DD`) | `prorate_days`: ⌊(U − R) × (M − D + 1) / M⌋ | `a` |
| **F1N** ⟂near | As F1A; p as F1B. | `{name} ({customer_id}) has a per-seat contract and was promised {p}% off their seats. Their {Month YYYY} invoice billed the seats above the contract rate and without the discount. The correct seat charge is seats x contract rate, less {p}% of that amount rounded down to the cent. Credit the difference between the billed seat line and the correct seat charge.` | `customer_id`, `period`, `rule`, `percent` | `per_seat_percent`: N × U − (N × R − ⌊N × R × p / 100⌋) | `p` |
| **F1X** ⟂far | Per-seat C, R = P1 ∈ [1500, 7000] (first-tier price); P2 = P1 − g, g ∈ [150, 1500]; n ∈ {10, 20, 25, 40, 50}; N = n + e, e ∈ [3, 120]; U = P1. | `{name} ({customer_id}) has tiered seat pricing: the first {n} seats cost {P1$} each and every seat beyond {n} costs {P2$}. Invoice {invoice_id} charged every seat at the first-tier price. Credit the overcharge.` | `customer_id`, `invoice_id`, `rule`, `tier_seats`, `tier1_price`, `tier2_price` | `tiered`: N × U − (min(N, n) × P1 + max(0, N − n) × P2) | `a` |

### F2 Duplicate or extra payment → `refund_payment`

**Oracle (all F2):** refund payment p (below). Final: p `refunded`, `note` set; T `amount_due` goes from `−p.amount` to 0, status `paid`; one `refund` entry.
**Core:** C, T, every payment on T. T = P + 0–2 add-ons, billed B.
**Units:** `a` a failed payment on T of p's amount, before T's first non-failed payment (1; not F2X) · `b` another invoice of C paid once, for T's lines (2) · `c` another customer with an invoice paid once (3) · `d` another invoice of C whose duplicate was **already refunded** (3) · `e` another invoice of C paid in 2–3 unequal installments, no duplicate (3–4) · `s` a twin with an invoice **in T's month** paid in 2–3 installments, no duplicate (4–5) · `n` a near-duplicate contact (a twin; F2C phone variant: with a look-alike phone) with its **own genuine duplicate**: charged twice (F2C, 4) or one installment twice (F2N, 5–6) · `f` · `z`. `s` and `n` are required-only.
**Solver:** customers matching `customer_id` / `customer_name` / every `lookup` pair; their invoices filtered by `invoice_id` / `period`; per invoice, by `rule`:
- `duplicate`: every succeeded p with an earlier succeeded q of equal amount on the same invoice (SPEC.md §4 F2).
- `overpayment`: succeeded payments sorted by `created_at`; the first at which the running total exceeds `amount_billed` (none if it never does).

| Template | Scenario and generation | Request | `params` | Required |
|---|---|---|---|---|
| **F2A** | 4–5 payments on T, all of amount B, 1 min–1 day apart: the original, the duplicate (both succeeded), ≥ 1 failed and ≥ 1 already refunded attempt; the original is the earliest non-failed one. p = the duplicate. | `{name} ({customer_id}) says one of their invoices was charged more than once. Refund the duplicate charge that has not been refunded yet.` | `customer_id`, `rule="duplicate"` | `d` |
| **F2B** | B split into k ∈ [2, 4] distinct installments, each ≥ 10% of B, 1 h–3 days apart; installment i charged again 1 min–2 days after its first charge. p = the later copy. | `{name} paid their {Month YYYY} invoice in installments, but one installment went through twice. Refund the duplicate.` | `customer_name`, `period`, `rule="duplicate"` | `s` |
| **F2C** | T charged twice in full, P2 = P1 + 1 min–1 day. Contact is C's phone with share 0.5, else C's email. | email: `Support inbox message from {email}: "Hi, I was charged twice for one of my invoices. Please refund the extra payment."` · phone: `Voicemail from {phone}: "Hi, I was charged twice for one of my invoices. Please refund the extra payment."` | `lookup={"email"}` or `{"phone"}`, `rule="duplicate"` | `n` |
| **F2N** ⟂near | F2B's payments; customer by email. | `Support inbox message from {email}: "Hi, I paid one of my invoices in installments and one installment went through twice. Please refund the duplicate."` | `lookup={"email"}`, `rule="duplicate"` | `n` |
| **F2X** ⟂far | B split into k ∈ [2, 3] distinct installments (as F2B, no duplicate), then an extra payment x = ⌊B × q / 100⌋, q ∈ [5, 60] (raised by 1 until it equals no installment), 1 h–3 days after the last; with share 0.5 a failed retry (an installment's amount) 1 min–1 day after x. p = x. | `{name} ({customer_id}) paid more than invoice {invoice_id} was billed. Refund the payment that pushed their total paid above the billed amount.` | `customer_id`, `invoice_id`, `rule="overpayment"` | `e` |

### F3 Dispute → `resolve_case`

**Oracle (all F3):** open case K on T (open, unpaid). Adjustment A by `params.rule`; `2 ≤ A < T.amount_due`. Final: K `resolved`, `resolution = A`, `note` set; T `amount_due −= A`, stays open; one `dispute_credit` entry.
**Core:** C, T, K.
**Usage lines** (`USAGE_LINES`): `Usage: API calls (per 1k)`, `Usage: Storage (per GB)`, `Usage: SMS messages`, `Usage: Compute hours`, `Usage: Data transfer (per GB)`. **Usage invoice:** P + 1–2 usage lines (qty X ∈ [120, 4000], unit U ∈ [3, 400]) + 0–2 add-ons, non-plan lines shuffled; the first usage line drawn is the disputed line L. **Used units** Y = ⌊X × u / 100⌋, u ∈ [25, 95], clamped to [1, X − 1]. **Usage summary:** `Customer says they used only {Y} of the {X} units billed on the '{L}' line.`; `disputed_amount = L.total`.
**Units:** `a` another invoice of C with a resolved case (2) · `b` another customer with an invoice and an open case (3) · `c` another invoice of C (1) · `d` an open case of C on another invoice, other month (F3B/N/X: a usage invoice) (2) · `e` a resolved (rejected) case on T (1) · `s` a twin with an open usage case on an invoice of **another month** (3, required-only) · `f` · `z`.
**Solver:** open cases filtered by `case_id`, and to invoices of customers matching `customer_name` and invoices matching `invoice_id` / `period`; A by `rule`; one call each.

| Template | Scenario and generation | Request | `params` | `rule`: A | Required |
|---|---|---|---|---|---|
| **F3A** | Flat C. T = P + 2–4 add-ons with qty 1–12; L (an add-on) disputed: `Customer disputes the '{L}' charge.`, D = L.total. p ∈ {15, 35, 45, 55, 65, 85}; raw = ⌊D × p / 100⌋. Cap c (whole dollars): with share 0.5 (when raw ≥ 300) binding, c = 100·j, j ∈ [1, ⌊(raw − 1)/100⌋]; else c = 100·j, j ∈ [⌊raw/100⌋ + 1, ⌊raw/100⌋ + 1 + max(1, ⌊raw/200⌋)]. | `Resolve the open dispute on invoice {invoice_id}. Policy: approve {p}% of the disputed amount, rounded down to the cent, but no more than {c$}.` | `invoice_id`, `rule`, `percent`, `cap` | `percent_cap`: min(⌊D × p / 100⌋, c) | `e`, `d` |
| **F3B** | Flat C, usage invoice, usage summary. | `{name} disputes the usage billed on their {Month YYYY} invoice. Resolve the open case: credit only the unused units, at the invoice's unit price.` | `customer_name`, `period`, `rule`, `used_units`, `line_description` | `unused_units`: (X − Y) × U | `s`, `d` |
| **F3C** | C flat or (share 0.5) per-seat (N ∈ [12, 180], R ∈ [800, 9000]); T = P + 0–2 add-ons. Outage h ∈ [3, 72] hours: `Customer reports a {h}-hour service outage on {Month} {day} and asks for a credit.`; D = P.total. | `Resolve case {case_id}. Outage policy: credit the plan line's total x outage hours / hours in the invoice's billing month, rounded down to the cent.` | `case_id`, `rule`, `hours` | `outage_hours`: ⌊P.total × h / (M × 24)⌋ | `e` |
| **F3N** ⟂near | As F3B; p as F3A. | `Resolve case {case_id}. Policy: take the unused units at the invoice's unit price, then approve {p}% of that amount, rounded down to the cent.` | `case_id`, `rule`, `used_units`, `line_description`, `percent` | `unused_units_percent`: ⌊(X − Y) × U × p / 100⌋ | `d` |
| **F3X** ⟂far | As F3B; allowance F = max(1, ⌊Y × a / 100⌋), a ∈ [10, 60] (so F < Y). | `Resolve case {case_id}. The customer's plan includes {F} free units on the '{L}' line each month; only usage above that allowance is billable. Credit the billed units minus the billable units, at the invoice's unit price.` | `case_id`, `rule`, `used_units`, `allowance_units`, `line_description` | `allowance`: (X − max(0, Y − F)) × U | `a` |

### F4 Billing error → `adjust_invoice`

**Oracle (all F4):** T open, unpaid; `new_amount` by `params.rule`, ≠ B. Final: `amount_billed = amount_due = new_amount`, status open, `note` set; one `adjustment` entry.
**Core:** C, T.
**Discounted lines (F4A, F4N):** P + 3–5 add-ons with qty 1–24 (positive subtotal Q), then 1–2 `Discount: …` lines (qty 1): first −⌊Q × q / 100⌋, q ∈ [5, 20]; second (optional) −min(r, ⌊Q / 10⌋), r ∈ [500, 5000].
**v1 error injection** = SPEC.md §4 F4 "Error injection".
**Units:** `a` another invoice of C summing correctly, open (1) or paid (2) · `b` another customer with a correct invoice (2) · `c` an open invoice of C with T's lines at other quantities (±5, discounts and a flat plan kept), summing correctly (1) · `s` a twin with a correct open invoice (F4A: **in T's month**) (2) · `m` (F4X) another open invoice of C billed per the commitment rule, `max(Σ, minimum)` (1) · `f` · `z`.
**Solver:** **open** invoices of customers matching `customer_name`, filtered by `invoice_id` / `period`; correct total by `rule` (S = Σ line totals, discount lines included); keep those whose correct total ≠ `amount_billed`; one call each.

| Template | Scenario and generation | Request | `params` | `rule`: new_amount | Required |
|---|---|---|---|---|---|
| **F4A** | Flat C, discounted lines. B uniform over the kinds with a value > 0 and ≠ S: discounts omitted (Q) · one discount applied twice (S + d) · one discount added as a charge (S − 2d) · v1 error injection. | `{name}'s {Month YYYY} invoice has the wrong total. Recompute it from the line items, including discounts.` | `customer_name`, `period`, `rule="sum"` | `sum`: S | `s`, `a` (open) |
| **F4B** | Flat C; P + 2–4 add-ons with qty 1–30; B = S (tax missing); rate t ∈ {4.25, 5, 6, 6.25, 7.25, 7.75, 8.25, 8.75, 9.5, 10.25}% (as basis points t·100). | `Invoice {invoice_id} for {name} is missing sales tax. Set its total to the sum of its line items plus {t}% tax on that sum, with the tax rounded down to the cent.` | `invoice_id`, `rule="tax"`, `tax_bp` | `tax`: S + ⌊S × tax_bp / 10000⌋ | `a` |
| **F4C** | C per-seat (share 0.5; N ∈ [20, 300], R ∈ [800, 9000]) or flat. P + 3–5 add-ons with qty 10–250 (4–6 lines). B = v1 error injection. | `One of {name}'s open invoices has a total that doesn't match its line items. Correct it.` | `customer_name`, `rule="sum"` | `sum`: S | `s`, `a` (open) |
| **F4N** ⟂near | Flat C, discounted lines, rate t as F4B. B uniform from {S (tax missing), Q (discounts and tax missing), Q + ⌊Q × tax_bp / 10000⌋ (discounts missing)}. | `Invoice {invoice_id} ({name}) has the wrong total. Recompute it from the line items, including discounts, then add {t}% sales tax on that amount, with the tax rounded down to the cent.` | `invoice_id`, `rule="tax"`, `tax_bp` | `tax`: S + ⌊S × tax_bp / 10000⌋ | `a` |
| **F4X** ⟂far | Flat C; P + 1–3 add-ons (qty 1). With share 0.7: minimum m = 100 × (⌊S × r / 10000⌋ + 1), r ∈ [105, 160] (m > S) and B = S. Else m = 100 × ⌊S × r / 10000⌋, r ∈ [40, 90] (m < S) and B uniform from {m, v1 error injection} (if that equals S: B = m). | `{name}'s contract has a minimum monthly commitment of {m$}: each invoice's total is the sum of its line items or the commitment, whichever is higher. Invoice {invoice_id} does not follow this rule. Correct its total.` | `invoice_id`, `rule="minimum"`, `minimum` | `minimum`: max(S, m) | `m` |

### F5 Account change → `update_customer`

**Oracle (all F5):** the one customer C matching every `lookup` pair exactly; `update_customer(C.id, field, new_value)`. Final: one field changed, no ledger entry.
**Core:** C.
**Values:** V is fresh and valid for its field (SPEC.md §3 regexes), differs from C's current value, and its G3 perturbation (§7) differs from it too. email: `firstlast@`, `f.last@` or `first.last.home@` a random domain; address and phone: fresh, canonical format.
**Corrections (F5B, F5N):** the message first asks for a wrong value W, then corrects it. With share 0.6 a **partial** correction fixes one part of W (`typo` = two adjacent digits swapped, or one digit changed):
- address: W = V with a typo in the house number → `Sorry, typo: the house number is {V's}, not {W's}.`
- phone: W = V with a typo in the last four digits → `Sorry, typo: the last four digits are {V's}, not {W's}.`
- email: W = V on another domain → `Oops, the domain is {V's}, not {W's}.`

Otherwise W is a fresh value and the correction restates V: one of `Actually, scratch that - please use {V} instead.` · `Sorry, I made a typo there: the correct one is {V}.` · `Wait, ignore that: it should be {V}.`
**Ask:** `Please change my {label} to {X}.` with label `email address` / `mailing address` / `phone number`.
**Units:** `t` a twin with a look-alike phone (1) · `l` a twin whose email has C's local part on another domain (1) · `a` a customer with another name whose email has C's local part (1) · `b` an invoice of C (1) · `f` · `z`. Twins required: n ∈ [1, 2].
**Solver:** SPEC.md §4 F5 (v1's `solve_f5`): customers matching every `lookup` pair; one call each with `params.field`, `params.new_value`.

| Template | Field | Request | `params` | Required |
|---|---|---|---|---|
| **F5A** | email or address | `Support inbox message: "Hi, this is {name}. My phone number on file is {phone}. {ask V}"` | `lookup={"name","phone"}`, `field`, `new_value=V` | `t` × n |
| **F5B** | email, address or phone | `Support inbox message from {email}: "Hi! {ask W} {correction}"` | `lookup={"email"}`, `field`, `new_value=V` | `l` |
| **F5C** | phone or address | `Support inbox message: "Hi, this is {name} ({email}). {ask V}"` | `lookup={"name","email"}`, `field`, `new_value=V` | `l` × n |
| **F5N** ⟂near | email or address | `Support inbox message: "Hi, this is {name}. My phone number on file is {phone}. {ask W} {correction}"` | `lookup={"name","phone"}`, `field`, `new_value=V` | `t` × n |
| **F5X** ⟂far | phone | `Support inbox message from {email}: "Hi, please change my phone number to {V shown}." Save the number in the same format as the phone numbers already on file.` V shown = V (`+1-AAA-555-NNNN`) written as one of `(AAA) 555-NNNN`, `AAA.555.NNNN`, `AAA 555 NNNN`, `AAA555NNNN`, `1-AAA-555-NNNN`, `+1 AAA 555 NNNN`, `1 (AAA) 555-NNNN`. | `lookup={"email"}`, `field="phone"`, `new_value=V` (canonical) | `a` |

⟂near / ⟂far = held out. Never in `train`, `sametype` or `broken`.

**Near templates use trained steps only:** F1N = F1A's per-seat difference + F1B's "p% rounded down" + month lookup (F1A/F1B) · F2N = F2B's installment duplicate + F2C's email lookup · F3N = F3B's unused units + F3A's "p% rounded down" · F4N = F4A's discount lines + F4B's tax rule · F5N = F5A's phone-among-twins lookup + F5B's correction. **Far rules never trained:** price tiers (F1X), overpayment (F2X), free allowance (F3X), minimum commitment (F4X), phone format conversion (F5X).

## 5. Difficulty knobs (one tuning pass, ADR 0007 §5)

The calibration gate allows one tuning pass on **number ranges, line counts and distractors only**. Every such knob is a module-level UPPER_CASE constant; a tuning pass is an edit of these constants (and the matching numbers in §4), then `make data-v3 acceptance-v3`.

| Where | Knobs |
|---|---|
| `envlab/tasks/v3/common.py` | `K_CHOICES` (distractor count), `MIN_DISTRACTORS` (trim floor), `ADDON_UNIT`, `USAGE_LINES` |
| `envlab/tasks/v3/f1.py` | `SEAT_RATE`, `SEATS`, `SEAT_OVER`, `T_ADDONS`, `PERCENTS`, `F1B_PER_SEAT_SHARE`, `DOWNGRADE_GAP`, `TIER_SEATS`, `TIER_EXTRA`, `TIER1_PRICE`, `TIER_GAP` |
| `envlab/tasks/v3/f2.py` | `T_ADDONS`, `F2A_PAYMENTS`, `INSTALLMENTS`, `F2X_INSTALLMENTS`, `F2X_EXTRA_SHARE`, `F2C_PHONE_SHARE` |
| `envlab/tasks/v3/f3.py` | `PERCENTS`, `DISPUTED_QTY`, `T_ADDONS`, `CAP_BINDING_SHARE`, `USAGE_QTY`, `USAGE_UNIT`, `USAGE_LINES_ON_T`, `USED_SHARE`, `ALLOWANCE_SHARE`, `OUTAGE_HOURS`, `SEAT_SHARE_C`, `SEATS`, `SEAT_RATE` |
| `envlab/tasks/v3/f4.py` | `DISCOUNT_ADDONS`, `DISCOUNT_QTY`, `DISCOUNT_LINES`, `DISCOUNT_PCT`, `FIXED_DISCOUNT`, `TAX_ADDONS`, `TAX_QTY`, `TAX_BP`, `LARGE_ADDONS`, `LARGE_QTY`, `LARGE_SEAT_SHARE`, `SEATS`, `SEAT_RATE`, `MIN_ADDONS`, `F4X_BELOW_SHARE` |
| `envlab/tasks/v3/f5.py` | `TWINS`, `FIELDS`, `PARTIAL_CORRECTION_SHARE`, `CORRECTIONS`, `PARTIAL_CORRECTIONS`, `PHONE_FORMATS` |

Levers by direction: **easier** = smaller quantities/prices, fewer lines, smaller k, `TWINS = (1, 1)`, `PARTIAL_CORRECTION_SHARE` lower, `CAP_BINDING_SHARE` lower; **harder** = the reverse. Keep k within the prompt budget (§10). Request phrasings and rules are not knobs.

## 6. Graders

Unchanged (SPEC.md §6). Leaky still ignores amounts and values; it fails when the call hits **another customer's** record (K4), so v3's same-name customers and look-alike contacts are what keep leaky below saturation.

## 7. Quality gates

G1–G3 as SPEC.md §7. **Acceptance:** every broken instance fails ≥ 1 gate, and ≥ 99% of valid instances pass all three, pooled over `train`, `heldout` and `sametype` (`python -m envlab.verify.acceptance --data data/v3` picks these splits from a v3 manifest).
- **G2** uses the v3 solvers (§4 Solver lines, `envlab.verify.solvers_v3`); `solve` dispatches on the template id.
- **G3 wrong-value call:** SPEC.md §7 table, with one v3 extension for F2: if T has no other succeeded payment of p's amount (F2X), refund the earliest other succeeded payment on T instead. (Every v1 instance has an equal-amount original, so v1 behaviour is unchanged.)

| Family | Wrong call |
|---|---|
| F1 | same invoice, `amount − 1` |
| F2 | earliest succeeded payment on T with p's amount, id ≠ p; else (F2X) earliest other succeeded payment on T |
| F3 | `A + 1` if `A + 1 ≤ invoice.amount_due`, else `A − 1` |
| F4 | `new_amount + 1`, or `+ 2` if that equals `amount_billed` |
| F5 | email `"x" + V`; address `V + " Apt 2"`; phone last digit d → `(d + 1) % 10` |

## 8. Broken variants

100 instances, 25 per type, 5 per family per type, built from train templates only. `impossible`, `grader_bug` and `env_bug` are SPEC.md §8's family-level mutations unchanged (`envlab.tasks.broken.make_broken`); `ambiguous` is v3's own.

| Type | Source templates | Mutation | Expected gate failure |
|---|---|---|---|
| `impossible` | F1A F2A F3A F4A F5A | SPEC.md §8 | G1(b), G2 |
| `ambiguous` | F1A F2C F3A F4C F5C | add a second valid target: F1A another open invoice of C **in T's month** with the plan overcharged (U kept, more seats) · F2C another invoice of C charged twice · F3A a second open case on T (another line) · F4C another open mis-summed invoice of C (v1 error injection) · F5C another customer with C's name and email. Records appended; ledger rebuilt; `oracle_final_state` recomputed | G2 (`multiple_solutions`) |
| `grader_bug` | F1B F2B F3B F4B F5B | SPEC.md §8 (perturbed oracle final state) | G1(c) |
| `env_bug` | F1C F2C F3C F4C F5C | SPEC.md §8 | G1(a) (+ G1(b) for F1–F4) |

Seeds `derive_seed(7003, kind, template, index)`; ids `{template}-broken-{kind}-{index:03d}`.

## 9. Data build and splits

`python -m envlab.tasks.build --set v3 --out data/v3` (`make data-v3`). Deterministic: same code ⇒ byte-identical files. The default `--set v1` builds v1 exactly as before; `data/manifest.json` is never touched.

| Split | File | Templates | Count | Base seed |
|---|---|---|---|---|
| `train` | `train.jsonl` | 15 train (A, B, C) | 6,000 = 400 each | 7001 |
| `heldout` | `heldout.jsonl` | 5 near + 5 far | 1,000 = 100 each | 7002 |
| `sametype` | `sametype.jsonl` | 15 train, unseen instances | 500, split evenly (34 for the first 5 templates, 33 for the rest) | 7005 |
| `broken` | `broken.jsonl` | §8 | 100 | 7003 |

- Instance seeds, ids and "split evenly" as SPEC.md §9: `derive_seed(base, template, index)`, ids `{template}-{split}-{index:05d}` (e.g. `F1A-train-00007`, `F3X-heldout-00012`).
- **`sametype`** is disjoint from `train` by `content_key` (SPEC.md §9: request + shown records, order-insensitive); a colliding draw moves to `derive_seed(7005, template, index, "redraw", k)`.
- `data/v3/splits/{split}.txt` lists ids. As in v1, only `manifest.json` and `splits/` are committed; the `.jsonl` files are rebuilt.
- **`data/v3/manifest.json`** (`sort_keys`, indent 2): `spec_version` `"v3.0"`, `adr`, `label`, `seeds`, `counts` per split and template, `files` (sha256 of every jsonl and split file), `heldout_templates` (`{template: "near" | "far"}`), `far_templates`, `broken_labels`, `sametype` (`content_key`, `disjoint_from` with train's sha256, `redrawn` ids), `prompt_budget`, `prompt_estimate` (min / median / max per template).
- The build asserts for every non-broken instance: initial invariants, oracle call ok, oracle final state matches, final invariants, prompt estimate ≤ 2,000.

## 10. Prompt budget

Format exactly SPEC.md §10. **Budget:** real prompt ≤ **2,048 tokens** (`V3_PROMPT_TOKEN_LIMIT`); the generator keeps the CI-safe estimate `850 + ceil(len(user_message) / 2.5)` ≤ **2,000** (`V3_ESTIMATE_BUDGET`), the same margin as v1. GPU-side code must count real tokens and treat a v3 prompt > 2,048 as an error. Context for training: 2,048 prompt + 1,024 answer = 3,072 (ADR 0007 §6).

## 11. Python interfaces

```python
# envlab/tasks/v3/__init__.py   registry (fixed): FAMILIES, TRAIN/NEAR/FAR/HELDOUT_TEMPLATES, TEMPLATES,
#                               SPEC_VERSION = "v3.0", DATA_DIR = "data/v3", is_v3_template, heldout_kind
# envlab/tasks/v3/common.py     knobs, finalize_v3, twins, look-alike phones, installments, text helpers
# envlab/tasks/v3/f1.py … f5.py one build(d, template, variant) per family + pure oracle-rule functions
# envlab/tasks/v3/generate.py
def generate(template: str, seed: int) -> TaskInstance
def generate_split(split: str, *, train: list[TaskInstance] | None = None) -> list[TaskInstance]
SPLITS, SPLIT_SEEDS, SPLIT_ORDER = ("train", "heldout", "sametype", "broken"), BROKEN_SEED = 7003
# envlab/tasks/v3/broken.py
SOURCES: dict[str, tuple[str, ...]]
def make_broken_v3(base: TaskInstance, kind: str, seed: int) -> TaskInstance
def generate_broken() -> list[TaskInstance]
# envlab/tasks/v3/build.py
def build_dataset(out_dir: Path) -> dict          # writes §9 files; returns the manifest
# envlab/verify/solvers_v3.py
SOLVERS_V3: dict[str, Callable[[TaskInstance], list[ToolCall]]]   # used by envlab.verify.solvers.solve
# envlab/prompts.py
V3_PROMPT_TOKEN_LIMIT = 2048; V3_ESTIMATE_BUDGET = 2000             # v1's constants unchanged
```

## 12. Design notes

- **Difficulty targets** (ADR 0007 §5; not measurable without a GPU, checked by the calibration gate): base 4B thinking-on in the 10–90% band on ≥ 9 of 15 train templates (aim 30–60%), leaky pass ≤ 90%; base 2B ≥ 6 of 15 in band.
- **Where difficulty comes from.** Arithmetic: F1, F3, F4 (multi-step, stated rounding, larger numbers, 4–6 lines). Customer finding: F1A, F2B, F2C, F3B, F4A, F4C, F5A, F5B, F5C always show a same-name customer whose records look like a valid target in all but one detail (month, contact, sum). Record finding: F2A (failed and refunded attempts, another invoice with an already-refunded duplicate), F3A (a rejected case on T). Values: F5B partial corrections.
- **Expected easier templates:** F2A, F2C, F5A, F5C (no arithmetic; exact matching). They help the 2B band count and keep some leaky failures (wrong twin). If 4B saturates them, the tuning levers are `TWINS`, `K_CHOICES` and `PARTIAL_CORRECTION_SHARE`.
