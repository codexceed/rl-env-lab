# Final plan: RL environment and training lab

Oct 11, 2026 · @Sarthak Joshi

## At a glance

Train Qwen3-0.6B with GRPO inside a small billing simulator, twice: once with a robust reward, once with a leaky one. Measure what it really learned.

|  |  |
| --- | --- |
| **Dates** | Mon 12 to Sun 18 Oct 2026 |
| **Effort** | 6 active hours a day; training and evals run unattended overnight |
| **Compute** | Scenario A (preferred): up to 3 × 8 H200 on Lightning AI. Scenario B: one rented 24 GB GPU. MacBook for development in both |
| **Budget** | Scenario B: $100 cloud compute plus about $15 API; never above $200 in total |
| **Core experiment** | 2 matched GRPO runs: robust vs leaky reward, same start, same budget |
| **Must ship** | Tested simulator, one real GRPO run, held-out eval, rerunnable public repo |

**Success on 18 Oct:** I can show the environment, explain how a model was trained in it, and separate the reward it optimised from the skill we meant to teach. A better 0.6B model is not the goal; negative results count if measured correctly.

## Locked decisions

Agents treat these as fixed. Changing one needs a fallback trigger from the last section, logged in `DECISIONS.md`.

| Area | Decision |
| --- | --- |
| Model | `Qwen/Qwen3-0.6B`, thinking disabled; Scenario A may promote 4B or 8B on Wednesday; fallback `Qwen3-1.7B` |
| Training | GRPO via Hugging Face TRL `GRPOTrainer` + PEFT LoRA (full fine-tune allowed for 4B and 8B in Scenario A) |
| Environment | Pure-Python, in-memory, deterministic; no network, no database server |
| Action format | Exactly one tool call per task, in Qwen's native `<tool_call>{json}</tool_call>` format |
| Reward | Binary 0 or 1, computed from simulator state after executing the call |
| True success | The robust verifier, applied to held-out templates only |
| Exploit | A trajectory the leaky verifier passes and the robust verifier fails |
| Held-out split | 1 of 3 templates per family, frozen on Monday, never trained on |
| Matched runs | Robust and leaky runs share config, seed, start weights and step count; only the reward function differs |
| Language and tooling | Python 3.11, `uv`, `pytest`, GitHub Actions CI; code is written on the Mac, and every number reported comes from CUDA GPUs |
| Licence | MIT; all data synthetic |

## System

&#91;embedded content: rollout and eval flow · one tool call per task\]

Training uses one verifier per run; evaluation always applies both, so every held-out sample is classed as success, exploit or failure.

## Environment and tasks

A billing back office with 5 entities, 6 write tools, and 5 task families of 3 templates each.

**Entities:** `Customer` (id, email, credit\_balance) · `Invoice` (id, customer\_id, amount\_billed, amount\_due, status) · `Payment` (id, invoice\_id, amount, status) · `LedgerEntry` (id, customer\_id, invoice\_id, amount, kind) · `Case` (id, customer\_id, invoice\_id, status, notes)

**Tools:** `apply_credit(invoice_id, amount, reason)` · `refund_payment(payment_id, reason)` · `adjust_invoice(invoice_id, new_amount, reason)` · `update_customer(customer_id, field, value)` · `resolve_case(case_id, adjustment, note)` · `void_invoice(invoice_id, reason)`. Every money-moving tool posts a ledger entry.

**Prompt:** system message with the tool schemas; user message with the request plus a JSON snapshot of the relevant records and 2 to 4 distractor records. Output that fails to parse, or names an unknown tool, scores 0.

| Family | Request | Correct action | Skill tested |
| --- | --- | --- | --- |
| F1 Overcharge | Billed above the contract price | `apply_credit` for the exact difference | Find record, subtract |
| F2 Duplicate payment | Customer paid twice | `refund_payment` on the later duplicate | Pick the right record |
| F3 Dispute | Open case with a disputed amount | `resolve_case` with the correct adjustment and a note | Compute and close |
| F4 Billing error | Line items don't sum to the invoice | `adjust_invoice` to the correct total | Sum and correct |
| F5 Account change | Customer asks to change a field | `update_customer` with the exact new value | Extract and apply |

**Splits:** templates T1 and T2 train, T3 held out, per family. Seeded generation: 2,000 train instances, 250 held-out, 100 broken.

**Broken variants (25 of each, labels in `manifest.json`):** impossible (referenced record missing) · ambiguous (two valid targets) · grader bug (oracle end state wrong) · env bug (tool raises on valid input).

**Quality gates, run on every instance:**

1. **G1 Oracle:** the reference action passes the robust verifier.
2. **G2 Unique:** exactly one candidate action in the snapshot passes.
3. **G3 Exploit reachable:** a scripted wrong-value action passes leaky and fails robust.

Acceptance means every broken variant is rejected, and at least 99% of valid instances pass.

## Rewards, metrics and hypotheses

Two verifiers, one metric sheet, four hypotheses committed to `PREREG.md` on Monday before any training.

| Verifier | Passes when |
| --- | --- |
| **Robust** | Final state matches the oracle on every touched record, the ledger balances, and nothing else changed |
| **Leaky** | The right tool hit the right customer, and status and note are set. It ignores amounts, values, the ledger, and which payment was refunded |

| Metric | Definition | Logged |
| --- | --- | --- |
| Training reward | Mean reward per step | Every step |
| Dead-group share | Share of prompt groups whose 8 rewards are all equal | Every step |
| Entropy, completion length | As logged by TRL | Every step |
| **True success** | Robust pass rate on held-out, 4 samples per instance at temperature 0.7, 95% bootstrap CI | Base, every 50 steps, final |
| **Exploit rate** | Share of held-out samples passing leaky but failing robust | Same as true success |
| Format failures | Share of outputs that don't parse | Same as true success |
| Cost | GPU hours and API dollars | Per run |

| Hypothesis | Prediction | Falsified if |
| --- | --- | --- |
| **H1** The environment teaches | Robust run: held-out true success rises at least 10 points over base | The 95% CI of the gain includes 0 |
| **H2** A leaky reward misleads | Leaky run: training reward at least the robust run's; true success at least 10 points lower; exploit rate at least 10% and at least 2× base | Any of the three parts fails |
| **H3** Difficulty matters *(exploratory)* | Families with base pass rate of 10 to 90% gain more than those outside | Reported, not tested |
| **H4** Triage works *(target)* | Macro-F1 at least 0.70 on broken-variant labels | F1 below 0.70 |
| H5 Bigger models exploit faster (Scenario A only) | Leaky runs: final exploit rate rises with size across 0.6B, 1.7B and 4B, averaged over 3 seeds | The order is not increasing, or the 4B vs 0.6B gap has a 95% CI that includes 0 |

H1 and H2 are the core experiment. Triage classes: agent error, env bug, spec ambiguity, grader error, format.

## Training configuration

One config file, `configs/grpo.yaml`, shared by both runs. TRL defaults apply to anything not listed.

| Setting | Value |
| --- | --- |
| LoRA | r 16, alpha 32, dropout 0.05, all linear layers |
| Learning rate | Chosen Monday from 1e-5, 3e-5 or 1e-4: the highest where toy reward rises with no entropy collapse; then frozen |
| Generations per prompt | 8 |
| Prompts per step | 8 (64 completions) |
| Max lengths | Prompt 1,536 tokens; completion 128 tokens |
| Sampling | Temperature 1.0 during training |
| KL penalty (beta) | 0.0 |
| Steps | 300, or fewer so a run fits in 8 hours (set from Monday's step timing) |
| Checkpoints | Every 50 steps; held-out eval runs on each from a separate script |
| Precision and memory | bf16, gradient checkpointing |
| Generation backend | vLLM colocated if it works within 1 hour on Monday; otherwise plain HF generate |
| Seed | 42 |

**SFT warm-up (fallback only):** 1 epoch on 500 oracle completions from train templates. Both RL runs then start from that same checkpoint.

## Compute scenarios

The scenario is decided Monday night. Monday to Wednesday are identical in both, so neither choice causes rework.

|  | **A: up to 3 × 8 H200 on Lightning AI** (preferred) | **B: $100 cloud** |
| --- | --- | --- |
| Main model | Largest ladder model with at least 6 of 10 templates in band on Wednesday; expected 4B or 8B, full fine-tune | Qwen3-0.6B, LoRA |
| Ladder | 0.6B, 1.7B, 4B, 8B | 0.6B, 1.7B (one robust and leaky pair) |
| Tasks | Single-turn; multi-turn (2 to 6 tool calls) as stretch | Single-turn |
| Seeds | 3 per condition | 1 |
| Core hypotheses | H1, H2, H5 | H1, H2 |
| Stack | TRL, vLLM in server mode; a multi-turn framework only for the stretch | TRL, vLLM colocated |
| Hardware rule | One node per run; never one run across nodes | One 24 GB GPU (RTX 4090 or L4); A100 80GB fallback |
| Spend | Free to me; idle nodes still auto-stop (about $37 per node-hour) | About $85 GPU (roughly 40 GPU-hours) plus about $15 triage API |

**Scenario A lanes**

| Node | Lane | Runs |
| --- | --- | --- |
| 1 | Core: main model, robust vs leaky, 3 seeds | 6 runs, one node each: 2 GPUs generate, 6 train |
| 2 | Ladder: 0.6B, 1.7B, 4B, single-turn, robust vs leaky | Up to 8 runs at once, 1 GPU each |
| 3 | Ablations and every held-out eval | SFT vs GRPO, LoRA vs full, difficulty-aware sampling |

**Go for A only if all three hold by Monday night:** access confirmed in writing; written confirmation that it is a personal grant with no Lightning IP claim; the toy GRPO run works on one node. Otherwise B.

**A-only rules:** distributed setup gets Tuesday night only. If single-node training of the main model fails by Wednesday noon, the core lane drops to the largest model that trains on one GPU. Keep checkpoints every 100 steps plus the final one.

## The week

&#91;embedded content: week plan · 12 to 18 Oct, gates per day\]

A failed gate triggers its fallback before the next day starts. In band means a base pass rate of 10 to 90%.

## Agent work packages

Monday sets up everything the week depends on. Each package ends in a test an agent can run.

**Monday 12 Oct: run M1, M2, M3 and M5 in parallel after M0**

| ID | Owner | Do | Accept when |
| --- | --- | --- | --- |
| M0 | Me, 15 min | Scenario A: confirm H200 access and the IP position in writing. Scenario B: rent one 24 GB on-demand GPU with per-second billing, idle auto-stop and a spend alert | Script prints the GPU name, memory and CUDA available; auto-stop tested once |
| M1 | Agent | Repo scaffold (layout below), `uv` env with pinned torch, transformers, trl, peft, datasets; CI with pytest and ruff | CI green |
| M2 | Agent | `train/smoke_toy.py`: Qwen3-0.6B + LoRA, reward = exact answer to two-number addition; 100 steps at each of the 3 learning rates | Reward rises for at least one LR; LoRA weights changed; step time logged |
| M3 | Agent drafts, I review | `SPEC.md`: entities, tool effects, ledger rules, 15 templates with oracle rules, both verifiers, gates G1 to G3, prompt format | I sign off; no open TODOs |
| M4 | Me | `PREREG.md`: H1 to H5 thresholds, held-out template IDs, eval protocol | Committed before any task training |
| M5 | Agent | `eval/metrics.py`: true success, exploit rate, bootstrap CI, on fake data | Unit tests pass |
| M6 | Me, end of day | Record the scenario, chosen LR, step count and backend in `DECISIONS.md` | Committed |

**Tuesday to Sunday**

| Day | Packages (parallel agents where marked) | Output |
| --- | --- | --- |
| Tue | Simulator and tools ∥ verifiers and gates ∥ generators and broken variants; conformance tests | `src/envlab/`, `tests/` |
| Wed | Acceptance report; baseline of the base model, 50 instances per template × 8 samples | `reports/acceptance.md`, `reports/baseline.md` |
| Thu | Robust run; checkpoint eval runner | `runs/robust/`, `eval/heldout.py` |
| Fri | Leaky run; robust checkpoints evaluated; `eval/compare.py` | `runs/leaky/`, results tables |
| Sat | H1 and H2 verdicts; triage classifier plus 30 hand-labelled real failures | `reports/results.md`, `triage/` |
| Sun | Clean-clone rerun from the README; report; field notes v2 | Public repo |

**A healthy run at step 100:** mean reward over steps 80 to 100 is above steps 0 to 20; dead-group share is below 80%; no NaN or crash.

```
rl-env-lab/
  README.md  SPEC.md  PREREG.md  DECISIONS.md
  configs/       grpo.yaml  sft.yaml
  src/envlab/
    sim/         entities, tools, ledger, state hash
    tasks/       f1..f5 templates, generate, broken
    verify/      robust, leaky, gates
    prompts.py   parse.py
  train/         smoke_toy.py  grpo.py  sft.py
  eval/          heldout.py  metrics.py  compare.py
  triage/        classify.py  labels/
  reports/       acceptance  baseline  results
  tests/
  data/  runs/   (gitignored, except manifest and split IDs)
```

## Fallbacks, cut order and guardrails

A failed gate triggers its row below the same day; every switch is logged in `DECISIONS.md`.

| Trigger | Response, in order |
| --- | --- |
| Chosen GPU unavailable, or CUDA broken | Switch provider, or move up to an A100 80GB; GPU spend capped at $150 |
| Toy reward flat at all 3 LRs | Check the reward function and parser first; then try 3e-4; then full fine-tune of 0.6B |
| Fewer than 6 of 10 train templates in the 10 to 90% band | SFT warm-up; then drop the hardest template per family; then Qwen3-1.7B |
| Out of GPU memory | Prompts per step 8 → 4; completion 128 → 96 tokens; then an A100 80GB |
| Run unhealthy at step 100 | Halve the LR and restart from step 0; a second failure means SFT warm-up |
| Run crashes | Resume from the last 50-step checkpoint |
| No exploit appears in the leaky run | Report it as a finding; stretch: widen the leak |
| Triage overruns Saturday | Ship a code-only exploit detector; the classifier moves to week 2 |

**Cut order:** stretch → triage classifier → template count. Never cut: the robust run, held-out evaluation, the reproducible repo.

**Stretch (only after Saturday's gate):** easy vs medium training · SFT vs GRPO at matched budget · curriculum sampling · multi-turn tasks.

**Guardrails:** personal accounts and own time · commits dated before 19 Oct · synthetic data only · nothing confidential from any employer or vendor in public files · API keys in `.env`, never committed · every unattended job has a timeout and a spend cap, and stops its instance on exit.

