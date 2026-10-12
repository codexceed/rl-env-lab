# DECISIONS

One-line ledger of decisions. **The reasoning lives in ADRs** (`docs/adr/`); everything up to 2026-10-12 is in [ADR 0001](docs/adr/0001-foundational-decisions.md). `PROGRESSION.md` is the execution plan.

Format: one row per decision. A changed decision gets a new row; the old row is marked superseded. A row backed by an ADR links its file in the Status column (`[ADR NNNN §X](docs/adr/NNNN-title.md)`), so the reasoning is one click or one `Read` away.

| # | Date | Area | Decision | Why | Status |
|---|---|---|---|---|---|
| 1 | 2026-10-12 | Deadline | Fri 16 Oct 2026, 12:00; training cutoff Fri 06:00 | Availability | Locked |
| 2 | 2026-10-12 | Compute | 2 × 8 H200 on Lightning AI | Availability | Locked |
| 3 | 2026-10-12 | Model family | Qwen3.5 (0.8B / 2B / 4B); fallback Qwen3 (0.6B / 1.7B / 4B) | Same-family ladder; 2B fits 1 GPU/run | Pending §0.3 compat |
| 4 | 2026-10-12 | Main model | Qwen3.5-2B | All 3 pairs fit in one block | Pending §2.2 band check |
| 5 | 2026-10-12 | Weights updated | Full fine-tune, all sizes | Fair size comparison | Pending Mon confirm |
| 6 | 2026-10-12 | Learning rate | **1e-6**, constant, all sizes | 2B toy (11-digit addition, 100 steps): 1e-6 reward 0.82→0.87, no entropy collapse; 3e-6 collapsed at step ~28, 1e-5 at step 3, each after vLLM-vs-trainer logprob gaps of 10+ nats. The gap is now logged per step | Locked (owner) |
| 7 | 2026-10-12 | Step count | **1,000**, all sizes | Benchmark: 300 steps = 22–40 min per run; 1,000 = 4 passes over 2,000 tasks, matrix ~5 h | Locked · [ADR 0003](docs/adr/0003-step-count-backend-and-4b-layout.md) |
| 8 | 2026-10-12 | Generation backend | **vLLM, colocated**; HF generate fallback for train + eval | Compat: generation + weight sync pass at all sizes and in server mode | Locked · [ADR 0003](docs/adr/0003-step-count-backend-and-4b-layout.md) |
| 9 | 2026-10-12 | 4B GPU layout | **4 ZeRO-3 ranks, colocated vLLM** (1 server + 3 train infeasible: 64 ∤ 3) | 4.4 s/step vs 8.1 s for 1 server + 2 train | Locked · [ADR 0003](docs/adr/0003-step-count-backend-and-4b-layout.md) |
| 10 | 2026-10-12 | Hypotheses | H1, H2, H5 tested; H3 exploratory; H4 cut | Deadline | Locked |
| 11 | 2026-10-12 | Scope cuts | Triage classifier, multi-turn cut; ablations optional | Deadline | Locked |
| 12 | 2026-10-12 | Node shutdown | Coordinator only, after training + eval + transfers | Runs share nodes | Superseded by #13 |
| 13 | 2026-10-12 | Node release | No auto-release. Coordinator stops runs at cutoff and marks idle; machines stay attached through dips; owner releases manually | Re-acquiring H200s costs more than idle time | Locked |
| 14 | 2026-10-12 | Process | Major decisions recorded as ADRs in `docs/adr/`; agents prompt the owner for one on major architecture/planning/experiment-design changes | Progressive disclosure; decisions outlive chat | Locked · [ADR 0001](docs/adr/0001-foundational-decisions.md) |
| 15 | 2026-10-12 | Compute account | `bare-metal` ([rate] per 8× H200), set up on CPU first | Half the price of public-prod, available | Locked · [ADR 0001 §D6](docs/adr/0001-foundational-decisions.md) |
| 16 | 2026-10-12 | Python | Runtime 3.14; code + deps 3.12-compatible, CI tests both | Kernel fallback stays one line | Locked · [ADR 0001 §D3](docs/adr/0001-foundational-decisions.md) |
| 17 | 2026-10-12 | Model loading | Text-only export of Qwen3.5; fp32 master weights + bf16 compute | VL checkpoints break sync/resume; bf16 loses 1e-6 updates | Locked · [ADR 0001 §D2](docs/adr/0001-foundational-decisions.md) |
| 18 | 2026-10-12 | Tool-call format | Qwen3.5 native XML; JSON parsed as fallback; thinking always off | Native template; owner approved | Locked · [ADR 0001 §C2](docs/adr/0001-foundational-decisions.md) |
| 19 | 2026-10-12 | Run logging | Small run files synced to tracked `logs/` (git) and mirrored to Lightning Experiments (LitLogger) by a separate process; local JSONL stays the source of truth | Live view, n2→n1 transfer, public evidence; tracker failures can't touch runs | Locked · [ADR 0002](docs/adr/0002-run-logs-in-repo-and-lightning-experiments.md) |
| 20 | 2026-10-12 | Exploratory follow-up | Same-type eval on all checkpoints; thinking-on v2 round for 2B gated on a base diagnostic; pre-registered study unchanged | Held-out failures mix hacking with missing arithmetic; most groups dead | Locked · [ADR 0005](docs/adr/0005-exploratory-followup.md) |
| 21 | 2026-10-12 | Post-baseline amendments | Answer cap 128 → 256 tokens; H2 part 3 = leaky exploit ≥ base + 10 pts; train template F1T5 added (2B: 6 of 11 in band). PREREG §10 #10–#12 | Base evals: 4B answers truncated, 2× base unreachable, 2B 5 of 10 in band | Locked · [ADR 0004](docs/adr/0004-post-baseline-amendments.md) |
