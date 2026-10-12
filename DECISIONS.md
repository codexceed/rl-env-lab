# DECISIONS

Locked decisions and every change to them. `PROGRESSION.md` is the execution plan; this file records *why* a locked decision is what it is.

Format: one row per decision. A changed decision gets a new row; the old row is marked superseded.

| # | Date | Area | Decision | Why | Status |
|---|---|---|---|---|---|
| 1 | 2026-10-12 | Deadline | Fri 16 Oct 2026, 12:00; training cutoff Fri 06:00 | Availability | Locked |
| 2 | 2026-10-12 | Compute | 2 × 8 H200 on Lightning AI | Availability | Locked |
| 3 | 2026-10-12 | Model family | Qwen3.5 (0.8B / 2B / 4B); fallback Qwen3 (0.6B / 1.7B / 4B) | Same-family ladder; 2B fits 1 GPU/run | Pending §0.3 compat |
| 4 | 2026-10-12 | Main model | Qwen3.5-2B | All 3 pairs fit in one block | Pending §2.2 band check |
| 5 | 2026-10-12 | Weights updated | Full fine-tune, all sizes | Fair size comparison | Pending Mon confirm |
| 6 | — | Learning rate | `____` (from 1e-6 / 3e-6 / 1e-5) | | Pending §0.4 |
| 7 | — | Step count | `____` (common across sizes) | | Pending §0.4 benchmark |
| 8 | — | Generation backend | `____` (vLLM preferred; HF fallback for train + eval) | | Pending §0.3 |
| 9 | — | 4B GPU layout | ~~1 vLLM server + 3 train~~ infeasible (64 ∤ 3 ranks); candidates 1 server + 2 train, or 4-GPU colocated | | Pending §0.4 benchmark |
| 10 | 2026-10-12 | Hypotheses | H1, H2, H5 tested; H3 exploratory; H4 cut | Deadline | Locked |
| 11 | 2026-10-12 | Scope cuts | Triage classifier, multi-turn cut; ablations optional | Deadline | Locked |
| 12 | 2026-10-12 | Node shutdown | Coordinator only, after training + eval + transfers | Runs share nodes | Superseded by #13 |
| 13 | 2026-10-12 | Node release | No auto-release. Coordinator stops runs at cutoff and marks idle; machines stay attached through dips; owner releases manually | Re-acquiring H200s costs more than idle time | Locked |
