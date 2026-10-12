# 0003. Step count, generation backend and 4B layout from the benchmark

- **Status:** Accepted
- **Date:** 2026-10-12
- **Deciders:** owner (proposed after the §0.4 benchmark)

## Context

ADR 0001 left four settings to Monday's GPU checks: LR (#6, now 1e-6), step count (#7), generation backend (#8) and the 4B layout (#9). The plan assumed runs of about 2 h (0.8B), 4 h (2B) and 8 h (4B) at 300 steps.

The benchmark (`make bench`, every prompt at the full 1,536 tokens and every answer at the full 128) measured far less:

| Run | Step time | 300 steps |
|---|---|---|
| 0.8B, 1 GPU colocated | 4.4 s | 22 min |
| 2B, 1 GPU colocated | 6.2 s | 31 min |
| 4B, 1 vLLM server + 2 trainers | 8.1 s | 40 min |
| 4B, 4 GPUs colocated | 4.4 s | 22 min |

At 300 steps the whole matrix would take about 2 h and leave most compute idle. And with LR 1e-6, a leaky run might not move far enough to show reward hacking.

## Decision

- **#7 Step count: 1,000 steps for every size.** That is 4 passes over the 2,000 training tasks. The whole matrix takes about 5 h. The per-step budget is unchanged, so H5 still compares like with like.
- **#8 Backend: vLLM**, colocated for every run. It passed generation and weight sync at all sizes, and in server mode. HF generate stays the fallback.
- **#9 4B layout: 4 ZeRO-3 ranks with colocated vLLM.** 4.4 s/step, against 8.1 s for 1 server + 2 trainers. That is also 17.6 vs 24.3 GPU-seconds per step.
- **Timeouts** are about 2× the benchmark at 1,000 steps. Eval cadence is unchanged (every 50 steps, PREREG).
- **Training blocks shrink** to about 2 h for block 1 and about 1.5 h each for blocks 2 and 3. The extra time becomes buffer, for reruns and optional ablations.

## Alternatives considered

| Option | Why not |
|---|---|
| 300 steps | Uses about 5% of the available compute; little movement at LR 1e-6 |
| 2,000 steps | 8 passes over 2,000 tasks raises the risk of memorising them; 1,000 leaves room to extend |
| 1 vLLM server + 2 trainers for 4B | 1.8× slower and leaves 1 GPU idle |
| A higher LR instead of more steps | 3e-6 and 1e-5 collapsed in the toy sweep (DECISIONS #6) |

## Consequences

- **Easier:** the matrix finishes well before the cutoff, so a failed pair can be rerun in full.
- **Harder:** eval falls behind during block 1 (2 GPUs handle 58 checkpoints/h; six fast runs make 70–97). Final checkpoints go first and the backlog clears after the runs end. Idle GPUs join eval.
- **Watch:**
  - The vLLM-vs-trainer logprob gap over 1,000 steps, since the toy collapses followed gap spikes.
  - Overfitting to the training tasks: compare held-out with train pass rates.
  - Real step times. Real prompts and answers are shorter than the benchmark's maximum lengths.
