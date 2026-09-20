# Historical A100 Timing Evidence

Reviewed 2026-09-17. This is a historical measurement record, **not approval of the current candidate**. Current blockers and proposed work are in [HISTORY_FIXES.md](HISTORY_FIXES.md); operational instructions are in the [launch guide](README_A100_LAUNCH.md).

## Source and limits

The interrupted 2026-09-16 Modal run used an NVIDIA A100-SXM4-40GB, with 39.5 GiB visible VRAM and PyTorch 2.5.1+cu124.

Original local sources:

- Report: `/Users/phucdang/Downloads/LegalIR_A100_Timing-ScaleDown_Report_2026-09-16.md`.
- Raw log: `/tmp/modal-a100-launch.log` (temporary local path; may not remain available).
- Recorded raw-log size: 244,771 bytes, 2,028 lines.
- SHA-256: `4ad86f1ba1a65eedfc49916b3faf4813c9a87319204b36d99ecba92e03102de7`.

Line 1921 reports `Stopping app - local client disconnected`. This establishes the logged disconnect event, not its underlying network or remote-terminal cause. Earlier documents disagreed about total elapsed time (1.7 versus 3.9 hours); neither is retained as a verified full-run duration. The run did not produce a completed five-fold benchmark.

## Measured baseline

| Measurement | Recorded value |
|---|---:|
| Micro chunks processed by PyVi | 934,416 |
| PyVi indexing/tokenization | 2,648.6 s / 44.14 min |
| Dense macro chunks | 219,460 |
| Fold 0 pair pool | 6,102 positive + 67,643 negative = 73,745 |
| Fold 0 training updates | 700; batch 8, accumulation 2, BF16 |
| Fold 0 held-out queries | 1,400 |
| Fold 0 evaluation duration | 3,103.2 s / 51.72 min |
| End-to-end evaluation throughput | **0.451 queries/s**, or **2.217 s/query** |
| Fold 0 Recall@5 | 82.54% |
| Fold 0 Precision@5 | 17.60% |
| Fold 0 candidate recall@50 | 98.05% |
| Fold 0 candidate recall@150 | 98.89% |

The earlier reading of **2.2 queries/s** inverted the units. At the observed rate, five equally sized folds would take **258.6 minutes for evaluation alone**. This is an extrapolation, not a completed measurement. Historical 11–15-hour overall runtime and approximately 15-minute fold training figures were estimates, not reliable cold end-to-end timings.

## Mechanisms now present and what they do not establish

| Mechanism | Scope / limitation |
|---|---|
| Interleaved query-balanced sampling | Removes the old all-positive then all-negative ordering; changes training exposure. Does not prove convergence or improved score. |
| Shared static mining candidates | Reduces repeated label-independent mining searches. Held-out inference and other paths can still retrieve/load indexes separately. |
| Multi-query reranking | Flattens/scatters pairs across queries. Actual A100 throughput and memory headroom remain unmeasured. |
| Parallel PyVi tokenization | Can reduce CPU-bound indexing time if resources suffice. No measured >50% speedup is established. |
| Missing-rank sentinel handling | Prevents absent branches from adding artificial RRF mass. Ranking-affecting, not just a runtime optimization. |
| Completed-stage markers | Persist stage outputs. Identity gaps remain; fresh Modal attempts do not automatically reuse previous directories. |
| Revision/cache/coverage repairs | Local candidate improvements under review; not covered by the older committed release receipt. |

Do not describe sampler, sentinel, model-revision, or coverage changes as score-equivalent performance refactors. They can change predictions or training behavior and require renewed evaluation.

## Five-hour feasibility remains open

The planning budget is **270 minutes of work plus 30 minutes of contingency**, detailed in `fix.md`. It includes cold acquisition/setup/indexing, mining, all required training/evaluation jobs, final inference, validation, and durable delivery. It must not move precomputation outside the timer to manufacture a cold-run speedup.

For scale: 7,000 held-out queries in 55 minutes require about **2.12 queries/s**, approximately **4.70×** the old measured rate. Document-disjoint evaluation increases the workload further. Faster neural forward passes alone do not establish this whole-stage improvement.

The >96% Recall@5 objective is also unproven. High candidate recall and a corpus top-five capacity of 100% provide headroom, not achieved ranking quality. Use the actual candidate top-five oracle and complete honest held-out evaluation before making a quality claim.

## Existing release evidence is narrower

Committed release `d39792836482f29bd6d5e691235690c738cdde3d` binds runtime `373e8791917915da36864b7eb9b2f457493b4a0e`. Its genuine Kaggle dual-T4 receipt records three optimizer updates in 27.05 seconds, finite training losses, positive weight delta, and adapter reload. That is a short correctness gate, not evidence for the uncommitted repairs, A100 resource sizing, five-hour completion, or >96% Recall@5.

See the [release workflow](REPRODUCIBLE_TRAINING_WORKFLOW.md) for the next qualification sequence. No new GPU run was performed for this documentation review.
