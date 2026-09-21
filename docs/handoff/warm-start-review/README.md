# Handoff — LegalIR private-round readiness review

Review date: 2026-09-21. Reviewer: Claude (code + logic review; **no GPU run,
no `modal run`**). Baseline: release `2a0d17e` / runtime `180b15e`.

> **OPERATOR DIRECTIVE (2026-09-21):** *score is the objective; runtime is not a
> constraint.* Maximize Recall@5 as far as possible. Runtime-reduction tasks are
> therefore **demoted**, and score levers previously rejected as "too expensive"
> are back on the table. See **`08-score-maximization.md`** — start there.

Grounded in two real measurements:
- Run 4 receipt (`runs/run-04-oof89.6/reports/submission_manifest.json`)
- Live fold 0: `R@5 91.33% | Cand@50 98.05% | Cand@150 98.89% (2758.2s)`
- Live fold 1: `R@5 90.90% | Cand@50 97.20% | Cand@150 98.56% (2470.1s)` → mean **+0.54pp** vs Run 4, projected OOF **~90.15%**

## Verdict: **BLOCKED** — one correctness defect, one strategy correction

Private-round verification was **executed** (not just reviewed) — see
`07-private-round-verdict.md`:

| Requirement | Result |
|---|---|
| 1. Private-round configuration | **PASS** (7/7 discovery, fail-closed proven) |
| 2. Submission format & validity | **PASS**, 1 latent gap (validator allows 1–5, should be ==5) — BLOCKED-ON-CODE |
| 3. Time scaledown | **WAIVED** by operator directive — runtime is not a constraint (`07` §3) |
| 4. Recall policy parity | **PASS** (shared fixed RRF + 59 parity tests) |
| 5. Proof | **PASS** (2080 fixtures green; strict approval PASS) |

Top blocker: **warm-start leaks into OOF and document-disjoint evaluation**
(`01-CRITICAL-oof-leakage.md`). It does not crash; it silently invalidates the
only metrics used to judge a submission.

Runtime is **not** a blocker — that was my error, corrected in `05`.

## What changed from my earlier reporting (corrections)

| Earlier claim | Status | Corrected by |
|---|---|---|
| "These settings reach >0.96 Recall@5" | **Wrong.** Measured +0.24pp → ~89.85% | `06` |
| "Warm-start cuts training time 30–40%" | **Wrong.** Step count is coverage-pinned; training is 11% of runtime | `02`, `05` |
| "New config will time out at ~9.6h" | **Wrong.** Live data shows 1.23×, ~6.4h | `05` |
| "candidate_k 200 raises ceiling to 99.2%" | **Overstated.** Cand@150 already 98.89%; upside ≤1.11pp | `06` |
| candidate_k/rerank_k/max_length/max_chunks/pos_weight are wired | **Correct** — verified in source | `02` |
| Timeout 25,200s consistent across 3 files | **Correct** | `02` |

## The two findings that decide the strategy

**A. The loss is in ranking, not retrieval.** Gold is in the candidate pool
98.89% of the time but reaches top-5 only 91.33% — the cross-encoder loses
**7.56pp**; only 1.11pp is reachable by retrieving more.

**B. 82.4% of the model budget is idle.** `parameter_audit.json`: 702.75M used
of a 4.0B limit — **3.30B headroom**. Capacity was never the constraint, and
under the new directive neither is compute. Bigger LoRA rank, full-pool
reranking, and multi-model ensembles all fit comfortably.

## The one-paragraph strategic finding

The retriever is not the problem. The gold document is in the candidate pool
**98.89%** of the time, but only reaches top-5 **91.33%** of the time — the
cross-encoder loses **7.56pp** of recall that retrieval already earned. Only
1.11pp is reachable by retrieving more. Effort spent widening `candidate_k` is
spent on the wrong stage; the objective mismatch (pointwise BCE training a
ranking task) is where the headroom is.

## Files

| File | Contents |
|---|---|
| `01-CRITICAL-oof-leakage.md` | The blocker: mechanism, evidence, blast radius, fix options |
| `02-claims-audit.md` | Every claim re-checked against source: true vs false, with citations |
| `03-implementation-plan.md` | Ordered tasks with exact line numbers and code shapes |
| `04-verification-protocol.md` | Proof for each task + the release re-qualification chain |
| `05-runtime-reality.md` | Measured stage timings, my projection error, ranked time levers |
| `06-recall-bottleneck.md` | The 7.56pp reranker gap and how to close it |
| `07-private-round-verdict.md` | **Executed** private-round verification + READY/BLOCKED verdict |
| `08-score-maximization.md` | **START HERE** — ranked score levers under free runtime; 82% idle model budget |
| `11-road-to-096.md` | **How to reach 0.96** — arithmetic budget, lever ranges, honest probability |
| `10-fold-evidence-and-multigold.md` | **NEW** — fold 0+1 live evidence; arithmetic proof of multi-gold loss (~+2pp) |
| `09-open-items-and-ready-checklist.md` | Scope boundary, executed-vs-open proofs, dataset-gap scripts, 13-box READY checklist |

Read order: `08` (what to change) → `01` (why measurement must be fixed first) → `06` (evidence) → `03` → `04`. (`02` and `05` are evidence appendices.)

## Immediate action for the currently running job

Grep the live log for:

```
[+] Warm-start: loading existing LoRA adapter     # -> metrics CONTAMINATED, discard OOF
[!] Warning: Failed loading warm-start adapter    # -> cold-started, metrics OK
```

If neither line appears the run predates `180b15e` and its OOF number is
trustworthy. See `05-runtime-reality.md` for why this matters.

## ⚠️ Do not commit this folder on top of release `2a0d17e`

`validate_runtime_release_lineage` (`src/release/provenance.py:104`) uses a
**strict allowlist**: any file between runtime and release commit not in
`RELEASE_ONLY_DIFF_ALLOWLIST` invalidates the gate evidence. `docs/` is not on it.

Verified empirically on a scratch branch — a docs-only commit produced:

```
lineage ok: False
ERROR: Non-evidence changes between runtime 180b15e and release ca359ed
       invalidate gate evidence: ['docs/handoff/warm-start-review/...']
```

Same failure mode as "Production freeze lineage rejected" at launch. This
folder is therefore left **uncommitted** on `main`. Commit it with the Task 1
**runtime** commit (runtime commits may touch any path; only the runtime→release
diff is restricted), or keep it out of git.

## Hard constraints

- Never weaken or delete a gate to make it pass. A FAIL is a finding.
- Any edit under `src/`, `configs/`, `scripts/` invalidates the release: a new
  Kaggle dual-T4 gate + fresh freeze tuple is mandatory before A100 (`04`).
- Do not dispatch `modal run` without explicit user spend authorization.
- Private queries: raw question text only. Never into splits, qrels, mining, or
  any tuning path.
