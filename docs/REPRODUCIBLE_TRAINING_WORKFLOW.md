# Reproducible Training Workflow

Reviewed 2026-09-17. **The current uncommitted repair candidate is not a qualified release.** Read [HISTORY_FIXES.md](HISTORY_FIXES.md) before starting this sequence. Operational commands and supervision belong in the [A100 launch guide](README_A100_LAUNCH.md).

## 1. Finish and verify the candidate locally

Resolve the blockers in `fix.md`, preserve the canonical data and evaluation boundaries, and run:

```bash
.venv/bin/python scripts/verify_prepush.py
.venv/bin/python scripts/generate_notebooks.py --check-drift
.venv/bin/python scripts/check_notebook_parity.py
.venv/bin/python scripts/check_no_fallbacks.py
.venv/bin/python scripts/audit_parameters.py --check-only
```

The prepush gate covers compilation, regression suites, notebook checks, parameter budget, fallback policy, offline smoke, and working-tree hygiene. A direct pytest run alone is not the entire prepush gate.

Review all changed and untracked files. Commit/push only with authorization. Do not discard or stash repairs merely to launch an older release. Generated notebooks must come from `scripts/generate_notebooks.py`, not hand edits.

## 2. Verify CI on the runtime commit

Call the finalized runtime commit `R`. Check CI on that exact SHA, not just the newest green badge. Behavioral CI and strict release validation serve different purposes: a new runtime can pass behavioral CI while strict validation correctly rejects evidence belonging to an older runtime.

## 3. Run Kaggle dual-T4 smoke for R

This is the sole pre-A100 hardware smoke gate; Kaggle is not a FULL training backend.

- Notebook: `notebooks/kaggle_t4x2_smoke.ipynb` generated for the candidate.
- Canonical dataset: `phucdangg/legalir-task1-clean-data`.
- Accelerator: **two Tesla T4 GPUs**.
- Verify distinct dense/reranker CUDA device placement, real forward/backward updates, finite loss, positive weight delta, and adapter save/reload.
- Retrieve the genuine report; verify runtime SHA, dataset/config/profile identities, verdict, and report digest.
- Do not re-label an older receipt or extrapolate three update steps into a FULL runtime/quality claim.

Smoke allocation/publication still requires the appropriate user authorization. No smoke was launched during this review.

## 4. Create and verify the evidence-bearing release

Bind the genuine report and freeze to `R`. The evidence-bearing release commit `S` may differ from `R` only as allowed by the repository's runtime-to-release lineage validator. Do not bypass that validator, invent an evidence-generation command, or modify report fields to force acceptance.

On clean checkout of `S`:

```bash
.venv/bin/python scripts/verify_release_approval.py --repo-root .
git status --short
```

Verify CI on **S** as well. Record full SHAs and the exact CI URL. Any further runtime edit invalidates reuse of the old runtime receipt and restarts the relevant qualification steps.

### What the existing evidence actually covers

| Item | Reviewed identity |
|---|---|
| Committed runtime R | `373e8791917915da36864b7eb9b2f457493b4a0e` |
| Committed release S | `d39792836482f29bd6d5e691235690c738cdde3d` |
| Freeze | `artifacts/task1/freeze/production_freeze.json` |
| Smoke report | `artifacts/task1/gates/kaggle_t4x2_report.json` |
| Exact-release CI | https://github.com/silent9669/LegalIR/actions/runs/35231283958 |

The report records three optimizer steps in 27.05 seconds on dual T4s. This is the recorded short training interval, not total notebook runtime. These identities do **not** cover the uncommitted repairs reviewed in `fix.md`.

## 5. Qualify A100 resources and completion forecast

After release correctness checks and separate budget approval, measure a bounded real-model workload on the selected Modal or Colab profile. This is not permission for an automatic FULL launch.

Record cold setup/index time, host CPU/RAM, GPU/VRAM, actual batch size, optimizer throughput, end-to-end held-out queries/s, mining time, output delivery time, and immutable model identities. Project all five folds, document-disjoint training/evaluation, final training, inference, validation, and persistence. Count setup/precomputation rather than hiding it outside the cold-run timer.

Use the measured upper estimate plus reserve to decide whether the target fits. If it exceeds the approved window, stop and optimize; do not silently omit folds, shrink retrieval depth, or increase spending. Backend qualification is not automatically transferable to different Colab RAM/CPU allocations.

## 6. Supervise one approved FULL attempt and verify delivery

Use the [launch guide](README_A100_LAUNCH.md), select `S` explicitly, and record backend, attempt/session ID, start UTC, output path, approved ceiling, and stop procedure. Do not auto-retry or launch both providers concurrently.

Success requires more than process exit 0:

- Complete five-fold and document-disjoint evaluation with exact expected query coverage.
- Dedicated final model, compatible tokenizer/base-model identity, and successful final reload.
- Valid public submission with complete unique query coverage and valid corpus document IDs.
- Durable weights, effective configuration, evaluation reports, checksums, manifests, and truthful delivery receipts.
- Verified provider termination and no hidden active retry.

Later inference may reuse a validated final adapter. That is distinct from full training resume and must not contaminate held-out evaluation. Publication and competition submission are separate outward-facing actions requiring consent.
