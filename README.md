# LegalIR Task 1

Vietnamese legal document retrieval for the UIT Data Science Challenge 2026. The pipeline combines legal BM25, PyVi BM25, DEk21 dense retrieval, exact matching, fold-local question memory, and a BGE LoRA cross-encoder.

## Readiness — reviewed 2026-09-17

**Do not launch FULL yet.** The local repair candidate is uncommitted on top of release `d39792836482f29bd6d5e691235690c738cdde3d`. Existing release evidence binds runtime `373e8791917915da36864b7eb9b2f457493b4a0e`, not the uncommitted changes. See [fix.md](fix.md) for the review, remaining work, acceptance tests, and release sequence.

- **Kaggle:** bounded smoke tests only. The existing dual-T4 receipt records three optimizer steps in 27.05 seconds; it is not an A100 performance or final-quality benchmark.
- **Modal / Google Colab:** real-model qualification and FULL training, only with separate execution approval. Qualify each backend before claiming it meets the time budget. Do not launch both concurrently as an automatic retry strategy.
- **Targets, not demonstrated results:** >96% mean Recall@5 and <5-hour cold end-to-end delivery of a reloadable final model plus validated submission.
- **Parameter audit:** 702,754,049 parameters across the dense encoder and reranker, below the competition's 4B ceiling. Re-run the audit if the models change.
- A five-hour timeout can stop an unfinished run; it is neither a completion guarantee nor a total spending cap.

## Essential documents

| Document | Purpose |
|---|---|
| [fix.md](fix.md) | Current repair review, blockers, verification results, and next actions |
| [Architecture](docs/ARCHITECTURE.md) | Components, data boundaries, training/evaluation, and configuration sources |
| [Release workflow](docs/REPRODUCIBLE_TRAINING_WORKFLOW.md) | Local tests → CI → Kaggle smoke → evidence-bearing release → A100 qualification |
| [A100 launch guide](docs/README_A100_LAUNCH.md) | Modal/Colab supervision, consent, timeouts, recovery, and stop procedures |
| [Historical timing evidence](docs/A100_SCALE_DOWN_AND_OPTIMIZATION_REPORT.md) | Old A100 measurements and limits of the proposed optimizations |

Launch instructions live in the launch guide only. Historical reports and architecture descriptions are not launch approval.

## Pipeline

```text
Canonical Task 1 corpus and queries
  → lexical / dense / exact retrieval + fold-local question memory
  → candidate fusion and query-aware evidence selection
  → fold-specific BGE LoRA training and batched held-out inference
  → five-fold OOF + document-disjoint evaluation and fusion evaluation
  → dedicated final training on all training queries
  → final-model reload, public inference, submission validation
  → durable artifacts and verified delivery receipts
```

Static mining candidates can be reused across folds without reusing fold labels. This reduces some repeated searches; it does not eliminate every retrieval or index load. Batched inference and parallel PyVi indexing are implemented optimization mechanisms, not measured speedup guarantees.

The query-balanced sampler prioritizes an interleaved positive/negative pair per eligible query. Query coverage is not full exposure to every available pair, and neither guarantees model quality. The top-five oracle measures a ceiling, not achieved Recall@5.

## Data and evaluation rules

- Use only canonical Task 1 training queries, qrels, and legal corpus. Public queries are for inference only.
- No external legal corpus, Task 2 data, crawling, synthetic LLM examples, or external inference APIs.
- Preserve all five folds, document-disjoint evaluation, and fold-local supervised memory/mining.
- Never initialize honest held-out evaluation from a final adapter trained on all held-out labels.
- Parameter counts, dataset identities, model revisions, split identities, and effective configuration must be recorded and checked.

## Local verification

Use the repository virtual environment; no GPU allocation is needed for these commands:

```bash
.venv/bin/python scripts/verify_prepush.py
.venv/bin/python scripts/generate_notebooks.py --check-drift
.venv/bin/python scripts/check_notebook_parity.py
.venv/bin/python scripts/audit_parameters.py --check-only
.venv/bin/python scripts/verify_release_approval.py --repo-root .
git status --short
```

A strict verifier PASS on an unchanged HEAD does not certify uncommitted runtime edits. Publish the reviewed candidate through the release workflow before either remote backend can run those fixes. Do not bypass provenance validation or fabricate replacement smoke reports.

## Repository map

- `src/`: retrieval, evidence, training, ranking, evaluation, pipeline, and release contracts.
- `configs/`: algorithm, experiment, and backend runtime configuration.
- `scripts/`: local verification, notebook generation, hardware gates, and backend launchers.
- `notebooks/`: generated Kaggle smoke and Colab A100 notebooks; do not hand-edit.
- `tests/`: unit, contracts, dataset, notebook, parity, leakage, memory, integration, and release suites.
- `kaggle_dataset/`: canonical local dataset layout; heavy data is not a Git deliverable.
- `artifacts/task1/`: release evidence and run outputs; distinguish historical evidence from the active candidate.

Persisted stage artifacts are not automatic remote resume. Modal creates a new UUID attempt directory on every invocation; Colab recovery is best effort and does not survive VM loss by itself. Exact optimizer-state resume is not provided.
