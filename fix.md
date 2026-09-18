# Modal and Colab Readiness — Required Fixes

**Review date:** 2026-09-17  
**Verdict:** **NOT READY for the intended reliable FULL cold run.** The repairs improve correctness and pass the local regression suites, but they are not released, several reuse/provenance paths remain incomplete, and A100 completion/resource measurements are absent.

This is a local implementation handoff at the repository root, alongside `README.md`. It is not permission to launch, spend, upload, submit, commit, or push. This review changed documentation only; the source/test repairs described below were already present from another implementation session.

**Expanded acceptance objective:** establish **both cold end-to-end completion in <18,000 seconds and official pooled five-fold OOF Recall@5 >0.96 for the same frozen candidate and measured FULL attempt**, with complete document-disjoint evaluation, final-model reload and validated submission delivery. Sections 8–13 specify the target architecture, quality-development path, resource model, end-to-end tests and evidence contract. They are proposed implementation requirements, not a claim that either target has been met. Public-test Recall@5 cannot be established locally without public-test labels.

## Recommended execution strategy

**Plan refinement: 2026-09-18.** The code-review snapshot below remains dated 2026-09-17; this refinement is not a new source audit or benchmark.

Optimize the **whole delivered run**, not just optimizer steps. The historical evaluation bottleneck alone extrapolates to 258.6 minutes for five folds, while PyVi took 44.14 minutes. These two costs already exceed five hours before training, dense indexing, document-disjoint evaluation or delivery. The first implementation priority is therefore shared CPU/retrieval work and batched evaluation—not fewer folds or fewer learning exposures.

| Priority | Concrete implementation outcome | Evidence needed before promotion |
|---|---|---|
| P0: correct baseline | F2–F4/F7 resolved; fixed split/policy; effective configuration logged | Fresh local correctness gates; no stale/selection-biased acceptance |
| P1: remove repeated work | One build per immutable index; static retrieval/query encodings reused within the attempt; fold-local labels/memory applied separately | Call counts, cache hit/miss parity, exclusive stage timings and bounded RAM |
| P2: accelerate evaluation | Explicit inference batch independent of training; multi-query packing; tokenization/padding measured | Representative end-to-end queries/s, unchanged candidate/evidence membership and validated rankings |
| P3: accelerate training safely | Measure B8/G2 against B16/G1 at the same effective batch16, examples, updates and learning-rate schedule | Stable memory, measured seconds/update, learning-curve and quality comparison; not assumed bitwise equivalent |
| P4: improve recall per unit compute | Better evidence placement and sampled negatives first; greater rerank depth only when pruning losses justify it | Paired inner-validation recall and measured added pair/token cost |
| P5: prove delivery | Five folds + disjoint + dedicated final model + fresh reload + full submission | Same-attempt <18,000 seconds and >0.96 pooled official OOF recall, with valid receipts |

Keep the existing dense/reranker model families, LoRA rank and canonical chunking for the first candidate. Do not add another encoder, an ensemble, dense fine-tuning, online teacher inference or a broad resume subsystem before this simpler path is measured. Keep the 30-minute contingency for uncertainty; it is not a budget for extra experiments during confirmation.

**What this plan can ensure:** explicit tests prevent accepting a fast but incomplete run or a high score from an invalid evaluation. **What remains empirical:** whether the chosen architecture actually achieves both speed and recall targets. Until the real joint gate passes, readiness remains NOT READY.

## 1. Snapshot and evidence

| Item | Reviewed state |
|---|---|
| Branch / HEAD | `main` / `d39792836482f29bd6d5e691235690c738cdde3d` |
| Frozen runtime | `373e8791917915da36864b7eb9b2f457493b4a0e` |
| Working tree | Dirty: candidate source/test fixes plus documentation changes |
| Fresh offline regression | **486 passed, 2 skipped, 5 warnings, 212.68 seconds; exit 0** |
| Shell syntax | Modal and Colab wrappers passed `bash -n` |
| Generated notebook drift | PASS |
| GPU-gate forbidden-fallback scan | PASS |
| Strict release verifier | PASS for old committed release/runtime, **not for uncommitted fixes** |
| Direct local Colab bootstrap | PASS with expected HEAD while tree was dirty; confirms missing cleanliness safeguard |
| Prior exact-HEAD CI | [35231283958](https://github.com/silent9669/LegalIR/actions/runs/35231283958), for `d397928`, not candidate repairs |
| Existing Kaggle gate | Genuine dual-T4 PASS for `373e879`; three optimizer steps / 27.05 seconds |
| New candidate CI / GPU smoke / A100 qualification | Not performed in this review |

Regression command used:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
.venv/bin/python -m pytest -q \
  tests/unit tests/contracts tests/dataset tests/notebook tests/parity \
  tests/leakage tests/memory tests/integration tests/release
```

The five warnings are PEFT warnings in small training/integration fixtures: no config file found, vocabulary assumed unchanged. They were not test failures. The two skipped tests remain skipped; this result does not imply they were exercised.

A regression PASS is not a whole prepush PASS, a real-GPU qualification, or a performance benchmark. Live provider authentication, capacity, persistent storage, and publication permissions were not checked here.

## 2. What the repairs accomplished

| Area | Current assessment |
|---|---|
| Reranker reload revision | Improved: loader propagates explicit/manifest/registry revision; training records logical base model and revision. Real-GPU candidate save/reload still needs renewed smoke evidence. |
| Query embedding row identity | Improved: sidecar checks IDs, text, model/revision, row count and remaps reordered IDs. This fixes the original blind positional zip. |
| Expected training-query coverage | Improved for freshly mined FULL folds and document-disjoint jobs: missing expected queries, positive coverage <100%, and negative coverage <99% fail. |
| Fold reuse | Partial: validates several IDs/settings and some adapter integrity, but incomplete fingerprints/artifact requirements remain. |
| Final adapter reuse | Partial: revision/weights/training signals/counts checked; equal counts do not identify equal training data. |
| Document-disjoint reuse | Still weak: optional identity fields can permit stale reports to bypass current evaluation. |
| Modal detach | Added explicit `--detach`, with attached default and duplicate-flag rejection. Tests verify dispatch semantics, not live disconnect survival or a watchdog. |
| Modal resource sizing | Not repaired: no explicit CPU/RAM reservation; qualification remains absent. |
| Cross-attempt recovery | Not implemented: every Modal invocation creates a fresh UUID output directory. |

Code anchors for improvements:

- `src/ranking/reranker.py:214–286`: adapter revision resolution/loading.
- `src/training/train_reranker.py:156–218`: model/revision selection and recorded identity.
- `src/pipeline/predict.py:530–584,633`: reranker revision propagation.
- `src/pipeline/kaggle_train.py:191–283`: train-query cache sidecar validation/remapping.
- `src/pipeline/oof_runner.py:762–788,1191–1207`: FULL expected-query pair coverage.
- `tests/unit/test_reranker_revision.py`, `tests/unit/test_query_embedding_cache.py`, `tests/unit/test_fold_resume_identity.py`, `tests/contracts/test_modal_detach.py`: new test coverage included in the fresh run.

Line anchors refer to the reviewed candidate and will move after implementation.

## 3. Required work, in priority order

### F1 — Release the actual candidate, not the old HEAD

**Applies:** both backends, even with completely fresh output directories.

The existing freeze/report bind old runtime `373e879`. Local code changes do not alter Git HEAD. Modal's wrapper rejects a dirty tree; that is correct protection, not a guard to bypass. Colab can still select the old HEAD as described in F2.

**Action:** finish the necessary repairs, review the full diff, generate notebooks through the generator, run local verification, then obtain authorization to commit/push. Qualify the resulting runtime `R` through exact-SHA CI and genuine Kaggle dual-T4 smoke. Create an evidence-bearing release `S` with only allowed evidence lineage changes and verify strict approval plus CI on `S`.

**Acceptance:** clean tree; exact selected SHA; genuine report/dataset/config/profile hashes; valid runtime-to-release lineage; CI green on the relevant commits. Do not copy old reports into a new identity, fabricate receipts, relax checks, or stash away the fixes just to launch `d397928`.

### F2 — Reject dirty Colab launches before any allocation

**Files:** `scripts/colab/run_colab_cli.sh:243–273`; `scripts/colab/bootstrap.py`; relevant fixtures in `tests/contracts/test_shell_cli.py` and `tests/notebook/test_colab_bootstrap.py`.

**Confirmed path:** the wrapper takes `LEGALIR_COMMIT_SHA` or `git rev-parse HEAD`, runs bootstrap, then calls `colab new`. Bootstrap validates HEAD/evidence, not all working-tree edits. The generated notebook clones/checks out the selected commit. Thus local repairs may be present when preflight succeeds but absent from the remote runtime.

**Verification performed:** local, CPU-only bootstrap returned `Runtime SHA, freeze, algorithm config, and upstream gate reports match` while the source fixes were uncommitted. No Colab allocation was called.

**Minimal fix:** add a fail-closed working-tree check matching Modal before allocation/upload, including staged, unstaged, deleted and untracked runtime files. Keep ignored credentials out of Git and avoid printing their contents. Preserve exact-SHA and strict evidence validation. Update fake-repository test fixtures rather than weakening the new check to accommodate them.

**Tests:** each dirty case exits nonzero with zero fake `colab new/upload/exec` calls; clean pinned release reaches dispatch; SHA mismatch and stale evidence still fail before allocation; INT/TERM and bounded cleanup behavior remain intact. Use fake provider executables—no live GPU needed.

### F3 — Bind dense document embeddings to the immutable encoder actually used

**Files:** `src/retrieval/dense_macro.py:637–651,700–731`; `src/pipeline/kaggle_train.py:1362–1365,1421–1428`; model registry and new freeze evidence.

**Confirmed defects:**

- Dense `save()` omits `revision` from the index manifest.
- Dense `load()` accepts an explicit manifest revision such as legacy `main` when no revision argument is supplied.
- The pipeline loads an existing dense index without an explicit expected revision.
- The new query-cache sidecar compares against that runtime revision; matching `main` to `main` is internally consistent but does not prove immutable weights.
- The existing freeze also lists dense revision `main`. Do not infer from the registry pin that every cached index was built with it.

**Failure scenario:** a reused/prebuilt index can contain document vectors from one encoder snapshot while later query embeddings use another, or an unknown index revision gets implicitly attributed to today's registry revision. This affects retrieval correctness, not just metadata aesthetics. Exposure depends on whether the run loads such cached indexes; a fully fresh build does not establish the safety of the load path.

**Minimal fix:** persist the resolved immutable model revision with document vectors, model ID, relevant preprocessing settings, ordered chunk/corpus identity, dimension and integrity checks. Validate these against the selected release before use. Reject/rebuild legacy caches with missing, mutable, or mismatched identities. Ensure query-cache identity uses the same validated encoder contract.

**Important:** passing an expected revision to the query encoder is not sufficient if existing document embeddings were produced by unknown weights. Never relabel old vectors as newly pinned. Rebuild unless immutable provenance is independently established.

**Tests:** save/load round trip retains actual pin; `main`, missing revision, wrong model/revision, changed corpus/preprocessing, row/dimension mismatch invalidate reuse; correctly pinned matching artifacts load; reordered query rows still map correctly. Query-cache dimension is recorded but not validated by the reviewed loader—add shape/finite-value checks and rejection tests while closing this contract.

### F4 — Disable unsafe completed-stage reuse, or finish its contract

**Scope:** reused output directories and restored stage artifacts. **Not a deterministic failure of fresh Modal UUID attempts.** Do not claim an automatic cross-attempt resume exists.

For the fastest safe first-launch path, **disable completed-stage reuse in FULL until its contract is complete**. This avoids building a broad resume subsystem just to launch once. If reliable stage reuse is required now, implement the focused checks below. A future explicit resume feature should be a separate change, not bundled into this review's repair.

#### F4a. Random-fold reuse

**Source:** `src/pipeline/oof_runner.py:389–498`.

- Only completion marker, predictions and metrics are required.
- Candidate/features files are optional (`448–462`), yet downstream candidate metrics and fusion need them.
- Stored split SHA may be missing (`420–424`).
- Adapter checksum is checked only if recorded; base-model comparison does not require/compare base revision (`475–489`).
- Selected query IDs/settings do not identify query text, qrels, corpus/index, training pairs, or all effective training settings.

**Failure scenarios:** delete a feature/candidate file and reuse still succeeds; change data contents while preserving query IDs and old predictions can be reused; change the model pin and old stage metrics can remain accepted.

**Fix/tests:** require a versioned complete identity and every downstream-required artifact. Bind input data, split membership, model/tokenizer pin, effective training/inference settings, and adapter/artifact checksums. Validate unique and exact expected prediction IDs before any dictionary conversion; validate candidate/feature coverage. Missing/old fields mean cache miss, not compatibility approval. Test one changed identity at a time and each missing/corrupt file; positive-control matching artifacts should reuse without invoking training.

#### F4b. Document-disjoint reuse

**Source:** `src/pipeline/oof_runner.py:1048–1074`.

The fast path accepts marker + report with optional split SHA, model and smoke fields. It returns before loading/verifying the current document-disjoint split and omits required train/validation IDs, candidate/rerank depths, precision, model revision and adapter integrity.

**Fix/tests:** compute and verify current split identity/isolation before reuse; require complete current metadata and report/artifact validation comparable to ordinary folds. Missing stored SHA must reject reuse when a SHA is expected. Test legacy marker, changed split/data, FULL-vs-smoke mismatch, changed inference settings, missing adapter and truncated evaluation coverage. A positive-control reuse must preserve exactly the current held-out population and metrics.

#### F4c. Final adapter reuse

**Source:** `src/pipeline/kaggle_train.py:1630–1719`, especially `1689–1708`.

Current checks compare query and pair **counts**, not contents. A pair-count read failure is swallowed. Same-count changed labels, query text, pairs or training configuration can silently reuse an old final adapter.

**Fix/tests:** bind current query/text/qrels/corpus identity, canonical pair-content digest, effective training configuration and immutable model identity to the final training report; require a valid checksum. Missing metadata/read failures must be a cache miss or explicit integrity error, never permission to reuse. Test same-count changes to query IDs, text, labels, pair evidence and LoRA/training settings, plus corrupted pair files and checksum absence. Verify valid exact-match reuse as a positive control.

**Common rule:** write the completion marker last, after validating required artifacts. Checksums establish artifact integrity; matching input fingerprints establish applicability. Neither substitutes for the other.

### F5 — Qualify resource and time feasibility before FULL spending

**Source:** `scripts/modal/run_modal_a100.py`, especially the A100 function decorator; Colab runtime profile and launcher.

The Modal function requests A100 but no explicit CPU/RAM reservation. Parallel PyVi indexing, multiple indexes, tables, workers and repeated model loads can dominate host resources. Bounded evidence LRU does not cap total RAM. No measured cold-run throughput proves the optimization claims.

**Action:** first prepare a local benchmark/measurement protocol and tests. After separate authorization, run a bounded real-model A100 qualification, record actual host/GPU resources and peak memory, and select resource requests from the measurements—not arbitrary large reservations. Keep model/data/precision/evaluation semantics fixed for runtime-equivalence comparisons. Verify the configured worker count fits the available resources.

Measure cold acquisition/setup/indexing, static retrieval, supervised mining/evidence assembly, effective optimizer steps/sec, held-out queries/sec including retrieval/evidence/tokenization/scattering, final reload/inference, validation and durable delivery. Record cache hit/miss status so warm-cache numbers cannot be labeled cold results.

Project **all five folds + actual document-disjoint work + final model**. Derive doc-disjoint query/update counts from the actual split and coverage audit; do not hardcode the earlier illustrative 700 updates.

**Acceptance for the intended <5h run:** a measured upper forecast fits the work budget with reserve, memory headroom exists, and the delivery/stop paths are tested. Otherwise retain no-go, optimize the measured bottleneck, and request approval for any tradeoff. Do not omit evaluation or extend the timeout silently.

### F6 — Establish durable supervision and truthful recovery

The added Modal detach option is useful but not a complete supervisor. Attached disconnect remains hazardous; detached execution can outlive the local client and keep spending. Colab cleanup is bounded but cannot run after the orchestrator disappears.

**Before an approved run:** record app/session ID and attempt path, validate installed CLI stop syntax, set an approved progress/deadline/spend policy, arrange supervision that survives the expected client failure, and verify provider termination independently. Do not assume tmux prevents laptop sleep or network loss. No automatic relaunch.

**Acceptance tests:** local fake CLI coverage for attached/default, explicit detach, duplicate rejection, allocation failure, upload failure, execution timeout, cleanup precedence, failed stop and incomplete recovery. In an explicitly approved bounded provider exercise, demonstrate disconnect/lifecycle behavior and durable retrieval of required outputs. Preserve original failure status when delivery also fails.

A recovered adapter permits later inference only after provenance/reload validation. Stage reuse is not exact optimizer resume, and neither is automatically exposed by the current Modal launcher. Keep that limitation explicit rather than adding speculative resume infrastructure now.

### F7 — Remove validation-driven selection from the confirmatory quality claim

**Additional inspection for the expanded quality goal:** `src/ranking/train_fusion.py:113–174` passes outer validation labels into `eval_set` with early stopping and evaluates on that same fold. `src/ranking/fusion.py:296–317` actually installs the LightGBM early-stopping callback. The fusion trainer also selects learned versus RRF using aggregate OOF quality. Therefore, current winner OOF results are selection-influenced development measurements, not an untouched confirmatory score for the selected pipeline.

**Minimal recommended path:** predeclare one fixed fusion policy/configuration before the confirmation run; do not fit, early-stop, or choose it using the outer held-out labels. Keep those labels available only to the scorer after predictions are finalized. Publish alternative fusion scores as diagnostics, never substitute the best one into the acceptance report after looking at them.

**If learned fusion is retained:** use inner validation wholly within each outer training partition, including supervised feature generation. Reusing globally generated OOF features is not automatically nested-safe: feature-generating adapters for other folds may have seen the current outer validation labels. Audit the full dependency graph, not just the final LightGBM row mask. Fit the final production fusion only after confirmation using the predeclared training procedure, and include all extra inner-training cost in the cold-run budget. This is more expensive than the fixed-policy option and is not the recommended first-launch path.

**Tests:** replace outer held-out qrels while keeping training inputs fixed; model fitting, early-stopping decisions, candidate scores and predictions must not change—only scoring may change. Verify complete expected query coverage independently of the feature dataframe (current evaluation builds its scoring population from returned predictions). Test that a failed preselected method cannot be replaced by a better observed alternative while retaining a confirmatory PASS.

## 4. Proposed budget — not a benchmark

| Stage | Target minutes |
|---|---:|
| Cold startup/acquisition/indexes | 45 |
| Shared static retrieval | 25 |
| Supervised assembly | 20 |
| All required training jobs | 90 |
| Held-out evaluation, including document-disjoint | 55 |
| Fusion/reporting | 5 |
| Final reload/public inference | 20 |
| Validation/durable delivery | 10 |
| **Nominal work** | **270** |
| **Contingency** | **30** |

The outer cold-run clock includes setup and delivery; provider function/notebook timeout scope is narrower and must be accounted for separately. Historical PyVi alone took 44.14 minutes. Historical five-fold evaluation extrapolates to 258.6 minutes. Achieving 7,000 held-out queries in 55 minutes requires about 2.12 queries/s (4.70× the old 0.451); document-disjoint adds more. These figures explain why green tests alone cannot establish five-hour feasibility.

Quality target: >96% honest mean Recall@5. Historical one-fold Recall@5 was 82.54%, candidate recall@150 98.89%. A high candidate pool or 100% corpus capacity ceiling is not a ranking score. Report actual candidate top-five oracle, all fold scores, document-disjoint score, coverage and official Precision@5/Recall@5. Do not report the target as achieved.

## 5. Suggested next implementation sequence

1. **Close Colab preallocation cleanliness and dense-index provenance.** Verify with fake CLI/cache fixtures; no GPU.
2. **Choose minimal FULL reuse policy:** disable unsafe reuse first, or implement F4 identities/completeness with adversarial tests. Do not enable cross-attempt recovery accidentally.
3. **Close F7 and add timing/quality diagnostics.** Fix the evaluation procedure before selecting a quality winner; implement P1–P3 with parity and failure tests. Preserve existing coverage/data/leakage constraints.
4. **Run focused tests, then the full local gate.** Check notebook drift, parameter budget and fallback policy. Release the candidate with authorization: runtime CI → genuine Kaggle dual-T4 smoke → evidence-bearing release → strict verifier and release CI.
5. **Obtain a bounded A100 qualification/development budget.** Measure resources/throughput and supervision/delivery; choose one provider. Run only the bounded quality experiments justified in section10. Freeze the selected procedure and renew applicable release evidence after changes. No automatic promotion to FULL.
6. **Only if qualified and approved:** one supervised FULL attempt using the operational guide. The independent verifier must check both targets on that attempt, including final delivery and shutdown; qualification alone is not acceptance.

Do not implement a broad architecture rewrite before measuring these remaining bottlenecks. Keep equivalent performance work separate from training/ranking changes. Larger batches, new losses, retrieval-depth reductions and alternate models require their own quality/resource evidence.

## 6. Final acceptance checklist

- [ ] Actual candidate released, clean and exact-SHA validated; old receipts not reused under new identity.
- [ ] Both wrappers fail before allocation for dirty/mismatched/stale inputs.
- [ ] Dense document/query embeddings share verified immutable provenance; legacy caches reject/rebuild.
- [ ] FULL stage reuse disabled or all F4 adversarial/positive-control tests pass.
- [ ] Fresh regression, complete local gate, exact-SHA CI, and genuine dual-T4 smoke pass.
- [ ] Bounded A100 resource/throughput evidence supports the budget on the chosen backend.
- [ ] Operator approves one attempt, ceilings, visibility/publication and stop procedure.
- [ ] Five-fold + document-disjoint results cover all required held-out queries without leakage.
- [ ] Final adapter/tokenizer/base identity reloads; submission covers expected public IDs and valid documents.
- [ ] Required artifacts/checksums/receipts are durable; provider shutdown verified.
- [ ] F7 fixed: outer labels cannot select/early-stop fusion or replace the predeclared winner; prior development exposure disclosed.
- [ ] Runtime-only changes preserve candidate/evidence membership and pass ranking/scorer parity; training-semantic changes have separate quality evidence.
- [ ] Same-attempt independent verification reports **cold elapsed <18,000 seconds AND pooled official OOF Recall@5 >0.96**, without rounding across thresholds.
- [ ] Compare against the repaired baseline on the same development split/protocol; do not treat old single-fold 82.54% as an apples-to-apples current baseline.
- [ ] Actual runtime and quality reported honestly, even if targets are missed; failed targets retain FAIL/INCOMPLETE rather than a readiness PASS.

## 7. Documentation consolidation performed

- `README.md`: short entrypoint and truthful candidate/release distinction.
- `docs/ARCHITECTURE.md`: actual components and configuration; removed conflicting LoRA values, calibration/utilization guarantees and automatic-resume claims.
- `docs/REPRODUCIBLE_TRAINING_WORKFLOW.md`: one consistent local/CI/smoke/release/qualification sequence.
- `docs/README_A100_LAUNCH.md`: single operational runbook; removed direct-dispatch bypass examples and deleted-prompt reference.
- `docs/A100_SCALE_DOWN_AND_OPTIMIZATION_REPORT.md`: retained measured history; removed disputed elapsed time and unsupported speedups.
- Removed redundant `docs/A100_REARCHITECTURE_SPEC.md`; its useful constraints and proposed budget are preserved here and in the architecture/history references.

The already-deleted root launch guide and review prompt were pre-existing changes, not deletions performed by this review. No training code, tests, configuration, notebook, gate report or freeze was edited by this review. No GPU was allocated, no artifacts uploaded, and no commit or push was made.

## 8. Joint acceptance contract: what counts as established

### 8.1 Runtime gate T

Start a supervisor monotonic timer **before the first launch/setup action for the measured attempt**, including image/dependency preparation, allocation/wait, canonical data/model acquisition, and all attempt-specific preprocessing. Record matching UTC timestamps to correlate remote events. Stop only after:

1. All five OOF folds and document-disjoint evaluation finish.
2. The dedicated final model is trained and saved.
3. A fresh process reloads the saved final model/tokenizer with the pinned base revision.
4. That reloaded pipeline generates the full public submission.
5. Submission/manifest/checksum validation passes and required artifacts are durably delivered to the approved destination.
6. An independent delivery check confirms the required model and submission objects exist and match recorded checksums.

**PASS_T = complete required workload AND measured cold elapsed seconds <18,000.** Exactly 18,000 seconds fails the strict “under five hours” objective. Queue/network/setup are counted, not silently removed. Provider stop must subsequently be confirmed; record its latency and cost separately. The whole operation's spending cap remains independent from the runtime gate.

Cold means no reusable attempt-specific indexes, query vectors, candidate/evidence caches, trained adapters or stage outputs from an earlier attempt. Record provider-level image/model caches honestly; a run relying on a prebuilt image is a profile-specific cached-environment result unless its preparation is included in the cold clock. Development and qualification attempts are separate costs: report them, but do not present their sum as the runtime of one attempt or hide reused work from the cold test.

One measured PASS establishes an observed result for that hardware/configuration, not a guarantee against future network delays or eviction. Modal and Colab require separate measured receipts to claim both providers meet the objective; a successful Modal run does not certify Colab.

### 8.2 Quality gate Q

For all expected labeled training queries, gather exactly one prediction produced by a model trained without that query's outer validation labels. Require exactly five outer folds and their full expected validation ID sets before aggregation. Use the official scorer's per-query recall semantics (`Scoring-Program-Task-LegalIR/scoring.py:14–34`):

`R5 = mean_q(|gold(q) ∩ predicted_top5(q)| / |gold(q)|)`

The primary population is the concatenated complete OOF query set, not an average over whichever predictions happened to be returned. Require exact ID equality, no duplicates and valid unique document IDs before scoring. Report fold mean/std/min and each fold's sample count as well; do not let unequal fold sizes change the official query-level metric.

**PASS_Q = pooled official OOF Recall@5 >0.96 AND coverage/leakage/protocol checks pass.** Exactly 0.96 fails the “above 96%” objective. Report official Precision@5 and per-query scores, plus a deterministic query-bootstrap 95% interval as uncertainty information; a point estimate above 0.96 does not imply its lower confidence bound exceeds 0.96.

Document-disjoint evaluation is mandatory and must use the predeclared inference policy without tuning on its held-out labels. Report its Recall@5, Precision@5, query count, oracle and random-vs-disjoint gap separately. Its score is not blended into the OOF score to cross the threshold. The primary target here is **OOF >96%**; claiming **document-disjoint >96%** requires that result to independently exceed 0.96 too. Unlabeled public predictions cannot establish public/private leaderboard recall.

Previously inspected folds are development evidence. Freeze the procedure before the confirmation run and disclose any earlier tuning on these splits; rerunning the same known folds does not make them a new untouched holdout. If a fully independent generalization claim is required, reserve a genuinely untouched labeled partition before development or use a correctly nested evaluation. Do not claim that repeating CV erases selection bias.

### 8.3 Joint gate J

`PASS_J = PASS_T AND PASS_Q AND doc_disjoint_complete AND delivery_valid AND shutdown_confirmed AND provenance_valid`

All fields must refer to the **same runtime/release, dataset, effective configuration, model identities and FULL attempt**. No combining the fastest run with the best-scoring run. Missing metrics, failed folds, missing model artifacts, stale receipts or an unfinished attempt are FAIL/INCOMPLETE, never an assumed PASS.

## 9. Complete target architecture

This is the proposed implementation contract. Existing functions are integration points, not evidence that every optimization below already exists.

```text
Preflight + immutable run identity + cold supervisor timer
  ↓
Canonical data validation / frozen splits / resource inventory
  ↓
Once-per-run label-independent preparation
  ├─ legal BM25 + bounded parallel PyVi postings
  ├─ pinned DEk21 document vectors + exact FAISS index
  ├─ train/public query vectors
  └─ native-depth static branch candidates + lexical evidence/token caches
  ↓
Sequential isolated fold jobs 0…4
  train-only memory + pair mining → fresh fold adapter/optimizer
  → batched held-out retrieval/evidence/reranking → fixed fusion/top-5
  → immutable predictions/features/metrics + complete coverage checks
  ↓
Document-disjoint job, with its own isolated training state and evaluation
  ↓
Official pooled OOF scoring + separate document-disjoint diagnostics
  ↓
Dedicated final adapter on all permitted labeled training queries
  ↓
Save → fresh-process reload → public inference → submission validation
  ↓
Durable delivery → independent receipt/checksum checks → supervisor verdict
```

### A. One effective configuration and immutable identity

Resolve configuration once at startup and serialize the actual values used by every stage. Current source has materially different defaults: algorithm YAML specifies candidate/rerank depths 100/30, while FULL orchestration explicitly uses **150/50** (`kaggle_train.py:1489–1490,1856–1857`). Training experiment is BCE, LoRA r8/alpha16, batch8/accumulation2, maximum length384, nominal250 steps; runtime has additional settings and coverage can raise the effective updates. Never budget from YAML alone.

Start quality development from the actual FULL 150/50 policy, not a silent reduction to 100/30. Record candidate depth, neural-scored depth, evidence count, selector policy, model/tokenizer pins, training objective, sample schedule, effective updates and inference batch/window size separately. Persist a hash of this resolved contract in every receipt.

### B. Once-per-run immutable preparation

Keep existing canonical chunking and model families for the first candidate. Build each lexical index once, dense vectors/index once, and label-independent query encodings once. Reuse resident retrievers within the attempt rather than reloading them for every mining/evaluation call where equivalent behavior is verified.

Cache native retrieval results keyed by corpus/index identity, query ID/text, branch/model identity and exact search parameters/depth. A deeper search followed by slicing is allowed only when tested equivalent to native-depth behavior for that branch; do not assume aggregation/scoring is prefix-stable. Do not share fold-specific memory hits, gold-injected candidates, hard-negative label filters or adapters across fold boundaries.

Use bounded disk-backed evidence/token caches with measured hit rates, size limits and hit/miss parity. Include query text, document/chunk content, evidence policy, tokenizer revision, special-token/truncation policy and length in keys. Only label-independent evidence can share across folds; candidate provenance or fold-dependent features belong in separate keyed records. If the label-independent cache contract cannot be proved, recompute that portion rather than risk leakage.

### C. Predictable host/GPU lifecycle

Use one supervised A100 job; sequence neural training jobs rather than launching concurrent folds that duplicate base weights/indexes in memory. Keep large immutable CPU/index data resident where resource measurements permit. Bound PyVi workers to the qualified host allocation and avoid process duplication of the entire corpus.

Pin and download base snapshots once. Each fold starts from pristine base weights with a newly initialized adapter, optimizer, scheduler and random state. Reusing an in-memory base object is optional only after tests prove previous adapters/merged weights/optimizer state cannot survive reset; initially recreate the model from the local pinned snapshot. Reuse tokenization/index work, not learned fold weights. Unload/release dense GPU state before reranker training when all needed query/document encodings are prepared; preserve exact retrieval behavior.

### D. Coverage plus useful supervised learning

Preserve all expected training queries and the fold-local negative blacklist. Retain multiple gold documents in the pair pool; report which positives/negative difficulty strata are actually sampled across updates. One interleaved positive and negative per query is a coverage floor, not demonstrated adequate training.

Use the current corrected BCE schedule as the baseline. Quality development may test deterministic rotation among each query's positives and hard negatives, then one query-grouped pairwise ranking objective if ranking diagnostics justify it. Both are training-semantic changes: evaluate on training-only inner validation, account for extra updates and do not label them runtime-equivalent. Never remove a query with difficult/missing evidence to improve the metric or time.

**Training-speed qualification, without cutting exposure:**

- On one GPU, effective batch is microbatch × accumulation. Compare B8/G2 against B16/G1, both effective batch16, with identical ordered examples, update count, loss normalization, warmup and schedule. Handle partial final accumulation groups consistently. Different microbatch execution can change dropout/numerics, so validate learning and ranking rather than promise exact weight equality.
- Keep BF16 and length384 fixed initially. Measure forward/backward, data loading/tokenization, optimizer, checkpoint I/O and peak memory separately. Benchmark after warmup, but include warmup in whole-run timing. Record actual sampled examples/tokens per second as well as updates per second.
- Test gradient checkpointing off only if the measured host/VRAM envelope permits the entire representative workload with safety headroom. It trades memory for avoided recomputation; it is not automatically faster enough or safe on A100-40GB. Change one factor at a time; reject OOM/nonfinite/quality-regressing variants.
- Cache tokenized inputs only where truncation/evidence semantics match. Use bounded prefetch and workers selected from host measurements; an idle GPU may reflect a CPU input bottleneck, not an undersized neural batch.
- Do not select the shortest schedule solely from training loss or the coverage floor. During approved development, inspect a small predeclared set of exposure checkpoints on training-only inner validation. Count checkpoint evaluation time. Freeze an exposure-based schedule before confirmation; scale final training to all7,000 queries rather than copying the fold update count unchanged. No outer-fold early stopping.

For negative quality, begin with the existing sampler and mining rather than adding another teacher. Audit easy versus confusable negatives actually consumed, blacklist known positives from permitted training labels according to the leakage-approved contract (never read outer held-out qrels to mine negatives), and ensure a query cannot receive its own supervised answer through question-memory self-matching during mining. Precompute label-independent branch hits once; apply permitted label/memory filters per fold. A proposed hard-negative mixture or changed refresh frequency is a quality experiment, not a free speedup. Do not repeatedly cross-encode the full73,745-row historical pair pool unless profiling and quality evidence justify that cost.

### E. Token-budgeted multi-query reranking

Separate neural batch size from query window size. Existing flatten/scatter is the starting point. During qualification, test a small bounded batch sweep such as 8/16/32 pairs at fixed maximum length384 and BF16; accept only measured stable settings. Dynamic padding and length bucketing are candidate optimizations, not assumed defaults. Preserve pair identities, truncation semantics, document max/second-score aggregation and deterministic tie-breaking.

Measure the number of actual evidence pairs, padded tokens and forward passes—not just documents. `rerank_batch()` scores only the first `top_k` candidates and can emit multiple evidence pairs per document (`reranker.py:540–574`). A 50-document cutoff can therefore cost substantially more than 50 neural pairs/query.

If an OOM forces a smaller batch, record it, invalidate the faster forecast, recompute the remaining-time forecast and enforce the approved deadline. Never drop documents/evidence silently. CPU model fallback remains forbidden.

For evaluation only, detect identical fully tokenized query–evidence inputs **within the same adapter/model state**. Score an identical input once and scatter its score back to every original occurrence, preserving evidence metadata, multiplicity, max/second-score aggregation and tie handling. Measure duplicate rate before implementing this optimization. Do not deduplicate training examples or share neural scores across fold adapters; those changes alter learning or use the wrong model. Enable only after evaluation-mode/no-dropout and output-parity tests pass. Do not assume the present evidence pack actually contains duplicates.

### F. Fixed final policy for confirmation

Preselect exact-match handling, branch fusion weights, neural-score aggregation and top-five selector using development evidence before confirmation. Default to a fixed policy rather than expensive learned stacking for the first candidate. Preserve all mandatory evaluation stages, but disable validation-driven fusion winner selection per F7.

Keep development, confirmation and final-production model roles distinct. Final training uses all labels only after held-out predictions are finalized. The final production adapter is not the best fold adapter and cannot retroactively regenerate OOF predictions. No ensemble of all fold adapters is added by default: it would increase inference cost and deployment complexity.

## 10. Quality development path toward >96%

The observed old-fold gap from 82.54% to 96% is **13.46 percentage points**—not a small tuning adjustment. There is no evidence that batching or changing fusion alone closes it. Use a bounded, diagnostic sequence rather than unlimited hyperparameter search.

### 10.1 Measure where recall is lost

For each held-out query record these stages and their top-five oracle:

1. Each individual retrieval branch and the full union.
2. The fused candidate pool (initial FULL baseline:150).
3. The subset actually neural-scored (initial baseline:50).
4. Evidence presence and whether the relevant legal article survives length384 truncation.
5. Reranker document ranking, fixed fusion output and final top-five submission.

For set `C`, compute `min(5, |G ∩ C|) / |G|`, averaged over the full expected population. Ordinary candidate recall is not always the top-five oracle. Gold labels are used for diagnostics/scoring only, never injected into held-out candidates or evidence.

Classify misses into branch omission, fusion/pruning loss, missing/truncated evidence, wrong relevance ordering, aggregation/fusion error and selector/ID formatting error. Break down by multi-gold queries, legal-number references, paraphrases, document length and near-duplicate laws. Report counts, not just favorable examples.

If the maximum document set available to the final selector has oracle <=96%, >96% output is impossible for that configuration. Diagnose union150 and scored50 separately; the scored50 oracle is the reranker's opportunity, while fusion may still select unscored candidates from the union. Use >98.5% oracle at the relevant shortlist as a **development headroom objective**, not as an achieved result or guaranteed sufficient condition.

### 10.2 At most one focused change per diagnosed bottleneck

| Dominant loss | First experiment | Promotion evidence |
|---|---|---|
| Branch/union omission | Native-depth lexical/dense recall and legal-number matching; adjust union membership before adding models | Inner-validation oracle improves; exact semantics/data rules preserved; retrieval budget fits |
| Pruning before neural scoring | Compare fixed scored depths50 vs100;150 only if measured budget permits | More recoverable gold and actual top-five gains exceed added pair cost |
| Missing/truncated evidence | Query-aware article/header placement within the existing token cap; preserve negatives fairly | Token-level evidence inspection plus inner-validation ranking improvement |
| Wrong ranking despite good evidence | Positive/hard-negative rotation; then one pairwise-loss alternative to BCE | Paired inner-validation improvement, coverage maintained, total training budget fits |
| Fusion/selector regression | Compare predeclared fixed policies and ensure unique top-five IDs | No held-out fitting; independent scorer parity and consistent per-query improvement |

Use fixed seeds and the same permitted inner split for paired development comparisons. Keep an experiment ledger with configuration, training exposure, runtime, oracle, official recall/precision and rejection reason. Limit the first development round to the baseline plus at most two evidence-backed alternatives; further exploration requires a new time/spending decision.

Do not call an inner-validation >96% score proof of complete OOF >96%. Promote a single configuration before the full confirmation run. Do not tune on document-disjoint test labels or the public queries' unknown relevance. If no candidate has credible headroom and quality within the measured runtime budget, report the incompatibility rather than promise the threshold.

### 10.3 Select a faster, better candidate—not merely the fastest one

Establish a **repaired baseline** using the same data partition, pins, evaluation population and corrected sampling/fusion protocol as every comparison. Historical82.54% came from one old fold and cannot establish improvement from a new experiment on a different split.

Use a compact experiment ledger with: input/config digest; changed factor; total examples/tokens/updates; training seconds; end-to-end evaluation seconds; peak RAM/VRAM; pool/scored-subset oracles; official recall/precision; and per-query prediction differences. Compare paired per-query recall changes and their uncertainty, including multi-gold/legal-reference slices. A single small noisy gain is a candidate for confirmation, not proof of generalization.

Promotion rules:

1. **Execution optimization:** reduces measured wall time for the same workload without candidate/evidence loss or material ranking regression. Predeclare numerical tolerances; if changed GPU numerics alter top-five outputs, record the differences and evaluate quality instead of labeling the change exactly equivalent.
2. **Quality optimization:** improves paired development recall, respects leakage/coverage and still fits the conservative270-minute total. Spend saved time on better evidence/exposure only when justified; do not spend contingency to rescue an oversized design.
3. **Combined candidate:** retain configurations not dominated by another in both runtime and recall. Among those eligible for the five-hour target, favor the strongest supported recall, then lower runtime and simpler implementation. Freeze exactly one before the FULL attempt.
4. **No qualifying candidate:** stop with measured gaps. Do not silently choose a faster but worse model or claim >96% because the candidate oracle is high.

For a recall-first submission, evaluate a fixed **five unique valid documents** policy as the baseline. Adding a distinct document to an existing list of fewer than five cannot lower per-query recall, but it can lower precision. Report both official metrics and the competition's selection rules; do not call this free overall-score improvement. Do not use uncalibrated confidence thresholds to emit fewer answers merely for apparent precision, and never pad with duplicate or invalid IDs.

Keep depth50 as the initial neural cost baseline. Compare depth100 only if pruning losses warrant it; first try better evidence placement within length384 when that is the dominant loss. Do not automatically increase both depth and sequence length: they compound pair/token cost and obscure which change helped. Adaptive depth/early exits are deferred until fixed-depth evidence demonstrates a need and a leakage-safe policy can be qualified.

## 11. Workload-derived feasibility model

Keep the 270-minute nominal allocation in section4. Build a conservative forecast from measurements on the actual backend:

`T_total = T_setup_indexes + T_static + T_assembly + Σ_j(S_j × sec_per_update_j + model_load_save_j) + Q_eval / eval_qps + T_fusion + T_final_reload_public + T_delivery`

`j` includes five folds, document-disjoint and final training. Use actual coverage-enforced updates, not nominal250 or an assumed epoch count. Include any nested training if learned fusion is chosen. Stage clocks must be exclusive or explicitly represent overlap so work is neither omitted nor double-counted. Report both critical-path wall time and useful per-stage resource timings.

At B8/G2, one positive and negative exposure for5,600 queries implies a **minimum**700 updates, and7,000 queries implies875; actual orchestration/configuration may enforce more. Derive each job's actual update count and assert it matches the forecast before starting it.

Illustrative arithmetic only: if document-disjoint evaluation contains1,400 queries, evaluation covers8,400 queries, so55 minutes requires **2.545 queries/s**, about **5.64×** the historical0.451. At50 neural documents/query and2 evidence pairs/document, that corresponds to about254.5 evidence pairs/s including retrieval/tokenization/aggregation overhead. More evidence, longer padding or deeper scoring raises the workload. Replace these illustrative counts with measured values.

Promotion requires conservative measured stage bounds totaling <=270 minutes, leaving30 minutes contingency within the strict <300-minute acceptance window. Measure representative long-document and multi-gold cases, not only short easy queries. Report median and tail measurements, cold acquisition/index time and peak host/VRAM. A forecast is only eligibility to attempt confirmation; final PASS uses observed complete wall time.

At each stage, recompute remaining required work from actual throughput. If completion within the approved envelope is no longer credible, persist an INCOMPLETE receipt and stop safely; do not shrink validation, bypass reload, omit uploads, change the model or extend the ceiling to manufacture a PASS.

### 11.1 Convert stage budgets into engineering checks

| Budget | Required measurement / rejection condition |
|---|---|
| Setup/indexes45min | Time acquisition, corpus loading, PyVi, BM25 build and dense encoding separately. Old PyVi44.14min leaves almost no room for the others; this allocation is unqualified until those costs fit together. |
| Static retrieval25min | Count searches per distinct query/index/search contract. Repeated fold retrieval should hit validated within-attempt caches, not silently redo the same label-independent work. |
| Assembly20min | Measure per-fold memory/blacklist filtering and evidence/tokenization cost. Do not cache fold labels in static records. |
| Training90min | Sum actual updates and model load/save/validation overhead across all seven primary jobs; measure each proposed batching/checkpointing change. |
| Evaluation55min | Use actual OOF plus disjoint population, evidence-pair counts and complete queries/s, including static-cache misses. Pure model pairs/s is insufficient. |
| Fusion/reporting5min | Fixed-policy application and complete metrics; do not hide nested training here. |
| Reload/public20min | Load from the saved final artifact in a new process and score all1,000 public queries. Include any index reconstruction needed after reload. |
| Delivery10min | Include full required model/index bundle transfer and checksum confirmation. Measure bytes and sustained transfer rate; queueing/network uncertainty belongs in the forecast. |

Illustration, **not the current job schedule**: five700-update folds + one700-update disjoint job +875-update final job =5,075 updates. A90-minute training allowance permits only `5,400 / 5,075 ≈ 1.064 seconds/update` **before** model load/save or validation overhead. If actual effective steps are higher, the permissible seconds/update is lower. This check makes it impossible to assume a nominal250-step config proves feasibility.

For optimization proposals, record saved seconds on the affected stage and the resulting total forecast. Do not multiply independent headline speedups across the entire run: accelerating a fraction `f` of baseline wall time by factor`s` yields idealized total speedup `1 / ((1-f) + f/s)`, before new overhead. Measure the whole stage after integration.

**Implementation landing points:** static cache orchestration in `src/pipeline/kaggle_train.py` and `src/pipeline/oof_runner.py`; dense identity/encoding in `src/retrieval/dense_macro.py`; batching/deduplication in `src/ranking/reranker.py`; training execution/exposure in `src/training/train_reranker.py`; confirmatory fusion isolation in `src/ranking/train_fusion.py`; supervised lifecycle/resource profiles in existing Modal/Colab scripts. Extend those boundaries surgically rather than introducing a second training pipeline. Each work package needs correctness tests and timing counters before a GPU comparison.

## 12. End-to-end validation matrix

Proposed new test names/receipt files below do **not** yet exist unless independently implemented. They specify deliverables for the next coding session, not runnable commands today.

| Layer | Required checks | Evidence / acceptance |
|---|---|---|
| Unit/property | Cache identities, immutable dense index, pair coverage, native-depth parity, batch scatter/aggregation/ties, tokenizer/truncation parity | Deterministic CPU fixtures, adversarial mutations and positive controls |
| Leakage | Alter outer held-out labels; fitting, early stopping and predictions unchanged | Random-fold, document-disjoint and fusion dependency tests |
| Scorer parity | Compare internal metrics against official scorer on multi-gold/partial/empty cases; reject missing/duplicate/extra IDs and >5 outputs upstream | Same official query-macro recall and precision to numerical tolerance |
| Full orchestration contract | Small local fixture traverses five folds, disjoint, final training, fresh reload, submission validation and recovery | Tests control flow only; mock/tiny-model quality never presented as competition evidence |
| Fault injection | Dirty Colab tree, stale freeze, corrupted caches, partial marker, timeout, upload failure, failed stop, stale receipt | Fail before spending where possible; otherwise truthful FAIL/INCOMPLETE and no automatic retry |
| Kaggle hardware smoke | Real dual-T4 training update, pinned save/reload and real batch inference contract | Genuine candidate-bound report; smoke only |
| A100 qualification | Cold preparation, representative mining/training/inference, memory headroom, lifecycle/delivery | Separate approved bounded spend; forecast <=270 minutes, no FULL PASS claim |
| FULL confirmation | Actual canonical data, five folds, disjoint, all-query final adapter, fresh-process public inference and durable delivery | Same-attempt joint gate J; no dropped stages or substituted candidate |
| Artifact-only replay | Separate approved load of delivered final artifacts; deterministic small prediction reproduction | Demonstrates future inference reuse, not another training completion or new quality result |

### Independent acceptance verifier

Add a small CPU-only verifier (proposed `scripts/verify_end_to_end_acceptance.py`) and focused tests only after implementation is authorized. It must reconstruct the official OOF metric from persisted per-query predictions/qrels and fixed expected splits, not trust a claimed summary score. Use declared canonical paths without broad filesystem scans or credential reads.

Check: exact ID/fold coverage; no duplicate document IDs; five valid folds; disjoint report/input identity; immutable model/data/config identity; elapsed clock evidence; final fresh-process reload receipt; full public submission shape/IDs; artifact checksums and delivery confirmation; shutdown state; predeclared policy hash and selection protocol. Hard-fail inconsistent identities, rounded threshold comparisons, missing timing endpoints and unknown/incomplete states.

Verifier tests must include:17,999.9s vs18,000s;0.960001 vs0.96; high score on only6,999 queries; four folds; duplicate IDs; absent document-disjoint results; fast runA plus high-score runB; stale adapter; fake summary inconsistent with predictions; upload receipt missing final weights; and failed shutdown. Only the complete matching positive fixture passes.

### Machine-readable receipt contract

Persist a structured acceptance receipt alongside detailed stage/model/submission manifests. Before measurement, numeric results are `null` and verdict is `NOT_RUN`; do not seed plausible scores or durations. Required groups:

- **Identity:** schema version, attempt ID, runtime/release SHAs, dataset/split/config digests, model/tokenizer revisions, backend and hardware resources.
- **Protocol:** development versus confirmation, predeclared policy/config hash, selection method, earlier split exposure disclosure, cache inventory/cold definition.
- **Timing:** supervisor start/end UTC, monotonic elapsed seconds, exclusive stage intervals, setup/delivery boundaries, source log references.
- **Quality:** expected/evaluated query counts, per-fold counts/scores, pooled official OOF recall/precision, uncertainty interval, document-disjoint metrics, shortlist oracles.
- **Training:** per-job input/pair identity, sampled coverage, effective updates, adapter checksums and base-model identity.
- **Delivery:** fresh-process reload result, public ID/format validation, required artifact inventory, checksums and approved-destination confirmation.
- **Lifecycle/verdict:** provider shutdown evidence, T/Q/J booleans, reasons and final state (`PASS`, `FAIL`, `INCOMPLETE`, `NOT_RUN`).

Use receipt hashes to cross-reference outputs. Checksums do not independently authenticate the accuracy of a reported timer; retain supervisor logs and provider timestamps for audit. Publish only approved nonsensitive evidence; raw secrets never belong in receipts.

## 13. Implementation and completion contract

Implement in this order, with minimal changes to existing modules:

1. **Correctness foundation:** F1–F4 and F7; effective configuration/identity; disable unsafe FULL reuse; add scorer/coverage/leakage/launcher tests.
2. **Runtime architecture:** resident immutable indexes, once-per-run static/query work, bounded evidence caching, measured batched reranking and clean model lifecycle. Verify equivalent output before claiming a speed-only change.
3. **Quality development:** stage-loss diagnostics and bounded training-only experiments from section10. Freeze one procedure; do not run an unbounded search inside the confirmation attempt.
4. **Acceptance machinery:** supervisor timing, per-query evidence, fresh-process reload/delivery receipts and independent verifier with failure fixtures.
5. **Qualification/release:** full local gate, exact-SHA CI, genuine Kaggle smoke and evidence-bearing release; approved A100 profiling. Runtime changes after profiling/smoke require applicable requalification.
6. **Joint confirmation:** separately approved single-provider FULL attempt, verify J and report actual scores/time. Qualify the other provider separately if claiming support for both within the target.

Completion means an independently checked receipt from a real FULL run establishes the targets. A document, successful unit suite, three-step smoke, throughput extrapolation, favorable inner-validation result, or a partially completed model does not satisfy it.

If either target fails, retain all honest evidence and say which constraint failed. Recommend the next measured change; do not weaken the threshold, relabel warm work as cold, skip folds, claim public-test quality without labels, or launch another paid attempt without approval.
