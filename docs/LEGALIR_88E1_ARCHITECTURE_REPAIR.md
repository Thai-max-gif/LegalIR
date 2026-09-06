# LegalIR 88E1 — Architecture Convergence & Kaggle Final Hardening

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Use test-driven development for every behavioral repair and verification-before-completion before declaring a gate passed.

**Repository:** `silent9669/LegalIR`  
**Audited HEAD:** `88e1b87948d5b178d4aa695e89382433af6847f1`  
**Approved runtime recorded in release artifacts:** `33a0930e7d1ca1ee58000efdc28a79cf7107b9bf`  
**Goal:** Converge the project on the v16 Artifact-Factory → immutable bundle → thin Kaggle Final architecture without changing proven retrieval/ranking semantics.

## Executive Verdict

**DO NOT run `notebooks/kaggle_final.ipynb` as a competition production run yet.**  
**DO NOT use the monolithic `legalir_training.ipynb` FULL path as the preferred final run.**

Keep the retrieval/ranking design:

```text
Canonical Task1 v2
  ├─ Legal BM25
  ├─ PyVi BM25
  ├─ DEk21 Dense Macro
  ├─ Exact Matcher
  ├─ fold-local Question Memory (validation only)
  ├─ BGE reranker + LoRA
  ├─ OOF/doc-disjoint validation
  └─ frozen fusion / top-5
```

Converge execution to:

```text
CANONICAL V2
    ↓
RESUMABLE ARTIFACT FACTORY
    ├─ static train/public retrieval cache
    ├─ lazy macro evidence
    ├─ 5 fold OOF jobs
    ├─ doc-disjoint job
    ├─ fusion selection
    ├─ final all-7k training pairs
    ├─ public candidates/evidence
    ├─ dataset/static-cache/validation provenance
    └─ production_lock
    ↓
IMMUTABLE VERIFIED PRODUCTION BUNDLE
    ↓
KAGGLE FINAL
    ├─ verify runtime + dataset + bundle
    ├─ train exactly ONE real BGE LoRA
    ├─ fresh-reload adapter
    ├─ rerank public candidates
    ├─ apply exact frozen winner
    ├─ validate 1..5 unique official doc IDs
    └─ submission.zip
```

## Observed Drive State

The supplied Drive workspace has the canonical v2 dataset and a valid factory preflight, but the expensive production factory has **not** been materialized there yet.

Observed:
- canonical v2: 8,532 docs
- chunks: 1,153,876 = 934,416 micro + 219,460 macro
- train queries: 7,000
- qrels: 7,637
- public queries: 1,000
- duplicate groups: 4
- `factory/`: only `preflight.json`
- `task1/checkpoints/`: empty
- `task1/indexes/`: empty
- no visible immutable production bundle containing final pairs/public candidates/public evidence/fusion/lock

This means `approved_for_kaggle_full=true` is only an authorization artifact; it is not evidence that the new v16 production bundle or final production run already exists.

---

# P0 Repairs

## Task 1 — Eliminate accidental mock training from real Kaggle Final

**Files**
- Modify: `scripts/run_kaggle_final.py`
- Modify: `src/production/final_train.py`
- Modify: `src/training/train_reranker.py`
- Modify: `src/ranking/reranker.py`
- Tests: `tests/integration/test_final_production.py`
- Add: `tests/release/test_production_models_fail_closed.py`

### Problem

`run_kaggle_final.py` calls `train_final_adapter()` without a real runtime config.  
`train_final_adapter()` currently defaults:

```python
base_model = cfg.get("base_model_name", "mock")
```

Therefore a non-mock production run can train the tiny mock BERT instead of `BAAI/bge-reranker-v2-m3`.

In addition, both training and inference loaders catch real Hugging Face load failures and silently construct a random tiny BERT. Production must never fail open this way.

### Required behavior

- [ ] `mock_run=False` must make `base_model_name="mock"` illegal.
- [ ] Kaggle Final must derive the reranker configuration from the frozen production lock/bundle, not an implicit default.
- [ ] Real tokenizer/model load failure must raise a fatal error.
- [ ] Mock fallback is allowed only when an explicit test/smoke flag is true.
- [ ] Training manifest must record the exact real base model, PEFT config, optimizer steps, pair coverage, adapter SHA, and device.
- [ ] Fresh reload must prove the adapter is attached to the expected real base model.

### Red tests first

```text
test_nonmock_final_rejects_mock_base_model
test_run_kaggle_final_passes_frozen_reranker_config
test_nonmock_train_model_load_failure_is_fatal
test_nonmock_inference_model_load_failure_is_fatal
test_real_adapter_manifest_base_model_matches_production_lock
```

### Verification

```bash
pytest -q tests/integration/test_final_production.py \
  tests/release/test_production_models_fail_closed.py
```

Commit:
```bash
git add scripts/run_kaggle_final.py src/production/final_train.py \
  src/training/train_reranker.py src/ranking/reranker.py tests/
git commit -m "fix(production): fail closed and require real final reranker"
```

---

## Task 2 — Restore exact public query and frozen-fusion inference semantics

**Files**
- Modify: `scripts/run_kaggle_final.py`
- Modify: `src/production/public_rerank.py`
- Reuse: `src/ranking/fusion.py`
- Reuse: `src/ranking/oof_features.py`
- Tests: `tests/integration/test_final_production.py`
- Add: `tests/parity/test_public_fusion_parity.py`

### Problems

1. Official public JSON values are objects like:
   ```json
   {"question": "...", "answer": null}
   ```
   but the current final runner passes the whole object to public reranking. The reranker then stringifies the dictionary instead of receiving only the legal question text.

2. `fusion_model_path` exists in `public_rerank.py` but is not actually consumed.

3. Current public reranking implements a new simplified score:
   ```text
   static RRF + raw reranker logit * weight
   ```
   This is not equivalent to the proven `ReciprocalRankFusion`, which uses branch-specific ranks/weights and calibrated sigmoid treatment for reranker scores, and it cannot reproduce a learned LightGBM winner.

### Required behavior

- [ ] Normalize public data once to `dict[qid, question_string]`.
- [ ] Build the same candidate record fields used by OOF.
- [ ] Preserve the same post-rerank feature schema/version as OOF.
- [ ] If winner is RRF, instantiate and use `ReciprocalRankFusion` with the exact frozen `k` and weights.
- [ ] If winner is learned ranker, load the frozen model with `strict=True` and apply `extract_candidate_features()` using the frozen feature columns.
- [ ] Never silently substitute RRF when a learned model was selected.
- [ ] Never invent different public-time weights.
- [ ] Preserve deterministic tie breaking and strict top-5 sanitation.

### Red tests first

```text
test_public_json_is_normalized_to_question_text
test_public_reranker_never_receives_dict_as_query_text
test_public_rrf_matches_ranking_fusion_on_same_candidate_records
test_public_learned_fusion_matches_oof_feature_schema
test_public_learned_winner_requires_real_fusion_payload
test_public_top5_is_unique_complete_and_official
```

### Verification

```bash
pytest -q tests/parity/test_public_fusion_parity.py \
  tests/integration/test_final_production.py
```

Commit:
```bash
git add scripts/run_kaggle_final.py src/production/public_rerank.py tests/
git commit -m "fix(inference): restore frozen fusion and public query parity"
```

---

## Task 3 — Make the fusion artifact a real, single source of truth

**Files**
- Modify: `src/ranking/train_fusion.py`
- Modify: `scripts/select_production_config.py`
- Modify: `src/validation/promotion.py`
- Modify: `src/bundle/builder.py`
- Modify: `src/bundle/verifier.py`
- Tests: `tests/unit/`, `tests/integration/`

### Problems

- `build_production_bundle.py` requires `artifacts/factory/fusion/fusion_model.json`.
- The current fusion trainer emits `model_full.txt`, `manifest.json`, `fusion_comparison.json`, etc., but there is no coherent producer for the required `fusion_model.json`.
- `select_production_config.py` hard-codes RRF weights instead of freezing the actual OOF winner.

### Artifact contract

Create one schema-versioned descriptor:

```json
{
  "schema_version": 1,
  "winning_method": "reciprocal_rank_fusion | learned_ranker",
  "feature_schema_version": "...",
  "feature_columns": [],
  "rrf": {
    "k": 60,
    "weights": {}
  },
  "learned_model": {
    "file": "fusion_model.txt",
    "sha256": "..."
  }
}
```

Rules:
- RRF winner: `rrf` is mandatory; learned payload is absent.
- Learned winner: actual model payload + SHA is mandatory.
- Descriptor must be generated directly from the cross-fitted fusion selection result.
- Public inference reads this descriptor; no hard-coded alternative weights.

### Red tests first

```text
test_fusion_descriptor_matches_cross_fitted_winner
test_rrf_descriptor_roundtrip_preserves_exact_ranking
test_learned_descriptor_requires_hashed_payload
test_bundle_fusion_descriptor_is_consumable_by_public_inference
```

Commit:
```bash
git commit -am "fix(fusion): freeze and consume one production fusion artifact"
```

---

## Task 4 — Repair `production_lock.json` generation

**Files**
- Modify: `src/validation/promotion.py`
- Modify: `scripts/select_production_config.py`
- Tests: add/extend production-lock tests

### Problems

Current CLI still has:
```text
--runtime-commit default="a0efb25"
```

`create_production_lock()` still has a placeholder-style default:
```text
dataset_sha256="canonical_v2"
```

The CLI writes by default to:
```text
artifacts/bundle/production_lock.json
```

but the production bundle builder searches:
```text
artifacts/factory/production_lock.json
```

### Required behavior

- [ ] `--runtime-commit` required, exactly 40 hex characters.
- [ ] `--dataset-fingerprint` required, exactly 64 hex characters.
- [ ] no placeholder/default SHA values anywhere in strict production.
- [ ] default lock output: `artifacts/factory/production_lock.json`.
- [ ] lock must be derived from:
  - validation summary
  - actual cross-fitted fusion winner
  - exact reranker config
  - exact candidate/rerank/top-k
  - exact feature schema
- [ ] lock `config_sha256` must hash canonical serialized config.
- [ ] runtime SHA, dataset fingerprint, and config hash must later match the bundle manifest byte-for-byte.

### Red tests

```text
test_production_lock_rejects_short_runtime_sha
test_production_lock_rejects_placeholder_dataset_fingerprint
test_select_production_config_default_path_matches_bundle_builder
test_lock_winner_matches_fusion_descriptor
test_lock_config_hash_is_reproducible
```

---

## Task 5 — Upgrade bundle verification from file hashing to semantic verification

**Files**
- Modify: `src/bundle/verifier.py`
- Modify: `src/bundle/builder.py`
- Modify: `scripts/build_production_bundle.py`
- Tests: `tests/integration/test_production_bundle.py`

### Required semantic checks

After cryptographic file verification, fail unless:

```text
bundle.runtime_commit == production_lock.runtime_commit
bundle.dataset_fingerprint == production_lock.dataset_sha256
bundle.config_sha256 == production_lock.config_sha256
dataset_provenance fingerprint/counts == official canonical v2
static_cache_provenance says label-free and no qrels dependency
validation_summary contains 5/5 completed OOF folds
validation_summary contains completed doc-disjoint evaluation
all leakage counters == 0
all duplicate-equivalent negative violations == 0
final_training_pairs has required positive+negative coverage
final_training_pairs covers all expected eligible train queries
public_candidates covers all 1,000 official public qids
public_evidence covers every document that final reranking expects to score
fusion descriptor/payload hashes and schema are valid
```

Do not treat “64 hex characters” as sufficient evidence that hashes refer to the correct artifacts.

### Red tests

```text
test_bundle_rejects_lock_runtime_mismatch
test_bundle_rejects_lock_dataset_mismatch
test_bundle_rejects_lock_config_mismatch
test_bundle_rejects_incomplete_5fold_validation
test_bundle_rejects_missing_doc_disjoint
test_bundle_rejects_public_qid_gap
test_bundle_rejects_final_pair_coverage_gap
test_bundle_rejects_fusion_payload_hash_mismatch
```

---

## Task 6 — Repair release provenance: bind CI run ID to runtime SHA

**Files**
- Modify: `scripts/verify_release_approval.py`
- Reuse/merge: `scripts/verify_github_ci.py`
- Consolidate: `src/release/provenance.py`
- Modify: `.github/workflows/ci.yml`
- Tests: `tests/test_release_approval_head_gate.py`
- Add: `tests/release/test_release_ci_binding.py`

### Problem

The Drive/repo release approval records:
```text
runtime_sha = 33a0930...
ci.run_id   = 33527231890
```

but inspection of GitHub shows run `33527231890` executed on a different SHA (`a0efb25...`).

The current local release validator verifies that the JSON's `ci.runtime_sha` equals its own `runtime_sha`, but it does not prove that the recorded GitHub run ID actually belongs to that SHA.

### Required release model

Use the existing two-commit concept:

```text
Commit A = runtime commit
  ↓ CI on exact A must be GREEN
  ↓ real Colab T4 smoke on exact A must PASS
Commit B = release-only commit
  - approval artifact
  - Colab report
  - generated notebooks
  - parameter report
  - no scoring/runtime source changes
  ↓ CI on B verifies lineage and all proofs
```

No self-referential `release_sha` field is necessary inside the approval JSON.

### Required verification

- [ ] Query GitHub Actions for the recorded run ID.
- [ ] require workflow name `LegalIR CI`.
- [ ] require `head_sha == runtime_sha`.
- [ ] require `conclusion == success`.
- [ ] require Commit A to be ancestor of Commit B.
- [ ] require A→B diff to contain only release-allowlisted files.
- [ ] require Colab report runtime SHA == A.
- [ ] require Colab report SHA-256 == recorded hash.
- [ ] require all distributed notebook runtime pins == A.
- [ ] keep one authoritative provenance validator; remove schema disagreement between `src/release/provenance.py` and the CLI validator.

### Red tests

```text
test_release_rejects_ci_run_for_different_sha
test_release_rejects_failed_runtime_ci
test_release_rejects_wrong_workflow_name
test_release_accepts_runtime_A_release_only_B
test_provenance_validator_schema_is_single_source_of_truth
```

---

## Task 7 — Converge notebook/entrypoint surfaces

**Files**
- Modify: `scripts/generate_kaggle_notebook.py`
- Modify: `scripts/check_notebook_parity.py`
- Regenerate:
  - `legalir_training.ipynb`
  - `kaggle_kernel_task1/legalir_training.ipynb`
  - `kaggle_kernel/legalir_training.ipynb`
  - `kaggle_kernel/legalqa_gpu_pipeline.ipynb` only if still intentionally distributed
  - `notebooks/kaggle_final.ipynb`
- Modify: `README.md`
- Modify: `docs/OPERATING_RULES.md`
- Modify: `docs/CI_COLAB_KAGGLE_WORKFLOW.md`

### Required decision

There must be **one canonical competition Kaggle Final entrypoint**.

Recommended:
```text
scripts/run_kaggle_final.py
```

All notebooks intended for competition execution must be generated wrappers around it.

Keep `run_kaggle_pipeline()` only as a legacy/regression/smoke/reference orchestrator if needed; it must not remain the default advertised final path.

### Required checks

- [ ] canonical notebook generator takes exact runtime SHA.
- [ ] no notebook defaults to `main`.
- [ ] no competition notebook calls monolithic FULL OOF.
- [ ] no competition notebook reinstalls torch/CUDA.
- [ ] parity checker covers every distributed notebook surface or verifies that all are generated from the same canonical source.
- [ ] README/runbook describe the same architecture as executable code.

### Red tests

```text
test_all_distributed_kaggle_notebooks_use_final_runner
test_no_distributed_final_notebook_calls_run_kaggle_pipeline
test_all_distributed_notebooks_pin_same_full_runtime_sha
test_all_distributed_notebooks_avoid_torch_reinstall
test_notebook_parity_covers_every_distributed_surface
```

---

## Task 8 — Replace synthetic runtime-budget proof with measured telemetry

**Files**
- Modify: `tests/test_595e_drive_kaggle_final_blockers.py`
- Modify/add runtime estimator utility under `src/`
- Use: `artifacts/task1/colab_smoke_report.json`

### Problem

The existing runtime-budget test proves only that manually entered stage estimates add to less than 10.8 hours.

Current training coverage rules require approximately:
```text
5 OOF folds × ~700 optimizer steps = 3,500
doc-disjoint                  ~700
final all-7k                  ~875
-----------------------------------
monolithic total            ~5,075 optimizer steps
```

The inspected real T4 smoke used roughly 201.85 seconds for 10 optimizer steps (~20.2 s/step on that smoke workload). A naive linear extrapolation is ~28.5 hours for 5,075 steps. This is only a planning estimate, not a benchmark, but it is enough to show that the synthetic `<10.8h` assertion is not evidence of Kaggle feasibility.

### Required behavior

- [ ] Separate `factory_runtime_budget` from `kaggle_final_runtime_budget`.
- [ ] Use measured telemetry when estimating, with explicit uncertainty/safety factor.
- [ ] Kaggle Final estimate includes only one final adapter (~875 coverage-derived steps), public rerank, validation, packaging.
- [ ] Factory may span resumable GPU sessions and must not be forced into one Kaggle session.
- [ ] Runtime gate must fail if measured projection exceeds configured session budget.
- [ ] Never label a toy/mock smoke “READY FOR KAGGLE FINAL”.

### Tests

```text
test_runtime_projection_uses_measured_telemetry
test_monolithic_full_is_not_final_kaggle_entrypoint
test_kaggle_final_projection_excludes_oof_and_doc_disjoint
test_mock_smoke_cannot_authorize_final
```

---

# P1 Hardening

## Task 9 — Strengthen final model parameter and coverage audits

**Files**
- Modify: `src/production/final_train.py`
- Reuse: `src/models/parameter_audit.py`

- [ ] Final manifest must distinguish:
  - base model parameters
  - LoRA/trainable parameters
  - dense retriever parameters represented in precomputed system artifacts
  - total system learned parameters
- [ ] Require total system learned parameters `<4,000,000,000`.
- [ ] Do not use a fallback constant such as `50,000,000`.
- [ ] Require exact expected final query coverage from bundle provenance, not merely `unique_qids > 0`.
- [ ] Require positive and negative query coverage to meet production policy.

## Task 10 — Remove ambiguous legacy defaults after parity passes

Only after all above tests pass:
- remove stale short SHA defaults
- remove placeholder fingerprints
- remove duplicated release-validation logic
- remove/deprecate old notebook generation paths
- keep ranking math unchanged unless parity tests prove a required correction

Do **not** perform a broad cleanup/refactor of the 98KB legacy orchestrator before the production release. Competition risk is lower if it remains as tested reference code while the control plane is corrected around it.

---

# Factory Execution Gate

After code repair, run the real factory and materialize:

```text
artifacts/factory/preflight.json
artifacts/factory/static_cache/static_candidates_train.parquet
artifacts/factory/static_cache/static_candidates_public.parquet
artifacts/factory/static_cache/static_cache_provenance.json
artifacts/factory/folds/fold_0/...
artifacts/factory/folds/fold_1/...
artifacts/factory/folds/fold_2/...
artifacts/factory/folds/fold_3/...
artifacts/factory/folds/fold_4/...
artifacts/factory/doc_disjoint/...
artifacts/factory/final_training_pairs.parquet
artifacts/factory/evidence/public_evidence.parquet
artifacts/factory/fusion/fusion_model.json
artifacts/factory/validation_summary.json
artifacts/factory/dataset_provenance.json
artifacts/factory/production_lock.json
artifacts/bundle/production/...
```

The exact commands should come from the repaired runbook/CLIs. Do not fabricate missing artifacts to make the verifier pass.

---

# Mandatory Full Verification

Run from repository root:

```bash
python -m compileall -q src scripts
pytest -q
python scripts/audit_parameters.py
python scripts/check_notebook_parity.py
python scripts/verify_release.py
```

Then run targeted production verification against real artifacts:

```bash
python scripts/build_production_bundle.py \
  --bundle-dir artifacts/bundle/production \
  --runtime-commit <COMMIT_A_FULL_SHA> \
  --dataset-fingerprint <REAL_DATASET_SHA256> \
  --config-sha256 <REAL_PRODUCTION_CONFIG_SHA256> \
  --artifacts-dir artifacts/factory

python scripts/run_kaggle_final.py \
  --dataset-dir <CANONICAL_V2_DIR> \
  --bundle-dir artifacts/bundle/production \
  --output-dir <TEMP_OUTPUT> \
  --mock
```

The mock execution is only an orchestration test. It does not authorize production.

Then:
1. Commit A with source/runtime fixes.
2. Push A.
3. Require GREEN CI for exact A.
4. Run the real Colab T4 smoke on exact A.
5. Record real report + hash.
6. Generate release-only Commit B.
7. CI on B verifies A's exact CI run ID/SHA, Colab report, lineage, notebook pins, tests and bundle contracts.
8. Only then upload immutable canonical dataset + production bundle to Kaggle.
9. Run thin Kaggle Final with no mock flags.

---

# Release Acceptance Criteria

```text
HEAD tests                            PASS
all historical regressions           collected
real runtime CI run ↔ SHA             MATCH
Colab T4 report ↔ runtime SHA         MATCH
canonical identity                    PASS
static cache provenance               PASS, label-free
OOF folds                             5/5 PASS
OOF leakage                           0
doc-disjoint                          PASS
fusion winner                         frozen and reproducible
final pairs                           full required coverage
production lock                       real SHA/fingerprint/config
production bundle                     crypto + semantic PASS
real final base model                 BAAI/bge-reranker-v2-m3
mock fallback in production           impossible
effective batch                       16
fresh adapter reload                  PASS
total learned parameters              <4B
public queries                        1000/1000
public fusion parity                  PASS
submission validation                 PASS
competition notebook                  thin final only
torch/CUDA reinstall on Kaggle        NO
```

Final status must remain:

```text
READY FOR KAGGLE FINAL: NO
```

until all acceptance criteria above have fresh evidence.

---

# Required Agent Report

```markdown
# LegalIR 88E1 Production Repair Report

## Git
- audited HEAD:
- Commit A runtime SHA:
- Commit B release SHA:

## P0 repairs
- mock default eliminated:
- real loaders fail closed:
- public query normalization:
- frozen fusion parity:
- production lock:
- bundle semantic verification:
- CI run↔SHA binding:
- notebook convergence:
- measured runtime gate:

## Factory
- preflight:
- static cache:
- 5/5 folds:
- doc-disjoint:
- final pairs:
- public evidence:
- fusion artifact:
- production lock:
- bundle:

## Model
- base:
- LoRA:
- optimizer steps:
- effective batch:
- loss finite:
- param_diff:
- adapter hash:
- fresh reload:
- total learned params:

## Validation
- OOF Recall@5:
- OOF Precision@5:
- Candidate Recall@50/150:
- Doc-disjoint Recall@5:
- leakage violations:
- duplicate violations:

## CI / Colab
- pytest:
- runtime CI run id:
- runtime CI head SHA:
- Colab target SHA:
- Colab result:

## Kaggle Final
- notebook pin:
- final runner:
- production bundle hash:
- public queries:
- submission validation:

READY FOR KAGGLE FINAL: YES/NO
```
