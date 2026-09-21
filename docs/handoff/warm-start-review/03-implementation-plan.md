# Implementation plan (ordered)

Target: make warm-start safe, then re-qualify the release for A100.
Baseline: release `2a0d17e` / runtime `180b15e`.

Follow TDD: write the failing test first where a test is specified.

---

## Task 1 — Gate warm-start behind an explicit opt-in (CRITICAL)

**Goal:** fold and document-disjoint training always start from base weights;
only the final model may warm-start.

### 1a. `src/training/trainer.py` — `setup_peft_model()` (line 455)

Add a parameter and make the default refuse:

```python
def setup_peft_model(
    model: nn.Module,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
    pretrained_adapter: str | None = None,
    allow_warm_start: bool = False,      # NEW — fail-closed default
) -> tuple[nn.Module, dict[str, Any]]:
```

Change the guard at line 471 from `if pretrained_adapter:` to:

```python
if pretrained_adapter and allow_warm_start:
```

Keep the existing `hidden_size >= 256` mock guard inside.

### 1b. `src/training/trainer.py` — `RerankerTrainer.__init__` (line 631)

Read the opt-in from config and forward it:

```python
allow_warm = bool(self.config.get("allow_warm_start", False))
...
self.model, self.peft_meta = setup_peft_model(
    ...,
    pretrained_adapter=str(pretrained_adapter).strip() if pretrained_adapter else None,
    allow_warm_start=allow_warm,
)
```

### 1c. `src/training/train_reranker.py` — `train_reranker()` (line 33)

Add a keyword argument, defaulting to False, and write it into `cfg`:

```python
def train_reranker(
    *,
    pairs_file: str | Path,
    output_dir: str | Path,
    ...
    allow_warm_start: bool = False,      # NEW
) -> dict[str, Any]:
```

After the other cfg overrides (near line 78, before line 130 `cfg["max_steps"]`):

```python
cfg["allow_warm_start"] = bool(allow_warm_start)
```

This must **overwrite** any value in the YAML, so a stale config key cannot
re-enable warm-start behind the caller's back.

### 1d. Final-model call site opts in

`src/pipeline/kaggle_train.py:2071` — the `train_reranker(...)` call that
produces `final_reranker_dir`. Add:

```python
allow_warm_start=is_full,
```

Do **not** touch `src/pipeline/oof_runner.py:899` or `:1367`. They keep the
default `False` and therefore cold-start. That is the fix.

### 1e. Record provenance in the manifest

`peft_meta` already carries `warm_start` and `pretrained_adapter` on the warm
path. Ensure the fold/disjoint manifests show `warm_start` absent/False so a
reviewer can audit any run after the fact. No code change expected — verify.

---

## Task 2 — Regression test locking the isolation (CRITICAL)

New file: `tests/leakage/test_warm_start_isolation.py`

Required assertions:

1. `setup_peft_model(model, pretrained_adapter=<local adapter dir>)` with
   `allow_warm_start` omitted → returns a **cold** model
   (`meta.get("warm_start")` is falsy), and does not download anything.
2. Same call with `allow_warm_start=True` → `meta["warm_start"] is True`.
3. `train_reranker(...)` called **without** `allow_warm_start` writes a report
   whose `warm_start` is falsy, even when the YAML config sets
   `pretrained_lora_path`. This is the anti-footgun test — it is the one that
   would have caught the original defect.
4. Static guard: the OOF call sites do not pass `allow_warm_start=True`.
   Parse `src/pipeline/oof_runner.py` and assert no `allow_warm_start` truthy
   argument appears in the `train_reranker` calls.

Build the local adapter fixture by saving a tiny LoRA adapter to `tmp_path`
(the pattern already exists in `tests/unit/test_warm_start_lora.py`). Do not
hit the network in tests.

Red-green: confirm each test FAILS against current `HEAD` before Task 1 lands.

---

## Task 3 — Fix the Hugging Face model card

Repo `dangphuc2109/legalir-task1-reranker`, file `README.md`.

Replace:
```
- **Target Modules**: `q_proj`, `v_proj`
```
with the values from the published `adapter_config.json`:
```
- **Target Modules**: `query`, `key`, `value`, `dense`
- **Saved modules**: `classifier`, `score`
```

While editing, remove or qualify any claim that warm-start reduces training
time (see `02-claims-audit.md`, Finding 2).

---

## Task 4 — Correct the runtime narrative

Anywhere the repo or model card claims warm-start cuts training time, replace
with the accurate statement:

> Optimizer-step count is pinned by the query-coverage policy
> (`compute_coverage_required_steps`), so warm-start does not reduce training
> wall-clock. It changes the starting point, not the step budget. Runtime
> levers in this release are `inference_batch_size: 64` and static
> branch-cache reuse, both unmeasured on A100.

---

## Task 5 — Re-qualify the release

Mandatory after Tasks 1–2, because `src/` changed.
Full procedure in `04-verification-protocol.md`. Summary:

1. Local: `./.venv/bin/python scripts/verify_prepush.py` → all gates PASS.
2. Commit as a **runtime** commit; push.
3. Kaggle dual-T4 gate on that exact SHA → `verdict: PASS`, `weight_delta > 0`.
4. Update `production_freeze.json` (`git_sha`, `run_id`, canonical report hash).
5. Regenerate notebooks (zero drift).
6. Commit as a separate **release** commit; push.
7. `scripts/verify_release_approval.py` + `scripts/colab/bootstrap.py` PASS.

Only then is A100 dispatch defensible — and dispatch still needs the user's
explicit spend authorization.

---

## Task 6 — Switch to a pairwise ranking objective (HIGHEST SCORE VALUE)

Rationale and evidence: `06-recall-bottleneck.md`. The 7.56pp gap between
Cand@150 (98.89%) and Recall@5 (91.33%) is an ordering problem; pointwise BCE
does not optimize ordering.

All machinery already exists — this is config-level:

`configs/experiments/reranker_lora.yaml`
```yaml
loss_type: "pairwise_logistic"    # was "bce"
# pos_weight: 4.0                 # BCE-only knob; inert under pairwise — remove or mark unused
```

Verify the group path activates (`src/training/trainer.py:662`): the trainer
must build `RerankerGroupDataset` + `RerankerGroupCollator` +
`QueryBalancedGroupSampler`, not the pointwise dataset. Assert in a test.

**This is a scoring-policy change.** It must apply identically to OOF,
disjoint, and final inference, and the fusion/selector parity tests must be
re-run to prove identical ordering semantics (`tests/parity/`). It cannot be
tuned against private labels.

Risk: changes the objective on an otherwise unmeasured configuration. Land
Tasks 1–2 first so the OOF number that judges it is honest.

## Task 7 — Length-grouped batching (bit-identical time saving)

`src/ranking/reranker.py:429` already uses `padding=True` (dynamic padding to
the longest item in each batch) — good. What is missing is **length-grouped
batching**: no sort-by-length exists anywhere in the scoring path
(`grep -niE "sort.*len|argsort" src/ranking/reranker.py` → no match).

With random ordering, every batch pads to its longest member. Sorting pairs by
token length and batching adjacent items minimizes padding waste — typically
20–35% off variable-length batched inference, against the 25k+ seconds of
rerank work identified in `05-runtime-reality.md`.

Implementation: inside `score_pairs`, tokenize-length-estimate → `argsort` →
batch → score → **invert the permutation** before returning, so callers see the
original order.

**Invariance test is mandatory** (the constraints allow batching changes only
with one): scoring the same pair set with and without grouping must yield
identical *rank ordering*, and scores equal within bf16 tolerance. The
permutation inversion is the likely bug site — test it explicitly.

## Task 8 — Private-round proof obligations (not yet addressed)

None of this was covered by this review. It must be done before declaring READY.

1. **Disjointness proof** — train (7,000) / public (1,000) / private (2,080) ID
   sets pairwise disjoint across splits, qrels, mining inputs, and submission
   output. Assert, do not assume.
2. **Discovery order** — public-first by default; private selected *only* via
   `LEGALIR_TEST_PHASE=private` or `LEGALIR_TEST_FILE`; fail closed when the
   private file is absent (no silent fallback to the 1,000-query set).
   `tests/unit/test_private_test_discovery.py` exists — confirm it covers the
   fail-closed path, not just the happy path.
3. **Submission validity** — exactly 2,080 IDs, each with exactly 5 unique,
   valid corpus document IDs; zero duplicates/missing/invalid. Validate shape
   against `private-official.json`, never `public-official.json`.
4. **Acceptance fixtures** — extend the acceptance-verifier fixtures to a
   2,080-query case plus rejection cases (1,999 IDs, 4 answers, duplicate doc,
   unknown doc ID). Regression-test both the 1,000- and 2,080-query paths.
5. **Forecast** — recompute the runtime forecast with the private count and
   show nominal/strict fit, using `05-runtime-reality.md` measured numbers.

## Out of scope (deliberately not attempted)

- Per-fold adapters (Option C in `01-CRITICAL-oof-leakage.md`).
- Cascade reranking (score 100 @256, re-score top-20 @512). Attractive on paper
  but a policy change needing full parity re-testing; revisit only after
  Tasks 1, 6, 7 are measured.
- Reverting `candidate_k` 200 → 150 to buy runtime margin. The funnel in `06`
  says the recall cost is near zero, but leave it alone until there is a real
  runtime problem — `05` shows there is not.

---

## ⚠️ Re-prioritized 2026-09-21 — score-first directive

The operator has stated runtime is not a constraint and score is the objective.
Consequences for this plan:

| Task | Old status | **New status** |
|---|---|---|
| 1, 2 (leakage fix + lock) | CRITICAL | **CRITICAL — unchanged.** Prerequisite: contaminated OOF selects the wrong config. |
| 6 (pairwise loss) | "optional-but-recommended" | **MANDATORY — highest score lever** (`08`, Tier 1.1) |
| 7 (length-grouped batching) | recommended | **DROPPED** — pure runtime optimization, no score value |
| 8 (private-round proofs) | required | **required — unchanged** |
| 3, 4 (docs) | low | low |
| 5 (re-qualify release) | required | **required — unchanged** |
| *new* | — | **`08` Tier 1.2 `r: 32`, Tier 2.4 `rerank_k: 200`, Tier 3 ensemble** |

Read `08-score-maximization.md` for the full ranked lever list and the
recommended configuration block.

**New runtime caveat:** the `08` configuration will exceed the 25,200s Modal
timeout (`rerank_k` 100→200 roughly doubles the dominant stage). Raise
`MODAL_TIMEOUT_SECONDS` **and** `LEGALIR_TIME_GATE_SECONDS` together before
launch, and confirm `src/release/acceptance.py` agrees — otherwise the job is
killed before delivering. Raising a timeout deliberately is not weakening a
gate; silently lowering a threshold to make a check pass would be.

## Execution order (important)

Tasks 1, 2, 6, 7 all touch `src/` or `configs/`. Do **not** run the release
re-qualification (Task 5) after each one — batch them into a single runtime
commit, then qualify once. Each Kaggle gate cycle costs a push + kernel run.

```
Phase A (correctness)  Task 1 -> Task 2        # leakage fix + regression lock  [BLOCKER]
Phase B (SCORE)        Task 6 + 08 Tier 1/2    # pairwise loss, r=32, rerank_k=200, max_chunks=4
Phase C (private)      Task 8                  # disjointness, 2080 validity, fixtures
Phase D (docs)         Task 3, Task 4          # HF card, runtime narrative
Phase E (release)      Task 5                  # ONE runtime commit + ONE Kaggle gate + release commit
```

Phase B is now the point of the exercise, not an optional extra. Phase A is
what makes Phase B *measurable* — without an honest OOF you cannot tell which
Phase B lever helped.

**Tune on fold 0 only** (~46 min per configuration) before committing to a full
5-fold run. Record Recall@5 plus Cand@50/150/200 for each variant; the table at
the end of `08` is the scoreboard to fill in.

Task 7 is dropped. Revisit only if a runtime ceiling reappears.

### Verified preconditions (checked 2026-09-21, release `2a0d17e`)

| Claim the plan relies on | Verified |
|---|---|
| `PairwiseLogisticLoss` exists and instantiates via `get_loss_function("pairwise_logistic")` | yes — `src/training/losses.py:26` |
| `QueryBalancedGroupSampler` / `RerankerGroupDataset` / `RerankerGroupCollator` exist | yes — `src/training/trainer.py:235 / :337 / :381` |
| Trainer dispatches the group path for pairwise/listwise | yes — `src/training/trainer.py:662-665` |
| Reranker uses dynamic padding (`padding=True`) | yes — `src/ranking/reranker.py:429` |
| No length-grouped batching exists yet | confirmed — no `argsort`/length-sort in the scoring path |
| `os` imported in trainer (warm-start config read) | yes — `src/training/trainer.py:6` |
| HF adapter geometry matches repo LoRA config | yes — r=8, alpha=16, dropout=0.05, targets `{query,key,value,dense}` |
