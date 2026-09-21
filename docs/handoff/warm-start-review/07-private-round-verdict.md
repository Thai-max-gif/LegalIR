# Private-round readiness — executed verification and verdict

Executed 2026-09-21 on release `2a0d17e`. **No `modal run`, no FULL training,
no gate weakened.** Every row below was run, not read.

## VERDICT: **BLOCKED**

**Top blocker:** warm-start leaks into OOF and document-disjoint evaluation
(`01-CRITICAL-oof-leakage.md`). The private-round *mechanics* are sound; the
*metric used to judge the submission* is not.

| Requirement | Result | Evidence |
|---|---|---|
| 1. Private-round configuration | **PASS** | 7/7 discovery cases |
| 2. Submission format & validity | **PASS** (1 latent gap) | 7/8 cases; gap is non-active |
| 3. Time scaledown | **WAIVED** by operator directive | see §3 — runtime is no longer a constraint |
| 4. Recall policy parity | **PASS** | parity by construction + 59 tests |
| 5. Proof | **PASS** | 2080 fixtures exist and are green |

---

## 1. Private-round configuration — PASS

Executed against `discover_public_test_file`, private file absent then present:

| Case | Result |
|---|---|
| default (no env) → public | PASS |
| `LEGALIR_TEST_PHASE=private`, private file **missing** → **raises** | PASS (fail-closed) |
| `LEGALIR_TEST_PHASE=PRIVATE` (case-insensitive) → raises | PASS |
| `LEGALIR_TEST_PHASE=public` → public | PASS |
| `LEGALIR_TEST_PHASE=private`, file present → private | PASS |
| default still public after private exists (no drift) | PASS |
| `LEGALIR_TEST_FILE` explicit override | PASS |

**7/7.** Critically, the fail-closed case was tested *with
`public-official.json` present* — it still raised rather than silently falling
back. That is the exact trap that produced the 1,000-query submission.

Structural isolation of private data from training:
- `private-official.json` is gitignored (`.gitignore:63`).
- Training loads only `queries_train.parquet` / `qrels_train.parquet`
  (`kaggle_train.py:951,956,978`).
- No `private` reference exists in `src/training/build_pairs.py` or
  `src/data/splits.py` — private IDs cannot structurally reach pair mining or
  split construction.

**Residual gap (not closed):** a runtime assertion that the three ID sets
(train 7,000 / public 1,000 / private 2,080) are pairwise disjoint on the real
dataset was **not executed** — the canonical dataset is not present locally.
`tests/dataset/test_dataset_integrity.py:63` asserts the 2,080 count and runs
in CI where the data exists. Run it against the real dataset before launch.

---

## 2. Submission format and validity — PASS, with one latent gap

`validate_submission` against a synthetic 2,080-query private set:

| Case | Expected | Got | |
|---|---|---|---|
| valid 2080, 5 unique valid docs | valid | valid | PASS |
| missing 1 query (2079) | reject | reject | PASS |
| extra unknown query (2081) | reject | reject | PASS |
| **only 4 answers** | **reject** | **accepted** | **FAIL** |
| 6 answers | reject | reject | PASS |
| duplicate doc in answer | reject | reject | PASS |
| doc id not in corpus | reject | reject | PASS |
| regression: public 1,000 path | valid | valid | PASS |

### The gap: `1 <= len(answer) <= 5` should be `== 5`

`src/evaluation/submission.py:101` accepts any length 1–5.

Official scorer (`Scoring-Program-Task-LegalIR/scoring.py:31`):

```python
recall_q = |gold ∩ pred| / |gold|   if 0 < len(pred) <= 5   else 0
```

So a 4-document answer is **legal but strictly suboptimal** — it forfeits a
free slot, and recall can only rise (never fall) by adding a 5th document.
More than 5 scores **zero**. A length mismatch between prediction and
reference **raises**, which is precisely the Image-#8 server crash.

**Is it active?** No. The real Run-4 submission was checked directly:

```
queries in run-4 submission : 2080
answer-length distribution  : {5: 2080}
answers with duplicate docs : 0
```

So the pipeline does emit exactly 5. The validator gap is a
**defense-in-depth weakness**: a future regression that emitted 3 or 4
documents would pass local validation and silently bleed recall.

**Fix (small, safe):** tighten to `len(answer) == 5` for official phases, or
add a warning path. Do not loosen anything else.

---

## 3. Time scaledown — WAIVED (operator directive, 2026-09-21)

> *"quan trọng là điểm số chứ training time không sao đâu"* — score is the
> objective, runtime is not a constraint.

The 18,000s / 270-minute target is therefore **no longer binding**, and the
`08-score-maximization.md` configuration deliberately spends more runtime to
buy recall. The measurement below is retained as the baseline forecast.

**Action this creates:** raise `MODAL_TIMEOUT_SECONDS` and
`LEGALIR_TIME_GATE_SECONDS` together before launching the `08` configuration —
`rerank_k` 100→200 roughly doubles the dominant stage and will exceed 25,200s.
Raising a timeout on purpose is a deliberate budget decision; it is not the
same as weakening a correctness gate, and `src/release/acceptance.py` must be
re-checked so it agrees with the new ceiling.

### Baseline forecast (retained for reference)

Measured basis in `05-runtime-reality.md`. Projection from live fold 0
(2,758.2s) × 6 jobs plus measured non-OOF stages:

| Final-inference assumption | Total | vs 25,200s gate | vs 18,000s strict target |
|---|---:|---|---|
| ×1.23 (scales like folds) | 22,957s (6.38h) | +2,243s — **fits** | **+4,957s over** |
| ×1.60 | 23,698s (6.58h) | +1,502s — fits | over |
| ×2.00 | 24,499s (6.81h) | +701s — fits | over |

**Honest statement of the tension:** the run fits the **current** 25,200s gate
(raised to 7h at the user's explicit request) but does **not** meet the older
18,000s / 270-minute target. Meeting 18,000s would require giving back recall
depth (`rerank_k` 100 → 50) or landing Task 7 (length-grouped batching,
20–35% off rerank work — enough to reach ~18–19k s while keeping depth).

Do **not** lower the timeout to force the number.

---

## 4. Recall policy parity — PASS

`_OOF_FIXED_RRF = ReciprocalRankFusion()` (`oof_runner.py:55`) is shared by the
OOF folds (`:713`) and the document-disjoint pass (`:1395`). Final inference
default-constructs the same class with no arguments
(`predict.py:41`, `:688`) — identical parameters, therefore identical ordering
semantics by construction.

`tests/parity/` plus `tests/unit/test_disjoint_policy_parity.py`: **59 passed**.

Any change to the selector or loss (Task 6) is a policy change and must re-run
this suite.

---

## 5. Proof — PASS

```
tests/unit/test_private_test_discovery.py
tests/release/test_acceptance_verifier.py      <- includes a 2080-query case (:313)
tests/unit/test_submission_compliance.py
tests/unit/test_codabench_compat.py
tests/parity/
tests/unit/test_disjoint_policy_parity.py
-> 59 passed in 20.81s
```

Full pre-push gate (9 suites, parameter budget, notebook drift, fallback
detection, tree hygiene): **all PASSED** earlier this session.
Strict approval on HEAD `2a0d17e`: **PASS**.
Kaggle dual-T4 bound to runtime `180b15e`: **PASS**, Δw = 278.80, 2× Tesla T4.

---

## What must happen before READY

1. Fix the leakage (Tasks 1–2). Until then no OOF number is trustworthy.
2. Tighten the answer-length rule to exactly 5 (Task 8.3 — small).
3. Run `tests/dataset/test_dataset_integrity.py` against the **real** dataset
   to close the disjointness gap.
4. ~~Decide the 18,000s question~~ — **waived**. Instead: raise the Modal
   timeout and the acceptance time gate together to fit the `08` configuration,
   and confirm both agree before launch.
5. Apply `08-score-maximization.md` Tier 1–2 and tune on fold 0.
