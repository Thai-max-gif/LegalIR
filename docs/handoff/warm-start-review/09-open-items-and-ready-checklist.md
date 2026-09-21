# Open items, scope boundary, and the READY checklist

## Scope boundary (recorded deliberately)

The session's standing goal asks for the pipeline to be **reconfigured and
re-implemented**. The operator then instructed, on 2026-09-21:

> *"chỉ tiếp tục sửa docs và summary đừng sửa code"* — docs and summary only,
> do not modify code.

**No code was modified.** `git status` shows only `docs/handoff/` untracked;
`src/`, `configs/`, `scripts/`, `tests/` are byte-identical to release
`2a0d17e`, and strict approval still passes on HEAD.

Consequence: every item below marked **BLOCKED-ON-CODE** is fully specified
here but deliberately not implemented. A later agent (or a later instruction)
can execute them directly from `03-implementation-plan.md` and
`08-score-maximization.md`.

---

## Status of the five private-round requirements

| # | Requirement | Status | What remains |
|---|---|---|---|
| 1 | Private-round configuration | **PASS**, one residual | Disjointness on the *real* dataset — needs data, not code |
| 2 | Submission format & validity | **PASS**, one latent gap | `len == 5` tightening — BLOCKED-ON-CODE |
| 3 | Time scaledown < 18,000s | **WAIVED** by operator | Raise timeouts for the `08` config instead |
| 4 | Recall policy parity | **PASS** | Re-run parity after any `08` change |
| 5 | Proof | **PASS** | Re-run after Phase A/B land |

**Overall verdict: BLOCKED.** Top blocker unchanged — warm-start leaks into
OOF and document-disjoint evaluation (`01-CRITICAL-oof-leakage.md`).

---

## What was actually proven (executed this session)

| Proof | Result |
|---|---|
| Discovery: public default, private only via env, fail-closed **with public present** | 7/7 PASS |
| Submission validator: 2080 valid / 2079 / 2081 / 6-answers / duplicate / unknown-doc | 7/8 (the 8th is the `len==5` gap) |
| **Real Run-4 submission**: 2,080 queries, answer lengths, duplicates | `{5: 2080}`, 0 duplicates — exactly 5 everywhere |
| Parity: shared fixed RRF across OOF / disjoint / inference | by construction + 59 tests PASS |
| 2080 acceptance fixture exists and is green | `tests/release/test_acceptance_verifier.py:313` |
| Full pre-push gate (9 suites, budget, drift, fallbacks, hygiene) | ALL PASS |
| Strict approval on HEAD `2a0d17e` | PASS |
| Kaggle dual-T4 bound to runtime `180b15e` | PASS, Δw = 278.80, 2× Tesla T4 |

### Precision note on requirement 2

The real 2,080-query submission was verified for **count, answer length, and
duplicates**. It was **not** verified for *"every doc ID exists in the corpus"*
— the canonical corpus is not present locally. That check runs inside
`validate_submission(corpus_doc_ids=...)` during the pipeline and is covered by
`tests/unit/test_submission_compliance.py`; Run 4's manifest independently
reports `all_ids_valid: true`. Close it directly with the snippet below.

---

## The two gaps that need the real dataset (no code change)

Run both where the canonical dataset exists (Modal/Kaggle container, or locally
after `scripts/colab/bootstrap.py` fetches it).

### Gap A — pairwise disjointness of train / public / private ID sets

```python
import json, pandas as pd
from pathlib import Path
D = Path("<canonical_dataset_dir>")

train = set(pd.read_parquet(D/"queries_train.parquet")["query_id"].astype(str))
pub   = set(json.loads((D/"public-official.json").read_text()))
priv  = set(json.loads((D/"private-official.json").read_text()))

print("counts  train/public/private:", len(train), len(pub), len(priv))
assert len(train) == 7000 and len(pub) == 1000 and len(priv) == 2080
for a, b, na, nb in ((train,pub,"train","public"),
                     (train,priv,"train","private"),
                     (pub,priv,"public","private")):
    ov = a & b
    print(f"  {na} ∩ {nb}: {len(ov)}")
    assert not ov, f"LEAKAGE: {na}/{nb} overlap {sorted(ov)[:5]}"

qrels = set(pd.read_parquet(D/"qrels_train.parquet")["query_id"].astype(str))
assert not (qrels & priv), "LEAKAGE: private IDs present in qrels"
assert not (qrels & pub),  "LEAKAGE: public IDs present in qrels"
print("DISJOINTNESS: PASS")
```

Also run the existing CI test, which asserts the 2,080 count against real data:

```bash
./.venv/bin/pytest tests/dataset/test_dataset_integrity.py -v
```

### Gap B — corpus-membership of every submitted doc ID

```python
import json, pandas as pd
from pathlib import Path
D = Path("<canonical_dataset_dir>")
docs = set(pd.read_parquet(D/"documents.parquet")["doc_id"].astype(str))
sub  = json.loads(Path("<submission.json>").read_text())

bad = {q: [d for d in v["answer"] if d not in docs] for q, v in sub.items()}
bad = {q: v for q, v in bad.items() if v}
lens = {len(v["answer"]) for v in sub.values()}
print("queries:", len(sub), "| answer lengths:", lens, "| unknown-doc queries:", len(bad))
assert len(sub) == 2080 and lens == {5} and not bad
print("SUBMISSION VALIDITY: PASS")
```

---

## BLOCKED-ON-CODE items (specified, not implemented)

| Item | Where specified | Why it matters |
|---|---|---|
| Warm-start opt-in gate (`allow_warm_start`, fail-closed) | `03` Task 1 | **The blocker.** Contaminated OOF selects the wrong config |
| Regression test locking fold/disjoint isolation | `03` Task 2 | No test currently guards adapter provenance |
| Tighten `len(answer) == 5` | `03` Task 8.3, `07` §2 | Validator accepts 1–5; a regression emitting 4 would pass silently and bleed recall |
| `loss_type: "pairwise_logistic"` | `08` Tier 1.1 | Highest score lever — objective/metric mismatch |
| `lora r: 32, alpha: 64` | `08` Tier 1.2 | 82.4% of the 4B budget is idle |
| `rerank_k: 200` | `08` Tier 2.4 | 100 retrieved candidates are discarded unscored |
| Add **Cand@200** to fold diagnostics | `08` Tier 2.4 | Run reports only @50 and @150, so `rerank_k=200`'s benefit is unmeasurable |
| Raise `MODAL_TIMEOUT_SECONDS` + `LEGALIR_TIME_GATE_SECONDS` together | `07` §3, `08` | The `08` config exceeds 25,200s; without this the job is killed pre-delivery |
| HF model card `target_modules` correction | `03` Task 3 | Card says `q_proj/v_proj`; adapter is `query/key/value/dense` |

---

## READY checklist

READY requires every box ticked. Today: **boxes 1–3 and 5–6 are open.**

- [ ] 1. Warm-start gated; fold + disjoint provably cold-start (`03` Tasks 1–2)
- [ ] 2. Regression test red-green verified (fails before fix, passes after)
- [ ] 3. Validator enforces exactly 5 for official phases
- [ ] 4. Gap A disjointness proof green on the real dataset
- [ ] 5. Gap B corpus-membership proof green on the produced submission
- [ ] 6. `08` Tier 1–2 applied and tuned on fold 0, with Cand@200 reported
- [ ] 7. `tests/parity/` re-run green after every policy change
- [ ] 8. Timeouts raised consistently and acceptance gate agrees
- [ ] 9. Full pre-push gate green (`scripts/verify_prepush.py`)
- [ ] 10. New Kaggle dual-T4 gate bound to the new runtime SHA, `verdict: PASS`, Δw > 0
- [ ] 11. Freeze tuple refreshed; notebooks zero-drift; release commit is evidence-only
- [ ] 12. `verify_release_approval.py` + `colab/bootstrap.py` PASS on HEAD
- [ ] 13. Explicit operator authorization for Modal spend

Only then dispatch:

```bash
bash scripts/modal/run_modal_cli.sh --detach --hf-allow-public-repo --private
```

---

## One thing to check on the run in flight

Fold 0 improved only **+0.24pp** (91.09% → 91.33%) — too small for a model that
had supposedly pre-trained on its own evaluation queries. Grep the live log:

```bash
grep -E "Warm-start: loading existing LoRA|Failed loading warm-start" <logfile>
```

- **Neither line** → the run predates `180b15e`; its OOF is **honest and usable**.
- **First line present** → OOF is contaminated; discard the number, do not tune on it.
