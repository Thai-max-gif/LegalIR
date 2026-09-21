# FINDING 2 + claims audit — every performance claim re-checked against source

Each row was verified by reading the code at the cited location on release
`2a0d17e`. "Runtime measured" is **no** for every row: no A100 run has been
executed against this commit. Recall/time effects below are mechanism-level
reasoning, not measurements.

## Recall-affecting settings — VERIFIED PRESENT

| Setting | Value | Verified at | Effect (unmeasured) |
|---|---|---|---|
| `candidate_k` | 200 (full runs) | `src/pipeline/kaggle_train.py:1829` | Raises candidate ceiling above the 98.82% seen at k=150 |
| `rerank_k` | 100 | `configs/experiments/reranker_lora.yaml:24`, consumed `kaggle_train.py:1830` | Gold docs at ranks 51–100 are now scored instead of discarded |
| `max_length` | 512 | `configs/experiments/reranker_lora.yaml:28` | Long statutes no longer truncated mid-article |
| `max_chunks` | 3 | `src/pipeline/oof_runner.py:252` | More clauses per candidate document |
| `pos_weight` | 4.0 | `configs/experiments/reranker_lora.yaml:5` | Upweights positives against ~1:10 negative skew |
| `inference_batch_size` | 64 | config `:23`; consumed `kaggle_train.py:1844`, `:2139`, `oof_runner.py:174` | Better A100 utilization during scoring |

These are real. The recall argument for them stands on its own and does **not**
depend on warm-start.

## Timeout — VERIFIED PRESENT

| Constant | Value | Location |
|---|---|---|
| `TIMEOUT_SECONDS` | 25200 (7h) | `scripts/modal/run_modal_a100.py:59` |
| `STRICT_GATE_SECONDS` | 25200 | `src/pipeline/kaggle_train.py:126` |
| `TIME_GATE_SECONDS` | 25200 | `src/release/acceptance.py:23` |

All three agree, so the acceptance gate will not reject a run that the Modal
timeout permits. This was a real inconsistency risk and it is correctly wired.

## FINDING 2 — FALSE CLAIM: "warm-start reduces training time / step count"

**This claim is wrong. Retract it.**

`src/training/train_reranker.py:116-130`:

```python
req_steps = compute_coverage_required_steps(
    eligible_query_count=n_eligible,
    batch_size=batch_sz,
    gradient_accumulation_steps=grad_acc,
    target_coverage_pct=1.0,
    require_pos_and_neg=True,
)

if enforce_full_coverage_steps:
    cfg_steps = int(cfg.get("max_steps", 500))
    requested = max_steps or 0
    effective_steps = max(cfg_steps, requested, req_steps)
```

The optimizer-step count is **pinned by query-coverage policy**, not by
convergence. `enforce_full_coverage_steps` is `True` for folds
(`not self.smoke`) and for the final model (`is_full`). Warm-start changes the
starting weights; it changes **zero** steps. Wall-clock training time is
therefore unchanged.

Warm-start may still improve *quality at a fixed step budget* (better loss at
the same step count) — but that benefit is exactly what is unmeasurable right
now because the metric that would show it is the one being leaked into.

**Honest runtime story for this release:** the only defensible runtime levers
are `inference_batch_size: 64` (scoring throughput) and static branch-cache
reuse. Both are real but **unmeasured on A100**. Do not promise a specific
runtime reduction until a run produces `stage_timings`.

## FINDING 3 — HF model card lists wrong target modules

`README.md` on `dangphuc2109/legalir-task1-reranker` states:

> **Target Modules**: `q_proj`, `v_proj`

The published `adapter_config.json` actually contains:

```json
"target_modules": ["value", "key", "query", "dense"],
"r": 8, "lora_alpha": 16, "lora_dropout": 0.05,
"modules_to_save": ["classifier", "score"]
```

Low severity but public-facing and reproducibility-relevant — a judge copying
the card cannot rebuild the adapter. Fix the card text.

Note the good news: the adapter's `r`/`alpha`/`dropout`/`target_modules`
**match** `configs/experiments/reranker_lora.yaml` exactly, so there is no
geometry mismatch on load. Warm-start loading itself is mechanically sound
(verified live: 4,589,569 trainable params, forward pass returns a finite logit).

## Non-issues checked and cleared

- `os` is imported at module level (`src/training/trainer.py:6`) — the
  `os.environ.get` call in the trainer does not raise `NameError`.
- `str(None).strip().lower()` yields `"none"`, which is not in the
  `("auto","latest","true","1")` tuple, so a `None` config value is handled.
  Sloppy but not a bug.
- The mock guard (`hidden_size >= 256`) correctly prevents tiny test BERTs from
  attempting to load a 1024-dim adapter.
