# Score maximization — runtime is NOT a constraint

**Operator directive (2026-09-21):** *"quan trọng là điểm số chứ training time
không sao đâu — tôi muốn up scale điểm số cao nhất có thể."*
Score is the objective. Runtime is free. This file supersedes the runtime
prioritization in `03` and `05`.

## The two facts that drive everything

**Fact 1 — the loss is in ranking, not retrieval.** Live fold 0:

```
Cand@150 = 98.89%   gold IS in the candidate pool
Recall@5 = 91.33%   gold reaches top-5
--------------------------------------------
7.56pp lost by the reranker   |   1.11pp never retrieved
```

**Fact 2 — 82.4% of the model budget is unused.** From `parameter_audit.json`:

| | Value |
|---|---:|
| Used | 702,754,049 (0.703B) |
| Limit | 4,000,000,000 (4.0B) |
| **Headroom** | **3,297,245,951 (3.30B)** |
| Utilization | **17.57%** |

The competition permits ~4.7× the model currently deployed. Capacity has never
been the binding constraint — and with runtime free, neither is compute.

## Ranked levers

Ordered by (expected gain × confidence). Tier 1 attacks the 7.56pp gap; that is
where the headroom actually is.

### TIER 1 — attack the ranking gap

**1. Pairwise / listwise loss — highest expected value, config-only**

`loss_type: "bce"` optimizes *"is this pair relevant?"* per pair, independently.
The metric asks *"is the gold above the other four?"* — an ordering question.
Direct objective/metric mismatch, and the most plausible single cause of a
7.56pp gap behind a 98.89% pool.

Machinery already present (verified this session, all instantiate):
- `PairwiseLogisticLoss` — `src/training/losses.py:26`
- `ListwiseCrossEntropyLoss` — `src/training/losses.py:86`
- `QueryBalancedGroupSampler` — `src/training/trainer.py:235`
- `RerankerGroupDataset` / `RerankerGroupCollator` — `:337` / `:381`
- Trainer auto-switches to the group path — `:662-665`

```yaml
loss_type: "pairwise_logistic"   # or "listwise_ce"
# pos_weight: 4.0                # BCE-only; inert under pairwise — remove
```

Try `pairwise_logistic` first (more stable); `listwise_ce` optimizes the full
ranked list and may go further.

**2. Raise LoRA capacity — the budget is 82% idle**

`r: 8` with `alpha: 16` gives 4,589,569 trainable parameters (0.80% of the
reranker). Scaling is linear in `r`:

| `r` | approx. trainable | total model | budget used |
|---:|---:|---:|---:|
| 8 (now) | 4.6M | 702.8M | 17.6% |
| 16 | ~9.2M | ~707M | 17.7% |
| **32** | **~18.4M** | **~716M** | **17.9%** |
| 64 | ~36.7M | ~735M | 18.4% |

Even `r=64` barely moves budget utilization. Recommended: **`r: 32`,
`lora_alpha: 64`** (keep the conventional `alpha = 2r` ratio). Rank 8 is a
*low-resource* default; nothing here is low-resource.

⚠️ Raising `r` changes adapter geometry, so the published warm-start adapter
(r=8) **will no longer load** — `PeftModel.from_pretrained` will fail and fall
back to clean init. Choose one: keep `r=8` to warm-start, or raise `r` and
cold-start the final model. With runtime free, **raising `r` and training from
scratch is the stronger play**; warm-start's value was mostly saved epochs.

**3. Train longer — runtime is free**

`max_steps: 250` is a floor; the coverage policy raises it to whatever full
query coverage requires (Run 4 final: 875 steps, 1 epoch-equivalent, 427s).
Training is only ~11% of runtime, so multiplying it is cheap in wall-clock:

```yaml
max_steps: 2625      # ~3 epoch-equivalents
```

Watch the fold reports for overfitting; more steps is not monotonically better.

### TIER 2 — remove truncation losses (bounded but near-certain)

**4. Rerank the entire candidate pool — `rerank_k: 200`**

Currently `candidate_k=200` but `rerank_k=100`: 100 retrieved candidates are
discarded unscored (assigned `-999.0`). With runtime free there is no reason to
truncate at all. Set `rerank_k` = `candidate_k` = 200 and the rerank-depth
truncation loss goes to **zero** — the ceiling becomes Cand@200.

Bounded gain: Cand@50 (98.05%) → Cand@150 (98.89%) is +0.84pp, so scoring
50→150→200 recovers roughly that order plus whatever Cand@200 adds over
Cand@150 (unmeasured — the run only reports @50 and @150).

**Add Cand@200 to the diagnostics** so this is measured rather than assumed.

**5. Widen the candidate pool — `candidate_k: 300–400`**

Ceiling gain is small (≤1.11pp remains unretrieved at k=150) but with free
runtime it is nearly free recall. Keep `rerank_k = candidate_k`.

### TIER 3 — ensemble (reliable classic gain, expensive → now affordable)

**6. Multi-seed reranker ensemble**

Train N adapters with different seeds / negative samples, average their scores
before fusion. Typically +0.5–1.5pp on ranking tasks. The budget permits even a
**second full cross-encoder** (567.8M + 702.8M = 1.27B, still 32% of 4.0B).

Cost: N× rerank compute. Free under this directive.

⚠️ This is a **scoring-policy change** — it must be applied identically to OOF,
disjoint, and private inference, with `tests/parity/` re-run to prove identical
ordering semantics. It cannot be tuned on private labels.

**7. Warm-start the FINAL model only** (if `r` stays 8)

Legitimate — the final model trains on all 7,000 queries anyway, so there is no
leakage. Effectively an extra epoch on the model that produces the submission.
Folds must stay cold (`01-CRITICAL-oof-leakage.md`). Mutually exclusive with
lever 2 at `r≠8`.

### TIER 4 — evidence and selection

**8. `max_chunks: 4–5`** — more clauses per candidate inside the 512-token cap.
Run 4 used 2, current is 3. Diminishing once the cap saturates; measure.

**9. Multi-gold aware selection** — scoring is `|gold ∩ pred| / |gold|`, so a
3-gold query returning 1 gold scores 0.33. **Verify the distribution first** —
the "7.9% multi-gold / 95.87% ceiling" figure quoted earlier in conversation was
never checked:

```python
import pandas as pd
q = pd.read_parquet("<dataset>/qrels_train.parquet")
n = q.groupby("query_id").size()
print(n.value_counts().sort_index())
print("multi-gold share:", (n > 1).mean())
print("ceiling if only 1 gold ever found:", (1/n).mean())
```

If material, the selector must allow 2–3 documents from the same statute into
the top-5 rather than diversifying them apart.

## Recommended configuration (single highest-value change set)

```yaml
loss_type: "pairwise_logistic"   # Tier 1.1 — attacks the 7.56pp gap
lora:
  r: 32                          # Tier 1.2 — budget is 82% idle
  lora_alpha: 64
lora_r: 32
lora_alpha: 64
max_steps: 2625                  # Tier 1.3 — ~3 epoch-equivalents
rerank_k: 200                    # Tier 2.4 — no truncation
max_chunks: 4                    # Tier 4.8
# pos_weight removed — BCE-only, inert under pairwise
```
with `candidate_k: 200` (already set) and warm-start **disabled** (r changed).

Expected runtime: substantially above the 6.4h projection in `05` —
`rerank_k` 100→200 alone roughly doubles the dominant stage. Under this
directive that is acceptable, **but it will exceed the 25,200s Modal timeout.**
Raise `MODAL_TIMEOUT_SECONDS` / `LEGALIR_TIME_GATE_SECONDS` together before
launching, or the run is killed before it delivers. Do not lower gates — raise
the timeout deliberately and re-verify the acceptance gate agrees.

## Why the leakage fix is still priority 0 — even under "score only"

Every lever above is a **choice between configurations**. Choosing requires a
trustworthy comparison signal. With warm-start leaking into the folds
(`01-CRITICAL-oof-leakage.md`), OOF measures *"how well did the model memorize
queries it already trained on"* — it will favour whichever config memorizes
best, which is the opposite of what generalizes to the 2,080 private queries.

**Contaminated OOF does not merely report a wrong number — it actively selects
the wrong configuration.** Fix Tasks 1–2 first, then tune. There are no labels
on the private set; OOF is the only instrument available.

## Measurement discipline

Change **one tier at a time** and record fold-0 Recall@5 plus Cand@50/150/200:

| Config | Fold-0 Recall@5 | Cand@50 | Cand@150 | Cand@200 | Notes |
|---|---|---|---|---|---|
| run 4 baseline | 91.09% | — | — | — | cand150/rerank50/384/chunks2 |
| current (live) | **91.33%** | 98.05% | 98.89% | — | cand200/rerank100/512/chunks3 |
| + pairwise loss | ? | | | | Tier 1.1 |
| + r=32 | ? | | | | Tier 1.2 |
| + rerank_k=200 | ? | | | | Tier 2.4 |

Fold 0 alone is a fast, honest proxy (~46 min) — no need to run all five folds
to reject a losing configuration.
