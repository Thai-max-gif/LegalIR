# Where the recall actually goes — the reranker, not the retriever

This is the most important strategic finding in the handoff. It says the
optimization effort so far has been aimed at the wrong stage.

## The measurement

Live fold 0, current config:

```
Recall@5 = 91.33% | Cand@50 = 98.05% | Cand@150 = 98.89%
```

Read it as a funnel:

| Stage | Value | Meaning |
|---|---:|---|
| Gold document is in the candidate pool | **98.89%** | Retrieval did its job |
| Gold document survives into top-5 | **91.33%** | Reranker ordering |
| **Lost by the reranker** | **7.56pp** | Gold was available and got ranked ≥6 |
| Never retrieved at all | 1.11pp | The only part more retrieval can fix |

**87% of the remaining error is a ranking-quality problem.**
The gold document is already sitting in the candidate list; the cross-encoder
puts it below position 5.

## What this means for the changes already made

| Change | Intended effect | What the funnel says |
|---|---|---|
| `candidate_k` 150 → 200 | "raise the ceiling to 99.2%" | Ceiling was already 98.89% at k=150. Upside ≤1.11pp, realistically ~0.1–0.3pp. **Near-worthless.** |
| `rerank_k` 50 → 100 | "recover 1.82% dropped at ranks 51–100" | Real but bounded: Cand@50 98.05% → Cand@150 98.89%, so deepening 50→100 recovers **at most ~0.84pp** of pool coverage, and only if the reranker then ranks those items top-5. |
| `max_length` 384 → 512, `max_chunks` 2 → 3 | fuller statute context | Plausibly helps ranking quality — this one targets the right stage. |
| `pos_weight` 4.0 | counter negative skew | Targets the right stage. |

Measured combined effect of all four: **+0.24pp** (fold 0: 91.09% → 91.33%).
Extrapolated pooled OOF ≈ **89.85%**, against an 89.61% baseline.

**This does not reach 0.96. It is not close.** Any earlier statement from me
implying these settings would deliver >0.96 was unfounded — retract it.

## Closing a 7.56pp ranking gap

The gap is the cross-encoder's ordering ability. The levers that act on it,
best first:

### 1. Switch the training objective from pointwise to pairwise (HIGHEST VALUE)

`loss_type: "bce"` optimizes *"is this pair relevant?"* independently per pair.
The metric rewards *"is the gold above the other four?"* — a ranking question.
This is a direct objective/metric mismatch and it is the most plausible single
cause of a 7.56pp gap sitting behind a 98.89% pool.

**The machinery already exists — this is a config change, not new code:**

- `src/training/losses.py:26` `PairwiseLogisticLoss`
- `src/training/losses.py:58` `PairwiseMarginRankingLoss`
- `src/training/losses.py:86` `ListwiseCrossEntropyLoss`
- `src/training/trainer.py:663-665` already builds `RerankerGroupDataset` (defined :337) +
  `RerankerGroupCollator` + `QueryBalancedGroupSampler` when
  `loss_type in ("listwise","listwise_ce","pairwise_logistic","pairwise_margin")`
- `src/training/trainer.py:795` already branches the loss call for pairwise types

Set `loss_type: "pairwise_logistic"` in
`configs/experiments/reranker_lora.yaml`. Note `pos_weight` is a BCE-only knob
and becomes inert under a pairwise loss — remove it or document it as unused.

**This is a scoring-policy change.** Per the private-round constraints it must
be applied identically to OOF, disjoint, and final inference, and needs the
parity tests re-run. It cannot be A/B-ed on private labels.

### 2. Warm-start the FINAL model only (safe, and what the user asked for)

The final model legitimately trains on all 7,000 queries, so warm-starting it
from the published adapter is a genuine second epoch with no leakage. It is
the model that produces the submission. Folds must stay cold — see
`01-CRITICAL-oof-leakage.md`.

Expected: a better submission than the OOF number predicts. OOF becomes a
conservative lower bound, which is the safe direction.

### 3. Multi-gold queries

Scoring is `|gold ∩ pred| / |gold|`. A query with 3 gold documents where the
system finds 1 scores 0.33 even with a perfect top-1.

**Unverified in this session** — the "7.9% multi-gold / 95.87% ceiling" figure
quoted earlier in the conversation was never checked against the dataset. Before
building anything on it, compute from `qrels_train.parquet`:

```python
import pandas as pd
q = pd.read_parquet("<dataset>/qrels_train.parquet")
n = q.groupby("query_id").size()
print(n.value_counts().sort_index())
print("multi-gold share:", (n > 1).mean())
print("ceiling if only 1 gold ever retrieved:", (1/n).mean())
```

If the multi-gold share is material, the selector must be able to place 2–3
gold documents from the *same* statute into the top-5, rather than diversifying
them out.

### 4. Hard-negative mining quality

Negatives currently come from the candidate pool. Negatives mined from ranks
6–20 of the *current* model (the ones it actually confuses) teach more than
random pool negatives. Costly — `final_pair_mining` is already 674s.

## Recommended order

1. `loss_type: "pairwise_logistic"` — biggest expected move, config-level.
2. Final-model-only warm-start — safe, uses the published adapter as intended.
3. Verify multi-gold distribution; fix the selector only if the data warrants.
4. Consider reverting `candidate_k` 200 → 150 to buy runtime margin back; the
   funnel says it costs almost nothing in recall.
