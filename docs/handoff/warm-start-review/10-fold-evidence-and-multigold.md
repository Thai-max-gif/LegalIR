# Live fold evidence + the multi-gold discovery

Two folds of the in-flight run are now observed. Fold 1 carries more
information than fold 0 and surfaces a lever that was previously only asserted.

## Observed folds

```
Fold 0: Recall@5 91.33% | Prec@5 19.51% | MRR 0.7800 | nDCG@5 0.8046 | Cand@50 98.05% | Cand@150 98.89%  (2758.2s)
Fold 1: Recall@5 90.90% | Prec@5 19.29% | MRR 0.7589 | nDCG@5 0.7905 | Cand@50 97.20% | Cand@150 98.56%  (2470.1s)
```

### Trend vs Run 4 — better than fold 0 alone suggested

| Fold | Run 4 | Live | Δ | Time |
|---|---:|---:|---:|---:|
| 0 | 91.09% | 91.33% | **+0.24pp** | 2,758s |
| 1 | 90.07% | 90.90% | **+0.83pp** | 2,470s |
| mean | — | — | **+0.54pp** | 2,614s |

**Projected pooled OOF ≈ 90.15%** (baseline 89.61%). Better than the +0.24pp
fold-0 reading implied, still far from 0.96.

Fold 1 was also *faster* (2,470s), so the 6-job runtime estimate in `05` is
slightly conservative. Runtime remains a non-issue under the score-first directive.

## Funnel per fold — two separate losses

| Fold | Cand@50 | Cand@150 | R@5 | depth gap (50→150) | ordering loss | never retrieved |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 98.05% | 98.89% | 91.33% | **+0.84pp** | **7.56pp** | 1.11pp |
| 1 | 97.20% | 98.56% | 90.90% | **+1.36pp** | **7.66pp** | 1.44pp |

Two independent, separately-addressable losses:

1. **Ordering loss ~7.6pp** (consistent across folds) — the cross-encoder ranks
   an available gold below position 5. Addressed by `08` Tier 1
   (pairwise loss, higher LoRA rank, longer training).
2. **Depth gap 0.84–1.36pp** — candidates between rank 50 and 150 that
   `rerank_k` may never score. Fold 1's gap is **62% larger** than fold 0's,
   so this is fold-dependent and worse than fold 0 alone suggested. Addressed
   by `08` Tier 2 (`rerank_k` = `candidate_k` = 200).

**Note the blind spot:** the run reports Cand@50 and Cand@150 but the config
sets `candidate_k=200, rerank_k=100`. Neither reported depth matches either
configured depth. **Cand@100 and Cand@200 must be added to the diagnostics** —
without them the benefit of `rerank_k: 200` cannot be measured, only assumed.

## The multi-gold discovery — derived, not assumed

Earlier in this project a "7.9% multi-gold / 95.87% ceiling" figure was quoted
but never verified. The reported metrics now let us **prove multi-gold queries
exist and are being partially missed**, using only Prec@5 and Recall@5.

### The identity

```
avg |gold ∩ pred|  =  5 × Prec@5
avg recall         =  mean( |gold ∩ pred| / |gold| )
```

If **every** query had exactly one gold document, `|gold| = 1` and these two
quantities are **identical by definition**. They are not:

| Fold | 5 × Prec@5 | Recall@5 | gap |
|---|---:|---:|---:|
| 0 | 0.9755 | 0.9133 | **+0.0622** |
| 1 | 0.9645 | 0.9090 | **+0.0555** |

A strictly positive gap is only possible when queries with `|gold| > 1` are
found *partially* — the intersection counts each hit fully while recall divides
by the larger denominator. **This is arithmetic proof, not an assumption.**

### How much is recoverable

Model: fraction `p` of queries have 2 golds, the rest have 1; multi-gold
queries find `E[k]` of their 2 golds.

```
gap       = p · E[k] / 2
max gain  = p − gap          (achieved if every multi-gold query finds BOTH)
```

| If p = | fold 0 max gain | fold 1 max gain |
|---:|---:|---:|
| 7.9% | +1.68pp | +2.35pp |
| 10% | +3.78pp | +4.45pp |
| 12% | +5.78pp | +6.45pp |
| 15% | +8.78pp | +9.45pp |

At the previously-quoted 7.9%, **~+2pp is available from multi-gold handling
alone — with no reranker improvement whatsoever.** That is roughly four times
the +0.54pp all the current config changes delivered combined.

`p` must be > the gap for any headroom to exist (p > 5.6%), and the true value
requires `qrels_train.parquet`:

```python
import pandas as pd
q = pd.read_parquet("<dataset>/qrels_train.parquet")
n = q.groupby("query_id").size()
print(n.value_counts().sort_index())
print("multi-gold share p :", (n > 1).mean())
print("mean golds per query:", n.mean())
print("hard ceiling if only 1 gold is ever returned:", (1/n).mean())
```

**Run this first.** It is a one-line diagnostic that sizes the single largest
identified opportunity, and it needs no GPU and no code change.

### Why the selector may be discarding the second gold

Multi-gold documents for one query are frequently sibling articles of the
**same statute**. Any diversity/dedup behaviour in top-K selection that spreads
results across distinct documents will actively push the second gold out of the
top-5 — optimizing for apparent variety while destroying recall on exactly
these queries.

Audit `src/ranking/selector.py` and `src/task1/selector.py` for per-document
caps, MMR-style diversity, or same-parent suppression. If present, that is a
direct, cheap fix with no model retraining.

## Revised lever ranking

| Rank | Lever | Evidence | Expected |
|---|---|---|---|
| 1 | Pairwise/listwise loss (`08` T1.1) | 7.6pp ordering loss, both folds | largest, uncertain |
| 2 | **Multi-gold selection** | proven by Prec/Recall gap | **~+2pp at p=7.9%**, verify p first |
| 3 | LoRA `r: 32` (`08` T1.2) | 82.4% budget idle | moderate |
| 4 | `rerank_k: 200` (`08` T2.4) | depth gap up to 1.36pp | bounded, near-certain |
| 5 | Longer training (`08` T1.3) | runtime free | moderate |

Multi-gold moves **up to #2**: it is the only lever with an arithmetic proof of
existence and a closed-form bound, and it requires no retraining.
