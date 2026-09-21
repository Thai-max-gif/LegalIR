# Road to 0.96 — the arithmetic, the levers, and an honest probability

Docs-only analysis. No code changed. Baseline is the **measured** live run
(folds 0 and 1), not a projection from assumptions.

## What 0.96 actually demands

| Quantity | Value |
|---|---:|
| Current projected pooled OOF | **90.15%** |
| Candidate-pool ceiling (Cand@150, avg folds 0–1) | **98.72%** |
| Target | **96.00%** |
| Gap to close | **+5.85pp** |

Restated as a conversion problem — of the golds that *are* in the pool, how many
reach top-5:

| | Now | Needed for 0.96 |
|---|---:|---:|
| Pool→top-5 conversion | 91.31% | **97.24%** |
| Reranker error | 8.57pp | **2.72pp** |

**The reranker's error must shrink by 68%.** That is the whole problem stated in
one number. No amount of extra retrieval fixes it — only ~1.3pp of gold is
missing from the pool at all.

## The lever budget

| Lever | Gain (low–high) | Confidence | Why |
|---|---:|---|---|
| **Multi-gold selection fix** | +1.7 … +2.4pp | **high** | Arithmetically proven (`10`): `5×Prec@5 ≠ Recall@5` |
| **rerank_k 100 → 200** | +0.5 … +1.1pp | **high** | Bounded by measured depth gap (0.84 / 1.36pp) |
| Pairwise / listwise loss | +1.0 … +3.0pp | medium | Objective/metric mismatch; standard IR result |
| LoRA r=32 + longer training | +0.5 … +1.5pp | medium | 82.4% of the 4B budget idle |
| Multi-seed ensemble | +0.5 … +1.5pp | med-high | Classic reliable gain; budget allows a 2nd cross-encoder |
| candidate_k 300–400 | +0.2 … +0.5pp | high but small | Raises ceiling to ~99.3% |
| **Fine-tune the DEk21 bi-encoder** | +0.5 … +2.0pp | medium | **Currently unused lever** — see below |
| **Naive sum** | **+4.9 … +12.0pp** | | |

### Gains do not stack linearly

A better reranker already captures part of what deeper reranking would have
given; an ensemble overlaps with a bigger LoRA rank. Applying the usual overlap
discount:

| Stacking assumption | Resulting OOF | Reaches 0.96? |
|---|---|---|
| ×1.00 (no overlap — unrealistic) | 95.05% … 102.15% | only at the high end |
| **×0.65 (typical)** | **93.34% … 97.95%** | **only at the high end** |
| ×0.50 (heavy overlap) | 92.60% … 96.15% | only at the very top |

## Honest verdict on 0.96

**Reachable, but not likely from the current trajectory.** Under the realistic
×0.65 stacking assumption the expected landing zone is **93–95%**, with 96%
requiring nearly every lever to land at the top of its range simultaneously.

Anyone promising 0.96 as a consequence of a config edit is guessing. The
measured evidence says the current config set bought **+0.54pp**, and the
remaining gap is 10× that.

**What raises the odds most:**
1. The two **high-confidence** levers (multi-gold, full-pool reranking) are
   worth +2.2 … +3.5pp between them and are the cheapest to land.
2. The bi-encoder is untouched — the only *structural* lever left.
3. Honest OOF (the leakage fix) is what lets you tell which lever worked. With
   contaminated OOF you will stack changes blind and likely pick a config that
   memorizes rather than generalizes.

## The unused structural lever: fine-tune the dense retriever

Everything attempted so far tunes the **reranker**. `CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2`
(135.0M params) is used **off-the-shelf, never fine-tuned** on the 7,000
training queries. This is the largest untouched piece of the system.

Fine-tuning it with contrastive / MultipleNegativesRanking loss on
`(query, gold_chunk)` pairs would act on **both** losses at once:
- raises Cand@k (shrinks the 1.3pp never-retrieved)
- improves the dense branch's RRF ranks, so golds enter fusion higher and the
  cross-encoder starts from a better ordering

Budget: 702.75M used of 4.0B. Adding a fine-tuned bi-encoder costs **no extra
parameters** (same model, different weights). Runtime is free under the current
directive.

Caveats: it is a genuine training pipeline addition (new code, new gate cycle),
and it **must** respect fold isolation — a bi-encoder trained on all 7,000
queries and then used inside OOF folds reproduces exactly the leakage defect in
`01`, one layer deeper.

## Recommended sequence (highest expected value per unit of risk)

```
STEP 0  Verify p = multi-gold share from qrels_train.parquet     [1 line, no GPU]
        -> sizes the single largest proven lever before any code is written

STEP 1  Fix OOF leakage (03 Tasks 1-2)                           [BLOCKER]
        -> without this you cannot tell which later step helped

STEP 2  Multi-gold selection + rerank_k=200 + Cand@100/200 diag  [high confidence]
        -> expected +2.2 .. +3.5pp, lands ~92.4-93.7%

STEP 3  Pairwise/listwise loss, tuned on fold 0 only             [largest single lever]
        -> expected +1.0 .. +3.0pp, lands ~93.4-96.7%

STEP 4  LoRA r=32 + longer training                              [capacity]
STEP 5  Multi-seed ensemble                                      [if time allows]
STEP 6  Fine-tune DEk21 (fold-safe!)                             [structural, highest effort]
```

Evaluate each step on **fold 0 alone** (~46 min) before committing to a full
5-fold run. Reject anything that does not move fold 0.

## Scoreboard to fill in

| Config | Fold-0 R@5 | Prec@5 | Cand@50 | Cand@100 | Cand@150 | Cand@200 |
|---|---|---|---|---|---|---|
| Run 4 baseline | 91.09% | — | — | — | — | — |
| Current (live) | **91.33%** | 19.51% | 98.05% | *missing* | 98.89% | *missing* |
| + multi-gold fix | | | | | | |
| + rerank_k=200 | | | | | | |
| + pairwise loss | | | | | | |
| + r=32 | | | | | | |
| + ensemble | | | | | | |

**Add Cand@100 and Cand@200 to the diagnostics.** The run currently reports
@50 and @150 while the config uses `candidate_k=200, rerank_k=100` — neither
reported depth matches either configured depth, so the effect of changing
`rerank_k` is presently unmeasurable.

## One correction to keep in view

Two folds of measured evidence put this run at **~90.15% projected OOF**
(+0.54pp over Run 4). Earlier in this project — including by me — 0.96 was
described as a consequence of the `candidate_k`/`rerank_k`/`max_length` changes.
That was wrong, and the measurement disproved it. Plan from 90.15%.
