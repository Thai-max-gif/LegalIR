# Runtime reality — measured, not projected

All numbers here come from two real sources. No headline multipliers.

- **Source A — Run 4 receipt** (old config: `candidate_k=150, rerank_k=50, max_length=384, max_chunks=2`):
  `runs/run-04-oof89.6/reports/submission_manifest.json` → `metadata.stage_timings`
- **Source B — live fold 0** on the current config, observed 2026-09-21:
  `Fold 0 Results: Recall@5 = 91.33% | Cand@50 = 98.05% | Cand@150 = 98.89% (2758.2s)`

## Run 4 measured breakdown (total 19,418s = 5.39h, 2,080-query submission)

| Stage | Seconds | Share |
|---|---:|---:|
| **oof_cv** | **13,471** | **69.4%** |
| public_inference (2,080 queries) | 2,002 | 10.3% |
| dense_index | 1,845 | 9.5% |
| final_pair_mining | 674 | 3.5% |
| bm25_pyvi | 499 | 2.6% |
| final_reranker_training | 427 | 2.2% |
| bm25_legal | 184 | 0.9% |
| canonical_load | 149 | 0.8% |
| final_pipeline_load_audit | 127 | 0.7% |
| fusion_training / memory / packaging | 24 | 0.1% |

`oof_cv` decomposed (derived from `final_reranker_training` per-update cost and
`final_pair_mining` scaled to 5,600 train queries):

| Component | Seconds | Share of oof_cv |
|---|---:|---:|
| fold **evaluation** | 9,066 | 67.3% |
| fold pair mining | 2,697 | 20.0% |
| fold **training** (700 updates × 5) | 1,709 | 12.7% |

**Consequence:** all reranker *training* in the whole run is
1,709 + 427 = **2,136s = 11% of runtime**. Inference and indexing are 89%.
Any "training-time optimization" can address at most that 11%.

## Correction: my earlier runtime projection was wrong

I projected the new config at ~2.26× → 9.6h → timeout. **Live data refutes it.**

| | Run 4 per job | Live fold 0 | Actual factor |
|---|---:|---:|---:|
| oof_cv ÷ 5 jobs | 2,694s | 2,758s | **1.02×** |
| oof_cv ÷ 6 jobs (5 folds + disjoint) | 2,245s | 2,758s | **1.23×** |

**Why I was wrong:** I modelled the extra rerank work
(`rerank_k` 50→100 × `max_length` 384→512 = 2.67×) but ignored that
`inference_batch_size` rose to **64**. Run 4 fell back to
`batch_size` (8) — `kaggle_train.py:1844` reads
`inference_batch_size or batch_size, 16`. Going 8→64 on an A100 recovered
roughly the throughput the deeper reranking consumed. The two effects
nearly cancel.

**Lesson for the next agent:** on this pipeline, cross-encoder scoring was
*batch-starved*, not compute-bound. Measure before modelling.

## Revised total projection (6 jobs @ 2,758s)

| Final-inference assumption | Projected total | vs 25,200s timeout |
|---|---:|---|
| ×1.23 (scales like folds) | 22,957s (6.38h) | +2,243s margin — OK |
| ×1.60 (pessimistic) | 23,698s (6.58h) | +1,502s margin — OK |
| ×2.00 (worst case) | 24,499s (6.81h) | +701s margin — OK |

**Fits 7h in all three cases, but the margin is thin (3–9%).** It was a real
risk under the old 5h (18,000s) gate; the 25,200s gate is what makes it safe.
Do not reduce the timeout.

> **SUPERSEDED 2026-09-21 (score-first directive).** Runtime is no longer a
> constraint. The `08-score-maximization.md` configuration deliberately spends
> more runtime for recall and **will exceed 25,200s** — raise
> `MODAL_TIMEOUT_SECONDS` and `LEGALIR_TIME_GATE_SECONDS` together before
> launching it. The numbers in this file remain valid as the *baseline*
> forecast for the current config.

## Which commit is the live run actually on? — VERIFY THIS

Fold 0 improved only **+0.24pp** (91.09% → 91.33%). If the leaky warm-start
(`180b15e`) were active, the fold model would have already trained on its own
evaluation queries and the jump should be far larger. Two possibilities:

1. **The run is on `4d24633`** (candidate_k=200, rerank_k=100, max_length=512,
   max_chunks=3, pos_weight=4.0 — but *no* warm-start). Then fold 0 = 91.33%
   is an **honest, uncontaminated** number and the +0.24pp is the true effect
   of the depth/length changes. Most likely, given the run started before
   `180b15e` was pushed.
2. **The run is on `180b15e`/`2a0d17e`** and the HF adapter load silently
   failed — `setup_peft_model` catches the exception and falls back to clean
   init with a `[!] Warning: Failed loading warm-start adapter` line.

**Action:** grep the running log for
`[+] Warm-start: loading existing LoRA adapter` and for
`[!] Warning: Failed loading warm-start adapter`.
- Neither present → case 1, metrics are trustworthy.
- First present → metrics are contaminated; discard the OOF number.

## Time levers, ranked by measured value

| Lever | Target stage | Expected saving | Risk |
|---|---|---|---|
| Keep `inference_batch_size: 64` | all rerank | already banked (~2×) | none — done |
| **Length-grouped batching** | all rerank | 20–35% of rerank time | low; needs invariance test |
| Cache `dense_index` + `bm25_pyvi` on Modal Volume by dataset hash | indexing | 2,344s on warm re-runs | low; cold run unaffected |
| Static branch-cache reuse across folds | fold eval | part of the 2,697s mining | already partly implemented |
| Reduce `candidate_k` 200 → 150 | retrieval | small | **see `06`: costs ≤0.06pp recall** |

Not a lever: warm-start. Step count is pinned by coverage policy
(`train_reranker.py:124-130`), and training is only 11% of runtime anyway.
