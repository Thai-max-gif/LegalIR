# FINDING 1 — CRITICAL: warm-start adapter leaks into OOF and document-disjoint evaluation

Severity: **Critical — blocks A100 launch.**
Introduced by: commit `180b15e`.
Symptom: no crash. Metrics silently inflate.

## The defect in one sentence

`configs/experiments/reranker_lora.yaml` now sets
`pretrained_lora_path: "dangphuc2109/legalir-task1-reranker"`, and that same
config file is handed to the **per-fold** and **document-disjoint** trainers —
so every fold warm-starts from weights that already trained on that fold's
held-out evaluation queries.

## Evidence chain

**1. The published adapter saw all 7,000 training queries.**

`https://huggingface.co/dangphuc2109/legalir-task1-reranker/resolve/main/training_manifest.json`:

```json
"eligible_training_queries": 7000,
"unique_training_queries":   7000,
"actual_query_coverage_pct": 100.0
```

It also carries `"modules_to_save": ["classifier", "score"]`, so the trained
**scoring head** — the part that directly emits the relevance logit — is
restored too. That amplifies the leak rather than limiting it to the LoRA deltas.

**2. The fold trainer reads that config.**

`src/pipeline/oof_runner.py:895-899`
```python
reranker_cfg = self.reranker_config_path or self.config_path or "configs/experiments/reranker_lora.yaml"
...
train_report = train_reranker(
    pairs_file=pairs_dir / "reranker_pairs.parquet",
    config_path=reranker_cfg,      # <-- carries pretrained_lora_path
    output_dir=adapter_dir,
    fold=f_idx,
```

**3. The document-disjoint trainer reads the same config.**

`src/pipeline/oof_runner.py:1363-1367` — identical `reranker_cfg` resolution,
then `train_reranker(config_path=reranker_cfg, ...)`.

**4. `self.reranker_config_path` IS that file.**

`src/pipeline/kaggle_train.py:1311` (default), `:1581-1582` (resolve),
`:1827` (passed into the OOF runner).

**5. The config value reaches PEFT unconditionally.**

`src/training/trainer.py:631-637` reads `pretrained_lora_path` from the config
dict and forwards it to `setup_peft_model(...)`, which calls
`PeftModel.from_pretrained(model, pretrained_adapter, is_trainable=True)`
(`src/training/trainer.py:477`). There is no fold-awareness anywhere on this path.

## Blast radius

| Evaluation | Intended meaning | What it becomes |
|---|---|---|
| 5-fold OOF Recall@5 | Generalization to unseen queries | Inflated — model already trained on the eval queries |
| Document-disjoint Recall@5 | Robustness under **zero document leakage** | Void — its entire purpose is defeated |
| Learned fusion weights | Fit on honest OOF scores | Fit on contaminated scores; may transfer worse to private test |
| Final model + submission | Unaffected in itself* | Selected/tuned using corrupted signal |

\* The **final** model legitimately trains on all 7,000 queries, so warm-start
there is defensible. The defect is specific to the fold and disjoint paths.

**Practical consequence:** the run would likely report a higher OOF number
(possibly the >0.96 target) while true private-test performance is unchanged or
worse. The metric would no longer be a decision-making instrument.

## Why no gate caught it

There is no test anywhere in `tests/` asserting that fold or disjoint adapters
initialize from base weights. Verified:

```
grep -rniE "pretrained_adapter|warm_start|pretrained_lora" tests/
  -> only tests/unit/test_warm_start_lora.py (tests the feature, not the isolation)
```

The existing leakage suite (`tests/leakage/`) covers pair/query/document
splitting, not adapter weight provenance. This is a genuine blind spot, so the
full 556-test pass is not evidence against this finding.

## Fix options

### Option A — fold/disjoint always cold-start (RECOMMENDED)

Keep warm-start for the **final** model only. Folds and the disjoint split
initialize from base `BAAI/bge-reranker-v2-m3`.

- Preserves honest OOF and disjoint metrics.
- Keeps the warm-start benefit where it is legitimate (final model).
- Requires an explicit opt-in parameter so the default is safe.

Recommended shape: add `allow_warm_start: bool = False` to `train_reranker()`;
only the final-model call site in `src/pipeline/kaggle_train.py:2071` passes
`True`. `setup_peft_model` ignores `pretrained_adapter` unless allowed.

Fail-closed is essential: the default must be "no warm start", so a future
call site that forgets the flag stays correct.

### Option B — remove `pretrained_lora_path` from the shared config

Delete the key; pass the adapter path only at the final-model call site.
Simpler, but leaves the footgun: anyone re-adding the key silently re-breaks
OOF. Prefer A, or do B **plus** the regression test from Task 4.

### Option C — per-fold adapters trained only on that fold's queries

Correct in principle, but requires producing and publishing five fold-specific
adapters. Out of scope for the current deadline.

## Do NOT

- Do not "fix" this by disabling or relaxing the OOF/disjoint gates.
- Do not keep fold warm-start and annotate the metric as approximate —
  the disjoint split's only reason to exist is zero leakage.
