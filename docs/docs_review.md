# Review of the accessed DSC documentation

Only `DSC_2026_TECHNICAL_DOCUMENTATION(1).md` was accessed. The LegalIR repository, Notion page, dataset release, candidate/index artifacts, and official scorer were not accessed.

| Vấn đề | Mục docs | Cách sửa trong package | Kiểm tra |
|---|---|---|---|
| Dense encoder is written as `BKAI / DEk21`, but identity/provenance is not specified | 3.1, 2.2 | No alias is assumed. `index_identity.py` requires model ID, immutable revision, pooling, preprocessing and dimension before end-to-end retrieval can run. | `preflight_report.json.index_identity` |
| BGE is described as XLM-RoBERTa Large at about 560M parameters | 3.1, 3.4 | Do not rely on that architectural/parameter claim for audit. The current model card identifies `bge-reranker-v2-m3` as based on BGE-M3 and displays 0.6B parameters; the package counts the loaded tensors at the frozen revision. | Parameter-audit section of run report |
| Task 1 config uses `main` for a model revision | 4.1 | Config starts with `null`; runtime resolves it, and the freeze record must replace it with a concrete Hub commit. | Resolved config + freeze record |
| `document_disjoint_split.parquet` is used as `val_pairs_path` without its real schema | 4.1 vs. 2.2 | Training reads only `reranker_pairs.parquet`; split files are inspected, never interpreted as pairs without columns proving that contract. | Dataset schema report |
| LoRA target `dense` can match `classifier.dense` | 3.4 | The model is inspected first; only non-classifier `torch.nn.Linear` fully-qualified module names are handed to PEFT. `classifier` is saved separately. | `lora_target_modules` in report |
| Pairwise logistic and BCE are presented as alternatives without a schema decision | 4.2 | `loss_type=auto` infers only from actual pair schema and writes the decision; implementations are separate and unit-tested. | `training.loss_type` |
| BGE `num_labels=1` may cause implicit regression/MSE loss | 3.4, 4.2 | The runner never supplies model labels; it computes stable pairwise softplus or BCE explicitly. | Loss unit tests + run report |
| Flash-Attn/Liger guidance targets Task 2/Qwen and promises fixed gains | 3.5 | Neither is installed or enabled by default for BGE. A100 uses eager until a compatible, approved probe changes config. | Resolved config/runtime report |
| Notebook sample is LegalQA, production-default, and publishes automatically | 5.2 | Two LegalIR-only launchers use smoke/full commands separately, propagate exit codes, and contain no publishing call. | Notebook source |
| Example Git/data/HF hashes and metrics are illustrative and Task 2-specific | 5.4 | No example identity is copied. Reports populate only measured/runtime values. | Run artifacts |
| T4×2 “total 32GB” can be misconstrued as unified memory | 1.2, 3.5 | DDP checks each local GPU peak against 14,000,000,000 bytes and reports GiB. | Smoke report VRAM checks |

## Explicit implementation choices to freeze after the real release is inspected

- RRF weights currently `bm25=1.0`, `dense=1.0`, `rrf_k=60`, candidate budget 100, rerank budget 50, document aggregation `max` and top-k 5. They are configuration defaults, not measured or document-mandated values.
- The optimizer is `AdamW` through `torch.optim.AdamW`; the docs do not name an optimizer.
- Tie handling is deterministic by lexical ID after score; duplicate predicted documents are removed before metrics.
- The document-level qrels path is used for Recall/MRR/NDCG/Hit. Candidate recall must be added to the report only after the candidate schema is verified.
