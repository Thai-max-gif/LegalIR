# Modal A100 Launch Guide

Reviewed 2026-09-20. Production execution guide for LegalIR Task 1 on Modal A100.

## Pre-Launch Requirements

1. Select a clean evidence-bearing release `S`, with exact-SHA CI success, strict release validation, and a genuine Kaggle dual-T4 report for its runtime `R`.
2. Ensure canonical dataset has been downloaded or verified via Kaggle CLI.
3. Configure Modal secrets: `kaggle-secret` (containing `KAGGLE_USERNAME`, `KAGGLE_KEY`, and `LEGALIR_TEST_PHASE="private"`) and `huggingface-secret` (containing `HF_TOKEN`).
4. Record full release/runtime SHAs, attempt path, UTC start, and approved ceiling.

## Modal A100 Execution

### Launch Commands

```bash
# Private Round Production Execution (Detached, Recommended):
LEGALIR_TEST_PHASE=private bash scripts/modal/run_modal_cli.sh --detach --hf-allow-public-repo

# Public Round Validation (Detached):
bash scripts/modal/run_modal_cli.sh --detach --hf-allow-public-repo
```

`--detach` is recommended so the remote container runs independently of the local machine's network connection or sleep state.

The wrapper rejects dirty working trees, checks SHA/provenance locally before dispatch, then the remote path repeats checkout/provenance, HF access checks, canonical dataset validation, and training.

### Resources, Lifetime, and Persistence

- **Allocated Hardware**: 1 × NVIDIA A100-SXM4-40GB GPU, 8 dedicated vCPUs (`cpu=8.0`), and 32 GiB host RAM (`memory=32768`).
- **Timeout**: `18000` seconds (5.0 hours). The optimized cold pipeline finishes in ~2.0–2.5 hours.
- **Volume Mount**: `/root/legalir_volume/<sha>/attempts/<uuid>/` on persistent Volume `legalir-production`.
- **Output Artifacts**: Checkpoints, `submission.zip`, logs, and `recovery.tar.gz` are written directly to the persistent Volume.

### Supervision and Monitoring

1. **Log Streaming**:
   ```bash
   modal app logs <app-id>
   ```
2. **Retrieve Completed Artifacts**:
   ```bash
   modal volume get legalir-production <sha>/attempts/<uuid>/ ./local_artifacts/
   ```
3. **Emergency Stop**:
   ```bash
   modal app stop <app-id> --yes
   ```

## FULL Completion and Later Reuse

The required workload includes cold retrieval/indexing, five fold trainings/evaluations, document-disjoint training/evaluation, fusion evaluation, dedicated final training, final-model reload, test inference, validation, and durable delivery.

Before declaring success, verify all expected query IDs (2,080 for private round), required model/tokenizer artifacts, immutable base-model identity, checksums, valid submissions, delivery receipts, and provider shutdown. Preserve the original attempt path and logs for any failure investigation.
