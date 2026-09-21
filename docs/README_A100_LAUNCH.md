# Modal A100 Launch Guide

Reviewed 2026-09-20. Production execution guide for LegalIR Task 1 on Modal A100.

## Pre-Launch Requirements

1. Select a clean evidence-bearing release `S`, with exact-SHA CI success, strict release validation, and a genuine Kaggle dual-T4 report for its runtime `R`.
2. Ensure canonical dataset has been downloaded or verified via Kaggle CLI.
3. In workspace `zunuoivalutre`, environment `main`, configure `kaggle-secret` (`KAGGLE_API_TOKEN`, or legacy `KAGGLE_USERNAME` + `KAGGLE_KEY`) and `huggingface-secret` (`HF_TOKEN` with write access). Set `HF_REPO_ID` in the latter secret if using a destination other than `dangphuc2109/legalir-task1-reranker`. Phase and time gate are passed explicitly by the launcher; do not configure them in secrets.
4. Record full release/runtime SHAs, attempt path, UTC start, and approved ceiling.

## Modal A100 Execution

### Launch Commands

Recommended cross-platform entrypoint (PowerShell, Linux, or macOS), from the repository root:

```text
python scripts/modal/launch.py --check-only
python scripts/modal/launch.py --detach --private --hf-allow-public-repo
```

This uses the current Python's `modal` module, selects profile `zunuoivalutre`
and environment `main` explicitly, and validates the clean release before
invoking Modal. Install `modal` and `pyyaml` in that Python environment if
needed. The copied macOS `.venv` cannot be used on Windows. `--check-only`
does not build an image, inspect remote secret values, or allocate a GPU.
Omit `--hf-allow-public-repo` for a private HF destination.

Linux/macOS legacy shell entrypoint (requires working `.venv/bin` executables
and the intended Modal profile already selected):

```bash
# Private Round Production Execution (Detached, Recommended):
bash scripts/modal/run_modal_cli.sh --detach --hf-allow-public-repo --private

# Public Round Validation (Detached):
bash scripts/modal/run_modal_cli.sh --detach --hf-allow-public-repo
```

`--detach` is recommended so the remote container runs independently of the local machine's network connection or sleep state.
`--private` ensures fail-closed private test query evaluation (exactly 2,080 queries with 5 unique valid corpus documents).

The wrapper rejects dirty working trees, checks SHA/provenance locally before dispatch, then the remote path repeats checkout/provenance, HF access checks, canonical dataset validation, and training.

### Resources, Lifetime, and Persistence

- **Allocated Hardware**: 1 × NVIDIA A100-SXM4-40GB GPU, 8 dedicated vCPUs (`cpu=8.0`), and 32 GiB host RAM (`memory=32768`).
- **Timeout**: `25200` seconds (7.0 hours) by default; completion within this window still requires measurement. Override `MODAL_TIMEOUT_SECONDS` locally; the launcher passes this same value to the remote acceptance gate. If `LEGALIR_TIME_GATE_SECONDS` is also set locally, it must match. Image build/allocation and retries are outside this execution ceiling.
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
