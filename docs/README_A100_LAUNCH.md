# A100 Launch Guide

Reviewed 2026-09-17. **Do not launch the current uncommitted candidate.** Resolve [../fix.md](../fix.md) and complete the [release workflow](REPRODUCIBLE_TRAINING_WORKFLOW.md) first. The instructions below are a future operating procedure, not authorization to allocate a GPU or publish artifacts.

## Before either backend

1. Select a clean evidence-bearing release `S`, with exact-SHA CI success, strict release validation, and a genuine Kaggle dual-T4 report for its runtime `R`.
2. Resolve dense-index provenance and disable unsafe stage reuse or complete its identity checks. Do not copy old completion markers into a new production attempt.
3. Qualify actual A100 CPU/RAM/VRAM and throughput. Neither a five-hour completion forecast nor >96% Recall@5 is established.
4. Obtain approval for the backend, one-attempt limit, time and spending ceiling with reserve, publication target/visibility, and supervision/stop procedure. Private upload is still an external action; public visibility needs explicit opt-in too.
5. Confirm local CLI authentication and remote secret availability without printing credential values. Validate the installed CLI's logging/stop syntax before allocation.
6. Record full release/runtime SHAs, effective configuration, image/dependency identity, app/session ID when assigned, attempt path, UTC start, and approved ceiling.

Use one provider at a time. Colab is a separately approved fallback, not an automatic retry. Do not bypass wrapper preflight by calling the remote function directly.

## Modal

### Launch modes — after approval only

```bash
# Attached: supervise a stable, awake client until completion.
scripts/modal/run_modal_cli.sh

# Alternative: explicit detached execution, with independent supervision.
scripts/modal/run_modal_cli.sh --detach
```

Choose **one** command, not both. `--detach` is an explicit spending-supervision choice, not a speed option. The wrapper defaults to attached mode and rejects duplicate detach flags. If the approved HF target is public, the separate `--hf-allow-public-repo` opt-in is required; do not add it automatically.

The wrapper rejects dirty working trees, checks SHA/provenance locally before dispatch, then the remote path repeats checkout/provenance, HF access checks, canonical dataset validation, and training. Local success cannot establish remote credentials or hardware availability.

### Resources, lifetime, and persistence

- The function currently requests `A100` without explicit CPU/RAM reservation. Host-resource sizing and cold indexing throughput are **unqualified**; a GPU request alone is insufficient for this CPU/RAM-heavy pipeline.
- The function timeout is `18000` seconds. This caps that function's duration, not total setup/cleanup charges or repeated attempts. It can interrupt an unfinished FULL pipeline.
- Attached ephemeral execution can terminate remote tasks when the client disconnects. A persistent Volume does not keep the compute alive.
- Detached execution requires recorded app ID, active log monitoring, deadline/spend supervision, and verified explicit shutdown. The added flag does not install an independent watchdog or durable supervisor.
- Output location: `/root/legalir_volume/<sha>/attempts/<uuid>/` on `legalir-production`. Each invocation creates a **new UUID**; rerunning does not resume the previous attempt.
- Volume commits improve durability, but the last uncommitted bytes and unfinished training state can be lost. Completion markers are not optimizer/scheduler/RNG checkpoints.

Remote secret names expected by the launcher are `kaggle-secret` and `huggingface-secret`. Configure the supported Kaggle credential variables and write-capable HF credentials through provider secret management. Never put their values in Git or review documents.

### Supervision and stopping

Track meaningful stage progress, elapsed time, resource usage, output commits, and remaining approved budget. An “alive” heartbeat alone is not progress. Stop on fatal OOM/nonfinite failure, approved deadline/ceiling, or a confirmed stall. A proposed threshold is >20 minutes without meaningful progress in a normally chatty phase; approve thresholds before launch and account for legitimately quiet indexing operations.

Inspect `modal app logs --help` and `modal app stop --help` before the run. Use the recorded app ID with the installed CLI's supported stop command; do not stop unrelated apps. Confirm termination in provider state. Local client exit alone is insufficient evidence of shutdown, especially in detached mode. No automatic retry.

## Google Colab CLI

### Current preallocation limitation

The reviewed wrapper runs local provenance checks but does **not** reject dirty working trees. It selects HEAD, while the remote notebook checks out that committed SHA. Uncommitted fixes can therefore be absent remotely despite local preflight success. Add and test the clean-tree guard in `fix.md` before using this path. Merely passing the existing strict verifier does not close this gap.

### Launch — after repair, release, and approval only

```bash
./scripts/colab/run_colab_cli.sh A100
```

The wrapper requires the repository's supported `colab` CLI, local Python environment, and allowlisted credentials. Configure credentials privately; the wrapper uploads an allowlisted `.env` plus a required release-selection JSON. Public HF publication additionally requires explicit `HF_ALLOW_PUBLIC_REPO=1`; authentication is required regardless of visibility.

The wrapper validates mode/tools/deadlines, uses a private temporary directory, creates a unique A100 session, uploads required inputs, executes the generated notebook, and performs bounded recovery and stop. Record the chosen session name and independently confirm provider shutdown. If the supervising process/machine disappears, its cleanup cannot be relied on to send a stop request.

### Timeouts and recovery

- `COLAB_TIMEOUT` defaults to `18000` seconds for the externally bounded whole-notebook execution. The same value is passed as a per-cell timeout; the per-cell option alone cannot bound a multi-cell notebook.
- Allocation/setup and bounded cleanup are additional time; none is a platform billing cap. A timeout does not prove remote shutdown.
- Recovery downloads are individually bounded: 15 seconds each for manifest/log, 45 seconds for archive, and 15 seconds each for ZIP/JSON. Stop is bounded at 30 seconds.
- Recovery directory: `artifacts/task1/production/<session-name>/`.
- Exit precedence: primary failure wins; otherwise failed stop returns 70, incomplete recovery returns 74, and complete local handoff returns 0. INT/TERM cleanup is best effort.
- A recovered archive or exit 0 does not prove valid final weights. Inspect archive contents, checksums, provenance, and model reload results.
- VM eviction, local-machine loss, or network failure can defeat recovery. No automatic restart or cross-session optimizer resume is provided.

If cleanup cannot confirm shutdown, use the installed CLI's supported `colab stop -s SESSION` operation for the recorded session and check the provider view. Treat an unsuccessful stop as an operator incident, not permission to launch a replacement.

## Manual Colab fallback

Use only after a separate operational review; the CLI's local preallocation safeguards are not available in the web flow.

1. Select evidence-bearing release `S` whose strict verifier and exact-SHA CI pass.
2. Open `notebooks/colab_a100_train.ipynb` from `S`; do not hand-edit generated cells.
3. Set `LEGALIR_COMMIT_SHA` in Colab Secrets to the full 40-character SHA of **S**. Do not substitute the baked runtime pin `R`: the evidence-bearing release must be explicitly selected. Without explicit selection the generated notebook refuses checkout.
4. Configure required Kaggle and write-capable HF secrets, approved HF destination, and public opt-in only when authorized. Secrets override uploaded `.env` values.
5. Select A100 and appropriate RAM, knowing web allocation can occur before notebook provenance checks. Supervise execution and verify runtime termination manually.

## FULL completion and later reuse

The required workload includes cold retrieval/indexing, five fold trainings/evaluations, document-disjoint training/evaluation, fusion evaluation, dedicated final training, final-model reload, public inference, validation, and durable delivery. Coverage requirements can raise training steps above the nominal configured cap. Do not assume the whole workload fits five hours.

Before declaring success, verify all expected held-out/public query IDs, required model/tokenizer artifacts, immutable base-model identity, checksums, valid submissions, delivery receipts, and provider shutdown. Preserve the original attempt path and logs for any failure investigation.

A validated final adapter can support later inference without retraining. This is not permission to warm-start held-out evaluation from all-label weights. There is no general upload-only recovery CLI promised here; any manual recovery must validate provenance and obtain publication approval. Never fabricate a successful receipt.
