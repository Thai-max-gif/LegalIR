# LegalIR Task 1: A100 Training Guide

**Current Status (Updated 2026-09-17):**
- **Verified Runtime Commit:** `0ca7c135bcefb58b7fcb4f18ed9035a4e06d9428`
- **Verified Release Commit:** `aadd3f242f6258b2bacba5feef4783ccf373d84f`
- **Pre-A100 Hardware Gate:** Kaggle 2×T4 CUDA Smoke Gate (B1.1) is **VERIFIED PASS** on live Kaggle environment (`phucdangg/legalir-training`, Version 55, $\Delta w = 279.72 > 0$, runtime 31.6s).
- **CI Status:** GitHub Actions CI (`35220762427`) completely **GREEN** (`test`: 7m51s, `strict-release`: 2m2s).
- **Core Optimizations Active:**
  1. `QueryBalancedSampler`: 50/50 interleaved positive/negative windows for stable gradient updates.
  2. Shared static branch retrieval cache across folds (eliminates duplicate query searches across 5 folds + doc-disjoint + final).
  3. Contiguous multi-query batch reranking (`rerank_batch`) with score scattering.
  4. Bounded multiprocessing in PyVi BM25 indexing (`min(4, os.cpu_count() - 1)`).
  5. Missing-rank sentinel alignment (< 900.0) with RRF.
  6. Completed-stage checkpoint recovery (`complete.json`, `predictions.parquet`, `metrics.json`) for zero redundant work on restart.
  7. Fail-closed model/tokenizer loading with pinned revisions.
  8. Top-5 oracle feasibility verified: 100.0% corpus capacity ceiling (no query has >5 gold documents).

You have two backends. Repair both locally; qualify one candidate; use Modal first
after durable-output repair unless a human chooses otherwise. Do not launch Modal
and Colab concurrently. Colab is a separately reviewed fallback, not an automatic retry.

## Option 1: Modal Serverless (Recommended First Attempt)

Durable outputs use `/root/legalir_volume/<sha>/attempts/<id>/` on the
`legalir-production` Volume from the beginning. Completed adapter files and
`recovery.tar.gz` do not depend on a final copy allowlist. Background commits
improve durability while the process runs. Completed folds automatically persist
`complete.json`, predictions, and metrics; on restart or recovery, completed folds
and static retrieval caches are re-used directly to prevent redundant work.

- Function timeout: 5 hours (`timeout=18000s` in `scripts/modal/run_modal_a100.py`,
  preserved unless separately reviewed). The timeout caps duration per attempt,
  not total spend — retries and re-runs bill extra.
- Publication consent defaults to private-only on every path. Public repos
  require explicit opt-in, recorded in the manifest. Unknown visibility blocks.
- Recommended entrypoint (CPU gate before image build/dispatch):
  ```bash
  scripts/modal/run_modal_cli.sh
  scripts/modal/run_modal_cli.sh --hf-allow-public-repo
  ```
  The wrapper requires a clean tree, rejects SHA mismatch, runs
  `scripts/colab/bootstrap.py --expected-sha`, then dispatches with the same
  SHA via `modal run`. Direct dispatch is:
  ```bash
  modal run scripts/modal/run_modal_a100.py --no-hf-allow-public-repo
  modal run scripts/modal/run_modal_a100.py --hf-allow-public-repo
  ```
  Flag spelling confirmed via `modal run scripts/modal/run_modal_a100.py --help`
  (`--hf-allow-public-repo / --no-hf-allow-public-repo`).
- Remote order: checkout+provenance, then HF access, then dataset
  download/fingerprint, then train. HF failure precedes expensive acquisition.
  Local checks cannot validate remote secrets or GPU; remote repeats all checks.

1. Ensure you have the `modal` CLI installed and authenticated (`modal token new`).
2. In the Modal Dashboard → **Secrets** create:
   - **`kaggle-secret`**: `KAGGLE_USERNAME` and `KAGGLE_KEY` (or `KAGGLE_API_TOKEN`
     for `KGAT_` bearer tokens — code auto-promotes `KAGGLE_API_TOKEN`)
   - **`huggingface-secret`**: `HF_TOKEN_WRITE` (write scope required; read-only
     tokens fail preflight) and `HF_TOKEN`
3. Obtain explicit human approval for: backend choice, one-attempt limit, 5h
   limit, budget ceiling + reserve, HF target visibility + publication consent,
   image/dependency contract, timeout/supervision/stop procedure.
4. Launch via the wrapper above. Record app ID, attempt path, image identity,
   exact SHA, start UTC, and approved ceiling.
5. Watch remote logs + local log for meaningful phase/progress entries and the
   expected Volume path. Heartbeats that merely print “alive” must not reset a
   stall timer.

### FULL stages (unmeasured for 5h; no promise all work fits)

Retrieval/indexing, five fold-specific trainings and evaluations,
document-disjoint training/evaluation, fusion evaluation, final full-query
training (≈875 optimizer steps for 7,000 queries at current batch/accumulation;
fold counts computed separately), final inference, validation, HF artifact
upload, receipt upload, recovery finalization. `max_steps: 250` is increased to
satisfy coverage. Five-hour completion is unmeasured, not assured.

### Supervisor checklist

UTC start, app/session ID, last **meaningful progress** time, phase, elapsed
time, remaining approved budget, output/attempt path. Existing trainer prints
roughly ten progress updates per training loop; silence alone is not a reliable
measure of all pipeline phases. In less chatty phases inspect stage
timing/resource state rather than treating every long operation as deadlock.

### Kill criteria (user-provided, preserved)

OOM/nonfinite fatal error; >20 minutes without meaningful progress in a known
chatty phase; five-hour function limit; approved spend ceiling.

### Stop (Modal) and its limits

User-provided pattern: `modal app stop <id> --yes`; confirm current installed
CLI syntax before the paid run and use the identified app ID, not a broad stop
of unrelated apps. Stop the app and then the corresponding local client if
necessary; killing only the client is insufficient evidence of termination.
Independently confirm app termination, not just client exit. No automatic relaunch.

### Recovery without retraining (limited)

Immutable final-model artifacts can permit later delivery without retraining,
but no general upload-only recovery CLI or checkpoint resume is provided by
this repair. Any manual recovery must validate provenance and obtain
publication approval. Do not reconstruct a fabricated RELEASED receipt.

## Option 2: Google Colab CLI Automation (Supervised Fallback)

Supervised attempt with best-effort recovery. Does not survive VM eviction or
local-machine failure. Keep the supervising machine awake and connected. Record
the session ID before allocation. Confirm shutdown in the provider
session/runtime view, not only a local success string. Colab pricing/compute
units need separate approval (Modal dollar math does not apply).

- Execution wait: `COLAB_TIMEOUT=18000s` (5h) by default; cost-conscious wait,
  not a platform billing cap. Larger values need separate cost approval. A wait
  timeout does not kill the remote process without a confirmed stop.
- Cleanup: downloads bounded (15s manifest/log, 45s archive, 15s ZIP/JSON;
  105s total max plus overhead), stop bounded 30s, no unbounded retry.
  Recovery goes to `artifacts/task1/production/<session-name>/`, not shared
  filenames. Exit codes: primary nonzero wins; else failed stop=70; else
  incomplete recovery=74; else 0. INT=130, TERM=143, cleanup once.
- Consent: absent or `0` means private-only; literal `1` permits public
  (Secrets key `HF_ALLOW_PUBLIC_REPO` supported; Secrets win over uploaded
  `.env`, which is allowlisted by the wrapper). Authentication remains required
  even with consent.
- Local handoff success (manifest/log + archive or both submissions) is not
  proof the archive contains valid weights; downstream verification must inspect it.

1. Ensure you have the `colab` CLI tool installed (`pip install colab-cli`).
2. Ensure local `.env` has `HF_TOKEN_WRITE` (or `HF_TOKEN`), Kaggle vars, and
   optionally `HF_ALLOW_PUBLIC_REPO=1` for explicit public opt-in. The wrapper
   filters `.env` to an allowlist and preserves originals.
3. Launch:
   ```bash
   ./scripts/colab/run_colab_cli.sh A100
   ```
4. The wrapper validates mode/tools/timeout, uses a private temp dir
   (`umask 077`, `mktemp -d`), traps EXIT/INT/TERM, runs local provenance
   preflight, allocates a unique session, uploads required inputs (launch JSON
   always required; gate/freeze uploads required when overriding repo copies —
   failure never silently falls back), executes with the 5h wait, then bounded
   recovery + bounded stop with truthful exits. If allocation is ambiguous, no
   upload/exec runs but bounded stop is still attempted. If the orchestrator
   dies or network drops, `stop` is never sent — the VM burns until reclaimed.
   A failed stop is an incident needing immediate operator attention
   (`colab stop -s SESSION`).

Production config is unified: LoRA `r=8/alpha=16`, `max_length=384`, hybrid
`100/30`, RRF `k=60`, `bf16`.

## Final Review

Before launching, pass the `A100_Production_Review_Prompt.md` file to another
instance or senior ML engineer for final review. No “fully verified/safest/
guaranteed recovery” claim is supported by local evidence alone.
