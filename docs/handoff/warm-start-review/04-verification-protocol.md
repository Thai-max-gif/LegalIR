# Verification protocol

Rule: evidence before claims. Do not report a task complete without pasting the
command output that proves it. A FAIL is a finding, never something to route around.

## Per-task acceptance

### Task 1 — warm-start opt-in
```bash
./.venv/bin/pytest tests/leakage/test_warm_start_isolation.py -v
./.venv/bin/pytest tests/unit/test_warm_start_lora.py -v
```
Both green. Task 2's tests must have been seen RED before Task 1 landed.

Manual confirmation that fold training is cold — the fold log must NOT contain:
```
[+] Warm-start: loading existing LoRA adapter from ...
```
while the final-model stage MAY contain it exactly once.

### Task 2 — regression test
Red-green is mandatory:
```bash
git stash                      # remove the Task 1 fix
./.venv/bin/pytest tests/leakage/test_warm_start_isolation.py   # MUST FAIL
git stash pop
./.venv/bin/pytest tests/leakage/test_warm_start_isolation.py   # MUST PASS
```
A test that has never failed proves nothing.

### Task 3 — model card
```bash
./.venv/bin/python - <<'PY'
import json, urllib.request
u="https://huggingface.co/dangphuc2109/legalir-task1-reranker/resolve/main/"
cfg=json.load(urllib.request.urlopen(u+"adapter_config.json"))
rd=urllib.request.urlopen(u+"README.md").read().decode()
print("adapter target_modules:", cfg["target_modules"])
print("card mentions q_proj  :", "q_proj" in rd, "(must be False)")
PY
```

### Task 5 — full local gate
```bash
./.venv/bin/python scripts/verify_prepush.py
```
Expect all 7 gates PASS, including the 9 modular pytest suites, the <4B
parameter budget, notebook zero-drift, and forbidden-fallback detection.

---

## Release re-qualification chain (two-commit protocol)

Editing `src/` invalidates release `2a0d17e`. Skipping this produces the
"Production freeze lineage rejected" failure at A100 launch.

**Step 1 — runtime commit**
```bash
git add src/ tests/ configs/
git commit -m "fix(training): gate warm-start LoRA behind explicit opt-in to prevent OOF leakage"
git push origin main
RUNTIME_SHA=$(git rev-parse HEAD)
```

**Step 2 — pin notebooks to the runtime SHA and run the hardware gate**
```bash
./.venv/bin/python scripts/generate_notebooks.py --commit "$RUNTIME_SHA"
./.venv/bin/kaggle kernels push -p notebooks
./.venv/bin/kaggle kernels status phucdangg/legalir-training     # poll to COMPLETE
```

**Step 3 — retrieve and validate the receipt**
```bash
./.venv/bin/kaggle kernels output phucdangg/legalir-training -p /tmp/kaggle_gate
cat /tmp/kaggle_gate/artifacts/task1/gates/kaggle_t4x2_report.json
```
Required in the receipt — all four, no exceptions:
- `"verdict": "PASS"`
- `"git_sha"` equals `$RUNTIME_SHA`
- `"weight_delta" > 0` (proves real optimizer steps, not a mock)
- `"devices": ["Tesla T4","Tesla T4"]` and `"cpu_fallback_used": false`

If the SHA does not match, the gate ran against different code. Re-run; do not
hand-edit the receipt.

**Step 4 — refresh the freeze tuple**
```bash
cp /tmp/kaggle_gate/artifacts/task1/gates/kaggle_t4x2_report.json artifacts/task1/gates/
./.venv/bin/python - <<PY
import json
from pathlib import Path
from src.release.fingerprints import compute_canonical_json_hash
rep = json.loads(Path("artifacts/task1/gates/kaggle_t4x2_report.json").read_text())
h = compute_canonical_json_hash(rep)          # pass the dict, not the Path
fp = Path("artifacts/task1/freeze/production_freeze.json")
fz = json.loads(fp.read_text())
fz["git_sha"] = rep["git_sha"]
fz["run_id"]  = "task1-<YYYYMMDD>-" + rep["git_sha"][:7]
fz["gates"]["kaggle_t4x2"]["report_sha256"] = h
fz["gates"]["kaggle_t4x2"]["verdict"] = "PASS"
fp.write_text(json.dumps(fz, indent=2, sort_keys=True) + "\n")
print("freeze updated; report hash:", h)
PY
./.venv/bin/python scripts/generate_notebooks.py     # re-reads SHA from the freeze
```

Note: `compute_canonical_json_hash` takes the parsed **dict**. Passing a
`Path` raises `TypeError: Object of type PosixPath is not JSON serializable`.

**Step 5 — release commit**
```bash
git add artifacts/task1/freeze artifacts/task1/gates notebooks/
git commit -m "chore(release): evidence bundle for runtime <sha7> (kaggle PASS, genuine dual-T4)"
git push origin main
```
Keep this commit to evidence files only. Mixing `src/` changes into a release
commit is what triggers the allowlist rejection
(`RELEASE_ONLY_DIFF_ALLOWLIST`, `src/release/provenance.py:29`).

**Step 6 — final approval**
```bash
./.venv/bin/python scripts/verify_release_approval.py
./.venv/bin/python scripts/colab/bootstrap.py --expected-sha $(git rev-parse HEAD)
```
Both must succeed. `verify_release_approval.py` prints
`SUCCESS: current HEAD is approved for A100 launch validation (CPU only)` —
note its own caveat: this is not hardware proof and not spending approval.

---

## A100 dispatch

Requires explicit user authorization — it spends money.

```bash
bash scripts/modal/run_modal_cli.sh --detach --hf-allow-public-repo --private
```

`--private` sets `LEGALIR_TEST_PHASE=private`, which restricts test-file
discovery to `private-official.json` / `private.json` and fails closed if
absent (no silent fallback to the 1,000-query public set).

## Post-run checks

1. Submission covers exactly **2,080** query IDs, 5 unique valid corpus docs each.
2. `run_manifest.json` has `"status": "RELEASED"` and
   `huggingface.uploaded == true` with a 40-hex `manifest_commit_sha`.
3. Fold manifests show warm-start **off**; the final-model manifest may show it on.
4. Compare the new OOF Recall@5 against the 89.53% baseline. A large jump
   should be treated as suspect until fold cold-start is confirmed in the logs.
5. Read `stage_timings` for the first real runtime measurement — until this
   exists, no runtime-improvement claim is defensible.
