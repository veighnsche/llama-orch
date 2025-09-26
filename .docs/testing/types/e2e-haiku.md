# E2E Haiku (GPU) — Guide and Proof Bundle

## What
End-to-end streaming token run on a real GPU with metrics capture.

## Where
- `test-harness/e2e-haiku/`.
- Proof bundles: `<crate>/.proof_bundle/e2e-haiku/<run_id>/` for crates emitting artifacts; harness may also emit under its own crate.

## Env
- `REQUIRE_REAL_LLAMA=1` (must be set)
- `LLORCH_RUN_ID` (recommended)

## Artifacts (see template)
- `gpu_env.json`
- `sse_transcript.ndjson`
- `metrics_snapshot.json`
- `run_log_redacted.md`
- `test_report.md`

## Links
- Template: `.proof_bundle/templates/e2e-haiku/README.md`
- Index: `.docs/testing/TEST_TYPES_GUIDE.md`
