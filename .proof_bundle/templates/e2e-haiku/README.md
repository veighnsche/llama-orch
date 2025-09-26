# E2E Haiku Proof Bundle Template

Required generated artifacts

- gpu_env.json — device model, driver, CUDA details
- sse_transcript.ndjson — streaming tokens from the haiku job
- metrics_snapshot.json — performance snapshot
- run_log_redacted.md — redacted run log
- test_report.md — summary (pass/fail, constraints satisfied)

Notes

- See `/.specs/00_llama-orch.md` §5 (Haiku E2E on real GPU).
- REQUIRE_REAL_LLAMA=1 must be enforced.
