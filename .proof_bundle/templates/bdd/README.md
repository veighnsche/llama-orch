# BDD Harness Proof Bundle Template

Required generated artifacts

- features_report.md — feature/scenario pass/fail matrix
- sse_transcripts/ — directory of per-scenario SSE transcripts (*.ndjson)
- request_response_samples/ — redacted HTTP request/response samples
- metrics_snapshots/ — optional scrape samples if used
- logs_redacted/ — redacted logs relevant to scenarios

Notes

- Exercise only public orchestrator APIs. No internal state mutation.
- Reference: `/.specs/72-bdd-harness.md`.
