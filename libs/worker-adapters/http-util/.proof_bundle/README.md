# Proof Bundle — http-util

This directory contains audit artifacts for the `http-util` crate.
Populate these files during CI or local verification runs to attach to PRs or releases.

Contents
- `metadata.json`: Build and environment metadata
- `ci_manifest.json`: CI job identifiers and links
- `retry_timeline.jsonl`: Retry schedule events with seed disclosure (one JSON object per line)
- `streaming_transcript.ndjson`: Streaming transcript events (started/token/metrics/end), one JSON object per line
- `redacted_errors.log`: Sample redacted error logs to verify secret scrubbing
- `test_report.md`: Human-readable test summary and pointers to artifacts
- `seeds.txt`: RNG seeds used in tests (determinism disclosure)

Guidance
- Do not store secrets in any of these artifacts.
- Ensure all logs include redaction of Authorization, X-API-Key, and bearer-like tokens per `./.specs/40_ERROR_MESSAGING.md`.
- When applicable, include the seed used for retry jitter calculations and note the policy parameters.
