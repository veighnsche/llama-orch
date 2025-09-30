# TODO — test-harness-e2e-haiku

- Document REQUIRE_REAL_LLAMA and expected GPU/driver environment.
- Add golden transcripts and SSE recordings to proof bundle.
- Cover model reload scenario end-to-end.
- Add per-request observability assertions (logs/metrics).

## New (progress + purge + verbose SSE)

- Add CLI progress bars driven by SSE `metrics.prep` for concurrent steps:
  - Group by step ID prefix (`engine:*`, `model:*`).
  - Render percentage when `pct` present; otherwise compute from `bytes_done/bytes_total`.
  - Fall back to spinner if only `status` and `human` are present.
- Use AdmissionResponse `streams` to choose verbose SSE when `preparation.steps` is non-empty.
- Differentiate tokens vs logs:
  - Treat SSE `token` as model output only.
  - Treat narration/progress as `metrics` (`data.human`, `data.prep`).
  - Keep server JSON logs separate (filter by `X-Correlation-Id`).
- Implement a `purge` subcommand that calls `POST /v1/pools/{id}/purge` for retesting:
  - Flags: `--engine`, `--model <ref>...`, `--drain-first`, `--force`.
  - Show progress via narration/`metrics` if available; otherwise textual status.
- Proof bundle updates:
  - Include `admission.json` (with `streams` and `preparation`).
  - Include `sse_transcript.ndjson` with `metrics.prep` frames.
  - Include `logs.jsonl` filtered by correlation id.
  - Verification: check minute word present exactly once; ensure any `prep.*.status` reach `completed` before first token.
