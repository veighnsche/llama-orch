# worker-adapters/http-util — Shared HTTP Client & Helpers

Status: Draft
Owner: @llama-orch-maintainers

Purpose
- Provide a tuned shared `reqwest::Client` for all adapters (keep-alive, pool tuning, timeouts).
- Offer retry/backoff helpers with jitter and streaming decode utilities.
- Centralize secret redaction behaviors for adapter logs.

Spec Links
- `.specs/proposals/2025-09-19-adapter-host-and-http-util.md` (ORCH-3610..3613)

Refinement Opportunities
- Add per-request override knobs (timeouts/pool hints) with safe defaults.
- Provide a zero-copy SSE/stream decoder for token deltas.
