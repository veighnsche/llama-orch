# CLI — Implementation Plan (Developer UX / Consumer)

Scope: Command-line client for submit/stream/cancel, configuration, and quickstart flows. Supports CDC consumer tests and developer workflows.

## Stages and Deliverables

- Stage 1 — CDC Consumer usage
  - CLI drives pact interactions for enqueue, stream, cancel, sessions; emits snapshots for examples.

- Stage 6 — Product Journeys
  - Commands: `submit`, `stream`, `cancel`, optional `session` operations; correlation-id echo and pretty SSE rendering.
  - Config: server URL, auth, engine hints; environment overrides.

- Stage 11 — Config & Quotas
  - Validate config files; display effective quotas and budget warnings.

## Tests

- CLI snapshot tests under `cli/llama-orch-cli/tests/` (insta where applicable).
- Integration tests against orchestrator stub and live dev server.

## Acceptance Criteria

- Commands work end-to-end against OrchQueue v1; helpful error messages and retry hints.
- Snapshots are stable; CDC examples are easily reproducible.

## Backlog (initial)

- Progress bars and token-rate indicators for streams.
- Auth helpers and default profile management.
