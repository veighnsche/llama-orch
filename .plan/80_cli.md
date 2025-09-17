# CLI Plan — Home Profile

## Goals
- Provide a CLI that can submit tasks, watch SSE output, manage sessions, and inspect catalog/artifacts.
- Use CLI as the primary consumer for pact tests.

## Features
1. **Config Management**: load API token and endpoint, print effective settings.
2. **Task Submission**: stream tokens with live metadata (queue position, budgets).
3. **Session Tools**: list, inspect, delete sessions; show remaining budgets.
4. **Catalog Commands**: upload model metadata, trigger verify, flip state Active↔Retired.
5. **Artifacts**: upload plan snapshot, fetch by ID, list recent uploads.
6. **Diagnostics**: fetch `/v1/capabilities`, `/v1/pools/{id}/health`, `/metrics` snippet.

## Testing
- Pact tests in `cli/consumer-tests/tests/orchqueue_pact.rs` kept in sync with OpenAPI.
- Integration tests hitting `orchestratord` stub or mock server.
- Snapshot tests for CLI output (insta).

## Backlog
- Helper command to establish SSH tunnel automatically.
- Friendly output for artifact diff rendering.
