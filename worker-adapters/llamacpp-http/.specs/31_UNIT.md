# llamacpp-http — Unit Tests (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- Request serialization, header construction, error mapping.

## Test Catalog

- Request Serialization
  - Build HTTP request with correct path, headers (including `Authorization?`), and body fields (prompt, max_tokens, ctx, seed, sampler_profile_version)
  - Unknown fields rejected or ignored per adapter policy; stable serialization order for determinism

- Header Construction
  - Required headers present; sensitive headers marked for redaction in debug

- Error Mapping
  - Table-driven mapping from llama.cpp error/status shapes to `WorkerError` variants
  - Non-JSON/malformed responses map to a generic typed error with captured context (redacted)

## Execution

- `cargo test -p worker-adapters-llamacpp-http -- --nocapture` (adjust package name if needed)
- Keep tests hermetic; no network I/O in unit scope

## Traceability

- Aligns with `worker-adapters/http-util` client usage and retry policy
- Error taxonomy alignment: ORCH‑3330/3331

## Refinement Opportunities

- Table-driven error mapping for HTTP status/code to WorkerError.
