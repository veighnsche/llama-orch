# worker-adapters — Unit Tests (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- Trait conformance, retries/backoff (capped+jittered), error taxonomy mapping, redaction behavior.

## Test Catalog

- Trait Conformance
  - Implementations satisfy required adapter traits; compile-time checks augmented with minimal runtime assertions.

- Retry/Backoff Policy
  - Configure policy and assert computed delays (cap, base, jitter bounds) without sleeping.
  - Ensure retries stop on non-retriable errors.

- Error Taxonomy Mapping
  - Table-driven mapping from HTTP status/body codes to `WorkerError` variants.
  - Include upstream-specific error body shapes where applicable.

- Redaction Utilities
  - Ensure Authorization and sensitive params are redacted in logs and error messages.

## Execution

- `cargo test -p worker-adapters -- --nocapture`
- Keep tests hermetic; do not perform network I/O in unit scope.

## Traceability

- ORCH‑3330/3331: Error-class taxonomy.
- `worker-adapters/http-util` retry/backoff helpers.

## Refinement Opportunities

- Table-driven tests for HTTP error mapping helpers.
