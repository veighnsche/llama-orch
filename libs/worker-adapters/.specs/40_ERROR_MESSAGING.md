# worker-adapters — Error Messaging (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Errors

- Map upstream HTTP/gRPC errors to `WorkerError` taxonomy consistently.
- Redact sensitive headers/keys in all error messages.

## Refinement Opportunities

- Centralize error mapping helpers in `http-util` with examples per adapter.
