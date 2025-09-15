# Investigation — CDC (Pact) Coverage Plan

Status: done · Date: 2025-09-15 (plan)

## Desired Coverage

- POST /v1/tasks (202, 400, 429)
- GET /v1/tasks/{id}/stream (SSE frames)
- POST /v1/tasks/{id}/cancel (204)
- Sessions GET/DELETE

## Notes

- Include correlation-id headers in pact interactions where applicable.
- 429 pact body example includes `policy_label`, `retriable`, `retry_after_ms`.

## Next Steps

- Add/extend consumer tests and commit pact JSON under `contracts/pacts/`.
