# Policy Host Plan — Home Profile

## Goals
- Provide a lightweight policy hook to allow/deny outbound HTTP tooling for agents.
- Keep policy execution deterministic and fast (WASI plugin or in-process hook).

## Tasks
1. Define minimal policy snapshot schema (request metadata, target URL, environment context).
2. Implement default “allow all” policy with logging.
3. Add configuration knobs in `contracts/config-schema` to point to a WASI module or enable built-in policies.
4. Emit audit logs for every policy decision (`allow`/`deny`, reason, correlation ID).
5. Add BDD scenarios to prove allow/deny paths.

## Testing
- Unit tests for policy host bridge (allow/deny, timeout handling).
- BDD feature `test-harness/bdd/tests/features/tooling/policy.feature`.

## Risks
- Ensure policy execution cannot block admission (add time budget / fallback behaviour).
- Provide clear error message to the CLI when a request is denied.
