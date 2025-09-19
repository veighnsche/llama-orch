# Minimal Auth Hooks (Home-Lab, Non-Enterprise)

Status: Accepted
Owners: @llama-orch-maintainers
Date: 2025-09-19
Affected Components: orchestrator API, worker registration, CLI network client, docs

## Current Situation Sketch (≤10 lines)
- CLI and tests call `orchestratord` HTTP endpoints (Axum) directly; SSE used for token streaming.
- `orchestratord` binds via env `ORCHD_ADDR`, default `0.0.0.0:8080` (see `orchestratord/src/main.rs`, `app/bootstrap.rs`).
- No authentication is enforced today; control/data planes are open locally; logs are JSON (tracing-subscriber).
- Config elsewhere follows a mix of env vars and schema-driven files (see `contracts/config-schema`).
- Workers/adapters connect back to `orchestratord`; there is no dedicated browser UI in the repo.
- Error taxonomy is JSON with stable codes; SSE frames follow `started|token|metrics|end|error`.
- This proposal defines a minimal auth seam only; no implementation changes.

---

## Motivation & Non-Goals

Motivation:
- Provide simple safety for non-loopback exposure; prevent drive-by scans and spoof workers.
- Leave identity breadcrumbs in logs (localhost vs token fingerprint) for traceability.
- Establish a future seam for reverse-proxy-provided auth without importing enterprise gravity.
- Practice least-privilege with negligible operational burden in a home-lab GPL project.

Non-Goals (explicitly out of scope):
- NO OIDC/OAuth2/SSO/MFA/sessions/refresh tokens/users/roles/tenants/policy engine/cookie auth.
- NO external IdP and NO mandatory reverse proxy.
- NO RBAC or multi-user concepts; a single shared secret only.
- NO code in this change; SPEC/docs only.

## Terminology
- AuthN: Authentication (who is calling). AuthZ: Authorization (what they can do).
- Loopback: 127.0.0.0/8 or ::1 bind/listen and same-host connections.
- Token fingerprint (fp6): a non-reversible 6-character identifier derived from a token (e.g., first 6 hex of SHA-256(token)).

## Contract (RFC-2119)
- [AUTH-1001] Server MUST accept a single static Bearer token when configured.
- [AUTH-1002] Server MUST refuse to start if bound to a non-loopback address and no token is configured.
- [AUTH-1003] Worker registration MUST require a valid token.
- [AUTH-1004] If `AUTH_OPTIONAL=true`, requests from loopback MAY skip auth; all other sources MUST present the token.
- [AUTH-1005] The system SHOULD NOT define roles/permissions; all authenticated calls share the same privilege.
- [AUTH-1006] The system MAY trust a reverse proxy that injects `Authorization` when `TRUST_PROXY_AUTH=true`; risks MUST be documented.
- [AUTH-1007] Token comparisons MUST use timing-safe equality.
- [AUTH-1008] Logs SHOULD add `identity=localhost` for loopback or `identity=token:<fp6>` for authenticated requests; never log the full token.

## Config Surface (aligns with existing style)
- `BIND_ADDR` (aka `ORCHD_ADDR` for orchestrator): default MUST match current server defaults `0.0.0.0:8080`.
  - NOTE: While loopback is recommended for local dev, this SPEC does not change runtime defaults.
- `AUTH_TOKEN` (string): required for non-loopback binds, and required for workers.
- `AUTH_OPTIONAL` (bool, default `false`): when `true`, loopback requests MAY skip auth.
- `TRUST_PROXY_AUTH` (bool, default `false`): when `true`, the system MAY trust upstream `Authorization` headers (documented risks).
- Tokens are shared-secret only; no multi-user concept.

## Request Header & Error Model
- Header: `Authorization: Bearer <token>`.
- Error mapping (HTTP/JSON), aligned with repository conventions:
  - 40101 `MISSING_TOKEN` — message, hint.
  - 40102 `BAD_TOKEN` — message, hint.
  - 40301 `NON_LOOPBACK_WITHOUT_TOKEN` — message, hint.
- Response body fields: `{ code: string|int, message: string, hint?: string }`.

## Security Considerations & Threat Model
- Stops: casual network scans, spoof worker registrations, accidental LAN exposure without a token.
- Does not cover: user management, delegation, granular policy, secret rotation.
- Risks: `TRUST_PROXY_AUTH=true` misconfiguration can allow header spoofing if proxy boundary is not airtight.

## Operational Guidance
- Local dev: bind to loopback; `AUTH_OPTIONAL=true` may be used for convenience.
- Exposing beyond loopback: configure `AUTH_TOKEN` and keep `TRUST_PROXY_AUTH=false` unless behind a trusted reverse proxy.
- Reverse proxy posture: terminate TLS/restrict access at proxy; inject `Authorization` if desired and set `TRUST_PROXY_AUTH=true` knowingly.
- Token hygiene: generate sufficiently random tokens; store in env/secret files; rotate by restarting with a new token; never log full tokens.

## Compatibility & Migration
- Default behavior for existing users remains unchanged on loopback.
- Enforced refusal only applies when running on a non-loopback bind without `AUTH_TOKEN`.
- Logging additions (`identity`) are additive and non-breaking.

## Verification Plan (Spec-level)
```gherkin
@spec @auth_min
Scenario: Loopback + AUTH_OPTIONAL=true + no token
  Given the server binds to 127.0.0.1
  And AUTH_OPTIONAL is true
  When a request without Authorization is sent from localhost
  Then it is accepted (200)
  And logs include identity=localhost

@spec @auth_min
Scenario: Loopback + AUTH_OPTIONAL=false + no token
  Given the server binds to 127.0.0.1
  And AUTH_OPTIONAL is false
  When a request without Authorization is sent from localhost
  Then it is rejected with 401 MISSING_TOKEN (40101)

@spec @auth_min
Scenario: Non-loopback + no token at startup
  Given BIND_ADDR is 0.0.0.0:8080
  And AUTH_TOKEN is unset
  When the server starts
  Then startup is refused with an explicit error

@spec @auth_min
Scenario: Wrong token
  Given AUTH_TOKEN is configured
  When a request presents a different token
  Then it is rejected with 401 BAD_TOKEN (40102)

@spec @auth_min
Scenario: Correct token
  Given AUTH_TOKEN is configured
  When a request presents the correct token
  Then it is accepted (200)
  And logs include identity=token:<fp6>

@spec @auth_min
Scenario: Worker registration without token
  Given AUTH_TOKEN is configured
  When a worker attempts to register without a token
  Then it is rejected with 401 MISSING_TOKEN (40101)
```

## Design Choice: Module vs. Crate (No Code)
Option B is recommended: a tiny internal workspace crate (e.g., `auth-min/`) shared by the server, worker, CLI, and tests. Centralizing header parse/validation, timing-safe comparison, and token fingerprinting avoids duplicate logic and fragile drift across binaries. It also enables a single feature flag for `TRUST_PROXY_AUTH` behavior and keeps dependencies minimal (no TLS/IdP). The crate remains private (publish = false), with a narrow public API returning normalized decisions for callers (allow/deny + identity breadcrumb). This aligns with existing workspace boundaries and prevents cyclic deps. Option A (per-crate modules) risks divergence and duplicated tests.

## Refinement Opportunities
- Consider flipping the default bind to loopback in a future breaking change.
- Provide a one-liner CLI helper to mint tokens (without committing to user management).
- Add a structured log field `auth_decision` for easier filtering in observability tools.
- Document a minimal Nginx/Caddy reverse-proxy posture as an appendix.

## Open Questions / Decisions Needed
- Exact namespacing of config keys to match existing conventions (retain `ORCHD_ADDR` vs. introduce `BIND_ADDR`?).
- Health/metrics endpoints exempt or not? Default stance in this SPEC: not exempt. Confirm repo policy.
- CLI environment variable names and precedence (env vs. config file) for the token.
- Whether worker adapters read the same token or a separate `WORKER_AUTH_TOKEN` (shared-secret vs. split secrets).
- Token fingerprint algorithm specifics (e.g., `sha256` then first 6 hex).
