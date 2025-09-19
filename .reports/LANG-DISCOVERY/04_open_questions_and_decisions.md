# 04 — Open Questions and Decisions

## Unresolved Questions (clustered)

- Control flow semantics and transport robustness
  - Summary: Incremental SSE vs current transcript build, cancel token scope, heartbeat, micro-batching.
  - Candidate options: Keep current buffered transcript; adopt incremental SSE with backpressure and cancel-on-disconnect; add heartbeats.
  - Pros/cons: Incremental reduces memory and matches spec intent but requires new plumbing; heartbeats improve idle visibility but must remain compatible.
  - Sources: .specs/proposals/2025-09-19-token-streaming-and-cancel-robustness.md:7–24,35–76; .specs/20-orchestratord.md:32–37,26–31

- Placement policy and consumer overrides
  - Summary: Centralize decision logic and support pin/prefer/avoid/device-mask with fallback.
  - Candidate options: Keep current least-loaded heuristic; adopt centralized `policy::decide` + OpenAPI `TaskRequest.placement`.
  - Pros/cons: Central policy improves determinism/observability; adds contracts/tests and wiring work.
  - Sources: .specs/proposals/2025-09-19-centralized-placement-and-priority-policy.md:35–73,159–186

- Capability discovery surface
  - Summary: `/v1/capabilities` exists but examples and version pinning across CLI/client need firming.
  - Candidate options: Enrich payload (concurrency, features), add `x-examples`, tie to OpenAPI `info.version`.
  - Pros/cons: Better client ergonomics; requires regen and tests.
  - Sources: .specs/20-orchestratord.md:76–81; README.md:191–201

- Metrics and naming consistency
  - Summary: `decode_time_ms` vs `decode_ms` mismatch; SSE latency buckets unspecified.
  - Candidate options: Standardize on `decode_time_ms`; add metrics bucket guidance.
  - Pros/cons: Consistency aids tooling; may touch tests and dashboards.
  - Sources: .specs/00_llama-orch.md:88–89; .specs/20-orchestratord.md:43; .specs/metrics/otel-prom.md:112–118

- Security/auth seam
  - Summary: Minimal Auth Hooks are accepted as a seam; crate specs and config need alignment.
  - Candidate options: Keep home-profile open; document Bearer seam; enforce on non-loopback binds via config validation.
  - Pros/cons: Improves safety for LAN; minimal runtime changes.
  - Sources: .specs/60-config-schema.md:36–49; .specs/00_home_profile.md:9–13

## Conflicts Between Docs

- `decode_time_ms` vs `decode_ms`
  - Doc A: .specs/00_llama-orch.md:88–89 says canonicalize `decode_time_ms`.
  - Doc B: .specs/20-orchestratord.md:43 shows `end` → `{ tokens_out, decode_ms }`.

- Discovery surface naming
  - Doc A: .specs/20-orchestratord.md:16–17 removes `/v1/replicasets` and mandates `/v1/capabilities`.
  - Doc B: .docs/HOME_PROFILE.md:32–33 mentions "either enriched /v1/replicasets or dedicated /v1/capabilities" (legacy phrasing).

- Cancel robustness status
  - Doc A: Proposal requires cancel-on-disconnect and bounded channels. .specs/proposals/2025-09-19-token-streaming-and-cancel-robustness.md:17–24,41–58.
  - Doc B: Server spec currently omits these as MUSTs. .specs/20-orchestratord.md:32–37.

## Decision Proposals (DRAFT; minimize server changes)

- DRAFT: Adopt incremental SSE emitter with buffered writer and bounded channel; enable cancel-on-disconnect; default heartbeats off; keep micro-batch disabled.
  - Sources: .specs/20-orchestratord.md:32–37; proposals/token-streaming…:17–24,41–58,67–76.

- DRAFT: Standardize `end.decode_time_ms` in SSE and logs; maintain backward compatibility by emitting both for one release.
  - Sources: .specs/00_llama-orch.md:88–89; .specs/20-orchestratord.md:43.

- DRAFT: Promote centralized placement policy entry-point (`policy::decide`) and add `TaskRequest.placement` with safe defaults; log `DecisionLog` and add placement metrics.
  - Sources: proposals/centralized-placement…:35–73,97–153,249–256.

- DRAFT: Clarify discovery to `/v1/capabilities` only; add `x-examples` and client pinning to `info.version`.
  - Sources: .specs/20-orchestratord.md:76–81; README.md:191–201.

## Traceability Index

| Requirement/Decision | Source doc path(s) |
|---|---|
| SSE incremental + cancel-on-disconnect | .specs/proposals/2025-09-19-token-streaming-and-cancel-robustness.md:17–24,41–58; .specs/20-orchestratord.md:32–37 |
| Canonical `decode_time_ms` | .specs/00_llama-orch.md:88–89; .specs/20-orchestratord.md:43 |
| Centralized placement + overrides | .specs/proposals/2025-09-19-centralized-placement-and-priority-policy.md:35–73,159–186 |
| Capability discovery via `/v1/capabilities` | .specs/20-orchestratord.md:76–81; .docs/HOME_PROFILE.md:30–33 (conflict noted) |
| Metrics contract fields/labels | .specs/metrics/otel-prom.md:5–14,15–61,76–98,107–118 |

## Top 5 Opportunities

- Ship a migration note to standardize `decode_time_ms` and update provider tests/examples accordingly.
- Add OpenAPI `x-examples` for admission/SSE/metrics and for placement overrides once accepted.
- Define SSE heartbeat semantics and client timeouts to improve resilience under long queues.
- Document the Minimal Auth seam in crate specs and config examples; enforce non-loopback token requirement via config validation.
- Publish a placement decision log schema and metrics names (`placement_decisions_total`, `placement_candidates_considered`).
