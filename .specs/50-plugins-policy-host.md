# Policy Host SPEC — WASI ABI, Determinism, Safety (v1.0)

Status: Stable (draft)
Applies to: `plugins/policy-host/`
Conformance language: RFC‑2119

## 0) Scope & Versioning

Requirements are versioned as `OC-POLICY-4xxx`.

## 1) ABI & Determinism

- [OC-POLICY-4001] Default plugin ABI MUST be WASI; functions MUST be pure/deterministic over explicit snapshots.
- [OC-POLICY-4002] ABI versioning MUST be explicit; incompatible changes MUST bump MAJOR.

## 2) Sandboxing & Safety

- [OC-POLICY-4010] Plugins MUST run in a sandbox with no filesystem/network unless explicitly granted.
- [OC-POLICY-4011] Host MUST bound CPU time/memory per invocation and abort on overuse.

## 3) Telemetry

- [OC-POLICY-4020] Host MUST log plugin id/version, decision outcome, and latency.

## 4) Tooling Proxy: HTTP Fetch/Search (Client Docs Access)

- [OC-POLICY-4030] The policy host SHOULD expose a mediated HTTP fetch/search tool to clients (e.g., CLI agents), enforcing allowlists/denylists of domains and MIME types.
- [OC-POLICY-4031] The tool MUST redact secrets and PII as configured and MUST bound response size and rate.
- [OC-POLICY-4032] The tool MUST emit audit logs that include request URL (redacted), policy decision, and byte counts.
- [OC-POLICY-4033] The ABI for invoking tools MUST be stable and versioned; tool invocation SHOULD be deterministic with respect to inputs and policy snapshot.

## 4) Traceability

- Code: [plugins/policy-host/src/lib.rs](../plugins/policy-host/src/lib.rs)
- SDK: [plugins/policy-sdk/src/lib.rs](../plugins/policy-sdk/src/lib.rs)

## Refinement Opportunities

- Extend tool proxy to include `hf.hub_download` with digest verification and domain allowlists for safer model/doc retrieval.
- Add correlation ID propagation across all tool calls and surface in host logs for end‑to‑end tracing.
- Define a minimal deterministic caching layer for HTTP GETs to improve offline workflows while keeping auditability.
