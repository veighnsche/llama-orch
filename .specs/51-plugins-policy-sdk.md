# Policy SDK SPEC — Surface, Compatibility (v1.0)

Status: Stable (draft)
Applies to: `plugins/policy-sdk/`
Conformance language: RFC‑2119

## 0) Scope & Versioning

Requirements are versioned as `OC-POLICY-SDK-4xxx`.

## 1) Stability & Compatibility

- [OC-POLICY-SDK-4101] Public SDK functions MUST be semver‑stable within a MAJOR.
- [OC-POLICY-SDK-4102] Breaking changes MUST be accompanied by a migration note and version bump.

## 2) Safety

- [OC-POLICY-SDK-4110] SDK MUST NOT perform network or filesystem I/O by default.
- [OC-POLICY-SDK-4111] SDK SHOULD provide typed helpers for invoking host-exposed tools (e.g., `fetch`, `search`) that accept policy hints (allowed domains, max bytes) and propagate correlation IDs.
- [OC-POLICY-SDK-4112] SDK MUST redact sensitive inputs/outputs according to configuration and provide hooks for additional user redaction.

## 3) Traceability

- Code: [plugins/policy-sdk/src/lib.rs](../plugins/policy-sdk/src/lib.rs)

## Refinement Opportunities

- Provide typed helpers for correlation ID propagation and tool proxy hints (allowed domains, max bytes), aligned with `orchestratord` correlation ID spec.
- Add example plugins illustrating deterministic tool usage and redaction hooks.
- Consider a schema/IDL for tool invocation requests/responses to stabilize CDC for plugin authors.
