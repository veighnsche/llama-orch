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

## 4) Traceability

- Code: [plugins/policy-host/src/lib.rs](../plugins/policy-host/src/lib.rs)
- SDK: [plugins/policy-sdk/src/lib.rs](../plugins/policy-sdk/src/lib.rs)
