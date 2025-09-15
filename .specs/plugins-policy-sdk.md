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

## 3) Traceability

- Code: [plugins/policy-sdk/src/lib.rs](../plugins/policy-sdk/src/lib.rs)
