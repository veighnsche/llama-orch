# Config Schema SPEC — Validation & Generation (v1.0)

Status: Stable (draft)
Applies to: `contracts/config-schema/`
Conformance language: RFC‑2119

## 0) Scope & Versioning

Requirements are versioned as `OC-CONFIG-6xxx`.

## 1) Validation

- [OC-CONFIG-6001] Config MUST be strictly validated; unknown fields rejected (strict) or logged (compat) per mode.
- [OC-CONFIG-6002] Examples in tests MUST validate without errors.

## 2) Generation

- [OC-CONFIG-6010] Schema generation MUST be deterministic and idempotent across runs.

## 3) Traceability

- Code: [contracts/config-schema/src/lib.rs](../contracts/config-schema/src/lib.rs)
- Tests: [contracts/config-schema/tests/validate_examples.rs](../contracts/config-schema/tests/validate_examples.rs)
