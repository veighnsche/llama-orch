# Determinism Suite SPEC — Test Semantics (v1.0)

Status: Stable (draft)
Applies to: `test-harness/determinism-suite/`
Conformance language: RFC‑2119

## 0) Scope & Versioning

Requirements are versioned as `OC-TEST-7xxx`.

## 1) Semantics

- [OC-TEST-7001] Suite MUST verify byte‑exact token streams across replicas with fixed seeds.
- [OC-TEST-7002] Engine‑specific settings MUST be applied (e.g., single‑slot modes) for determinism.
- [OC-TEST-7003] Seeds corpus MUST contain at least 64 seeds and MUST be stable.

## 2) Traceability

- Code: [test-harness/determinism-suite/src/lib.rs](../test-harness/determinism-suite/src/lib.rs)
- Tests: [test-harness/determinism-suite/tests/byte_exact.rs](../test-harness/determinism-suite/tests/byte_exact.rs)
