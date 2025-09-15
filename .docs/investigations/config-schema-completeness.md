# Investigation — Config Schema Completeness

Status: done · Date: 2025-09-15

## Changes

- Added `catalog` trust policy; `admission.fairness` with tenants and weights; `preemption` block.

## Test

- Added `contracts/config-schema/tests/validate_v32_fields.rs` covering new fields.

## Proofs

- `cargo xtask regen-schema && git diff --exit-code`
- `cargo test -p contracts-config-schema -- --nocapture`
