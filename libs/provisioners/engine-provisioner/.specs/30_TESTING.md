# engine-provisioner — Testing Overview (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- Plan generation, preflight detection, flag normalization, GPU-only fail-fast diagnostics.

## Test Catalog

- Plan Generation
  - Build plan for `ensure_engine(engine=llamacpp, version=X)` with normalized flags and environment derivation.
  - Arch/CachyOS policy: detect `pacman` presence for system-managed install paths (feature-gated); no runtime installs in unit/integration.

- Preflight Detection
  - CUDA runtime/driver checks → produce explicit diagnostic variants when GPU is missing or incompatible.
  - Disk/network prerequisites surfaced as typed diagnostics (never panic).

- Flag Normalization
  - Map diverse user flags to canonical set; unknown flags rejected with helpful error.
  - Table-driven coverage for aliases and defaults.

- Fail-Fast GPU Policy
  - When GPU not detected, return fail-fast diagnostic; no CPU fallback paths.

## Execution & Tooling

- Run: `cargo test -p engine-provisioner -- --nocapture`
- Keep tests hermetic; stub environment probes and filesystem checks.
- Consider `proptest` for normalization maps (keys/values fuzz) when beneficial.

## Traceability

- Home profile GPU-only policy: fail-fast expectations.
- Alignment with `pool-managerd` preload sequencing and `catalog-core` engine entries.

## Refinement Opportunities

- Add container-mode providers coverage and Arch/pacman install policy tests.
