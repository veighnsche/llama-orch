# engine-provisioner — Integration Tests (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- Preflight + ensure flows with stubbed pool-managerd; verify fail-fast GPU-only diagnostics.

## Test Catalog

- Ensure Flow (happy path)
  - GIVEN a stubbed pool-managerd and model-provisioner
  - WHEN `ensure_engine` is invoked for an engine
  - THEN the flow completes with `PreparedEngine` populated (engine, version, build_ref/digest) and the pool reports Ready

- GPU-Only Fail-Fast
  - GIVEN a fake environment with `cuda_present=false`
  - WHEN `ensure_engine` runs
  - THEN it fails fast with a diagnostic (no CPU fallback), and pool remains Unready

- Mode Variations
  - Source mode executes build steps; container mode pulls image; package mode respects policy gate and Arch/CachyOS mapping

- Error Handling & Retries (if present)
  - Transient failures in substeps trigger bounded retries with backoff; permanent errors surface immediately with typed diagnostics

## Fixtures & Mocks

- Stubbed pool-managerd client responding to `register_prepared_engine`/health queries
- Fake environment probe for CUDA/driver versions
- Builders for `PoolConfig` across modes

## Execution

- `cargo test -p engine-provisioner -- --nocapture`
- Keep integration hermetic (no network/clones); simulate steps

## Traceability

- GPU-only policy (home profile)
- Aligns with `pool-managerd` preload sequencing and `catalog-core` engine entries

## Refinement Opportunities

- Add container-mode integration with rootless podman.
