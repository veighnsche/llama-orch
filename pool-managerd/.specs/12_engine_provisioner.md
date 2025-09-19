# Wiring: pool-managerd ↔ engine-provisioner

Status: Draft
Date: 2025-09-19

## Relationship
- `pool-managerd` consumes `engine-provisioner` to prepare and start engine processes during preload and reload.

## Expectations on engine-provisioner
- Accept `PoolConfig` (engine, provisioning, ports, flags) and perform `ensure()` to build/spawn the engine.
- Delegate model staging to `model-provisioner`; do not flip readiness or supervise runtime.
- Log key steps (plan, git/cmake, CUDA diagnostics) and GPU-required fail-fast outcomes; return typed errors.
- Honor `allow_package_installs` strictly (Arch pacman-only); otherwise return remediation guidance.

## Expectations on pool-managerd
- Call `ensure()` only after model artifacts are present (via `model-provisioner`).
- Supervise the spawned process (health checks, restarts/backoff); manage drain/reload semantics.
- Record `engine_version` and `last_error` in the registry.
- Pass device masks and runtime flags according to pool policy.

## Data Flow
- Preload: `ensure_present(model) → ensure(engine) → health checks → ready=true`.
- Reload: drain → ensure_present(new model) → restart engine → health checks → ready=true.

## Error Handling
- Any `ensure()` error keeps `ready=false` and updates `last_error`; backoff policy applies to restarts.

## Refinement Opportunities
- Introduce `prepare() -> PreparedEngine` handshake so manager can inspect/approve before spawn.
- Standardize version discovery and publish to registry.
