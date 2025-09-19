# Wiring: engine-provisioner ↔ pool-managerd

Status: Draft
Date: 2025-09-19

## Relationship
- `pool-managerd` invokes `engine-provisioner` during preload/reload to prepare and start engine processes for a pool.

## Expectations on pool-managerd
- Call `EngineProvisioner::ensure(&PoolConfig)` only after model artifacts are present via `model-provisioner`.
- Supervise the spawned engine process (health checks, restarts/backoff) and manage drain/reload; do not assume `engine-provisioner` handles readiness.
- Provide device masks, ports, and flags via `PoolConfig` per policy; record `engine_version` in registry when known.

## Expectations on engine-provisioner
- Respect `PoolConfig.provisioning` (source/container mode, allow_package_installs, ports, flags) and return typed errors with remediation hints.
- Delegate model staging to `model-provisioner` and use `ResolvedModel.local_path`.
- Normalize runtime flags; enforce GPU-only behavior with fail-fast diagnostics when CUDA/GPU is unavailable (no CPU fallback).

## Data Flow
- Preload/reload: `ensure_present(model)` → `ensure(engine)` → health checks (manager) → `ready=true`.

## Error Handling
- Any error from `ensure()` keeps pool `ready=false` with `last_error` set; manager backoff policy decides retry cadence.

## Refinement Opportunities
- Introduce `prepare() -> PreparedEngine` to allow manager inspection/approval before spawn.
- Standardize engine version detection to publish to the registry.
