# Pool Managerd SPEC — Preload, Readiness, Restart/Backoff (v1.0)

Status: Stable (draft)
Applies to: `pool-managerd/`
Conformance language: RFC‑2119

## 0) Scope & Versioning

Requirements are versioned as `OC-POOL-3xxx`.

## 1) Preload & Ready Lifecycle

- [OC-POOL-3001] Workers MUST preload at `serve` start and MUST NOT expose Ready until success.
- [OC-POOL-3002] Preload MUST fail fast if VRAM/host RAM insufficient; Pool remains Unready with retry backoff.
- [OC-POOL-3003] Readiness endpoints MUST reflect preload state and last error cause.

## 2) Restart/Backoff & Guardrails

- [OC-POOL-3010] Driver/CUDA errors MUST transition Pool to Unready, drain, and backoff‑restart.
- [OC-POOL-3011] Restart storms MUST be bounded by exponential backoff and circuit breaker.
- [OC-POOL-3012] CPU inference spillover is disallowed; controller MUST fail fast.

## 3) Device Masks & Placement Affinity

- [OC-POOL-3020] Placement MUST respect device masks; no cross‑mask spillover.
- [OC-POOL-3021] Heterogeneous split ratios MUST be explicit and capped for smallest GPU.

## 4) Observability

- [OC-POOL-3030] Emit preload outcomes, VRAM/RAM utilization, driver_reset events, and restart counters.

## 5) Traceability

- Code: [pool-managerd/src/main.rs](../pool-managerd/src/main.rs)
- Tests: (to be created) `pool-managerd/tests/`

## Refinement Opportunities

- Managed engine mode: define how `pool-managerd` supervises engine processes (spawn, health probes, restart/backoff) when provisioning mode is `source|container|package|binary`.
- Preload diagnostics: enrich readiness with last preload error cause and suggested fixes (e.g., insufficient VRAM, missing CUDA), including actionable hints for Arch/CachyOS.
- Backoff policy tuning: add configurable caps for restart storms and per-error-class backoff multipliers.
