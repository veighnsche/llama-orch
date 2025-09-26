# pool-managerd — pool-managerd (core)

## 1. Name & Purpose

`pool-managerd` provides a lightweight registry for pool/replica lifecycle state that other daemons consult (e.g., `orchestratord`). It tracks liveness/readiness, version, last error, heartbeats, and an active lease count used for simple placement decisions. A stub binary is provided; the crate is primarily a library at this stage.

## 2. Why it exists (Spec traceability)

Traceability follows the leading workspace specs:

- Core orchestrator spec: [.specs/00_llama-orch.md](../.specs/00_llama-orch.md)
  - Pool readiness/liveness and lifecycle: ORCH-3010, ORCH-3011, ORCH-3038
  - Observability fields consumed by callers: ORCH-3027, ORCH-3028
  - Control plane interactions (drain/reload/health) consumed by `orchestratord`
- Home profile overlay: [.specs/00_home_profile.md](../.specs/00_home_profile.md) — single-host assumptions, fast reloads.


## 3. Public API surface

- Rust crate API (internal)

## 4. How it fits

- Part of the core orchestrator. Upstream: adapters, Downstream: workers.

```mermaid
flowchart LR
  callers[Clients] --> orch[Orchestrator]
  orch --> adapters[Worker Adapters]
  adapters --> engines[Engines]
```

#### Detailed behavior (High / Mid / Low)

- High-level
  - In-process registry that stores per-pool health, last error, version, heartbeats, and active lease counters. Used by API handlers to answer `/v1/pools/:id/health` and to influence placement decisions.

- Mid-level
  - Types: `health::HealthStatus { live, ready }`, `registry::Registry` with a `HashMap<String, PoolEntry>`.
  - PoolEntry fields: `health`, `last_heartbeat_ms`, `version`, `last_error`, `active_leases`.
  - Library-first: `src/main.rs` is a stub binary printing a message.

- Low-level (from `src/registry.rs`, `src/health.rs`)
  - Health: `set_health(pool_id, HealthStatus)` / `get_health(pool_id) -> Option<HealthStatus>`.
  - Errors: `set_last_error(pool_id, err)` / `get_last_error(pool_id) -> Option<String>`.
  - Version: `set_version(pool_id, v)` / `get_version(pool_id) -> Option<String>`.
  - Heartbeats: `set_heartbeat(pool_id, ms)` / `get_heartbeat(pool_id) -> Option<i64>`.
  - Leases: `allocate_lease(pool_id) -> i32`, `release_lease(pool_id) -> i32`, `get_active_leases(pool_id) -> i32` (never negative).

## 5. Build & Test

- Workspace fmt/clippy: `cargo fmt --all -- --check` and `cargo clippy --all-targets --all-features
-- -D warnings`
- Tests for this crate: `cargo test -p pool-managerd -- --nocapture`


## 6. Contracts

- None


## 7. Config & Env

- See deployment configs and environment variables used by the daemons.

## 8. Metrics & Logs

- Emits queue depth, latency percentiles, and engine/version labels.

## 9. Runbook (Dev)

- Regenerate artifacts: `cargo xtask regen-openapi && cargo xtask regen-schema`
- Rebuild docs: `cargo run -p tools-readme-index --quiet`


## 10. Status & Owners

- Status: alpha
- Owners: @llama-orch-maintainers

## 11. Changelog pointers

- None

## 12. Footnotes

- Specs:
  - Core: [.specs/00_llama-orch.md](../.specs/00_llama-orch.md)
  - Home overlay: [.specs/00_home_profile.md](../.specs/00_home_profile.md)
- Requirements: [requirements/00_llama-orch.yaml](../requirements/00_llama-orch.yaml)

### Additional Details
- Preload/Ready lifecycle, NVIDIA-only guardrails, restart/backoff behavior.


## What this crate is not

- Not a general-purpose inference server; focuses on orchestration.
