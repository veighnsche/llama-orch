# engine-provisioner — Contracts

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Provided Contracts

- Engine provisioning API (in-crate)
  - Trait `EngineProvisioner` with `plan(&PoolConfig) -> Plan` and `ensure(&PoolConfig) -> Result<()>`.
  - Function `provider_for(&PoolConfig) -> Box<dyn EngineProvisioner>`.
- Llama.cpp source provider
  - Ensures repo/ref is cloned, builds `llama-server` with CMake, maps deprecated flags, detects CUDA and falls back to CPU when needed, and spawns the server with normalized flags.
- Preflight tooling
  - Detects required tools; optionally installs via pacman when `allow_package_installs=true` and on Arch-like systems.

## Consumed Contracts (Expectations on Others)

- Model artifacts
  - `model-provisioner::ModelProvisioner::{ensure_present, ensure_present_str}` returns `ResolvedModel` with canonical path.
- Pool configuration
  - `contracts/config-schema` types (e.g., `PoolConfig`, `ProvisioningConfig`). The provider relies on these to choose source/container modes, ports, and flags.
- Pool manager lifecycle
  - `pool-managerd` supervises and performs health checks; this crate does not manage readiness or restart loops.

## Data Exchange

- Input: `PoolConfig` (engine selection, provisioning settings, ports, flags).
- Output: Process spawn (runtime) and plan steps for preview (build-time).

## Error Semantics

- Plan/ensure return structured `anyhow::Error` with context; missing tools or unsupported host configurations are surfaced with remediation hints.

## Versioning & Compatibility

- Trait and `provider_for` stable within the repo pre‑1.0; engine variants may expand.
- Flag normalization evolves with upstream engines; changes are documented and tested.

## Observability

- Providers log key steps (`git`, `cmake`, CUDA hints, flag normalization) and CPU/GPU fallback decisions.

## Security & Policy

- No secrets persisted; package installs only via system package manager when explicitly allowed.

## Testing Expectations

- Unit/integration: plan generation, preflight detection, CUDA→CPU fallback, flag normalization.

## Refinement Opportunities

- Container-first providers for vLLM/TGI/Triton; rootless podman by default.
- Return a `PreparedEngine` for supervision by `pool-managerd` before spawning.
