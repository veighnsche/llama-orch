# Engine Provisioner — Component Specification (root overview)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Purpose & Scope

Prepares engine binaries or images for pools based on configuration. Plans and executes steps to fetch/build (or pull) engines, ensures required tools (optionally installing via Arch/CachyOS pacman when explicitly allowed), and returns a running engine or a prepared artifact for `pool-managerd` to supervise.

Out of scope: model staging (delegated to `model-provisioner`), placement, HTTP APIs.

## Provided Contracts (summary)

- Trait `EngineProvisioner` with `plan(&PoolConfig) -> Plan` and `ensure(&PoolConfig) -> Result<()>`.
- Provider selection by engine (llamacpp, vLLM, TGI, Triton), feature-gated.
- PreparedEngine metadata (version, build ref, digest, flags, mode, binary path).

## Consumed Contracts (summary)

- `contracts/config-schema` (`PoolConfig`, `ProvisioningConfig`).
- `model-provisioner` (`ResolvedModel`) for artifact paths.
- System package manager (optional, Arch-only) when allowed by policy.

## Key Flows

1) Preflight tools and environment (Arch pacman optional path when allowed).
2) Plan: clone/pull, configure/build (CMake flags normalization), or pull container image.
3) Ensure: produce `llama-server` (or engine) binary, discover/configure CUDA, and fail fast if GPU/CUDA is unavailable. GPU is required.
4) Spawn: return running process (or prepared artifact) with normalized flags.

## Observability & Determinism

- Narration logs for preflight, build, CUDA diagnostics and fail-fast, spawn, version.
- Emit `engine_version` and `engine_digest` for registry and proof bundles.

## Security

- No secret logging; package installs only via system manager when explicitly allowed.

## Testing & Proof Bundles

- Unit/integration: plan generation, preflight detection, CUDA diagnostics and fail-fast behavior, flag normalization.
- Proof bundles: CMake configure/build logs, spawn command line (redacted), PreparedEngine metadata.

## Refinement Opportunities

- Feature-gated providers with minimized deps.
- Build cache reuse (ccache), CUDA hint caching.
- Engine catalog writes for reproducibility.
