# Proposal: UX/DX Improvements Prior to Contract Freeze

Status: draft · Date: 2025-09-15

## Problem

- Important UX/DX affordances are missing from the Data Plane and Control Plane contracts, making clients and tests more cumbersome:
  - No `x-examples` on key OpenAPI endpoints, making CDC and SDK generation less ergonomic.
  - 429/backpressure body lacks `policy_label` required by `.specs/00_llama-orch.md §6.2`.
  - `ErrorEnvelope` does not provide advisory `retriable` and `retry_after_ms` fields.
  - No standardized correlation ID behavior across requests/responses/SSE.

## Change Summary

- Add OpenAPI `x-examples` to data-plane endpoints: enqueue, stream (SSE frames), cancel, sessions.
- Extend 429 bodies with a policy label field.
- Extend `ErrorEnvelope` with optional advisory fields `retriable: boolean` and `retry_after_ms: int64`.
- Standardize `X-Correlation-Id` header handling (request optional, response echo mandatory) including SSE responses.

## Impacted Areas

- contracts/openapi/data.yaml
- `.specs/00_llama-orch.md` (normative updates to §6.2 and transport conventions)
- CDC tests (consumer pact examples, provider verification expectations)
- tools/openapi-client examples and generated clients

## Requirements (new/changed IDs)

- Changes are additive to existing requirements:
  - ORCH-2006 (ErrorEnvelope): update to include optional advisory fields.
  - ORCH-2007 (429 backpressure): update to require `policy_label` in JSON body in addition to HTTP headers.
  - New: ORCH-20XX (Correlation ID): define request/response header behavior and logging field alignment.

## Migration / Rollback

- Migration:
  - Clients may continue ignoring new optional fields; headers remain canonical.
  - SDKs can start leveraging `retry_after_ms` and `retriable` for improved retry logic.
- Rollback:
  - If needed, remove `x-examples` is non-breaking; optional fields can be ignored by servers and clients.

## Proof Plan

- OpenAPI regeneration is diff-clean on second run.
- CDC examples updated; consumer tests for 429 and correlation id; provider verification green.
- Link/spec lint scripts pass.

## References

- `.docs/investigations/ux-dx-pre-freeze.md`
- `.specs/00_llama-orch.md`
- `contracts/openapi/data.yaml`

## Managed Engine Provisioning UX (new)

### Problem
- Today, engines (e.g., `llama.cpp`) must be installed and started by the operator. We want the program to provision and manage engines automatically where allowed, improving out-of-the-box UX while keeping strong security and OS/package-manager preferences.

### Change Summary
- Introduce a managed provisioning layer with four backends (source preferred):
  - Source: build from git (repo+ref) using CMake/Make; supports CPU and GPU backends via build flags; caches artifacts; stamps `engine_version` with git commit.
  - Container: run an engine container (Docker/Podman/K8s) with volume-mapped model cache.
  - Package: install system packages (e.g., Arch AUR `llama.cpp`) when permitted.
  - Binary: fetch a pinned binary artifact with checksum verification into an app-local cache.
- Add a model fetcher (HF Hub/local registry) with digest verification and offline cache.
- Provide lifecycle commands: `orchctl engine up|down|status` and `pool-managerd` managed mode to spawn/monitor workers.

### Impacted Areas
- `contracts/config-schema` — new fields under `engine`/`pool`:
  - `provisioning.mode: source|container|package|binary`
  - `engine.id: llamacpp|vllm|tgi|triton`
  - `engine.version: string` (pin/constraint)
  - `engine.binary.url/checksum` (binary mode)
  - `engine.package.name` (package mode)
  - `engine.container.image/tag` (container mode)
  - `engine.source.git.repo` (URL), `engine.source.git.ref` (tag/branch/sha), `engine.source.submodules: bool`
  - `engine.source.build.cmake_flags: [string]`, `engine.source.build.generator: string`, `engine.source.cache_dir: path`
  - `model.ref` (repo+file or digest), `model.cache_dir`
  - `allow_package_installs: bool`, `allow_binary_downloads: bool`
  - `ports`, `env`, `flags`, `devices`

### Source backend specifics for llama.cpp (preferred default)

- Build inputs (examples): `cmake`, `make`, `gcc/g++` (or clang), optional GPU SDKs (CUDA/HIP/Vulkan) depending on target backend.
- Build flags (examples):
  - `-DLLAMA_BUILD_SERVER=ON` to produce `llama-server`.
  - GPU toggles such as `-DLLAMA_CUBLAS=ON` (CUDA), `-DLLAMA_HIPBLAS=ON` (ROCm), `-DLLAMA_VULKAN=ON`.
- Determinism-friendly server flags remain as in docs: `--metrics --no-webui --parallel 1 --no-cont-batching`.
- Version pinning: clone `engine.source.git.repo` at `engine.source.git.ref` (prefer full commit SHA). `engine_version` is reported as that SHA.
- Caching: reuse `engine.source.cache_dir` across runs; only rebuild when `git.ref` or build flags change.

### Requirements (new/updated IDs)
- New: ORCH-31XX Managed Provisioning — program MAY provision engines when allowed by policy; MUST verify checksums and pin versions/commits; MUST expose audit logs.
- New: ORCH-31XY Model Fetcher — MUST support HF transfer, digest verification, and offline cache.
- Update: ORCH-3027/3028 Observability — add engine provisioning/version labels and events.

### Security & Policy
- Non-root by default; builds run unprivileged. Elevate only when package installs are explicitly allowed.
- For source mode, verify remote URL against allowlist and pin to immutable commit SHA; optionally verify signed tags.
- Verify checksums/signatures for binaries and models; maintain allowlist of sources.
- Respect OS preference (e.g., Arch pacman/AUR) when `provisioning.mode=package`.
- Sandboxed runtime user for engine processes; dedicated cache dirs.

### Migration / Rollback
- Opt-in via config; default remains external engines. Rollback by switching mode to `external` without removing adapters.

### Proof Plan
- Add unit/integration tests for provisioner backends (mocked network/pm). Include source-mode builds in CI with a tiny target (CPU).
- E2E Haiku tests per adapter using managed provisioning (source mode by default; container/package as alternates).
- Config schema regen is diff-clean; docs updated; `orchctl engine up` demo for llama.cpp.
