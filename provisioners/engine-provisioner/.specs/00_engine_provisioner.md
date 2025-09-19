# engine-provisioner — Component Specification (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## 0) Purpose & Scope

`engine-provisioner` prepares engines for pools based on configuration. It plans and executes steps to fetch/build binaries or select container images, ensures required tools exist (optionally installing via the host package manager), and starts the engine process with normalized flags. Models are fetched via `model-provisioner`.

In scope:
- Planning (`Plan`, `PlanStep`) of provisioning actions for a pool.
- Engine‑specific ensure flows (e.g., llama.cpp source build) using CMake/git.
- Optional host package installs (Arch/CachyOS via pacman) when explicitly allowed.
- Normalizing engine runtime flags; CPU/GPU capability detection and fallbacks.
- Spawning the engine process.

Out of scope:
- Network model downloads (delegated to `model-provisioner`).
- Placement or health checks (delegated to `pool-managerd`/adapters).

## 1) Normative Requirements (RFC‑2119)

- [ORCH‑3600] The crate MUST expose a trait `EngineProvisioner`:
  - `fn plan(&self, pool: cfg::PoolConfig) -> Result<Plan>` returns a high‑level plan (`PlanStep { kind, detail }`).
  - `fn ensure(&self, pool: cfg::PoolConfig) -> Result<()>` executes the steps to prepare and start the engine.
- [ORCH‑3601] The function `provider_for(pool)` MUST return a provider matching `pool.engine` and MUST error on unknown engines.
- [ORCH‑3602] Llama.cpp (source mode):
  - The provider MUST clone the configured repo/ref if missing and MUST run CMake with `-DLLAMA_BUILD_SERVER=ON` into a dedicated build dir.
  - Deprecated flags `LLAMA_CUBLAS*` MUST be mapped to `GGML_CUDA=ON`; cached CMake entries MUST be invalidated if needed.
  - If CUDA is requested but `nvcc` is unavailable, the provider SHOULD attempt to discover a CUDA root and set `CUDAToolkit_ROOT` hints.
  - If CMake configure fails with CUDA, the provider SHOULD retry with a compatible host compiler (`gcc-13` or `clang`) and, failing that, MUST fallback to CPU‑only by forcing `-DGGML_CUDA=OFF`.
  - The resulting `llama-server` binary MUST exist before starting.
- [ORCH‑3603] Preflight tooling:
  - The provider MUST detect missing tools (git, cmake, make, gcc, optional `nvcc`).
  - The provider MUST NOT install packages unless `provisioning.allow_package_installs == true`.
  - When installs are allowed, the provider MUST restrict automatic installs to Arch‑like systems with `pacman`; other distros MUST return an instructive error.
  - If model ref scheme is `hf:` and `huggingface-cli` is missing, the provider MAY include it in the pacman install set when installs are allowed; otherwise MUST return an instructive error.
- [ORCH‑3604] Model artifacts:
  - The provider MUST delegate model staging to `model-provisioner::ModelProvisioner::ensure_present*` with the configured cache dir or default model cache dir.
- [ORCH‑3605] Runtime flags & spawning:
  - The provider MUST normalize llama.cpp flags, mapping legacy `--ngl/-ngl/--gpu-layers` to `--n-gpu-layers` and enforcing `--n-gpu-layers 0` when CPU‑only.
  - The provider MUST pass `--model <path>`, host, and port; it SHOULD write a PID file under a default run directory.
- [ORCH‑3606] Security:
  - The provider MUST avoid privilege escalation; package installs MUST use the system package manager and only when explicitly allowed.
  - Secrets MUST NOT be written to disk; environment hints are ephemeral.

## 2) Data Types & Semantics

- `Plan { pool_id: String, steps: Vec<PlanStep> }` — a human‑readable plan.
- `PlanStep { kind: String, detail: String }` — a step label with parameters.
- `cfg::*` — types from `contracts/config-schema` (`PoolConfig`, `ProvisioningConfig`, etc.).

## 3) Interfaces & Contracts

- API stability: `EngineProvisioner` and `provider_for` are stable within the repo pre‑1.0; changes MUST update call sites in the same PR.
- Integration: `pool-managerd` calls `ensure` during preload; `orchestratord` may call `plan` for dry‑run/preview.

## 4) Observability

- Providers SHOULD log plan steps and key sub‑commands (`git`, `cmake`) and report warnings when falling back to CPU‑only.
- Suggested metrics (emitted by daemons): build duration, restart counts, readiness transitions.

## 5) Security

- When installing packages, prefer system‑managed packages (Arch pacman/AUR) over language managers; do NOT use `rustup` in this crate.
- Avoid writing tokens/credentials to logs.

## 6) Testing & Proof Bundle

- Unit/integration tests SHOULD cover: plan generation, flag normalization, fallback paths (CUDA→CPU), preflight tool detection.
- Proof bundles SHOULD include logs of CMake configure/build and the final spawn command line (with redactions if needed).

## 7) Open Questions

- Introduce a `prepare()` API that returns `PreparedEngine` (metadata) separate from `ensure()`/spawn for tighter `pool-managerd` control?
- Container mode as the default for vLLM/TGI/Triton with rootless `podman`?

## 8) Refinement Opportunities

- Add container providers and unify preflight across engines.
- Improve CUDA discovery (toolchain configs) and surface better diagnostics.
- Cache build artifacts across pools; export binary version and build hash.
- Extend plan semantics with idempotency and conditional steps.
