# engine-provisioner — Exhaustive TODO

This provisioner prepares and manages engine runtimes per pool based on provisioning mode: `source | container | package | binary`. It must coordinate with `model-provisioner` to ensure models are present and with `pool-managerd` for supervised process lifecycle.

References
- `.specs/00_llama-orch.md` §2.12 Engine Provisioning & Preflight, §2.11 Model Selection & Auto‑Fetch (for interface with model-provisioner), §2.6 Catalog
- `.specs/20-orchestratord.md` (correlation id headers, backpressure payloads; for logs/observability)
- `.specs/71-metrics-contract.md`, `.specs/metrics/otel-prom.md` (labels, names)
- Arch/CachyOS preference: pacman/AUR when `allow_package_installs=true`

Current state (from code)
- `provider_for()` routes engines to provisioners: `llamacpp`, `vllm`, `tgi`, `triton` (`src/lib.rs`).
- `LlamaCppSourceProvisioner` fully implements `plan()` and `ensure()` (`src/providers/llamacpp.rs`). It:
  - Preflights tools and optionally installs via pacman if allowed (`preflight_tools()` in same file; Arch detection via `/etc/os-release`).
  - Caches at `~/.cache/llama-orch/llamacpp`; clones repo, checks out ref; configures CMake.
  - Maps deprecated `LLAMA_CUBLAS` to `GGML_CUDA` and retries with host compiler or CPU-only fallback.
  - Builds `llama-server`; ensures model via `huggingface-cli` if `hf:` ref; spawns server; writes pid file under `~/.cache/llama-orch/run`.
  - Normalizes flags (`normalize_llamacpp_flags()`).
- `util.rs` provides `default_cache_dir`, `default_models_cache`, `resolve_model_path`, `parse_hf_ref`, shell helpers; preflight packages include `cuda`, `clang`, `python-huggingface-hub`.
- `vllm.rs`, `tgi.rs`, `triton.rs` are stubs. `source.rs` is empty.
- Gaps: no container/package/binary modes implemented per engine; no use of `model-provisioner`/`catalog-core`; limited metrics/logging standardization; version stamping incomplete for non‑source modes.

Cross‑cutting tasks
- Interfaces & contracts
  - [ ] Define an engine-agnostic `EngineProvisioner` extension: `prepare(pool) -> PreparedEngine { bin, args, env, ports, workdir }` to allow `pool-managerd` to own spawn/supervision.
  - [ ] Replace in‑provisioner download path with `model-provisioner::ModelProvisioner` (remove direct `huggingface-cli` usage here).
  - [ ] Record `engine_version`: for source commit SHA; for containers image digest; for packages version string; for binaries artifact checksum.
  - [ ] Policy gates: apply outbound network/tooling allowlists (reuse policy hooks). No package installs unless `allow_package_installs=true`.
- Observability & logs
  - [ ] Structured logs with `job_id` (if applicable), `pool_id`, `engine`, `engine_version`, `policy_label`, `correlation_id`.
  - [ ] Emit metrics where applicable (counters for prepare success/failure; gauges for cache hits; optional timings for build/download).
- OS/package manager
  - [ ] Arch/CachyOS: prefer pacman and AUR (via `paru`/`yay` if present) for optional installs; keep current pacman path as baseline.
  - [ ] Non‑root flow: attempt `sudo -n` first; fall back to interactive `sudo` with explicit messaging; never require root-only without prompt.
- Concurrency & caching
  - [ ] Single‑flight lock per engine build dir to avoid duplicate builds.
  - [ ] Cache invalidation triggers: change in commit/ref, cmake flags, container tag/digest, package version, binary checksum.
- Security
  - [ ] Run engines as a non‑root sandbox user (created or configured); ensure dirs are owned accordingly.
  - [ ] Sanitize env (PATH, LD_LIBRARY_PATH) and pass only whitelisted vars.

Provisioning modes (all engines)
- Source
  - [ ] Preflight: `git`, `cmake`, `make`, C/C++ compiler, optional CUDA/HIP/Vulkan toolchains.
  - [ ] Build plan determinism: log inputs (repo, commit, flags), produce reproducible build steps in `Plan`.
  - [ ] Version stamping: write `engine_version` = commit SHA (+ build flags summary) to a metadata file.
- Container
  - [ ] Runtime selection: prefer `podman`, fallback `docker`.
  - [ ] Pull by digest where possible; verify digest; support authenticated registries.
  - [ ] Volume mounts for model cache; map ports from config; set device masks (NVIDIA).
  - [ ] Version stamping: image digest + label extraction (`org.opencontainers.image.revision`).
- Package
  - [ ] Package names per OS/distro; on Arch, prefer pacman/AUR names; honor `allow_package_installs`.
  - [ ] Version locking: install a specific version if available; record exact package version in `engine_version`.
- Binary
  - [ ] Download pinned URLs with strong checksum (sha256/sha512); unpack; store under `~/.cache/llama-orch/<engine>/bin`.
  - [ ] Version stamping: checksum + upstream version.

Per‑engine tracks

## llama.cpp
- High (must)
  - [ ] Keep source mode as default; use `model-provisioner` for models.
  - [ ] Normalize flags for CPU/GPU determinism; expose a consistent set of server args.
  - [ ] Detect/record GPU backend (CUDA/HIP/Vulkan) in `engine_version` metadata.
- Mid (should)
  - [ ] Container mode alternative using upstream/community images; set env/args accordingly.
  - [ ] Package mode (if available); binary mode (prebuilt `llama-server`).
- Low (nice‑to‑have)
  - [ ] Auto‑tune `--n-gpu-layers` heuristics per device when not provided; still deterministic via recorded computed value.

## vLLM
- High (must)
  - [ ] Container mode default (image + tag/digest); map model cache; device masks.
  - [ ] Source mode optional: Python env management (venv/conda) is in scope only if absolutely necessary; prefer containers.
- Mid (should)
  - [ ] Package/binary paths only when realistic; record exact versions and constraints.
- Low (nice‑to‑have)
  - [ ] Build flags/runtime knobs captured in metadata for determinism audits.

## HF TGI
- High (must)
  - [ ] Container mode default; set `--model-id` to staged model path where supported, or mount repo.
  - [ ] Health/readiness probing; port mapping and TLS options (future).
- Mid (should)
  - [ ] Source mode optional; package/binary variants if feasible.
- Low (nice‑to‑have)
  - [ ] Optimize startup flags for determinism and low‐latency first token.

## NVIDIA Triton / TensorRT‑LLM
- High (must)
  - [ ] Container mode default (NVIDIA runtime); mount model repo; device masks; MIG awareness (future).
  - [ ] Health/readiness; numa/cpu pinning options.
- Mid (should)
  - [ ] Package mode for TRT‑LLM where available; version pinning.
- Low (nice‑to‑have)
  - [ ] Binary mode for standalone TRT‑LLM server when practical.

Integration with model‑provisioner & pool‑managerd
- [ ] Replace direct download path with `model-provisioner::ensure_present` (HF/URL/S3/OCI).
- [ ] `pool-managerd` should call `prepare()` and own process spawn; provisioner returns `{ bin, args, env, ports, workdir }` and metadata.
- [ ] Capability reporting includes `engine_version`, declared concurrency, supported workloads.

CLI & UX (future)
- [ ] `orchctl engine up|down|status` orchestrates prepare and supervises via pool-managerd.
- [ ] Logs and errors are copy‑paste friendly with explicit commands and fixes.
- [ ] Arch/CachyOS users get pacman/AUR guidance when `allow_package_installs=false`.

Testing & Proof
- [ ] Unit tests per mode per engine (mock process/commands).
- [ ] Integration tests that perform a minimal CPU build for llama.cpp (CI‑suitable).
- [ ] BDD happy path: prepare -> pool ready -> simple decode.
- [ ] Metrics lint and linkcheck.

High/Mid/Low behavior (summary)
- High (must): source/container mode support per engine, model‑provisioner integration, deterministic version stamping, policy‑gated preflight.
- Mid (should): package/binary variants where feasible, richer observability, Arch/AUR helpers, container auth/digests.
- Low (nice‑to‑have): auto‑tuning helpers (deterministic), UX niceties, extended OS support.
