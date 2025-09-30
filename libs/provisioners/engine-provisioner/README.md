# provisioners-engine-provisioner — provisioners-engine-provisioner (tool)

## 1. Name & Purpose

provisioners-engine-provisioner (tool)

## 2. Why it exists (Spec traceability)

- See spec and requirements for details.
  - [.specs/00_llama-orch.md](../../../.specs/00_llama-orch.md)
  - [requirements/00_llama-orch.yaml](../../../requirements/00_llama-orch.yaml)


## 3. Public API surface

- Rust crate API (internal)

### High / Mid / Low behaviors

- **High**
  - Prepare and start engine processes (initially `llama-server`) with normalized flags.
  - Fail fast when GPU is unavailable; do not proceed without GPU (per repo policy).
  - Emit a machine-readable handoff file for orchestrator binding with engine URL and metadata.

- **Mid**
  - Plan steps (clone/build/ensure tools) and execute ensure (spawn).
  - Delegate model staging to model-provisioner; read resolved model path.
  - Map legacy flags (`--gpu-layers`, `--ngl`) to `--n-gpu-layers`.

- **Low**
  - Optional package installs via pacman/AUR only when allowed and on Arch-like systems.
  - PID file and simple restart on crash are out-of-scope for MVP (documented in specs as follow-up).

## Inputs / Outputs

- **Input**
  - Pool config (`contracts/config-schema::*`), engine selection, ports, flags, and provisioning settings.
  - Resolved model info from model-provisioner (`id`, `local_path`).

- **Output**
  - Running engine process (listening host:port).
  - Handoff file for orchestrator (see below).

## Orchestrator handoff format (file)

Location (default): `.runtime/engines/llamacpp.json`

Example:

```json
{
  "engine": "llamacpp",
  "engine_version": "b1234-cuda",
  "provisioning_mode": "source",
  "url": "http://127.0.0.1:8080",
  "pool_id": "default",
  "replica_id": "r0",
  "model": {
    "id": "local:/models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "path": "/abs/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
  },
  "flags": ["--parallel","1","--no-cont-batching","--no-webui","--metrics"]
}
```

Orchestrator should watch/read these handoff files and bind adapters accordingly (see `bin/orchestratord/.specs/22_worker_adapters.md`).

## 4. How it fits

- Developer tooling supporting contracts and docs.

```mermaid
flowchart LR
  devs[Developers] --> tool[Tool]
  tool --> artifacts[Artifacts]
```

## 5. Build & Test

- Workspace fmt/clippy: `cargo fmt --all -- --check` and `cargo clippy --all-targets --all-features
-- -D warnings`
- Tests for this crate: `cargo test -p provisioners-engine-provisioner -- --nocapture`

### CLI (MVP)

- Build and run the engine provisioner CLI to provision a llama.cpp pool from a config file:

```
cargo run -p provisioners-engine-provisioner --bin engine-provisioner -- \
  --config requirements/llamacpp-3090-source.yaml
```

- Options:
  - `--config <path>` YAML or JSON matching `contracts/config-schema::Config`.
  - `--pool <id>` optional pool id; defaults to first `engine: llamacpp` pool.

- On success, writes a handoff file at `.runtime/engines/llamacpp.json` with `{ engine, engine_version, provisioning_mode, url, pool_id, replica_id, model, flags }`.


## 6. Contracts

- None


## 7. Config & Env

- CachyOS/Arch preferred tooling; optional `allow_package_installs` gate for pacman/AUR.
- Engine flags include deterministic defaults for tests: `--parallel 1`, `--no-cont-batching`, `--metrics`, `--no-webui`.

## 8. Metrics & Logs

- Minimal logs.

## 9. Runbook (Dev)

- Regenerate artifacts: `cargo xtask regen-openapi && cargo xtask regen-schema`
- Rebuild docs: `cargo run -p tools-readme-index --quiet`


## 10. Status & Owners

- Status: alpha
- Owners: @llama-orch-maintainers

## 11. Changelog pointers

- None
## 12. Footnotes

- Spec: [.specs/00_llama-orch.md](../../../.specs/00_llama-orch.md)
- Requirements: [requirements/00_llama-orch.yaml](../../../requirements/00_llama-orch.yaml)

## Policy note

- VRAM-only residency during inference (weights/KV/activations). No RAM↔VRAM sharing, UMA/zero-copy, or host-RAM offload; tasks that do not fit fail fast with `POOL_UNAVAILABLE`. See `/.specs/proposals/GPU_ONLY.md` and `/.specs/00_llama-orch.md §2.13`.

## What this crate is not

- Not a production service.

## Known shims and TODOs (Owner C)

- [DONE] Engine version capture: best-effort `/version` probe parsed to populate `engine_version`, with fallback to `source-ref + -cuda|-cpu`.
- [MVP DONE] Graceful shutdown: `stop_pool()` now sends `TERM` and waits up to 5s before `KILL`; upstream drain hooks TBD.
- [TODO] Restart-on-crash: add supervision loop and tests; MVP does not restart the process on failure.
- [DONE] Health/status mapping: readiness wait treats `503` as transient during model load; `/metrics` sanity checked when `--metrics` is set.
