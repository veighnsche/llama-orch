# model-provisioner — Model provisioner: orchestrates resolve/verify/cache via catalog-core

## 1. Name & Purpose

Model provisioner: orchestrates resolve/verify/cache via catalog-core

## 2. Why it exists (Spec traceability)

- See spec and requirements for details.
  - [.specs/00_llama-orch.md](../../../.specs/00_llama-orch.md)
  - [requirements/00_llama-orch.yaml](../../../requirements/00_llama-orch.yaml)


## 3. Public API surface

- Rust crate API (internal)

### High / Mid / Low behaviors

- **High**
  - Resolve a `model_ref` to a local artifact path, register/update catalog entry, and return `ResolvedModel { id, local_path }`.
  - Prefer offline/local file-only mode for MVP; no network I/O by default.
  - Emit a machine-readable handoff (in-memory or file) to engine-provisioner.

- **Mid**
  - Implement `ensure_present_str` and `ensure_present` APIs; verify digest when provided; record verification result.
  - Normalize IDs for local-file refs; for `hf:` scheme, include org/repo[/path] (feature-gated later).

- **Low**
  - Optional shell-out to `huggingface-cli` when present and allowed by policy (future). Otherwise, return instructive error.

## Inputs / Outputs

- **Input**
  - `model_ref` (string or typed `ModelRef`) and optional expected digest.

- **Output**
  - `ResolvedModel { id, local_path }` for engine-provisioner.
  - Catalog registration/update with lifecycle: `Active` and digest recorded when provided.
  - Handoff JSON for engine-provisioner at a well-known path when invoked via `provision_from_config_to_handoff()`.

## Engine handoff (to engine-provisioner)

Engine-provisioner consumes `ResolvedModel { id, local_path }` to construct the final `llama-server` spawn command. The recommended integration is a direct function call or a small JSON alongside pool config for auditability.

Example handoff payload:

```json
{
  "model": {
    "id": "local:/models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "path": "/abs/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
  }
}
```

Handoff location recommendation (MVP): write to `.runtime/engines/llamacpp.json` adjacent to orchestrator config, per `TODO_OWNERS_MVP_pt2.md`. Use `DEFAULT_LLAMACPP_HANDOFF_PATH` or the convenience API below to avoid drift.

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
- Tests for this crate: `cargo test -p model-provisioner -- --nocapture`

### Selected deterministic Haiku model profile (MVP)

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` quantized `q4_k_m.gguf` (fits modest VRAM, fast startup)
- Example local file ref: `file:/models/TinyLlama-1.1B-Chat-v1.0-q4_k_m.gguf`
- Determinism notes: set seed and disable speculative/micro-batching at engine; provisioner only resolves path/metadata.

Config example (YAML):

```yaml
model_ref: "file:/models/TinyLlama-1.1B-Chat-v1.0-q4_k_m.gguf"
expected_digest: { algo: sha256, value: "<hex>" }
strict_verification: true
```

Programmatic use:

```rust
use model_provisioner::{
    provision_from_config_to_default_handoff, DEFAULT_LLAMACPP_HANDOFF_PATH
};
let meta = provision_from_config_to_default_handoff(
    "/etc/llorch/model.yaml",
    std::env::temp_dir(),
)?;
println!("handoff written to {} (model path: {})",
         DEFAULT_LLAMACPP_HANDOFF_PATH, meta.path.display());
```


## 6. Contracts

- None


## 7. Config & Env

- Not applicable.

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

## Refinement Opportunities

- Add GGUF header parsing to populate `ctx_max` and tokenizer info in `ModelMetadata`.
- Implement LRU cache accounting and eviction policy with provenance logs.
- Add `hf:` native fetcher (no shell-outs) gated by repo trust policy; support `pacman`/AUR packaged dependencies on Arch/CachyOS.
- Emit provenance bundle linking `catalog-core` entry and verification outcome into proof artifacts.

## Arch/CachyOS notes (optional network fetching)

- The crate prefers local file paths for MVP. For optional `hf:` shell-out support, install `huggingface-cli` via system packages.
- On Arch/CachyOS:
  - `sudo pacman -S python-huggingface-hub` provides the `huggingface-cli` tool.
  - If unavailable, prefer an AUR package or use a system-managed alternative (avoid ad-hoc pip installs in this repo).
  - If `huggingface-cli` is missing, calls to `hf:` will return an instructive error advising installation or using a local `file:` path.
