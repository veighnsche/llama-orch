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
