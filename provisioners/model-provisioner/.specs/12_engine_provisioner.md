# Wiring: model-provisioner ↔ engine-provisioner

Status: Draft
Date: 2025-09-19

## Relationship
- `engine-provisioner` is a consumer of `model-provisioner` to ensure model artifacts exist locally before starting engines.

## Expectations on engine-provisioner
- Call `ModelProvisioner::{ensure_present, ensure_present_str}` and consume `ResolvedModel { id, local_path }`.
- Do not write to the catalog directly; rely on model-provisioner to register/update entries via `catalog-core`.
- Surface staging errors with remediation hints (e.g., missing `huggingface-cli`).

## Expectations on model-provisioner
- Provide file-only default (`ModelProvisioner::file_only`) without network I/O.
- Optionally support `hf:` via shell-out when CLI is present; otherwise return an instructive error.
- Register/update `CatalogEntry` with `lifecycle=Active`.

## Data Flow
- engine-provisioner → model-provisioner → catalog-core → `ResolvedModel` → engine spawn flags.

## Error Handling
- Errors bubble to engine-provisioner; no partial index corruption.

## Refinement Opportunities
- Pluggable network fetchers behind features to avoid shell-outs.
- Expose a fast-path cache check (`exists/locate`) to reduce redundant work.
