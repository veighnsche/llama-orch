# Wiring: engine-provisioner ↔ model-provisioner

Status: Draft
Date: 2025-09-19

## Relationship
- `engine-provisioner` consumes `model-provisioner` to ensure model artifacts exist locally before starting the engine.

## Expectations on model-provisioner
- Provide `ModelProvisioner::{ensure_present, ensure_present_str}` returning `ResolvedModel { id, local_path }`.
- Register/update the catalog via `catalog-core` under the hood; do not spawn processes or install packages.

## Expectations on engine-provisioner
- Delegate model staging entirely; do not write to the catalog index directly.
- Use `ResolvedModel.local_path` to form engine command-line flags (e.g., `--model <path>` for llama.cpp).

## Data Flow
- Input: `PoolConfig.provisioning.model.ref` and optional `cache_dir`.
- Steps: engine-provisioner → model-provisioner → catalog-core → `ResolvedModel` → engine runtime.

## Error Handling
- Bubble up staging errors with context; include remediation hints (e.g., missing `huggingface-cli` if required by policy).

## Refinement Opportunities
- Add a fast-path `locate(ModelRef)` to bypass redundant fetch when artifacts exist.
- Surface digest verification outcomes to engine-provisioner logs for traceability.
