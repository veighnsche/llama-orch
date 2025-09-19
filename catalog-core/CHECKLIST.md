# catalog-core — Production Readiness Checklist

This checklist enumerates what `catalog-core` must provide to be production‑ready as the shared model catalog, indexing, and verification library for llama‑orch.

## Scope & References

- Specs: `.specs/00_llama-orch.md` (§2.6 Catalog, §2.11 Model Selection & Auto‑Fetch)
- Metrics: `.specs/71-metrics-contract.md`, `.specs/metrics/otel-prom.md`
- Separation of concerns: see `model-provisioner` and `engine-provisioner`

## Index & Storage Semantics

- [ ] Atomic, crash‑safe writes
  - [ ] Write temp file then atomic rename; fsync directory and file
  - [ ] Index schema version/id; verify at load, refuse downgrade silently
- [ ] Concurrency safety
  - [ ] Advisory lock (e.g., lock file) for writers; readers tolerate stale view
  - [ ] Document multi‑process behavior expectations
- [ ] Canonical layout rules
  - [ ] Content addressable directory structure under `~/.cache/models`
  - [ ] Normalize and validate `local_path` under root (no traversal)

## API Surface (hardening)

- [ ] CRUD & listing
  - [ ] `get/put/set_state/list/delete` kept stable; typed errors for NotFound/Busy
  - [ ] `exists(id|ref)` and `locate(ModelRef)` helpers
- [ ] Eviction & GC
  - [ ] `evict(id)` removes files and index; report bytes reclaimed
  - [ ] `gc()` sweeps orphans; limit runtime; return `GcReport`
- [ ] Lifecycle & trust
  - [ ] Distinguish `lifecycle` (Active/Retired) from `trust_state` (verified/warned/unverified)
  - [ ] `verify(entry, expected)` helper that wraps `verify_digest` and records outcome

## ModelRef & Resolution

- [ ] Parsing completeness
  - [ ] `hf:org/repo[/path]`, `file:` (absolute), relative path, and `Url` variants
  - [ ] Normalization rules (case, trailing slashes) documented
- [ ] Ensure‑present contracts
  - [ ] Keep only `FileFetcher` here; network fetchers are delegated to `model‑provisioner`
  - [ ] Single‑flight locking guidance for callers to avoid duplication

## Observability & Metrics

- [ ] Metrics emission (by callers, crate provides hooks/types)
  - [ ] `catalog_verifications_total{result,reason}` counters
- [ ] Logs
  - [ ] Structured logs at call sites include `model_id`, `model_ref`, `digest`, `bytes`, `duration_ms`, and correlation id

## Security

- [ ] No package installs, no process execution
- [ ] Path normalization and denial of traversal outside model root
- [ ] Redaction rules for logs (do not leak tokens/credentials in URLs)

## Testing Strategy

- [ ] Unit tests
  - [ ] Index round‑trip; `delete` removes files/dirs
  - [ ] `ModelRef::parse()` edge cases
- [ ] Concurrency/crash tests
  - [ ] Simulate interrupted writes; ensure index is readable or recoverable
- [ ] Interop tests
  - [ ] Round‑trip with `model‑provisioner` (ensure‑present then register entry)

## Documentation

- [ ] README documents High/Mid/Low behaviors and example usage
- [ ] TODO kept current; CHECKLIST used for release gating
- [ ] Cross‑references to specs and provisioners

## Release Gating Criteria

- [ ] All tests green: `cargo test -p catalog-core`
- [ ] Clippy/fmt clean: `cargo fmt --all -- --check`, `cargo clippy --all-targets --all-features -- -D warnings`
- [ ] Deterministic and crash‑safe index behavior
- [ ] Docs (README, TODO, CHECKLIST) updated
