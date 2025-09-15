# Investigation — Security & Tenancy

Status: done · Date: 2025-09-15

## Summary

- Auth: API key day‑1; quotas for concurrent jobs, tokens/min, KV‑MB enforced pre‑admission where applicable.
- Typed errors and quotas reflected in SPEC; Config includes tenant quotas via fairness.tenants.

## Proofs

- SPEC addendum and config schema changes present.
