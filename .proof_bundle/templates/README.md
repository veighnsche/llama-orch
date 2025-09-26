# Proof Bundle Templates — Root

This directory defines the required artifact templates per test type. Each suite MUST write its artifacts to the repository root `.proof_bundle/` (or a crate-local `.proof_bundle/` when specified by that crate’s spec), with file names and structures described below.

Test types and templates
- unit/ — unit tests (crate-local)
- integration/ — integration tests (crate-local or cross-crate without network)
- contract/ — contract/provider/consumer tests
- bdd/ — root BDD harness over orchestrator HTTP boundary
- determinism/ — determinism suite (GPU/replica matching)
- chaos/ — chaos/load scenarios
- e2e-haiku/ — Haiku E2E anti-cheat on real GPU
- home-profile-smoke/ — Home profile reference environment smoke

Follow `.docs/testing/TESTING_POLICY.md` and `/.specs/00_llama-orch.md` §5 Testing & Validation.
