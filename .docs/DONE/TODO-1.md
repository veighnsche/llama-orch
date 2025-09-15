# TODO — Test-First Execution Plan (per SPEC→SHIP Workflow v2)

Status: planning · Scope: move from pre-code scaffolds to test-first development across CDC, provider verify, properties, determinism, and E2E Haiku.

---

## 0) SPEC Traceability Tightening

- [x] Assign stable `ORCH-XXXX` IDs in `specs/orchestrator-spec.md` (normative requirements only)
  - AC: Each normative MUST/SHOULD has an `ORCH-XXXX` anchor.
  - Proof: Ran `cargo run -p tools-spec-extract` → requirements/index.yaml populated; re-run → unchanged; `COMPLIANCE.md` regenerated with spec links; diff clean on second run.
- [x] Plumb `x-req-id: ORCH-XXXX` into `contracts/openapi/*.yaml` for every endpoint/response touching those requirements
  - AC: OpenAPI validates; regen produces no change; COMPLIANCE.md shows links.
  - Proof: `cargo xtask regen-openapi` validated both specs; generated files unchanged; diff clean.

---

## 1) CDC (Pact) — Broaden Interactions (CLI consumer)

- [x] Happy-path + streaming skeleton
  - Files: `cli/consumer-tests/tests/orchqueue_pact.rs`
  - AC: Pact JSON includes streaming and backpressure interactions.
- [x] Error matrix coverage
  - AC: One pact entry per error symbol with representative request/response.
- [x] Session lifecycle
  - AC: Pact matches OpenAPI schemas; headers present when specified in spec.
- [x] Determinism knobs present in request
  - AC: Pact validates; types regenerate unchanged.

Proof: Ran `cargo test -p cli-consumer-tests -- --nocapture` → pact file updated under `contracts/pacts/` and tests passed.

---

## 2) Wiremock Stubs — Mirror Pact

- [x] Expand `cli/consumer-tests/tests/stub_wiremock.rs` with mocks that satisfy the new interactions (incl. SSE placeholder)
  - AC: All consumer tests pass against wiremock; basic shape asserts for SSE lines.
- [x] Snapshots
  - AC: Inline Insta snapshots added (admission, session info, SSE transcript); `cargo test` green (cargo-insta not required for inline snapshots).

Proof: `cargo test -p cli-consumer-tests -- --nocapture` passed, including SSE and snapshot tests.

---

## 3) Provider Verification — Tighten Against OpenAPI

- [x] Response schema validation
  - AC: All pact interactions validate shapes and headers per OpenAPI.
- [x] Header checks
  - AC: Tests assert numeric headers and seconds format.
- [x] Negative coverage
  - AC: Fails if pact contains unknown path or method.

Note: Provider tests remain on stubs; minimal logic comes later to satisfy them gradually.

Proof: `cargo test -p orchestratord --test provider_verify -- --nocapture` passed with schema and header assertions.

---

## 4) Orchestrator-Core Properties (scaffolds now, logic later)

- [x] Property test harness crate setup
- [x] Invariant: FIFO within same priority
- [x] Invariant: Priority fairness
- [x] Invariant: Reject vs Drop-LRU semantics

AC: Properties compile and are marked `#[ignore]` until logic lands; CI job `unit_props` runs them without failing the build (or guard via feature flag).

Proof: `cargo test -p orchestrator-core --tests -- --ignored` passed (placeholder properties compiled and are ignored).

---

## 5) Determinism Suite — Utilities & Format

- [x] Define stream snapshot format
- [x] Config fixtures (skeletons)
- [x] Runner skeleton

AC: All compile; `cargo test -p test-harness-determinism-suite` passes with ignored heavy tests; seeds count test keeps guard.

Proof: Ran `cargo test -p test-harness-determinism-suite -- --ignored` → helpers compiled; seeds guard and placeholder test passed.

---

## 6) E2E Haiku Harness — Flesh Out (Real model gate later)

- [x] Client helpers for OrchQueue v1
- [x] Metrics scrape
- [ ] Anti-cheat checks

AC: Test remains `#[ignore]` by default; documentation on how to run locally against a live worker.

Proof: `cargo test -p test-harness-e2e-haiku -- --ignored` passed; helpers compile and gated test skips without `REQUIRE_REAL_LLAMA=1`.

---

## 7) Metrics Contract Tests

- [x] Metrics linter runner
- [x] Label cardinality budgets

AC: Placeholder tests compile; CI job wires in but marked `continue-on-error` until emitters exist.

Proof: Placeholder tests created under `test-harness/metrics-contract/tests/`; compiled as part of workspace tests.

---

## 8) CI Wiring (tighten jobs)

- [x] Gate precommit
  - Proof: CI job fails on regen + spec-extract diffs; caching for Rust and models added.
- [x] CDC → provider → properties pipeline order
  - Proof: CI job order updated; `cdc_consumer` and `stub_flow` must pass before `provider_verify` runs.
- [x] Determinism and Haiku remain optional in CI until a GPU runner is available
  - Proof: CI job marked `continue-on-error: true`; documentation updated on how to flip the switch later.

---

## 9) Definition of Done Template (per ORCH-XXXX)

Copy this block into PR descriptions touching requirement(s):

```
ORCH-XXXX — <short title>
- Contracts: [ ] OpenAPI/Schema updated · [ ] Regen clean
- CDC: [ ] Pact interaction added · [ ] CLI snapshots updated
- Provider: [ ] Verify passes (path/method/status/schema/headers)
- Properties: [ ] Property tests green
- Determinism: [ ] 64 seeds byte-exact (engine/version pinned) [nightly]
- E2E: [ ] Haiku gate passes (GPU pref; CI CPU fallback allowed)
- Observability: [ ] Metrics/logs prove requirement
- Compliance: [ ] requirements/index.yaml maps req → tests → code
```

---

## Commands Reference

- `cargo xtask regen-openapi && cargo xtask regen-schema`
- `cargo run -p tools-spec-extract`
- `cargo test -p cli-consumer-tests -- --nocapture`
- `cargo test -p orchestratord --test provider_verify -- --nocapture`
- `cargo insta test` (when snapshots present)
- `cargo test -p orchestrator-core --tests -- --ignored` (properties scaffolds)
- `cargo test -p test-harness-determinism-suite -- --ignored`
- `cargo test -p test-harness-e2e-haiku -- --ignored`

---

## Notes

- Keep everything deterministic and diff-friendly; no business logic until tests exist.
- Prefer to land tests in small PRs (CDC → provider verify → properties) before implementing any logic to satisfy them.
