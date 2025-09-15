# NEXT TODO_2 — Test-First Execution Plan (per SPEC→SHIP Workflow v2)

Status: planning · Scope: move from pre-code scaffolds to test-first development across CDC, provider verify, properties, determinism, and E2E Haiku.

---

## 0) SPEC Traceability Tightening

- [ ] Assign stable `ORCH-XXXX` IDs in `specs/orchestrator-spec.md` (normative requirements only)
  - AC: Each normative MUST/SHOULD has an `ORCH-XXXX` anchor.
  - Proof: `cargo run -p tools-spec-extract` updates `requirements/index.yaml` (non-empty); `git diff --exit-code` after second run.
- [ ] Plumb `x-req-id: ORCH-XXXX` into `contracts/openapi/*.yaml` for every endpoint/response touching those requirements
  - AC: OpenAPI validates; regen produces no change; COMPLIANCE.md shows links.
  - Proof: `cargo xtask regen-openapi` then diff clean.

---

## 1) CDC (Pact) — Broaden Interactions (CLI consumer)

- [ ] Happy-path + streaming skeleton
  - Files: `cli/consumer-tests/tests/orchqueue_pact.rs`
  - Add interactions for `GET /v1/tasks/:id/stream` (SSE placeholder body), and `Retry-After`/`X-Backoff-Ms` headers.
  - AC: Pact JSON includes streaming and backpressure interactions.
- [ ] Error matrix coverage
  - Add interactions for each `ErrorKind`:
    - `ADMISSION_REJECT`, `QUEUE_FULL_DROP_LRU`, `INVALID_PARAMS`, `POOL_UNREADY`, `POOL_UNAVAILABLE`, `REPLICA_EXHAUSTED`, `DECODE_TIMEOUT`, `WORKER_RESET`, `INTERNAL`.
  - AC: One pact entry per error symbol with representative request/response.
- [ ] Session lifecycle
  - Interactions for `GET /v1/sessions/:id` (ttl fields), `DELETE /v1/sessions/:id`.
  - AC: Pact matches OpenAPI schemas; headers present when specified in spec.
- [ ] Determinism knobs present in request
  - Include `seed`, `sampler_profile_version`, `determinism` in happy-path interaction.
  - AC: Pact validates; types regenerate unchanged.

Proof: `cargo test -p cli-consumer-tests` writes updated pact files under `contracts/pacts/`.

---

## 2) Wiremock Stubs — Mirror Pact

- [ ] Expand `cli/consumer-tests/tests/stub_wiremock.rs` with mocks that satisfy the new interactions (incl. SSE placeholder)
  - AC: All consumer tests pass against wiremock; basic shape asserts for SSE lines.
- [ ] Snapshots
  - Add Insta snapshots for admission, session info, and an example SSE transcript (3–5 lines, placeholders).
  - AC: Snapshots committed and reviewed; `cargo insta test` green.

---

## 3) Provider Verification — Tighten Against OpenAPI

- [ ] Response schema validation
  - Extend `orchestratord/tests/provider_verify.rs` to validate pact response bodies against OpenAPI response schemas (status→schema), not just presence.
  - AC: All pact interactions validate shapes and headers per OpenAPI.
- [ ] Header checks
  - Enforce `Retry-After` and `X-Backoff-Ms` existence/format on 429 responses.
  - AC: Tests assert numeric headers and seconds format.
- [ ] Negative coverage
  - Add tests asserting provider doesn’t claim undeclared paths or statuses.
  - AC: Fails if pact contains unknown path or method.

Note: Provider tests remain on stubs; minimal logic comes later to satisfy them gradually.

---

## 4) Orchestrator-Core Properties (scaffolds now, logic later)

- [ ] Property test harness crate setup
  - Add `proptest` to `orchestrator-core` dev-deps; create `tests/props_queue.rs`.
- [ ] Invariant: FIFO within same priority
  - Generate sequences of enqueue/cancel; assert order property (placeholder assertions initially with `#[ignore]`).
- [ ] Invariant: Priority fairness
  - Property describing budget for `interactive` over `batch` under contention (placeholder now).
- [ ] Invariant: Reject vs Drop-LRU semantics
  - Encode decision boundary based on pool capacity; error codes must match.

AC: Properties compile and are marked `#[ignore]` until logic lands; CI job `unit_props` runs them without failing the build (or guard via feature flag).

---

## 5) Determinism Suite — Utilities & Format

- [ ] Define stream snapshot format
  - Create `test-harness/determinism-suite/src/lib.rs` with helpers to parse token streams and serialize snapshots (engine, seed, first-32 tokens, timing).
- [ ] Config fixtures (skeletons)
  - Commit per-engine config skeletons for two replicas with deterministic flags (comments only for now).
- [ ] Runner skeleton
  - Write `tests/byte_exact.rs` (ignored) invoking helpers; asserts byte-exactness for the 64 seeds.

AC: All compile; `cargo test -p test-harness-determinism-suite` passes with ignored heavy tests; seeds count test keeps guard.

---

## 6) E2E Haiku Harness — Flesh Out (Real model gate later)

- [ ] Client helpers for OrchQueue v1
  - Build small client helpers in `test-harness/e2e-haiku/src/` using `tools/openapi-client` to call `/v1/tasks` and stream.
- [ ] Metrics scrape
  - Add `/metrics` scrape util (prom parser) to compute token deltas.
- [ ] Anti-cheat checks
  - Repo grep for fixtures/haiku and minute+nonce combos; gate with `REQUIRE_REAL_LLAMA=1`.

AC: Test remains `#[ignore]` by default; documentation on how to run locally against a live worker.

---

## 7) Metrics Contract Tests

- [ ] Metrics linter runner
  - Add a small test binary or script that reads `ci/metrics.lint.json` and validates emitted metric names/labels from a synthetic registry (placeholder now).
- [ ] Label cardinality budgets
  - Document budgets per label and add guards (placeholder tests for now).

AC: Placeholder tests compile; CI job wires in but marked `continue-on-error` until emitters exist.

---

## 8) CI Wiring (tighten jobs)

- [ ] Gate precommit
  - Ensure regen + spec-extract diffs fail CI; add caching for Rust and models.
- [ ] CDC → provider → properties pipeline order
  - Ensure `cdc_consumer` and `stub_flow` must pass before `provider_verify` runs.
- [ ] Determinism and Haiku remain optional in CI until a GPU runner is available
  - Keep `continue-on-error: true`; document how to flip the switch later.

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
