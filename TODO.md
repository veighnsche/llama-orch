# TODO — Active Tracker (Spec→Contract→Tests→Code)

This is the single active TODO tracker for the repository. Maintain execution order and update after each task with what changed, where, and why.

## P0 — Blockers (in order)

- [ ] Spec ambiguity requiring proposal approval before code (consult `.specs/`; if unclear, open proposal under `.specs/proposals/` and reference IDs)
- [ ] OpenAPI/Schema divergence detected by regen (`cargo xtask regen-openapi && cargo xtask regen-schema`) — must reconcile before Stage 0 freeze
- [ ] Missing pact or failing CDC in `contracts/pacts/` blocking provider development
- [ ] CI or formatting gate broken (`cargo fmt`, `clippy`, `bash ci/scripts/check_links.sh`) — fix immediately
- [ ] Determinism suite failing or infra unavailable (`test-harness/determinism-suite/`)
- [ ] Haiku E2E environment unavailable or misconfigured (GPU/driver/network); set up before Stage 6

---

## Roadmap — 4 Weeks Plan (Spec→Contract→Tests→Code)

Discipline: Follow `README_LLM.md` strictly — Spec → Contract → Tests → Code. Never implement before contracts and tests. Update this tracker after each item with links to commits, spec IDs, and evidence.

### Week 1 — Spec audit, contract freeze (Stage 0), CDC consumer green (Stage 1)

__Goal__: Confirm specifications, freeze contracts, generate clients, and produce consumer contracts/pacts and snapshots.

- [ ] Inventory and stabilize requirement IDs
  - [ ] Read `.specs/00_llama-orch.md`, `.specs/10-orchestrator-core.md`, `.specs/50-plugins-policy-host.md`, `.specs/51-plugins-policy-sdk.md`, `.specs/metrics/otel-prom.md`
  - [ ] Ensure RFC‑2119 terms and stable IDs exist; add missing IDs as needed with a proposal
  - [ ] Run requirements extractor: `cargo run -p tools-spec-extract --quiet` and commit regen outputs
  - [ ] Create/refresh `requirements/*.yaml` linking req → tests → code (stubs now)

- [ ] Proposals for any ambiguous areas (minimal)
  - [ ] Draft under `.specs/proposals/` with: problem, change summary, impacted areas, new/changed IDs, migration/rollback
  - [ ] Reference proposal IDs in TODO and commit messages

- [ ] Contract regeneration and freeze (Stage 0)
  - [ ] OpenAPI: `cargo xtask regen-openapi` and ensure `contracts/openapi/` compiles; examples validated
  - [ ] Config schema: `cargo xtask regen-schema`; ensure `contracts/config-schema/` examples compile
  - [ ] Verify diff‑clean regen: `cargo run -p tools-spec-extract --quiet && git diff --exit-code`
  - [ ] Add CI guard if missing in `ci/pipelines.yml` for regen diffs

- [ ] Consumer contract tests and snapshots (Stage 1)
  - [ ] `cli/consumer-tests/` — write pact tests against OpenAPI for control and data paths
  - [ ] Generate pact: save to `contracts/pacts/cli-consumer-orchestratord.json` (update if schema changes)
  - [ ] Snapshot tests for golden OpenAPI examples in `tools/openapi-client/tests/`
  - [ ] Add `bash ci/scripts/check_links.sh` to CI stage if not already

- [ ] Tooling readiness
  - [ ] Ensure `tools/openapi-client/` supports codegen and round‑trip compile with current OpenAPI
  - [ ] Ensure `tools/spec-extract/` outputs requirement maps (PR links in Progress Log)

### Week 2 — Provider verify (Stage 2), Core invariants properties (Stage 3)

__Goal__: Provider passes pact verification; implement orchestration core skeleton with property tests for invariants.

- [ ] Provider verification for `orchestratord/`
  - [ ] Implement minimal providers for all endpoints specified in `contracts/openapi/*.yaml`
  - [ ] Add provider verification in `orchestratord/tests/provider_verify.rs` using the pact from `contracts/pacts/`
  - [ ] Achieve green provider verify in CI

- [ ] Orchestrator core skeleton (`orchestrator-core/`)
  - [ ] Define queue, admission, and scheduling interfaces per `.specs/10-orchestrator-core.md`
  - [ ] Add unit tests for basic state transitions referencing `OC-…` IDs
  - [ ] Integrate with OpenAPI handlers through `orchestratord/`

- [ ] Property‑based tests (Stage 3)
  - [ ] Add `proptest` suites in `orchestrator-core/tests/props_queue.rs` for invariants (e.g., no job lost, fairness bounds)
  - [ ] Introduce model‑based tests for scheduling against simplified model
  - [ ] Confirm determinism assumptions in properties (no time‑based flakes)

- [ ] BDD harness enablement
  - [ ] Use `test-harness/bdd/` binary `bdd-runner` (see `.docs/testing/` and memory) to run features under `tests/features`
  - [ ] Support `LLORCH_BDD_FEATURE_PATH` to target features/spec‑IDs
  - [ ] Add initial features covering happy path admission and queue lifecycle

### Week 3 — Determinism suite (Stage 4), Observability plumbing (Stage 5), Adapters and pool manager

__Goal__: Prove byte‑level determinism across replicas and implement metrics per contract; build worker adapters and pool manager scaffolding.

- [ ] Determinism suite (Stage 4)
  - [ ] Implement `test-harness/determinism-suite/` configs to run two replicas per engine
  - [ ] Ensure byte‑exact stream comparison and snapshot capture
  - [ ] Record and persist proof artifacts per `VIBE_CHECK.md`

- [ ] Observability per `.specs/metrics/otel-prom.md` (Stage 5)
  - [ ] Implement metrics emission with labels: `engine`, `engine_version`, `trtllm_version` etc.
  - [ ] Add metrics scraper tests; ensure Prometheus format validates
  - [ ] Wire metrics into major code paths (admission, scheduling, dispatch, worker latency)

- [ ] Pool manager and adapters scaffolding
  - [ ] `pool-managerd/`: basic worker registry, health, and lease protocols per spec
  - [ ] `worker-adapters/mock/`: mock adapter satisfying contract for local runs
  - [ ] `worker-adapters/llamacpp-http/`, `worker-adapters/tgi-http/`, `worker-adapters/triton/`: stub implementations with config validation
  - [ ] Add conformance tests for adapters using shared test suite and config schema

- [ ] Policy host + SDK integration points
  - [ ] `plugins/policy-host/` host interface per `.specs/50-plugins-policy-host.md`
  - [ ] `plugins/policy-sdk/` minimal SDK per `.specs/51-plugins-policy-sdk.md`
  - [ ] CDC tests for policy decisions influencing admission and routing

### Week 4 — Real model E2E (Stage 6), hardening, docs, release hygiene

__Goal__: Pass Haiku E2E on real hardware, finalize documentation, and close the loop with proof bundles and hygiene.

- [ ] E2E Haiku
  - [ ] Configure environment: `TZ=Europe/Amsterdam`, `REQUIRE_REAL_LLAMA=1`; ensure GPU reachable
  - [ ] Run `test-harness/e2e-haiku/` end‑to‑end; fix issues until green within budget
  - [ ] Capture metrics deltas demonstrating improvement/regression per `.specs/metrics/otel-prom.md`

- [ ] Hardening and reliability
  - [ ] Stress via `test-harness/chaos/` to validate resilience under failures
  - [ ] Timeouts, backpressure, circuit breakers per spec; add tests
  - [ ] Remove dead code; ensure no BC shims exist per Golden Rules

- [ ] Documentation and artifacts
  - [ ] Ensure `.docs/test-case-discovery-method.md` and `.docs/spec-derived-test-catalog.md` are updated to reflect new/changed tests
  - [ ] Update `.plan/` progress (see per‑spec plan stubs) and link to fulfilled milestones
  - [ ] Add or refresh READMEs in each crate with usage and contract notes

- [ ] Release hygiene and archiving
  - [ ] Run the PR checklist from `README_LLM.md` and ensure all gates pass
  - [ ] Update this `TODO.md` with final state; if complete, run `bash ci/scripts/archive_todo.sh` to move to `.docs/DONE/TODO-<next>.md` and create a fresh `TODO.md`

---

## Daily Developer Loop (run locally before pushing)

- [ ] `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings`
- [ ] `cargo xtask regen-openapi && cargo xtask regen-schema`
- [ ] `cargo run -p tools-spec-extract --quiet && git diff --exit-code`
- [ ] `cargo test --workspace --all-features -- --nocapture`
- [ ] `bash ci/scripts/check_links.sh`

---

## Acceptance Gates (Definition of Done per stage)

- [x] Stage 0 — Contract freeze: OpenAPI + config schema regenerated; CI fails on diffs; examples compile
- [x] Stage 1 — CDC + snapshots: Pact + insta tests green before provider code
- [x] Stage 2 — Provider verify: orchestrator passes pact verification
- [ ] Stage 3 — Properties: core invariants via proptest are green and meaningful
- [ ] Stage 4 — Determinism: two replicas per engine; byte‑exact streams proven
- [ ] Stage 5 — Observability: metrics exactly per `.specs/metrics/otel-prom.md`
- [ ] Stage 6 — Real‑model E2E (Haiku): pass within budget; metrics delta observed

---

## Cross‑Cutting Tasks and Hygiene

- [ ] CI: ensure `ci/pipelines.yml` runs format, clippy, tests, regen checks, link checks, pact verify
- [ ] CODEOWNERS up‑to‑date for all crates and contracts
- [ ] Security: review `SECURITY.md`, dependency updates, and license headers
- [ ] Deterministic regen tools are idempotent (second run is diff‑clean)
- [ ] Ensure all tests and code reference requirement IDs in names and comments
- [ ] Maintain `requirements/*.yaml` linking req → tests → code with coverage notes

## Progress Log (what changed)

- 2025-09-16 01:40 CEST — Expanded root TODO into 4-week roadmap aligned with `README_LLM.md` (Spec→Contract→Tests→Code). Added P0 blockers, Daily Developer Loop, Acceptance Gates, and Cross‑Cutting Hygiene.
- 2025-09-16 01:42 CEST — Ran developer loop: `cargo fmt`, fixed clippy warnings in BDD harness (`test-harness/bdd/src/main.rs`, `tests/traceability.rs`), ensured `steps::registry()` referenced to avoid dead_code.
- 2025-09-16 01:44 CEST — Regenerated contracts: `cargo xtask regen-openapi` and `regen-schema` — both validated and diff-clean.
- 2025-09-16 01:45 CEST — Extracted requirements with `tools-spec-extract`; updated `requirements/*.yaml` and `COMPLIANCE.md` deterministically.
- 2025-09-16 01:47 CEST — Verified CDC consumer tests and pacts under `cli/consumer-tests/`; provider verification tests under `orchestratord/tests/provider_verify.rs` passed; trybuild OpenAPI client UI tests passed.
- 2025-09-16 01:49 CEST — Full workspace tests passed (`cargo test --workspace --all-features -- --nocapture`). Acceptance Gates updated: Stage 0, 1, 2 complete; later stages scaffolded.
