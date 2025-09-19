# Proposal: Testing Ownership & Scope — Per‑Crate Behavior, BDD for Cross‑Crate Integration

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025‑09‑19

## 0) Motivation

Consolidate a clear testing policy so that:
- Each crate owns its behavior and unit tests.
- The root BDD harness and outer test suites validate integration, contracts, and end‑to‑end flows between crates, not internal behavior of a single crate.
- CI gates become faster and more reliable by reducing duplicated behavior tests across layers.

This aligns with README_LLM.md “Golden Rules” (Spec → Contract → Tests → Code) and the home profile focus.

## 1) Scope

In scope:
- Test ownership boundaries per crate vs. cross‑crate.
- Layers: unit, property, integration (within crate), provider verify/contract tests, BDD, determinism, metrics lint.
- CI signals and PR discipline.

Out of scope:
- Specific test implementations for every crate (covered by each crate’s TODO/CHECKLIST).

## 2) Normative Requirements (RFC‑2119)

IDs use the ORCH‑32xx range (testing ownership and scope).

- [ORCH‑3200] Each crate MUST provide its own unit and behavior tests colocated within that crate (e.g., `src/`, `tests/`).
- [ORCH‑3201] Cross‑crate behavior MUST NOT be tested inside a single crate; it MUST be covered by the outer BDD/integration layers.
- [ORCH‑3202] The root BDD harness (`test-harness/bdd`) MUST focus on cross‑crate integration and end‑to‑end flows (e.g., admission→stream, catalog→reload, health→capabilities) and MUST avoid asserting inner‑crate implementation details.
- [ORCH‑3203] Provider verification and contract tests MUST remain aligned to the contracts (`contracts/openapi/*`, `contracts/config-schema`) and MAY be run from the orchestrator’s test targets or the outer harness, but MUST assert contract‑level behavior only.
- [ORCH‑3204] The determinism suite (`test-harness/determinism-suite`) MUST validate byte‑exactness and replica consistency across the system and MUST NOT duplicate unit‑level sampler/engine tests.
- [ORCH‑3205] The metrics contract lint (`.specs/metrics/otel-prom.md` + `ci/metrics.lint.json`) MUST gate metric names/labels at emission sites; unit tests MAY assert per‑crate wiring but NOT full scrape outputs.
- [ORCH‑3206] CI MUST gate merges on: crate unit tests, contract/provider tests, BDD integration flows, determinism smoke (when enabled), and metrics lint.
- [ORCH‑3207] PRs that change a crate’s behavior MUST include tests in that crate. PRs that change cross‑crate flows MUST include or update BDD/contract tests.
- [ORCH‑3208] GPU‑required tests SHOULD be marked and scoped to relevant profiles (e.g., determinism/haiku), and MUST NOT block minimal CPU‑only unit tests.

## 3) Layer Responsibilities

- Per‑crate (owned by each crate)
  - Unit + behavior tests (logic, edge cases, properties, error taxonomy mapping).
  - Narrow integration tests within the crate boundary (e.g., storage backend selection, flag normalization logic), avoiding external network I/O unless explicitly mocked.
- Contracts (owned by `contracts/*` + orchestrator tests)
  - OpenAPI and schema consistency; provider verification.
- System harnesses (root)
  - BDD: orchestratord↔core↔adapters↔provisioners cross‑crate flows; SSE traces; cancel/429 behavior; catalog lifecycle; drain/reload; health/capabilities.
  - Determinism: byte‑exact streams under fixed seeds; per‑replica determinism; mixed‑GPU seeds corpus.
  - Metrics lint: names/labels and optional scrape checks in dev mode.

## 4) Mapping (Examples)

- `orchestrator-core`
  - Unit: queue invariants; backpressure policies; placement tie‑breakers; compatibility predicate.
  - NOT in BDD: internal queue order specifics; those are crate‑level tests.
- `pool-managerd`
  - Unit/integration: registry updates; health transitions; device mask validation; preload gating (mocked provisioners).
  - BDD: orchestrator health endpoints and capabilities reflecting `pool-managerd` state (cross‑crate).
- `catalog-core`
  - Unit: index round‑trips; delete semantics; `ModelRef::parse` edge cases.
  - BDD: catalog lifecycle only as cross‑crate flow via HTTP, not inner index semantics.
- `orchestratord`
  - Unit/integration: error envelope mapping; provider verify aligned to OpenAPI.
  - BDD: admission→stream tokens/metrics; cancel; reload; capabilities.

## 5) CI & Tooling Requirements

- [ORCH‑3210] `cargo test -p <crate>` MUST pass for each modified crate.
- [ORCH‑3211] `cargo test --workspace --all-features -- --nocapture` SHOULD pass as an aggregate smoke.
- [ORCH‑3212] `cargo xtask dev:loop` MUST include: fmt, clippy, regen (openapi/schema/spec‑extract), workspace tests, metrics linkcheck.
- [ORCH‑3213] BDD and determinism suites MAY be split into fast and full profiles; the fast subset MUST remain green on PRs.

## 6) Migration Plan

- Audit current tests and relocate:
  - Move inner‑behavior tests from `test-harness/bdd` into their respective crates.
  - Ensure BDD scenarios assert cross‑crate contracts and flows only.
- Update contributing docs and crate READMEs to state test ownership and expectations.
- Add CI jobs per crate (or matrix) to ensure modified crates run their suites.

## 7) Acceptance Criteria

- Repo builds CI with per‑crate unit tests, contract/provider verify, BDD integration, determinism smoke (optional), and metrics lint.
- No BDD scenarios assert inner behavior of a single crate.
- PRs modifying crate behavior fail without corresponding crate tests.

## 8) Refinement Opportunities

- Add a `make test-fast` profile that runs crate tests + minimal BDD.
- Provide a template for per‑crate test sections (README: High/Mid/Low behavior + test pointers).
- Tag BDD scenarios by feature (`@admission`, `@catalog`, `@reload`) to support selective runs in CI.
