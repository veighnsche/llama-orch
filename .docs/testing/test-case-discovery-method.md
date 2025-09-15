# Spec-driven Test Case Discovery Method (SPEC→TEST)

Status: living document
Scope: Figure out all test cases for the program directly from the specs in `.specs/NN_*.md` and produce a complete, traceable test catalog.

## Principles (from README_LLM.md)

- Spec is the source of truth. Work order is Spec → Contract → Tests → Code.
- Requirements use RFC-2119 language with stable IDs (e.g., ORCH-xxxx, OC-CORE-1xxx).
- Tests must reference requirement IDs in names/docs and map to code via `requirements/*.yaml`.
- Regeneration must be idempotent; quality gates must pass before merge.

## Inputs

- Specs: `.specs/00_llama-orch.md`, `10-orchestrator-core.md`, `20-orchestratord.md`, `30-pool-managerd.md`, `40-43-worker-adapters-*.md`, `50-plugins-policy-host.md`, `51-plugins-policy-sdk.md`, `60-config-schema.md`, `70-determinism-suite.md`, `71-metrics-contract.md`.
- Contracts: `contracts/openapi/*.yaml`, `contracts/config-schema/*`.
- Existing requirement maps: `requirements/*.yaml`.
- CI linters: `ci/metrics.lint.json`.
- Test harness dirs: `test-harness/*`, package tests, BDD features.

## Method (how to discover all test cases)

1) Collect normative statements
   - Parse all `.specs/*.md` for RFC-2119 terms (MUST/SHOULD/MUST NOT/…).
   - Extract or assign stable requirement IDs adjacent to each normative statement.
   - If any normative statement lacks an ID, open a spec proposal to add one before proceeding.

2) Normalize into atomic testable requirements
   - Split combined requirements into atomic assertions (one behavior per ID or per test clause).
   - Define acceptance criteria for each assertion: inputs, expected outputs/side-effects, and error classes.

3) Classify test type for each requirement
   - Contract/API (OpenAPI provider/consumer, CDC).
   - BDD end-to-end and scenario-driven behavior (test-harness/bdd).
   - Property/invariant tests (queue fairness, determinism, placement).
   - Metrics/logs contract (names, labels, budgets, presence).
   - Configuration validation (schema strictness, examples).
   - Chaos/failover and performance SLO checks.

4) Map requirement → concrete test artifact(s)
   - Choose the package and path (e.g., `orchestrator-core/tests/props_queue.rs`, `orchestratord/tests/provider_verify.rs`, `test-harness/…`).
   - Propose test names that include the requirement ID.
   - For metrics, map to linter entries and BDD assertions where applicable.

5) Ensure traceability
   - For each requirement ID, add an entry to a test catalog with: ID, spec section, short description, test type(s), suggested test path, and cross-links to contracts.
   - Keep `requirements/*.yaml` aligned; add or update entries as needed.

6) Validate coverage and regenerate
   - Ensure every spec file contributes entries; no RFC-2119 statement left unmapped.
   - Run the regeneration tool (non-breaking, idempotent): `cargo run -p tools-spec-extract --quiet && git diff --exit-code`.
   - Run CI gates as in README (fmt, clippy as errors, tests, link checker).

7) Review discipline
   - If gaps or ambiguities are found, propose spec updates first (minimal proposals with impact and IDs). Then update contracts/tests.

## How I would do it (concrete plan)

- Step A: Extract normative statements and IDs using `tools/spec-extract` (or grep fallback) and compare with `requirements/*.yaml`.
- Step B: For each spec file, enumerate all requirement IDs and draft test cases per ID, classified by test type.
- Step C: Produce a human-readable catalog `.docs/spec-derived-test-catalog.md` that maps IDs → test stubs/locations and references to specs.
- Step D: Keep the catalog diff-clean across regenerations and update it on spec changes via proposals.

## Combinatorial coverage (cross-spec interactions)

Why: Many defects emerge at the intersection of requirements spanning multiple specs (e.g., scheduling × deadlines × adapters). Exhaustive cross-product is infeasible; use t-wise (pairwise/3-wise) testing with constraints.

How:

- Identify factors and domains from specs (examples):
  - Engine: `llamacpp | vllm | tgi | triton` (OC-ADAPT-5xxx)
  - Queue full policy: `reject | drop-lru | shed-low-priority` (ORCH-3005)
  - Fairness: `WFQ weights by tenant/priority` on/off; quotas on/off (ORCH-3075/3077)
  - Deadlines: `feasible | infeasible` (ORCH-3079)
  - Preemption: `off | soft | hard(capability)` (ORCH-3085/3086)
  - Lifecycle: `Active | Deprecated | Retired` (ORCH-3069..3073)
  - Artifact trust: `signed | unsigned` under `trust_policy=strict` (ORCH-3060..3065, 3093)
  - Auth: `apikey-present | missing` (OC-CTRL-2040)
  - Host/placement: `Ready | Unready`, `device mask respected` (OC-POOL-3001..3021)

- Define constraints to prune invalid combos:
  - Hard preemption only when adapter proves `interruptible_decode` (ORCH-3086).
  - CPU-only hosts cannot serve inference (ORCH-1101).
  - No cross-mask spillover (ORCH-3011/OC-POOL-3020).
  - OpenAI-compatible endpoints are internal-only (OC-ADAPT-5002/5021).
  - No determinism assumed across engine/model updates (ORCH-3047).

- Generate a minimal pairwise set (IPOG/PICT or equivalent). Elevate to 3-wise for high-risk triads:
  - (Fairness WFQ) × (Deadlines EDF) × (Preemption mode)
  - (Lifecycle Deprecated/Retired) × (Data-plane admission) × (Typed errors MODEL_DEPRECATED)
  - (Trust strict × unsigned artifact) × (Control plane ingest) × (UNTRUSTED_ARTIFACT)

- Prioritize by impact/severity; tag each combo to requirement IDs and planned test artifacts.

Artifacts:

- Keep a living matrix at `.docs/spec-combination-matrix.md` listing factors, constraints, and the prioritized pairwise/3-wise set with links to tests.
- Add a "Cross-Spec Interaction Tests" section to the catalog with the selected combos and their mappings.

## Deliverables

- This method document.
- A generated catalog: `.docs/spec-derived-test-catalog.md` with a complete, traceable list of tests tied to spec IDs.

## Ownership

- Maintained by the orchestrator team; updated whenever specs change or new components are added.
