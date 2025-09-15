# TODO — Per‑Crate SPECs (RFC‑2119) — Contract‑First, Proof‑Driven

This plan produces crate‑scoped SPECs with stable requirement IDs, wired into tests/CI/docs with idempotent regeneration.

---

## P0 — Blockers (in order)

### 0) Extend requirements extractor to support per‑crate SPECs (tools/spec-extract)

Decision: MUST — Foundation for traceability and proofs

- Rationale: We need machine‑readable requirements from each crate SPEC to enforce contract‑first changes and map req → code → tests.
- Files/dirs to touch:
  - `tools/spec-extract/src/main.rs` — extend to scan `<crate>/SPEC/SPEC.md` in addition to `.specs/orchestrator-spec.md`
  - `requirements/` — write one file per crate, e.g., `requirements/<crate>.yaml` (to be created)
  - `COMPLIANCE.md` — aggregate table gains per‑crate sections (to be created if missing)
- Acceptance Criteria:
  - The extractor detects RFC‑2119 requirements with stable IDs (e.g., `OC-CORE-1001`) from crate SPECs and emits deterministic YAML (sorted keys, stable order).
  - Second run produces no diffs when inputs unchanged (idempotent).
  - Requirements link back to SPEC section anchors and to code/tests paths referenced in the SPEC.
- Proof (run after implementation):
  - `cargo run -p tools-spec-extract --quiet && git diff --exit-code`
  - `bash ci/scripts/check_links.sh`

Notes:

- Anchor/ID convention to support: `\[(OC-[A-Z0-9\-]+)\]` in headings or list items; IDs MUST be unique per crate.
- Output path schema: `requirements/<crate>.yaml` derived from `package.name`.

---

### 1) CI job for SPECs (docs_specs)

Decision: MUST — Prevent spec/code drift and broken links

- Files/dirs to touch:
  - `ci/pipelines.yml` — add job `docs_specs` (to be created) that runs:
    - `cargo run -p tools-spec-extract --quiet`
    - `bash ci/scripts/check_links.sh`
    - `git diff --exit-code`
- Acceptance Criteria:
  - Job present and green on branch; re‑runs produce no diffs.
- Proof:
  - `grep -n "docs_specs" ci/pipelines.yml`
  - `cargo run -p tools-spec-extract --quiet && bash ci/scripts/check_links.sh && git diff --exit-code`

---

### 2) Create SPEC skeletons for MUST crates

Decision: MUST — Highest‑value crates require explicit normative contracts

Crates:

- `orchestrator-core` — queue invariants, fairness, capacity policies, determinism hooks.
- `orchestratord` — control/data plane surfaces, SSE framing, error envelopes, backpressure.
- `pool-managerd` — preload/ready lifecycle, restart/backoff, NVIDIA‑only guardrails.
- `plugins/policy-host` — WASI ABI, safety/purity, versioning and compatibility.

- Files/dirs to touch (each crate):
  - `<crate>/SPEC/SPEC.md` (to be created)
- SPEC skeleton (all crates):
  - Scope & versioning
  - Normative requirements (RFC‑2119) with IDs (e.g., `OC-CORE-1xxx`, `OC-Daemon-2xxx`, `OC-Plugin-3xxx`)
  - Error taxonomy & IDs (if applicable)
  - Observability (metrics/logs/audit)
  - Determinism/ordering/rollback semantics (if applicable)
  - Contracts (schemas/OpenAPI/CLI flags), generation hooks
  - Traceability map (req → code → tests → contracts)
- Acceptance Criteria:
  - SPEC.md exists with the sections above and at least 10 concrete MUST/SHOULD requirements per crate.
  - Requirement IDs unique per crate, anchored, and parsable by `tools-spec-extract`.
  - Link checker passes.
- Proof (after extractor support lands):
  - `cargo run -p tools-spec-extract --quiet && git diff --exit-code`
  - `bash ci/scripts/check_links.sh`

---

## P1 — High Priority

### 3) SPEC skeletons for SHOULD crates

Decision: SHOULD — Valuable contracts but lower blast radius

Crates and rationale:

- `plugins/policy-sdk` — SDK surface and compatibility. Decision: SHOULD.
- `contracts/config-schema` — normative configuration rules. Decision: SHOULD.
- `test-harness/determinism-suite` — deterministic verification scope. Decision: SHOULD.
- `test-harness/metrics-contract` — metric names/labels and budgets. Decision: SHOULD.
- `worker-adapters/*` (llamacpp-http, vllm-http, tgi-http, triton) — engine mapping tables, determinism knobs. Decision: SHOULD.

- Files/dirs to touch:
  - `<crate>/SPEC/SPEC.md` (to be created)
- Acceptance Criteria:
  - SPECs include clear requirement IDs and traceability to tests and/or OpenAPI/schema as applicable.
  - Extractor emits `requirements/<crate>.yaml` deterministically.
- Proof:
  - `cargo run -p tools-spec-extract --quiet && git diff --exit-code`
  - `bash ci/scripts/check_links.sh`

---

### 4) Wire SPEC IDs into code/tests/docs (traceability)

Decision: SHOULD — Make requirements discoverable in source and tests

- Files/dirs to touch (illustrative, adjust per crate):
  - Source modules: add `// OC-<ID>` comments next to key logic
  - Tests: reference `OC-<ID>` in `#[test]` names or doc comments
  - Docs: link from README “Why it exists” to SPEC anchors
- Acceptance Criteria:
￼
￼

## P2 — Medium Priority

### 5) SPECs for MAY crates (document why skipping or add minimal spec)

Decision: MAY — Lower value, internal tooling/harness

Crates and decision:

- `contracts/api-types` — MAY (types derive from OpenAPI; contract tracked there)
- `tools/openapi-client` — MAY (client gen validated by trybuild/UI tests)
- `cli/consumer-tests` — MAY (test harness; CDC behavior covered by tests)
- `test-harness/{e2e-haiku, chaos, bdd}` — MAY (harness glue; behavior lives in upstream specs)
- `worker-adapters/mock` — MAY (illustrative only)
- `tools/readme-index` — MAY (documentation tool)

- Files/dirs to touch:
  - If skipping: add a one‑paragraph rationale to the crate `README.md` under “What this crate is not” and link to upstream SPEC.
  - If adding minimal SPEC: create `<crate>/SPEC/SPEC.md` with 3–5 MUSTs relevant to the tool/harness.
- Acceptance Criteria:
  - Rationale present or minimal SPEC in place; link checker passes.
- Proof:
  - `bash ci/scripts/check_links.sh`

---

## P3 — Nice‑to‑Have

### 6) SPEC lint script (structure/IDs/anchors)

Decision: NICE — Quality gate for authors

- Files/dirs to touch:
  - `ci/scripts/spec_lint.sh` (to be created) — verify RFC‑2119 words present, IDs match `OC-<AREA>-<NNNN>`, anchors resolve
  - `ci/pipelines.yml` — add to `docs_specs` job (to be created)
- Acceptance Criteria:
  - Lint script returns non‑zero on violations; green on the branch.
- Proof:
  - `bash ci/scripts/spec_lint.sh && git diff --exit-code`

---

## SPEC Style Guide (appendix)

- Use RFC‑2119 conformance language; define a short glossary if needed.
- Assign stable, human‑meaningful IDs per crate area, e.g., `OC-CORE-1xxx`, `OC-CTRL-2xxx`, `OC-POLICY-3xxx`, `OC-ADAPT-4xxx`, `OC-TEST-5xxx`.
- One requirement per bullet/list item with a unique ID; avoid bundling.
- All IDs MUST appear in code/tests comments where enforced.
- SPECs MUST only use repo‑relative links and fit within 100 columns (link checker enforced).
- Regeneration of requirements is idempotent; second run must be diff‑clean.

---

## Execution Checklist (run top‑to‑bottom)

1) Extend `tools/spec-extract` to scan `<crate>/SPEC/SPEC.md` and emit `requirements/<crate>.yaml` (to be created)
2) Add CI job `docs_specs` to run extractor + linkcheck + diff‑check (to be created)
3) Create SPEC skeletons for MUST crates and commit
4) Run extractor and link checker; ensure diff‑clean on second run
5) Create SPEC skeletons for SHOULD crates and commit
6) Wire requirement IDs into code/tests/docs for MUST crates; repeat for SHOULD
7) Optional: add `ci/scripts/spec_lint.sh` and wire into CI (to be created)

---

## Proof Commands (quick)

- Workspace hygiene: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings`
- Tests: `cargo test --workspace --all-features -- --nocapture`
- Requirements regen: `cargo run -p tools-spec-extract --quiet && git diff --exit-code`
- Links: `bash ci/scripts/check_links.sh`

All commands above exist today, except those explicitly marked “to be created”. Ensure idempotency: re‑run all regenerations and verify `git diff --exit-code`.
