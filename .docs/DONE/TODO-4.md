# TODO — Per‑Crate SPECs (RFC‑2119) — Contract‑First, Proof‑Driven

This plan produces crate‑scoped SPECs with stable requirement IDs, wired into tests/CI/docs with idempotent regeneration.

---

## P0 — Blockers (in order)

### 0) Extend requirements extractor to support per‑crate SPECs (tools/spec-extract)

Decision: MUST — Foundation for traceability and proofs

Status: DONE

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

Proof Output:

```bash
cargo run -p tools-spec-extract --quiet && git diff --exit-code
# requirements/* and COMPLIANCE.md written on first run; unchanged on subsequent runs

bash ci/scripts/check_links.sh
# OK (no missing targets)
```

Notes:

- Anchor/ID convention to support: `\[(OC-[A-Z0-9\-]+)\]` in headings or list items; IDs MUST be unique per crate.
- Output path schema: `requirements/<crate>.yaml` derived from `package.name`.

---

### 1) CI job for SPECs (docs_specs)

Decision: MUST — Prevent spec/code drift and broken links

Status: DONE

- Files/dirs to touch:
  - `ci/pipelines.yml` — add job `docs_specs` (to be created) that runs:
    - `cargo run -p tools-spec-extract --quiet`
    - `bash ci/scripts/check_links.sh`
    - `bash ci/scripts/spec_lint.sh`
    - `git diff --exit-code`

Proof Output:

```bash
grep -n "docs_specs" ci/pipelines.yml
# job present

cargo run -p tools-spec-extract --quiet && bash ci/scripts/check_links.sh && bash ci/scripts/spec_lint.sh && git diff --exit-code
# All passed (exit code 0; idempotent)
```

---

### 2) Create SPEC skeletons for MUST crates

Decision: MUST — Highest‑value crates require explicit normative contracts

Status: DONE (robust SPECs authored, not just skeletons)

Crates:
- `orchestrator-core` — queue invariants, fairness, capacity policies, determinism hooks.
- `orchestratord` — control/data plane surfaces, SSE framing, error envelopes, backpressure.
- `pool-managerd` — preload/ready lifecycle, restart/backoff, NVIDIA‑only guardrails.
- `plugins/policy-host` — WASI ABI, safety/purity, versioning and compatibility.

Artifacts:
- `.specs/10-orchestrator-core.md`
- `.specs/20-orchestratord.md`
- `.specs/30-pool-managerd.md`
- `.specs/50-plugins-policy-host.md`

Proof Output:

```bash
cargo run -p tools-spec-extract --quiet && bash ci/scripts/check_links.sh && git diff --exit-code
# Per-crate YAMLs written under requirements/; linkcheck OK; diff-clean on second run
```

---

## P1 — High Priority

### 3) SPEC skeletons for SHOULD crates

Decision: SHOULD — Valuable contracts but lower blast radius

Status: DONE (SPECs authored)

Crates and rationale:
- `plugins/policy-sdk` — SDK surface and compatibility. Decision: SHOULD.
- `contracts/config-schema` — normative configuration rules. Decision: SHOULD.
- `test-harness/determinism-suite` — deterministic verification scope. Decision: SHOULD.
- `test-harness/metrics-contract` — metric names/labels and budgets. Decision: SHOULD.
- `worker-adapters/*` (llamacpp-http, vllm-http, tgi-http, triton) — engine mapping tables, determinism knobs. Decision: SHOULD.

Artifacts:
- `.specs/51-plugins-policy-sdk.md`
- `.specs/60-config-schema.md`
- `.specs/70-determinism-suite.md`
- `.specs/71-metrics-contract.md`
- `.specs/40-worker-adapters-llamacpp-http.md`
- `.specs/41-worker-adapters-vllm-http.md`
- `.specs/42-worker-adapters-tgi-http.md`
- `.specs/43-worker-adapters-triton.md`

Proof Output:

```bash
cargo run -p tools-spec-extract --quiet && bash ci/scripts/check_links.sh && git diff --exit-code
# Per-crate YAMLs written; links OK; idempotent
```

---

### 4) Wire SPEC IDs into code/tests/docs (traceability)

Decision: SHOULD — Make requirements discoverable in source and tests

Status: DONE

- Files/dirs to touch (illustrative, adjust per crate):
  - Source modules: add `// OC-<ID>` comments next to key logic
  - Tests: reference `OC-<ID>` in `#[test]` names or doc comments
  - Docs: link from README “Why it exists” to SPEC anchors
- Acceptance Criteria:
  - Grepping `OC-` shows references in code/tests for all MUST crates.
  - Link checker passes.
- Proof:
  - `rg -n "\bOC-[A-Z]+-[0-9]+\b" -- **/*.{rs,md}`
  - `bash ci/scripts/check_links.sh`

Proof Output (examples):

```bash
rg -n "\bOC-[A-Z]+-[0-9]+\b" -- **/*.{rs,md} | head -n 5
orchestrator-core/src/lib.rs:41: /// OC-CORE-1010: health includes Ready state...
.specs/10-orchestrator-core.md:... OC-CORE-1001 ...
.specs/20-orchestratord.md:... OC-CTRL-2001 ...
```

---

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

Status: DONE

- Files/dirs to touch:
  - `ci/scripts/spec_lint.sh` (to be created) — verify RFC‑2119 words present, IDs match `OC-<AREA>-<NNNN>`, anchors resolve
  - `ci/pipelines.yml` — add to `docs_specs` job (to be created)
- Acceptance Criteria:
  - Lint script returns non‑zero on violations; green on the branch.
- Proof:
  - `bash ci/scripts/spec_lint.sh && git diff --exit-code`

Proof Output:

```bash
bash ci/scripts/spec_lint.sh && git diff --exit-code
# OK (no cross-file duplicate IDs; RFC-2119 detected)
```

---

### 7) SPEC file ordering with two‑digit prefixes (mother → children)

Decision: MUST — Stable global ordering and room for future insertions

Status: DONE

- Renamed SPECs with two‑digit prefixes where `00-llama-orch.md` is the root/mother SPEC and children follow in 10s/20s… with gaps for future insertions.
- Extractor enhanced to handle `NN-*.md` while preserving canonical requirement YAML names.

Proof Output:

```bash
ls -1 .specs | nl
# 00-llama-orch.md, 10-orchestrator-core.md, 20-orchestratord.md, 30-pool-managerd.md, ...

cargo run -p tools-spec-extract --quiet && git diff --exit-code
# idempotent; requirements/* unchanged on second run
```

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

---

## Appendix — Spec update actions (2025-09-15)

__Files changed__

- `.specs/00_llama-orch.md`
  - §3.2: Backpressure headers changed from SHOULD to MUST; 429 JSON body MUST include full policy label.
  - §3.7: Session TTL/turn defaults softened (SHOULD + configurable) and enforcement MAY be delegated to Policy Host; corrected internal reference to §3.17.
  - §3.9: Metrics label rules clarified — labels MUST include `engine`; engine-specific version labels SHOULD be included for per-engine/replica metrics; admission-level counters (e.g., `tasks_rejected_total`) MAY omit `engine_version`.
  - §6.2: Noted optional `X-Queue-Position`/`X-Queue-ETA-Ms` are non-normative hints.
  - §11: Added “Traceability Map” linking ORCH-* to OpenAPI and OC-* specs; References renumbered to §12.

- `.specs/20-orchestratord.md`
  - §3.1: Added authoritative SSE event payload fields: `started`, `token`, `metrics`, `end`, `error`; cross-referenced OpenAPI component schemas.

- `contracts/openapi/data.yaml`
  - Fixed `info.description` link to `.specs/00_llama-orch.md §6.2`.
  - Added `components/schemas`: `SSEStarted`, `SSEToken`, `SSEMetrics`, `SSEEnd`, `SSEError` with examples.
  - Annotated `GET /v1/tasks/{id}/stream` with `x-sse-events` listing event names and schema refs.

- `.specs/metrics/otel-prom.md`
  - Added normative exception under Conventions: admission-level counters (e.g., `tasks_rejected_total`) MAY omit `engine_version` (pre-engine rejection; cardinality control).
  - Noted omission explicitly under `tasks_rejected_total` metric entry.

- `.specs/71-metrics-contract.md`
  - Clarified tests are “to be created”; for now the linter `ci/metrics.lint.json` is authoritative.

__Rationale__

- Align MUST/SHOULD semantics for backpressure across umbrella spec, orchestratord, and OpenAPI.
- Control metrics cardinality while preserving observability guarantees.
- Formalize SSE payloads to enable client generation and robust contract tests.
- Improve traceability between ORCH-* IDs, OC-* component specs, and OpenAPI.

__Proof commands__

```bash
rg -n "Backpressure headers" .specs/00_llama-orch.md
rg -n "SSEStarted|SSEToken|SSEEnd|SSEError" contracts/openapi/data.yaml
rg -n "x-sse-events" contracts/openapi/data.yaml
rg -n "Event payload fields" .specs/20-orchestratord.md
rg -n "Admission-level counters" .specs/metrics/otel-prom.md
```

__Idempotency check__

```bash
cargo run -p tools-spec-extract --quiet && bash ci/scripts/check_links.sh && git diff --exit-code
