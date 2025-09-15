# TODO — Per‑Crate READMEs & Root README Consolidation (Contract‑First, Proof‑Driven)

**Read this first:**

- `.specs/orchestrator-spec.md` — normative MUST/SHOULD with ORCH‑IDs (trace all claims)  
- `requirements/index.yaml` & `COMPLIANCE.md` — req → tests → code mapping (auto‑generated)  
- `contracts/openapi/*.yaml` — control/data plane (OpenAPI‑first, x‑req‑id)  
- `tools/spec-extract` — extracts requirements / anchors
- Existing TODOs for workflow context: `TODO-0.md`, `TODO-1.md`, `TODO-3.md`

> Goal: Every crate has a crisp, accurate `README.md` that tells **what it is**, **why it exists** (with ORCH‑IDs), **how to build/test it**, and **where it sits in the system**. Then the **root `README.md`** gets an **auto‑generated index** consolidating all crate READMEs so a developer instantly knows where to look.

---

## 0) Discovery & Inventory

- [x] Enumerate all workspace members from the top‑level `Cargo.toml` `[workspace]` list.
- [x] For each path, detect crate **kind** (lib/bin/mixed) and **role** (core, adapter, plugin, tool, test‑harness, contracts).
- [x] Build a JSON manifest (`tools/readme-index/readme_index.json`) with fields:
  - `path`, `name`, `kind`, `role`, `description`, `owner`, `spec_refs` (ORCH‑IDs), `openapi_refs`, `schema_refs`, `binaries`, `features`, `tests`, `docs_paths`.

**AC:** JSON manifest is deterministic (stable order) and checked into repo; re‑running the generator yields no diff when nothing changed.

**Proof:** `cargo run -p tools-readme-index --quiet && git diff --exit-code`

```bash
cargo run -p tools-readme-index --quiet && git diff --exit-code
# Passed (exit code 0)
```

---

## 1) Standard README Template (single source of truth)

Create a single canonical template at `tools/readme-index/TEMPLATE.md` used by all crates. Sections:

1. **Name & Purpose** — 1‑2 sentences explaining the crate’s role.
2. **Why it exists (Spec traceability)** — bullet list of ORCH‑IDs this crate helps satisfy, with anchor links.
3. **Public API surface** — brief; link to OpenAPI or Rust docs where applicable.
4. **How it fits** — short system diagram or ASCII block and bullets (upstream/downstream crates).
5. **Build & Test** — commands relevant to this crate (unit, CDC, provider verify, trybuild, BDD, determinism, e2e where applicable).
6. **Contracts** — OpenAPI/Schema/Pacts paths and how to regenerate (via `cargo xtask`).
7. **Config & Env** — key env vars, feature flags, config schema pointers.
8. **Metrics & Logs** — metric names/labels this crate emits; link to dashboards.
9. **Runbook (Dev)** — common tasks (run locally, regenerate artifacts).
10. **Status & Owners** — stability (alpha/beta/stable), CODEOWNERS/github handles.
11. **Changelog pointers** — link to `CHANGELOG_SPEC.md` if impacted.
12. **Footnotes** — additional references.

**AC:** Template uses only relative links (repo‑local), wraps at 100 cols, and passes the link checker.

**Proof:** `bash ci/scripts/check_links.sh`

```bash
bash ci/scripts/check_links.sh
# Passed (no output, exit code 0)
```

---

## 2) Generate README.md for Every Crate

Implement a small generator (suggest `tools/readme-index`) that:

- [x] Loads the JSON manifest and fills the template with crate‑specific data.
- [x] Pulls **spec anchors** from `.specs/orchestrator-spec.md` and `requirements/index.yaml` to auto‑populate the **Why it exists** section.
- [x] Pulls **OpenAPI** operation IDs & tags for crates that expose servers/clients.
- [x] Pulls **Schema** paths for config‑schema crate.
- [x] Includes **Commands** that are truly relevant per crate (e.g., `cargo test -p <crate>`, `cargo xtask regen-openapi`).
- [x] Adds **badges** (optional): crate kind (lib/bin), test status (textual), and stability.
- [x] Writes/updates `README.md` in each crate directory.

**AC:** All generated READMEs are idempotent and formatted with `mdformat`/`prettier` if configured. No TODO words left in final docs.

**Proof:** Re-run generator twice → no diff; `bash ci/scripts/check_links.sh` → OK.

```bash
cargo run -p tools-readme-index --quiet
git diff --exit-code
bash ci/scripts/check_links.sh
# All passed (exit code 0, diff-clean)
```

---

## 3) Hand‑Curated Enhancements per Crate (minimal but required)

For these crates, add extra details:

- **`/orchestrator-core`**: queue invariants & property tests overview; fairness; capacity policies.
- **`/orchestratord`**: data/control plane routes; SSE framing; backpressure headers; provider verify entry points.
- **`/pool-managerd`**: preload/ready lifecycle; NVIDIA‑only guardrails; restart/backoff behavior.
- **`/worker-adapters/*`**: per‑engine endpoint mapping tables (native/OpenAI‑compat → adapter calls); determinism knobs; version capture.
- **`/contracts/*`**: how to regenerate types, schemas, and validate; pact files location and scope.
- **`/plugins/*`**: WASI policy ABI and SDK usage; example plugin.
- **`/test-harness/*`**: when tests are ignored vs required; how to run real‑model Haiku; determinism suite scope.
- **`/tools/*`**: responsibilities, inputs/outputs; how “determinism” and “idempotent regen” are enforced.

**AC:** Each enhanced README contains a short “What this crate is **not**” to prevent misuse.

---

## 4) Root `README.md` — Consolidated Index (Auto‑Generated)

Add a section **“Workspace Map”** with a generated table listing every crate:

| Path | Crate | Role | Key APIs/Contracts | Tests | Spec Refs |
|------|------|------|---------------------|-------|-----------|
| `orchestrator-core/` | `orchestrator-core` | core lib | — | properties | ORCH‑3004, ORCH‑3005, … |
| … | … | … | … | … | … |

- [x] Insert per‑crate one‑liners (from template section 1) below the table as a quick glossary.
- [x] Link each row to the crate’s `README.md`.
- [x] Add **Getting Started** that points contributors to the right crate depending on the task (e.g., “adapter work,” “contracts,” “core scheduling”).

**AC:** Root README section is marked with begin/end comments so the generator can update it in place. Link checker passes.

**Proof:** `cargo run -p tools-readme-index --quiet && bash ci/scripts/check_links.sh`

```bash
cargo run -p tools-readme-index --quiet
bash ci/scripts/check_links.sh
# Passed (exit code 0)
```

---

## 5) CI & Quality Gates

- [x] Add a CI job `docs_readmes` that runs the generator and fails on uncommitted diffs.
- [x] Include the link checker and a simple “README lint” (max line length, section presence).

**AC:** CI is green on a fresh run; re‑runs produce no diffs.

**Proof:** `ci/pipelines.yml` updated; CI passes on branch with only README work.

```bash
grep -n "docs_readmes" ci/pipelines.yml
cargo run -p tools-readme-index --quiet
bash ci/scripts/check_links.sh
bash ci/scripts/readme_lint.sh
git diff --exit-code
# All passed locally; CI job present in pipelines.yml
```

---

## 6) Acceptance Checklist (per crate)

Copy into PR description when a crate README is first generated or significantly updated:

```
README(x) — <crate path>
- Template sections present and accurate
- Spec refs (ORCH‑IDs) resolve to anchors
- Commands runnable and minimal
- Contracts & regen steps correct
- Tests links correct (CDC/provider/properties/etc.)
- Links pass checker; doc is idempotent on regen
```

---

## 7) Stretch (Nice‑to‑Have)

- [x] Generate a small **ASCII system diagram** per crate using Graphviz/mermaid and embed it (rendered in GitHub).
- [x] Generate **Rustdoc TOC badges** linking to docs.rs (when published).

---

## Execution Order

1) Inventory → JSON manifest  
2) Template → TEMPLATE.md  
3) Per‑crate README generation  
4) Root README consolidation section  
5) CI job & link checks

---

## Commands (suggested)

- `cargo run -p tools-readme-index --quiet`  
- `bash ci/scripts/check_links.sh`
- `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test --workspace --all-features -- --nocapture`
