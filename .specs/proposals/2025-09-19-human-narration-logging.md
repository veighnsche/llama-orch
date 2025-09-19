# Proposal: Human‑Readable Narration in Logs (Repo‑Wide)

**Status:** Approved
**Owner:** @llama-orch-maintainers
**Date:** 2025‑09‑19

## 0) Motivation & Goals

Humans must be able to skim the story of any scenario from logs while preserving machine‑readable structure. This proposal introduces a repo‑wide, human‑readable narration surface that complements existing structured fields and works in both pretty console and JSON outputs.

Goals:

* Human‑readable narration appears alongside structured fields in all relevant logs.
* Works with current `orchestratord` JSON logging and adds an opt‑in pretty mode for local dev without breaking CI defaults.
* Enforceable/observable via BDD and unit tests, without breaking current tests that read `state.logs`.

## 1) Non‑Goals

* No removal of existing structured fields or log levels.
* No forced OpenTelemetry (OTEL) adoption in this change (future‑gated only).
* No breaking changes to public APIs unless explicitly justified by follow‑up specs.

## 2) Current State (Brief)

Refer to discovery: `REPORTS/logging_narration_discovery.md` and `REPORTS/logging_narration_discovery.yaml`.

* `orchestratord/src/app/bootstrap.rs::init_observability()` initializes `tracing_subscriber` with JSON formatter and `EnvFilter` from `RUST_LOG`.
* Other binaries largely do not initialize `tracing`; they use `println!/eprintln!`.
* `orchestratord` emits ad‑hoc JSON strings into an in‑memory `state.logs` (e.g., `api/data.rs` pushes `{ "queue_position": 3, "predicted_start_ms": 420 }`) used by BDD assertions.
* SSE assembly (`services/streaming.rs`) logs `tokens_out` and `decode_ms` (note mismatch vs README’s `decode_time_ms`).
* No OTEL setup in manifests; no exporters configured.

## 3) Design Options Considered

A) New shared crate (observability/telemetry‑core)

* Pros: Single surface for narration API, taxonomy enforcement, init helpers, future OTEL exporters, and test shims.
* Cons: Cross‑crate wiring; must keep deps light.

B) Extend only `orchestratord` (e.g., `infra/logging.rs`)

* Pros: Fastest for primary service; small blast radius.
* Cons: Not shared; inconsistent across crates; duplicates over time.

C) Hybrid (Chosen): tiny shared core + thin per‑binary init

* Pros: Shared narration API/taxonomy with minimal deps; binaries adopt a small init to choose pretty vs JSON; `orchestratord` implements first.
* Cons: Slight upfront design; must bridge current `state.logs` capture to avoid BDD breakage.

## 4) Normative Requirements (RFC‑2119)

IDs use the ORCH‑33xx range (narration & logging).

* \[ORCH‑3300] Every significant event/span emitted by services and APIs MUST attach a short, plain‑English narration string, under a consistent field (e.g., `human`).
* \[ORCH‑3301] Narration MUST preserve all existing structured fields and log levels; it is additive.
* \[ORCH‑3302] Narration MUST NOT include secrets or PII. The logging surface MUST offer helpers to redact common sensitive tokens.
* \[ORCH‑3303] Narration MUST function in both pretty console and JSON outputs.
* \[ORCH‑3304] The logging surface MUST define a minimal field taxonomy so narration complements structure (at least: `actor`, `action`, `target`, relevant IDs: `req_id|job_id|task_id|session_id|worker_id`, and contextual keys such as `error_kind`, `retry_after_ms`, `backoff_ms`, `duration_ms`).
* \[ORCH‑3305] Narration SHOULD be ≤ \~100 characters, present tense, subject‑verb‑object, and include actor + intent/outcome.
* \[ORCH‑3306] The logging surface SHOULD provide a test‑time capture adapter that allows BDD/unit tests to assert narration presence without breaking the current `orchestratord` BDD steps that read `state.logs`. While migration is in progress, production code MAY continue to push compatibility lines to `state.logs`.
* \[ORCH‑3307] BDD SHOULD gain step‑scoped spans derived from step text and parity checks asserting narration exists for key flows; an optional “story snapshot” MAY be produced for golden tests.
* \[ORCH‑3308] An environment/config toggle MAY switch pretty vs JSON formatting without removing narration; CI defaults to JSON unchanged.
* \[ORCH‑3309] OTEL export of the narration field MAY be added behind a feature flag in the future.
* \[ORCH‑3310] The spec MUST reconcile the `decode_ms` vs `decode_time_ms` naming: the canonical field name is `decode_time_ms` (per `README_LLM.md`). Implementations MUST maintain backward compatibility during migration (e.g., include both fields transiently or map internally) and MUST update tests/specs accordingly in follow‑up work.
* \[ORCH‑3311] `orchestratord` MUST continue to emit JSON logs as today; pretty mode is optional and off by default.

## 5) Chosen Design & Wiring (Hybrid)

* **Shared narration surface:** introduce a tiny, dependency‑light core (new crate or minimal shared module) providing one call/macro to attach `human` alongside structured fields, plus redaction helpers and a test capture adapter.
* **Per‑binary init:** binaries keep their chosen formatter; `orchestratord` retains JSON via `init_observability()`, gains an optional pretty mode toggle for local dev.
* **Field taxonomy:** standardize `actor`, `action`, `target`, IDs (`req_id|job_id|task_id|session_id|worker_id`), and contextual keys (`error_kind`, `backoff_ms`, `retry_after_ms`, `duration_ms`). Existing SSE/metrics fields remain intact.
* **BDD coupling:** add step‑scoped spans, narration parity checks, optional story snapshots, and an emitted coverage metric.
* **Bridge for current tests:** continue or shim `state.logs` until narration capture fully replaces it in BDD.

## 6) Migration Plan

**Phase 1 — Foundations**

* Add shared narration surface + test capture adapter.
* Adopt in `orchestratord` admission (`api/data.rs::create_task`), streaming (`services/streaming.rs`), and cancel (`api/data.rs::cancel_task`).
* Keep `state.logs` lines or bridge into capture so current BDD keeps passing.

**Phase 2 — Provisioners & Engines (high value, low risk)**

* Replace `println!/eprintln!` in `provisioners/engine-provisioner` with narration wrappers for key events (preflight, CUDA fallback, spawn messages).

**Phase 3 — Wider Adoption**

* Opt‑in other crates (adapters, pool manager, CLI, harnesses).

**Phase N — OTEL (optional)**

* Feature‑gate export of narration field to OTEL when ready.

## 7) CI & Tooling

* Add a narration coverage stat to BDD outputs (e.g., ≥80% initially). Make the threshold configurable and ratchet over time.
* Optional BDD tags: `@narration-required`, `@silent-ok` to control expectations per scenario.
* Keep existing CI gates unchanged; add a lane that reads the narration coverage stat but do not fail initially (informational), then enforce when stable.

## 8) Risks & Mitigations

* **BDD reliance on `state.logs`** → Provide a capture layer and keep compatibility lines until fully migrated.
* **SSE hot path** → Use structured fields and avoid heavy formatting; the narration macro must be zero‑alloc or low‑alloc on hot paths.
* **Cross‑crate deps** → Keep the shared surface dependency‑light (no Axum), so CLI/tools can consume it.
* **Naming mismatch** (`decode_ms` vs `decode_time_ms`) → Canonicalize as `decode_time_ms` and plan a gradual migration with compatibility and tests.

## 9) Acceptance Criteria

* Spec merged following `README_LLM.md` mechanics.
* A single, documented mechanism to attach human narration exists (shared surface + init guidance).
* `orchestratord` emits narration for admission/stream/cancel in both pretty and JSON outputs (pretty optional) and tests confirm presence (via capture or `state.logs` bridge).
* BDD shows a narration coverage stat; at least one scenario demonstrates a “story snapshot”.

## 10) Companion Artifacts (required by this SPEC; implement in follow‑ups)

* `docs/LOGGING_GUIDE.md` — style, taxonomy, examples.
* `docs/BDD_NARRATION.md` — how tests assert narration; story snapshots; coverage.
* `docs/NARRATION_TAXONOMY.yaml` — machine‑readable keys with brief definitions.
* (Optional) `REPORTS/narration_coverage.json` — emitted by BDD runs for CI consumption.

## 11) Mapping to Repo Reality (anchors)

* `orchestratord/src/app/bootstrap.rs` — keep JSON init; add pretty toggle.
* `orchestratord/src/api/data.rs` — admission log and cancel path (current `state.logs` lines).
* `orchestratord/src/services/streaming.rs` — SSE event metrics and `decode_ms` field; plan reconciliation to `decode_time_ms`.

## 12) Refinement Opportunities

* Improve narration detection from naive regex presence to semantic verb/actor checks or embeddings.
* OTEL attribute export of narration and taxonomy fields under a feature flag.
* Automated redaction patterns and audits for sensitive fields.
* Naming reconciliation plan for `decode_ms` vs `decode_time_ms` across SSE transcripts, logs, README, and tests.
