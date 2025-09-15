# TODO — IDE AI Pre‑Code Artifacts & Stubs (Contract‑First, TDD)

**Read this first:**

* `docs/workflow.md` (SPEC→SHIP Workflow v2)
* `specs/orchestrator-spec.md` (Orchestrator SPEC v3)

> Goal: Produce **all documents, contracts, scaffolds, and stubs** needed to implement the orchestrator, **before** writing real business logic. Rust stubs are allowed where specified. Every task has acceptance criteria (AC). Do tasks in order; do not skip gates.

---

## 0) Repo Hygiene & Scaffolding

* [x] **Create workspace layout** (no business logic yet)

  * AC: Top‑level `Cargo.toml` with `[workspace]` members for: `contracts/api-types`, `contracts/config-schema`, `orchestrator-core`, `orchestratord`, `pool-managerd`, `worker-adapters/llamacpp-http`, `worker-adapters/vllm-http`, `worker-adapters/tgi-http`, `worker-adapters/triton`, `worker-adapters/mock`, `plugins/policy-host`, `plugins/policy-sdk`, `cli/consumer-tests`, `test-harness/e2e-haiku`, `test-harness/determinism-suite`, `test-harness/chaos`, `tools/spec-extract`, `tools/openapi-client`, `xtask`.
  * Proof: `cargo build --workspace` → Finished dev; `cargo test --workspace` → ok (stubs only).
* [x] **Add repo meta**

  * AC: `.editorconfig`, `.gitignore`, `CODEOWNERS`, `LICENSE` (MIT OR Apache‑2.0 or dual), `SECURITY.md`, `CONTRIBUTING.md`, `docs/PROJECT_GUIDE.md`.
  * Proof: Files created; `git status -s` shows tracked/unstaged where applicable.
* [x] **Set up `xtask` CLI** for multi‑file ops

  * AC: `xtask` binary with subcommands (stubs only):

    * `cargo xtask regen-openapi`
    * `cargo xtask regen-schema`
    * `cargo xtask spec-extract`
    * `cargo xtask ci:haiku:gpu` (prefers NVIDIA GPU worker over LAN) and `ci:haiku:cpu` (CI‑only fallback; starts llama.cpp locally and runs the haiku test)
    * `cargo xtask ci:determinism`
    * `cargo xtask pact:verify` / `pact:publish` (local broker optional later)
  * Proof: `cargo xtask spec-extract` → "xtask: spec-extract (stub)"; other subcommands print stub banners.

---

## 1) Spec Extraction & Requirements Map

* [x] **Implement `tools/spec-extract` (scaffold)**

  * Input: `specs/orchestrator-spec.md`.
  * Output: `requirements/index.yaml` mapping `ORCH-IDs → {title, section, must/should, links}`.
  * AC: Deterministic output; CI fails on uncommitted diffs.
  * Proof: `cargo run -p tools-spec-extract` → wrote `requirements/index.yaml`; re-run → unchanged; `git diff --exit-code` → 0.

---

## 2) Contracts (OpenAPI‑First) & Types

* [x] **Author OpenAPI contracts** (no handlers yet)

  * Files: `contracts/openapi/control.yaml`, `contracts/openapi/data.yaml`.
  * Must embed `x-req-id` fields referencing `ORCH-…` IDs as per spec.
  * Public API is **OrchQueue v1** only. Define endpoints and schemas for:
    * `POST /v1/tasks` (queue admission)
    * `GET /v1/tasks/:id/stream` (SSE)
    * `POST /v1/tasks/:id/cancel`
    * `GET /v1/sessions/:id`, `DELETE /v1/sessions/:id`
  * Include typed errors: `ADMISSION_REJECT`, `QUEUE_FULL_DROP_LRU`, `INVALID_PARAMS`, `POOL_UNREADY`, `POOL_UNAVAILABLE`, `REPLICA_EXHAUSTED`, `DECODE_TIMEOUT`, `WORKER_RESET`, `INTERNAL`. Error envelopes MUST include the `engine` context where applicable.
  * Define an explicit `engine` enum used across Task/Pool and related types: `llamacpp | vllm | tgi | triton`.
  * AC: `oapi-codegen` runs clean.
  * Proof: `cargo xtask regen-openapi` → validated specs; generated deterministic types/client; re-run → unchanged.
* [x] **Generate Rust API types & server stubs**

  * Target crates: `contracts/api-types` (shared types), `orchestratord` (server stubs), `tools/openapi-client` (client for tests).
  * AC: Build succeeds; no `todo!()` except in handlers.
  * Proof: `cargo build --workspace` OK; `cargo clippy -D warnings` OK; handlers use `unimplemented!()` only.
* [x] **Config Schema (code‑first)**

  * Create `contracts/config-schema` Rust types and emit `contracts/schemas/config.schema.json` via `schemars`.
  * Include fields: `engine` (enum), `devices` (GPU indices), optional `tensor_split` (per‑GPU ratios), and `require_same_engine_version`.
  * AC: Schema validates basic examples; CI enforces regen diffs.
  * Proof: `cargo xtask regen-schema` → wrote then unchanged; `cargo test -p contracts-config-schema` → 1 test passed.

---

## 3) Metrics Contract

* [x] **Write metrics contract** in `specs/metrics/otel-prom.md`

  * Include names, types, units, labels (with cardinality budgets), and `ORCH-IDs` links.
  * Labels MUST include `engine` and engine‑specific version labels (e.g., `engine_version`, `trtllm_version`).
  * AC: Add a metrics linter config in `ci/metrics.lint.json`.
  * Proof: Created `specs/metrics/otel-prom.md` + `ci/metrics.lint.json`.

---

## 4) Consumer‑Driven Contracts (Pact) & Stubs

* [ ] **CLI consumer Pact tests (scaffold)**

  * Crate: `cli/consumer-tests`.
  * Target API: **OrchQueue v1** only — `POST /v1/tasks`, `GET /v1/tasks/:id/stream`, `POST /v1/tasks/:id/cancel`, session endpoints.
  * AC: Pact interactions cover happy path + errors; pact JSON written to `contracts/pacts/`.
* [ ] **Stub provider with wiremock‑rs**

  * Crate: `worker-adapters/mock` + a small `stub-server` binary (inside `cli/consumer-tests` or `test-harness`).
  * AC: Stubs satisfy pact interactions; add **insta** snapshots of CLI transcripts.
* [ ] **Provider verification tests**

  * Crate: `orchestratord`.
  * AC: Loads pact files and verifies handler stubs (no real logic yet) match contract.

---

## 5) Worker Adapter Interfaces (Stubs Only)

* [ ] **Define `WorkerAdapter` trait** (no logic)

  * Location: `worker-adapters/common` (or `orchestrator-core`).
  * Methods: `health()`, `props()`, `submit() -> stream`, `cancel()`, `engine_version()`.
  * AC: Trait compiles; no net I/O.
* [ ] **llamacpp-http adapter skeleton**

  * Crate: `worker-adapters/llamacpp-http` with request/response types wired to `contracts/api-types`.
  * AC: HTTP client plumbing typechecks; methods `unimplemented!()`.
* [ ] **vllm-http adapter skeleton**

  * Crate: `worker-adapters/vllm-http` (OpenAI‑compatible endpoints).
  * AC: Methods present as per `WorkerAdapter` and `unimplemented!()`.
* [ ] **tgi-http adapter skeleton**

  * Crate: `worker-adapters/tgi-http` (TGI custom API + optional OpenAI compat).
  * AC: Methods present as per `WorkerAdapter` and `unimplemented!()`.
* [ ] **triton adapter skeleton**

  * Crate: `worker-adapters/triton` (Triton/TensorRT‑LLM HTTP/gRPC frontend or OpenAI‑compat where applicable).
  * AC: Methods present as per `WorkerAdapter` and `unimplemented!()`.

---

## 6) Determinism & Session Policy (Docs + Tests Skeleton)

* [ ] **Determinism suite scaffold** (`test-harness/determinism-suite`)

  * AC: Launch two logical replicas per engine (config files only). For llama.cpp set `--parallel 1`, `--no-cont-batching`; for other engines, run single‑slot/single‑request mode or equivalent to disable cross‑request batching.
  * Add test placeholders for 64 seeded prompts; write snapshot format for token streams; compare streams per engine.
* [ ] **Session policy doc**

  * File: `docs/session-policy.md` summarizing TTL ≤ 10m, ≤ 8 turns, no KV migration; metrics to emit.

---

## 7) Real‑Model E2E (Haiku) — Test Harness Before Code

* [ ] **Model cache & downloader** for CI

  * Script: `ci/scripts/fetch_model.sh` to pull **Qwen2.5‑0.5B‑Instruct (GGUF, Q4\_K\_M)**; fallback to TinyLlama 1.1B.
  * AC: Uses `HF_HUB_ENABLE_HF_TRANSFER=1` and caches under `~/.cache/models`.
* [ ] **llama.cpp server runner** (CPU) script

  * Script: `ci/scripts/start_llama_cpu.sh` (starts `llama-server --metrics --no-webui`).
* [ ] **Haiku test harness** (`test-harness/e2e-haiku`)

  * AC: Implements minute‑in‑words + 8‑char nonce; asserts ≥ 3 lines, substrings, and `/metrics` token delta > 0; fails on mock. Interact with the orchestrator via **OrchQueue v1** (`POST /v1/tasks`, then `GET /v1/tasks/:id/stream`). Prefer targeting a GPU worker over LAN; if GPU unreachable in CI, run CPU llama.cpp as a CI‑only fallback.
  * Enforce `TZ=Europe/Amsterdam` and `REQUIRE_REAL_LLAMA=1`.

---

## 8) CI Pipeline Files (No business logic)

* [ ] **Create `ci/pipelines.yml`** (GitHub Actions by default)

  * Jobs: `precommit`, `cdc_consumer`, `stub_flow`, `provider_verify`, `unit_props` (empty now), `determinism` (per engine), `e2e_haiku` (prefer GPU; CI‑only CPU fallback; drive via OrchQueue v1), `docs_compliance`.
  * AC: Pipeline runs through with stubs/skeletons (mark some as `continue-on-error` until code lands).
* [ ] **Dashboards & alerts**

  * Add `ci/dashboards/*.json` (placeholders referencing metric names).

---

## 9) Compliance & Docs Generators

* [ ] **Auto‑generate `COMPLIANCE.md`**

  * Tooling: extend `tools/spec-extract` to emit a coverage report linking `ORCH-ID → {openapi, tests, crates}`.
  * AC: `COMPLIANCE.md` builds and is linked from README.
* [ ] **Linkcheck** for SPEC/Workflow

  * Add `ci/scripts/check_links.sh` to validate internal anchors.

---

## 10) Gates Before Real Coding Begins

* [ ] `requirements/index.yaml` exists and is up‑to‑date.
* [ ] OpenAPI + Schema regenerate cleanly via `cargo xtask` (engine enum present across Job/Pool/types; schema includes engine/devices/tensor_split fields).
* [ ] Pact consumer tests target **OrchQueue v1** and produce pact files; provider verification passes against stubs.
* [ ] Wiremock stubs + insta snapshots approved.
* [ ] Determinism suite & Haiku harness compile and run via **OrchQueue v1** against a live NVIDIA GPU worker when available; CI‑only CPU fallback allowed (no orchestrator logic asserted yet beyond plumbing).
* [ ] CI pipeline green on stub flow; nightly jobs disabled or `continue-on-error` until logic lands.

---

## 11) Nice‑to‑Haves (Optional, still pre‑code)

* [ ] **RACI matrix** in `docs/RACI.md` (Spec, Contracts, Testing, Runtime owners).
* [ ] **Threat model** in `docs/threat-model.md` (Auth, quotas, PII, least privilege).
* [ ] **Release & rollback** SOP in `docs/release-ops.md`.

---

## 12) After these tasks

* Only after all boxes above are checked, begin implementing **orchestrator-core** logic and adapter network calls (`llamacpp-http`, `vllm-http`, `tgi-http`, `triton`). Keep TDD: extend pact/snapshots/properties first, then code. Use `cargo xtask` to regenerate artifacts on every PR.
