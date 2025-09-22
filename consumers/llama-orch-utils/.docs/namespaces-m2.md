# M2 Namespaces → Applets Map (Runtime/TS)

> **STATUS: DRAFT** — Namespaces, function names, types, and signatures in this doc are provisional. They may change during implementation. They will be explicitly marked **LOCKED** only after the implementation and the Bun consumer checks pass.

Objective: Track the **current intended** runtime/TS namespaces and function names for M2 without changing code. This document records the mapping from intended runtime names to the current Rust module tree and notes any minimal alignment actions (plan-only). **All content is DRAFT until explicitly marked LOCKED later.**

Namespaces: `{ fs, prompt, model, params, llm, orch }`

Intended functions:
- fs.readFile, fs.writeFile
- prompt.message, prompt.thread
- model.define
- params.define
- llm.invoke
- orch.response_extractor

## Current mapping (Rust → Intended)

| Namespace | Intended function       | Rust module path today                                   | Status                | Minimal action (if any) |
|-----------|-------------------------|-----------------------------------------------------------|-----------------------|-------------------------|
| fs        | fs.readFile             | `src/fs/file_reader/file_reader.rs` (fn `run`)           | Needs alias/rename    | Export TS alias `readFile` mapping to `fs.file_reader` (no Rust rename). |
| fs        | fs.writeFile            | `src/fs/file_writer/file_writer.rs` (fn `run`)           | Needs alias/rename    | Export TS alias `writeFile` mapping to `fs.file_writer` (no Rust rename). |
| prompt    | prompt.message          | `src/prompt/message/message.rs` (fn `run`)               | Aligned               | None.                   |
| prompt    | prompt.thread           | `src/prompt/thread/thread.rs` (fn `run`)                 | Aligned               | None.                   |
| model     | model.define            | `src/model/define/define.rs` (fn `run`)                  | Aligned               | None.                   |
| params    | params.define           | `src/params/define/define.rs` (fn `run`)                 | Aligned               | None.                   |
| llm       | llm.invoke              | `src/llm/invoke/invoke.rs` (fn `run`)                    | Aligned               | None.                   |
| orch      | orch.response_extractor | `src/orch/response_extractor/response_extractor.rs` (fn `run`) | Aligned          | None.                   |

Notes:
- The only mismatches are naming for `fs.readFile`/`fs.writeFile` vs the current Rust folders `file_reader`/`file_writer`. To avoid churn in Rust, we will introduce TypeScript-level aliases (e.g., export `readFile` that calls into the `file_reader` applet) when we add the runtime package. No behavior or signatures change.
- All other namespaces and applets already align with intended names.

## Current DRAFT names for M2
We propose the following runtime/TS surface for M2 (DRAFT): `fs.readFile`, `fs.writeFile`, `prompt.message`, `prompt.thread`, `model.define`, `params.define`, `llm.invoke`, and `orch.response_extractor`. These names may change during implementation and consumer validation. The Rust module tree remains as-is; minimal TypeScript aliases will be added for `fs.readFile` and `fs.writeFile` only. No code changes are required at this step; this file serves as the source of truth to prevent naming drift.

## Step 2 status — Public + serializable I/O types

All listed applets expose public request/response types (or equivalent) suitable for serde and TS generation. The public types derive `serde::{Serialize, Deserialize}` and are annotated with `#[cfg_attr(feature = "ts-types", derive(ts_rs::TS))]` for conditional TypeScript generation.

| Applet (intended)              | Canonical request type(s)          | Canonical response type(s) | Notes |
|--------------------------------|------------------------------------|----------------------------|-------|
| fs.readFile                    | `fs::file_reader::ReadRequest`     | `fs::file_reader::ReadResponse` | Data blobs via `fs::file_reader::FileBlob` |
| fs.writeFile                   | `fs::file_writer::WriteIn`         | `fs::file_writer::WriteOut`     | Overwrites by design (M2) |
| prompt.message                 | `prompt::message::MessageIn`       | `prompt::message::Message`      | Sources via `prompt::message::Source` |
| prompt.thread                  | `prompt::thread::ThreadIn`         | `prompt::thread::ThreadOut`     | Items via `prompt::thread::ThreadItem` |
| model.define                   | (inline fields: `model_id`, `engine_id?`, `pool_hint?`) | `model::define::ModelRef` | Input is inline params; output is `ModelRef` |
| params.define                  | `params::define::Params`           | `params::define::Params`        | Identity normalizer |
| llm.invoke                     | `llm::invoke::InvokeIn`            | `llm::invoke::InvokeOut`        | Result shape `llm::invoke::InvokeResult` |
| orch.response_extractor        | `llm::invoke::InvokeResult`        | `String`                        | Returns best-effort text |

Visibility check: All types are declared `pub` within their respective modules and are importable via the crate module tree (e.g., `llama_orch_utils::fs::file_reader::ReadRequest`). No re-exports are required for M2.

## Step 3 — Public function signatures (DRAFT)

This section records the single public entry function per applet and the intended TS name it maps to. Signatures are the Rust boundary we will rely on for the M2 runtime surface. No code changes performed in this step.

| Applet (TS name)            | Rust function (module path)                           | Signature |
|-----------------------------|--------------------------------------------------------|-----------|
| fs.readFile                 | `fs::file_reader::run`                                 | `pub fn run(req: ReadRequest) -> std::io::Result<ReadResponse>` |
| fs.writeFile                | `fs::file_writer::run`                                 | `pub fn run(input: WriteIn) -> std::io::Result<WriteOut>` |
| prompt.message              | `prompt::message::run`                                 | `pub fn run(input: MessageIn) -> std::io::Result<Message>` |
| prompt.thread               | `prompt::thread::run`                                  | `pub fn run(input: ThreadIn) -> std::io::Result<ThreadOut>` |
| model.define                | `model::define::run`                                   | `pub fn run(model_id: String, engine_id: Option<String>, pool_hint: Option<String>) -> ModelRef` |
| params.define               | `params::define::run`                                  | `pub fn run(p: Params) -> Params` |
| llm.invoke                  | `llm::invoke::run`                                     | `pub fn run(client: &OrchestratorClient, input: InvokeIn) -> Result<InvokeOut, llama_orch_utils::error::Error>` |
| orch.response_extractor     | `orch::response_extractor::run`                        | `pub fn run(result: &InvokeResult) -> String` |

Mapping notes:
- The TS-facing names are DRAFT per Step 1. Rust functions remain named `run` within their applet modules; TS aliases will provide `fs.readFile`/`fs.writeFile` as needed without renaming Rust.
- All parameters and return types are the public I/O types confirmed in Step 2, or references thereto.

## Step 3 — Signatures + Error/Async model (DRAFT)

This section records the public function signatures (DRAFT), their error types, and sync/async model for M2. No code changes were performed; any inconsistencies are noted with plan-only follow-ups.

| Applet (TS name)        | Rust function (module path)                    | Signature (Rust)                                                                 | Error type                    | Sync/Async |
|-------------------------|-------------------------------------------------|-----------------------------------------------------------------------------------|-------------------------------|-----------|
| fs.readFile             | `fs::file_reader::run`                         | `pub fn run(req: ReadRequest) -> std::io::Result<ReadResponse>`                  | `std::io::Error`             | Sync      |
| fs.writeFile            | `fs::file_writer::run`                         | `pub fn run(input: WriteIn) -> std::io::Result<WriteOut>`                        | `std::io::Error`             | Sync      |
| prompt.message          | `prompt::message::run`                         | `pub fn run(input: MessageIn) -> std::io::Result<Message>`                        | `std::io::Error`             | Sync      |
| prompt.thread           | `prompt::thread::run`                          | `pub fn run(input: ThreadIn) -> std::io::Result<ThreadOut>`                       | `std::io::Error`             | Sync      |
| model.define            | `model::define::run`                           | `pub fn run(model_id: String, engine_id: Option<String>, pool_hint: Option<String>) -> ModelRef` | (infallible for M2)          | Sync      |
| params.define           | `params::define::run`                          | `pub fn run(p: Params) -> Params`                                                 | (infallible for M2)          | Sync      |
| llm.invoke              | `llm::invoke::run`                             | `pub fn run(client: &OrchestratorClient, input: InvokeIn) -> Result<InvokeOut, llama_orch_utils::error::Error>` | `llama_orch_utils::error::Error` | Sync      |
| orch.response_extractor | `orch::response_extractor::run`                | `pub fn run(result: &InvokeResult) -> String`                                     | (infallible for M2)          | Sync      |

Notes on error model
- No `panic!`, `unwrap`, or `expect` are present on public paths for these applets (verified in src/ tree).
- Non-SDK applets prefer `std::io::Result<...>` in M2. Two exceptions are currently infallible by design:
  - `model::define::run(...) -> ModelRef`
  - `params::define::run(p: Params) -> Params`
  Plan-only: if needed for uniformity post-M2, wrap these in `std::io::Result<...>` while returning `Ok(...)` for valid inputs.
- `orch::response_extractor::run(&InvokeResult) -> String` is infallible best-effort text extraction. Plan-only: could become `io::Result<String>` without behavior change if uniformity is required.

llm.invoke UNIMPLEMENTED (DRAFT)
- Current status (M2): returns `Result<InvokeOut, llama_orch_utils::error::Error>` and immediately returns the typed variant below (no SDK calls).
- Variant path (DRAFT): `llama_orch_utils::error::Error::Unimplemented`.
- Exact message string (DRAFT for M2): `"unimplemented: llm.invoke requires SDK wiring"` (subject to change until SDK wiring is decided).

## Transition to LOCKED
These DRAFT names, types, and signatures will be promoted to **LOCKED** only after:
1) All non-SDK applets are implemented and pass tests;
2) The Bun consumer type-checks against `@llama-orch/utils` successfully;
3) Any discovered misalignments have been addressed.
At that point a separate commit will change this document’s status to **LOCKED** and record the exact command to regenerate `index.d.ts` plus the consumer check command.

Async model
- All applets listed are synchronous for M2 (no `async fn`).
- Future streaming support (e.g., `llm.invoke`) can be added behind a new function or return type without breaking these M2 sync boundaries.

## Phase A status (DRAFT)
- llm.invoke: ⛔ UNIMPLEMENTED — returns typed `Unimplemented` error with message: `unimplemented: llm.invoke requires SDK wiring` (DRAFT).
- No `panic!/unwrap/expect` on public applet paths policy is in effect (DRAFT).
- fs.readFile: ✅ Implemented (DRAFT) — as_text=true decodes using `encoding.unwrap_or("utf-8")` with UTF-8 lossy fallback; as_text=false returns raw bytes. Order preserved.
- prompt.message: ✅ Implemented (DRAFT) — sources: text|lines|file; dedent removes common leading whitespace; file read is UTF-8 lossy; errors via io::Result.
- prompt.thread: ✅ Implemented (DRAFT) — composes ordered messages from items; sources text|lines|file; per-item dedent; errors via io::Result (missing-file propagates).
- model.define: ✅ Implemented (DRAFT) — constructs ModelRef deterministically (no I/O, no SDK); no validation beyond struct construction.
- params.define: ✅ Implemented (DRAFT) — normalization/clamping rules:
  - temperature: default 0.7; clamp to [0.0, 2.0]
  - top_p: default 1.0; clamp to [0.0, 1.0]
  - max_tokens: default 1024
  - seed: passthrough (no default)

## Phase A summary (DRAFT)

| Namespace | Function | Status | Notes | Tests (file) |
|-----------|----------|--------|-------|--------------|
| fs | readFile | ✅ Implemented (DRAFT) | text=UTF-8 lossy; binary=bytes; order preserved | `src/fs/file_reader/tests.rs` |
| fs | writeFile | ✅ Implemented (DRAFT) | overwrite; create_dirs optional | `src/fs/file_writer/tests.rs` |
| prompt | message | ✅ Implemented (DRAFT) | sources text|lines|file; per-call dedent; errors via io::Result | `src/prompt/message/tests.rs` |
| prompt | thread | ✅ Implemented (DRAFT) | ordered composition; per-item dedent; file errors propagate | `src/prompt/thread/tests.rs` |
| model | define | ✅ Implemented (DRAFT) | pure ModelRef; no I/O | `src/model/define/tests.rs` |
| params | define | ✅ Implemented (DRAFT) | defaults/clamps (see rules above) | `src/params/define/tests.rs` |
| llm | invoke | ⛔ UNIMPLEMENTED (DRAFT) | returns typed Unimplemented error with message: `unimplemented: llm.invoke requires SDK wiring` | `src/llm/invoke/tests.rs` |
| orch | response_extractor | ✅ Implemented (DRAFT) | choices[0].text → first non-empty → ""; no trim/aggregation | `src/orch/response_extractor/tests.rs` |

### Drift guard (crate-side)
The compile-only drift guard at `src/tests_draft_surface.rs` imports all public applet symbols and asserts signatures via typed pointers. Any rename or removal will fail `cargo test`.

### How to validate Phase A

```
cargo test -q -p llama-orch-utils -- --nocapture
```

### Next phase preview (DRAFT, do not execute yet)
- Phase B: TS generation via `ts-types` feature and lib exporter → write `npm-build/index.d.ts`; add tracked `npm-build` assets (`package.json`, `index.js`, `api.d.ts`).
- Phase C: Bun consumer points to `file:../../llama-orch-utils/npm-build`; `bun install`; `bunx tsc --noEmit`.
- Phase D: After consumer proof, flip DRAFT → LOCKED; add CI drift guards for TS + docs.

### Phase B status (DRAFT)
- B1: `ts-types` Cargo feature present.
- B2: `export_ts_types()` added behind `--features ts-types`; writes `npm-build/index.d.ts` and appends API declarations; not executed yet.
- B3: `npm-build/` folder created and gitignored (no files generated yet).
- B4r: removed `.npm/` templates; exporter now programmatically generates `npm-build/{index.d.ts, index.js, package.json}` (DRAFT). Not executed yet.

## Phase N status (DRAFT)
- N1: Bindings crate scaffolded with napi-rs at `consumers/llama-orch-utils-node/`; core crate untouched; exposes `probe()` for toolchain validation. Built with `cargo build -q -p llama-orch-utils-node --release`.
- N2: `fs.readFile` exported via napi-rs — sync function `fs.readFile(input: ReadRequest) -> ReadResponse`; input/output shapes mirror core types via object wrappers; errors mapped to `napi::Error(Status::GenericFailure, "fs.readFile failed: …")`; Rust-side unit tests validate mapping & determinism; no panics on public paths.
- N3: `prompt.message` and `prompt.thread` exported via napi-rs — sync functions `prompt.message(input: MessageIn) -> Message` and `prompt.thread(input: ThreadIn) -> ThreadOut`; `Source` represented in N-API as a discriminated object `{ kind: "Text"|"Lines"|"File", text?: string, lines?: string[], path?: string }` which converts to the core enum; errors mapped to `napi::Error(Status::GenericFailure, "prompt.message failed: …"/"prompt.thread failed: …")`; Rust-side unit tests validate conversions and determinism; no panics on public paths.
- N4: napi-rs exports added — `model.define(model_id, engine_id?, pool_hint?) -> ModelRef`, `params.define(p: Params) -> Params` (core enforces defaults/clamps), and `orch.response_extractor(result: InvokeResult) -> string` (deterministic choice rules). Conversions are pure and total; conversion failures would map via `napi::Error(Status::GenericFailure, "…")` (not expected for current shapes). Rust-side unit tests added for each wrapper; no panics on public paths.
- N5: `llm.invoke` exported in bindings but returns a typed napi error with message: "unimplemented: llm.invoke requires SDK wiring". No panics. No SDK calls. Shapes are present for parity only.
- N5 (Path B): Node smoke test for `llm.invoke`. Packaging scaffold added to `consumers/llama-orch-utils-node/` (local only). Built via `@napi-rs/cli` (`npm run build`). Smoke script `.smoke/llm_invoke.mjs` calls `llm.invoke` and verifies it throws the exact message: `unimplemented: llm.invoke requires SDK wiring`. Rust unit tests do not call napi exports; Node-level smoke covers the behavior. No publishing; artifacts are gitignored.

- N6: ESM facade exported (`esm.js`); CJS via `index.cjs`. Exports the namespaced runtime surface `{ fs.readFile, prompt.{message,thread}, model.define, params.define, llm.invoke, orch.response_extractor }` and default aggregates the same. Node smokes added and passing: `.smoke/fs_read_file.mjs`, `.smoke/prompt_message.mjs`, `.smoke/llm_invoke.mjs`.

- N7: napi-rs generates `index.d.ts` (untracked via .gitignore); package exposes `"types": "./index.d.ts"`. Added a local TS smoke under `consumers/llama-orch-utils-node/ts-smoke/` that imports from `../esm.js` and type-checks calls to `fs.readFile`, `prompt.message`, `model.define`, `params.define`, `orch.response_extractor`, and `llm.invoke` (with nullables). `pnpm run smoke:types` compiles clean with `tsc --noEmit`.
