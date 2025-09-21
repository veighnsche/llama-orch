# M2 Namespaces → Applets Map (Runtime/TS)

Objective: Lock the runtime/TS namespaces and function names for M2 without changing code. This document records the mapping from intended runtime names to the current Rust module tree and notes any minimal alignment actions (plan-only).

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

## Final, locked names for M2
We lock the following runtime/TS surface for M2: `fs.readFile`, `fs.writeFile`, `prompt.message`, `prompt.thread`, `model.define`, `params.define`, `llm.invoke`, and `orch.response_extractor`. The Rust module tree remains as-is; minimal TypeScript aliases will be added for `fs.readFile` and `fs.writeFile` only. No code changes are required at this step; this file serves as the source of truth to prevent naming drift.
