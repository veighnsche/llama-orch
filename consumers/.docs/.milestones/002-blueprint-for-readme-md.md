# M2 — Blueprint for README.md (Utils Applets)

## Scope

This milestone delivers the **first Blueprint**: transforming a human-authored **seed file** into a `README.md` using applets from `llama-orch-utils`.

The Utils library **MUST**:

* Provide applets to read files, assemble prompts, define models/params, invoke the SDK, extract responses, and write outputs.
* Be fully independent of the SDK internals (all llama-orch communication must go through the SDK).
* Expose a process definition (YAML/JSON) that wires applets into a deterministic pipeline.

The goal is to prove that a Blueprint can be composed entirely of Utils applets, run end-to-end, and produce a reproducible `README.md`.

## Non-Goals

* No multi-seed merging.
* No scaffolded repos beyond `README.md`.
* No write guardrails or rollback strategies (always overwrite).
* No model auto-selection or GPU scheduling.
* No advanced prompt templating beyond minimal helpers.
* No CLI integration (capabilities codegen remains out of scope here).

## Exit Criteria

* A minimal `llama-orch-utils` crate exists.
* Utils defines and exports the M2 applet set (below).
* Blueprint CLI can run a process definition that wires these applets and produces `README.md`.
* Proof bundle is emitted with:

  * Seed input file + hash.
  * Prompt, params, model/engine metadata.
  * Transcript/trace from llama-orch via SDK.
  * Output `README.md` + diff.
* Golden fixture ensures locked inputs produce byte-identical `README.md`.

## Required Artifacts

* `proof-bundle/manifest.json`
* `proof-bundle/events.jsonl` (trace of applet invocations)
* `proof-bundle/prompts/` (prompt + params snapshot)
* `proof-bundle/outputs/README.md`
* `proof-bundle/diffs/` (filesystem mutations)

## M2 Applet Set

1. **File Reader**

   * **Library:** `utils`
   * **Namespace:** `fs`
   * **Input:** `{ paths: [string] }`
   * **Output:** `{ files: [{ path, content, sha256 }] }`

2. **Prompt Message**

   * **Library:** `utils`
   * **Namespace:** `prompt`
   * **Input:** `{ role: "system"|"user"|"assistant", content: string|fileRef|[string] }`
   * **Output:** `{ message: { role, content } }`

3. **Prompt Thread**

   * **Library:** `utils`
   * **Namespace:** `prompt`
   * **Input:** `{ messages: [{ role, content }...] }`
   * **Output:** `{ messages: [{ role, content }...] }`

4. **Model Define**

   * **Library:** `utils`
   * **Namespace:** `model`
   * **Input:** `{ model_id: string, engine_id?: string, pool_hint?: string }`
   * **Output:** `{ model_ref }`

5. **Params Define**

   * **Library:** `utils`
   * **Namespace:** `params`
   * **Input:** `{ temperature?: number, seed?: number, top_p?: number, max_tokens?: number }`
   * **Output:** `{ params_ref }`

6. **LLM Invoke**

   * **Library:** `utils` (wrapper), calls `llama-orch-sdk` under the hood
   * **Namespace:** `llm`
   * **Input:** `{ messages, model_ref, params_ref? }`
   * **Output:** `{ transcript_ref, trace_ref }`

7. **Response Extractor**

   * **Library:** `utils` (wrapper), uses SDK response models
   * **Namespace:** `orch`
   * **Input:** `{ transcript_ref }`
   * **Output:** `{ text }`

8. **File Writer**

   * **Library:** `utils`
   * **Namespace:** `fs`
   * **Input:** `{ files: [{ path, content }] }`
   * **Output:** `{ wrote: [{ path, sha256 }] }`
   * **M2 Behavior:** Always overwrite, no guardrails.

## Risks & Controls

* **Risk:** Applets duplicating SDK logic → Controlled by enforcing SDK-only network calls.
* **Risk:** Non-determinism → Controlled by params lock and proof bundle capture.
* **Risk:** File overwrite → Allowed in M2, planned guardrails for later milestones.
* **Risk:** Misaligned prompt structure → Controlled by separating `prompt.message` and `prompt.thread`.

## Rollback

* Remove `llama-orch-utils` crate.
* Remove Blueprint example process.
* Retain SDK (from M1) as baseline.

## Changelog

* `2025-09-21 — Initial draft of Blueprint for README.md milestone (M2).`