# ADR-001: Library Location, Language Targets, and Scope

## Status

Accepted

## Context

We need a first, foundational decision for the **agentic toolkit** that will let users orchestrate LLM-driven, stepwise transformations from a small set of user-supplied files into a complete software project. This library should live alongside `llama-orch` so it can reuse its capabilities (tasks, sessions, artifacts, placement, determinism) while staying shippable to external ecosystems (crates.io and npm).

Key forces:

* Keep **tight integration** with llama-orch while avoiding vendor lock-in in the public API.
* **Artifact-first** workflows and deterministic replays are non-negotiable.
* The library must be **consumable in Rust** and **in JS/TS via WASM**.

## Decision (RFC-2119)

* Replace the single "toolkit" with three artifacts under `consumers/`:
  * `llama-orch-sdk/` (**SDK**) — typed clients/models for llama-orch HTTP + SSE.
  * `llama-orch-utils/` (**Utils**) — Blueprint-oriented applets and Runner.
  * `llama-orch-cli/` (**CLI**) — capability codegen and Blueprint initialization.
* **SDK** **MUST** own all llama-orch transport: HTTP and SSE clients, typed requests/responses, streaming envelopes, and error types. It **MUST** be publishable to **crates.io** and **SHOULD** provide a **WASM build** consumable via **npm** (browser/Node).
* **Utils** **MUST** organize capabilities into **namespaces** (e.g., `spec.*`, `plan.*`, `scaffold.*`, `code.*`, `tests.*`, `ci.*`, `provenance.*`), each containing **applets** (small, composable units). The **Runner API** **MUST** execute declarative **processes** (linear or small DAG) referencing applets by `namespace.applet`, with `needs:` dependencies, retries, timeouts, and budgets. Utils **MUST** call llama-orch **only via the SDK**.
* Each **applet** in Utils **MUST** implement a common **Applet trait/contract** with:
  * declared **input/output schemas**,
  * a pure **`execute()`** entry using provided handles (ModelClient via SDK, SafeFs, ArtifactSink),
  * **side-effects only** through provided handles (no raw FS/VCS/network).
* **CLI** **MUST** use the **SDK** to discover capabilities and catalogs and **MUST** generate static bindings (TS/JS/Rust) for Blueprints. **CLI** **MUST** target native-only.
* Build targets: **SDK** and **Utils** **SHOULD** build for native and `wasm32-unknown-unknown` where feasible; **CLI** is native-only.
* All steps **MUST** emit **artifact trails** (inputs, prompts, params, outputs, diffs) and compile a **proof bundle** manifest at run end.
* Filesystem guardrails: For **M1**, follow ADR-004’s exception and ADR-005 governance — write applets **MUST** write exactly where instructed (no guardrails); later milestones **MAY** add opt-in guardrails. This ADR does not impose a blanket library-wide restriction.
* RFC-2119 keywords in this ADR **MUST** be interpreted per RFC-2119.

## Consequences

**Positive**

* Clear monorepo location and packaging story (crates.io + npm).
* Consistent applet model encourages reuse and testability.
* Artifact-first pipeline enables reproducibility and auditing.
* WASM build broadens adoption to JS/TS environments.

**Negative**

* Maintaining dual targets (native + WASM) increases CI complexity.
* Strong sandboxing and artifact capture add overhead to simple applets.
* Adapter abstraction must balance power vs. portability.

**Trade-offs**

* Tight coupling to llama-orch for the default path, with an abstraction to keep options open.
* Small-DAG support adds complexity vs. strictly linear flows but unlocks parallelizable steps.

## Alternatives Considered

* **Separate repo** for the toolkit: cleaner isolation but weaker day-1 integration and slower iteration. Rejected for now.
* **JS/TS-first implementation** with Rust later: faster browser path but worse performance/control on FS and determinism. Rejected.
* **No WASM target** initially: simpler build but blocks JS/TS users. Rejected.

## Artifacts / Proof Bundle Links

* To be produced with the first end-to-end demo run:
  * `proof-bundle/manifest.json`
  * `events/run.jsonl` (tokens/metrics/decisions)
  * `diffs/` for filesystem mutations
  * `prompts/` and `params.lock.json`

## References

* ADR-004 (Applet Ground Rules) — Filesystem side-effects and M1 exception.
* ADR-005 (Milestone Governance) — Guardrails policy and milestone control.
* ADR-006 (Library Split) — Defines SDK/Utils/CLI layering.
* Llama-orch capabilities (tasks, sessions, artifacts, placement, determinism).
* Internal spec drafts for Applet trait, Runner API, Process schema (to be authored next).
* RFC-2119 (“Key words for use in RFCs to Indicate Requirement Levels”).

## Changelog

- 2025-09-21 — Amended by ADR-006: replace single toolkit with SDK/Utils/CLI; HTTP/SSE owned by SDK; SDK & Utils target native+wasm; CLI native-only; filesystem guardrails aligned with ADR-004/ADR-005 (M1 exception).
