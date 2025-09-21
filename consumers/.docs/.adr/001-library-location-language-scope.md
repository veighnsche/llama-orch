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

* The library **MUST** be implemented as a Rust crate located at:
  `/home/vince/Projects/llama-orch/consumers/llama-orch-toolkit`.
* The crate **MUST** be publishable to **crates.io** and expose a stable, semver’d public API.
* The crate **MUST** provide a **WASM build** consumable via **npm** (package name TBA), suitable for Node and modern browsers.
* The library **MUST** organize capabilities into **namespaces** (e.g., `spec.*`, `plan.*`, `scaffold.*`, `code.*`, `tests.*`, `ci.*`, `provenance.*`), each containing **applets** (small, composable units).
* Each **applet** **MUST** implement a common **Applet trait/contract** with:

  * declared **input/output schemas**,
  * a pure **`execute()`** entry using provided handles (ModelClient, SafeFs, ArtifactSink),
  * **side-effects only** through provided handles (no raw FS/VCS/network).
* The **Runner API** **MUST** execute declarative **processes** (linear or small DAG) referencing applets by `namespace.applet`, with `needs:` dependencies, retries, timeouts, and budgets.
* The toolkit **MUST** integrate with llama-orch via an adapter, using tasks/streaming, sessions, artifacts, placement, and determinism controls.
* All steps **MUST** emit **artifact trails** (inputs, prompts, params, outputs, diffs) and compile a **proof bundle** manifest at run end.
* The library **SHOULD** remain backend-agnostic at the public surface (allow alternate adapters), while the default adapter targets llama-orch.
* The library **MUST NOT** write outside a configured safe project root; all writes **MUST** be atomic with diffs captured.
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

* Llama-orch capabilities (tasks, sessions, artifacts, placement, determinism).
* Internal spec drafts for Applet trait, Runner API, Process schema (to be authored next).
* RFC-2119 (“Key words for use in RFCs to Indicate Requirement Levels”).
