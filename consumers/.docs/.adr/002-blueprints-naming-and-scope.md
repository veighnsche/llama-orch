# ADR-002: “Blueprints” — Project Type & Naming
## Status
Accepted
## Context
We need a precise name for projects that use the **llama-orch Utils** (with the **CLI** for generated bindings) to programmatically turn a small set of seed files into a complete software project via applet-driven processes, using **SDK**-provided server types. These aren’t ordinary software projects—they are **inputs and orchestration** for software generation.
## Decision (RFC-2119)
* Such projects **MUST** be called **Blueprints**.
* A **Blueprint** **MUST** be a repo/workspace initialized by our CLI with:
  * minimal seed files (e.g., `Spec.md`, `process.yml` or equivalent),
  * a generated **capabilities binding** for the target language (TS/JS/Rust) that reflects the llama-orch server’s declared capabilities at generation time, expressed using types from `llama-orch-sdk`,
  * scaffolding to run **Utils applets** end-to-end (incl.  emission).
* Blueprints **SHOULD** be deterministic and artifact-first (, JSONL traces, prompt/params lock).
* Blueprints **MAY** include language-specific helpers, tests, and CI templates.
* Tooling and docs **MUST** use “Blueprint(s)” consistently (UI, flags, folder names).
* The term **MUST NOT** be used for ordinary, hand-written application repos that don’t rely on the **Utils** generation flow.
## Consequences
**Positive**: Clear identity; consistent DX; paves the way for language-specific generators and typed server bindings.
**Negative**: New term to teach; migration of any prior wording.
**Trade-offs**: Stronger branding vs. generic terminology.
## Alternatives Considered
* “Recipes” — overlapped with build/CI jargon; weaker sense of project. Rejected.
* “Scaffolds” — implies only initial layout, not full programmatic build. Rejected.
* “Workflows” — describes the *process*, not the *project*. Rejected.
## Artifacts / Proof Bundle Links
* First canonical Blueprint will include: `/manifest.json`, `events/run.jsonl`, `prompts/`, `params.lock.json`, and per-step diffs.
## References
* llama-orch capabilities & discovery (for generated bindings): `/v1/capabilities` returns engines and supported workloads; the CLI also snapshots models (id, digest, state) and pools.
* Catalog/models, pools, and artifacts endpoints used by Blueprints/**Utils**; generated bindings are typed using `llama-orch-sdk` types.
**Suggested path:**
`consumers/.docs/.adr/002-blueprints-naming-and-scope.md`
