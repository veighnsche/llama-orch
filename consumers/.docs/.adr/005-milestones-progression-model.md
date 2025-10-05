# ADR-005: Milestone Governance (Generic—No Enumerations)
## Status
Accepted
## Context
We need a way to progress the project through **milestones** without polluting the docs with guesses or speculative scope. This ADR defines **how milestones are created, documented, reviewed, and accepted**, **without** listing any specific future milestones.
## Decision (RFC-2119)
### 1) What a “Milestone” is
* A **Milestone (Mx)** **MUST** describe a bounded, reviewable increment of capability.
* A milestone **MUST** be independently buildable, demo-able, and auditable.
* Milestones **MUST NOT** be speculative. If a milestone is not approved, it **MUST NOT** be referenced or listed anywhere.
### 2) Where milestones live
* Each milestone **MUST** have a single source file at `SPEC/milestones/Mx.md`.
* That spec file **MUST** be the only canonical definition of the milestone’s scope and gates.
* ADRs **MUST NOT** enumerate or sketch future milestones. They **MAY** reference **only** milestones that already exist in `SPEC/milestones/`.
### 3) How a milestone is defined
Every `SPEC/milestones/Mx.md` **MUST** contain exactly these sections (no extras, no placeholders with fake content):
* **Scope** — plain-language description of what is included.
* **Non-Goals** — explicit out-of-scope items.
* **Exit Criteria** — verifiable gates (bullets only; no numbers until approved).
* **Required Artifacts** —  elements (names only; no sample data).
* **Risks & Controls** — known risks + mitigations.
* **Rollback** — how to revert the milestone safely.
### 4) Acceptance & promotion
* A milestone **MUST** be accepted via an ADR (status: *Accepted*) that references its `SPEC/milestones/Mx.md`.
* The ADR **MUST NOT** add details beyond linking and confirming acceptance.
* Promotion to the next milestone **MUST** occur **only after** all Exit Criteria are met and Required Artifacts are attached to the acceptance record.
### 5) Evidence & artifacts
* Evidence **MUST** be provided as a **** (hashes, logs, transcripts, params lock, diffs) linked from the milestone spec.
* Evidence **MUST** be sufficient for a reviewer to reproduce results.
### 6) Change control
* Any change to a milestone spec **MUST** be via PR and recorded at the bottom of `SPEC/milestones/Mx.md` under **Changelog**.
* Breaking changes to a milestone **MUST** be approved by a new ADR that **Supersedes** the prior acceptance ADR.
### 7) Prohibitions (to prevent pollution)
* **MUST NOT** list, hint, or template future milestones (e.g., “M2 will…”) anywhere except in their **own** `SPEC/milestones/M2.md` once proposed.
* **MUST NOT** include “example” gate values or mock artifacts for milestones that are not approved.
* **MUST NOT** auto-generate milestone files with placeholder content.
### 8) Minimality & determinism
* Milestones **SHOULD** be small enough to complete, review, and accept within a short iteration.
* Where applicable, deterministic settings (seed, params lock) **MUST** be specified in artifacts to make the result reproducible.
## Consequences
**Positive:** Keeps the docs clean, prevents misinformation, and ensures each milestone is precise, auditable, and independently reviewable.
**Negative:** Requires discipline—no forward-looking content in ADRs.
**Trade-off:** We optimize for correctness and clarity over speed of speculative planning.
## Alternatives Considered
* “Roadmap ADR” that lists all future milestones — **Rejected** (invites drift and misinformation).
## References
* ADR-002 (Blueprints)
* ADR-004 (Applet Ground Rules)
* ADR-006 (Library Split) — SDK/Utils/CLI layering; context for M1 filesystem guardrails relaxation referenced by ADR-004.
---
### Appendix (Templates)
These are **templates only**. They contain **no example content**. Copy as-is and fill **only** with approved information.
**Template — `SPEC/milestones/Mx.md`:**
```md
# Mx — <Short Title>
## Scope
<what is included>
## Non-Goals
<explicitly excluded items>
## Exit Criteria
- <verifiable gate 1>
- <verifiable gate 2>
## Required Artifacts
- <artifact name 1>
- <artifact name 2>
## Risks & Controls
- <risk> — <control>
## Rollback
<how to revert>
## Changelog
- <yyyy-mm-dd> <summary of change>
```
**Template — ADR to accept a milestone:**
```md
# ADR-0NN: Accept Milestone Mx — <Short Title>
## Status
Accepted
## Context
Milestone Mx has a defined spec at SPEC/milestones/Mx.md.
## Decision
- Accept Milestone Mx as defined in SPEC/milestones/Mx.md.
- Exit Criteria satisfied with attached (s).
## Consequences
Milestone Mx is now the baseline. Subsequent work MUST not contradict it.
## References
- SPEC/milestones/Mx.md
- : <link or ID>