# Mx — <Milestone Title>

## Scope

* What this milestone delivers (clear, bounded).
* Keep language tight and action-oriented.

## Non-Goals

* Explicitly state what is out of scope, even if related.
* Prevents drift and over-expectation.

## Exit Criteria

* List verifiable gates that **MUST** be met.
* Each bullet should be objectively checkable.
* Do not speculate about future milestones.

## Required Artifacts

* Proof bundle elements required to accept this milestone. Examples:

  * `proof-bundle/manifest.json`
  * `events/run.jsonl`
  * `params.lock.json`
  * `diffs/` for file mutations
* Artifacts **MUST** be hashable, reproducible, and linked in acceptance ADR.

## Risks & Controls

* Known risks of this milestone.
* Controls or mitigations planned.
* Keep concrete; avoid hand-waving.

## Rollback

* How to undo or disable this milestone cleanly if acceptance fails.
* Specify whether rollback is code-level (revert) or config-level (disable flag).

## Changelog

* `YYYY-MM-DD — <summary of change>`
* Every modification to this spec **MUST** be logged here.
