# observability-narration-core — Shared Narration Helper

Status: Draft
Owner: @llama-orch-maintainers

Purpose
- Provide a tiny, dependency-light helper to attach a `human` narration string alongside structured tracing fields.
- Unify narration across `orchestratord`, provisioners, adapters, and manager.

Spec Links
- `.specs/proposals/2025-09-19-human-narration-logging.md` (ORCH-33xx)

Detailed behavior (High / Mid / Low)

- High-level
  - Offers a small facade to emit human-readable narration strings co-located with structured fields.
  - Keeps a consistent taxonomy of fields across crates (see `README_LLM.md`).

- Mid-level
  - Provides helpers for common emission points: admission, placement, stream start/end, cancel.
  - Redaction utilities ensure secrets are never present in narration; integrates with error taxonomy.
  - Optional capture adapter enables tests/BDD to assert narration presence and content.

- Low-level
  - Minimal dependencies to remain usable by CLIs and provisioners; integrates with `tracing` when available.
  - Outputs are structured-first; narration string is an adjunct field (e.g., `human`).

## Style Guide — Human-Friendly Narration

- MUST be natural language and human-friendly. Do not dump opaque identifiers into `human`.
- Keep raw IDs (UUIDs, hashes) in structured fields such as `job_id`, `session_id`, etc.
- Use present tense, subject–verb–object, ≤ ~100 characters.
- MUST NOT include secrets or PII; use redaction helpers when in doubt.
- Prefer descriptive subjects (e.g., pool or model labels) to opaque IDs.

Examples

- Good: "Asked pool-managerd for readiness of pool 'default'"
- Good: "Accepted request; queued at position 3 (ETA 420 ms) on pool 'default'"
- Good: "Canceled streaming upon client request"
- Bad:  "task=00000000-0000-0000-0000-000000000001 queued pos=3"
- Bad:  "pool=3f4b6e readied" (opaque ID with no context)

Refinement Opportunities
- Add test capture adapter for BDD assertions.
- Add redaction helpers and taxonomy constants.
