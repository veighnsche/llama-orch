# ADR-004: Applet Ground Rules (Contracts, Scope, Determinism, Side-Effects)

## Status

Accepted

## Context

We need crisp, enforceable rules so applets are **small, composable tools** that empower building **Blueprints** (ADR-002) without exploding into thousands of single-purpose variants or, conversely, becoming vague “do-everything” blobs. This ADR defines **what every applet MUST/SHOULD/MUST NOT do**, independent of namespaces.

## Decision (RFC-2119)

### 1) Contract & IO

* Every applet **MUST** declare a **stable JSON input schema** and **stable JSON output schema** with examples; unknown/extra fields **MUST** be rejected by default.
* Inputs **MUST** be explicit. **No magic locations** or implicit discovery. If a path/glob is needed, the caller **MUST** provide it.
* Applets **MUST NOT** depend on hidden global state (env, CWD, project layout). Any optional project config **MUST** be passed explicitly.
* Applets **MUST** be usable as **pure transforms** from inputs → outputs. Side-effects (FS/network) **MUST** happen only in applets whose purpose is that side-effect.

### 2) Scope & Coupling (the “fine line”)

* An applet **MUST** do **one crisp thing** and expose **only the minimal parameters** needed to do that thing well.
* Over-specialization is forbidden: **Do not** create “read\_seed\_file\_x.md”-style variants. A single “read files” applet (as a class of applet) **MUST** support **one or many paths** provided by the caller.
* Over-generalization is also forbidden: an applet that converts **llama-orch responses** to text **MUST** be **specific to llama-orch response types** and **MUST NOT** claim to “convert arbitrary content to text.”
* Applets **SHOULD** avoid embedding domain assumptions (e.g., “seed”) unless that domain is passed in as data. Templates/prompt builders **MUST** accept multiple input forms (literal string, array of lines, file reference, or dedent helper) **without** assuming any file is a “seed.”

### 3) Determinism & Reproducibility

* If an applet calls a model, it **MUST** accept a **seed/determinism mode** and **MUST** pass through model/engine params verbatim. It **MUST NOT** mutate prompts/params implicitly.
* Applets that call models **MUST** emit enough metadata (prompt, params, model id, engine id, seed) so runs can be reproduced.
* Non-model applets **SHOULD** be order-insensitive/pure where possible, or document ordering constraints explicitly.

### 4) Observability & Proof

* Each applet **MUST** log a compact **operation record**: inputs (redacted as configured), outputs, decision points, and references to any created artifacts.
* Where relevant, applets **SHOULD** emit artifacts (e.g., transcript/trace) using the shared artifact mechanism so Blueprints can assemble proof bundles.
* Error paths **MUST** include structured error codes and actionable messages (what the caller can fix vs. re-tryable conditions).

### 5) Errors, Idempotency, Retries

* Pure-transform applets **MUST** be **idempotent** for identical inputs.
* Side-effecting applets **MUST** document idempotency behavior (e.g., “overwrite,” “skip if exists,” or “fail if exists”) and accept a **mode** parameter to control it.
* Transient failures (e.g., network hiccups to llama-orch) **SHOULD** be retryable with backoff; permanent failures **MUST** fail fast with clear diagnostics.

### 6) Side-Effects & Filesystem

* Only applets whose purpose is **writing** **MAY** write to disk. All other applets **MUST NOT** write.
* **M1 exception (per project scope):** the file-writing applet **MUST** write exactly where instructed, **with NO guardrails**. Later milestones may add opt-in guardrails.
* Applets that read files **MUST** accept **one or many** explicit paths/globs and return content + stable identifiers (e.g., hashes). They **MUST NOT** infer locations.
* Any applet that claims to “merge” content **MUST** describe the merge strategy deterministically and return the merged value as data; it **MUST NOT** write unless it is a writer.

### 7) Model Invocation (llama-orch)

* The applet that **sends prompts to llama-orch** **MUST** have a **simple signature** (messages + required model id; engine/pool/params optional) and hide transport complexity internally.
* The applet that **extracts text from a llama-orch response** **MUST** be **specific** to the llama-orch response envelopes (stream or batch) and produce **exactly** the text payload the downstream step expects. It **MUST NOT** be a generic “content-to-text” utility.

### 8) Streaming vs Batch

* Applets that handle streaming **MUST** define how they expose partials/finals (e.g., buffered final string, plus optional token events for callers who care).
* Downstream applets **MUST** specify whether they require the **final** value (e.g., full text) or can run incrementally.

### 9) Configuration & Defaults

* Required arguments (e.g., **model id**) **MUST** be required in the schema. Defaults **MAY** be supplied for optional params (temperature, engine id, pool hint).
* Precedence **MUST** be explicit: inline inputs > provided config object > library defaults. No environment-driven surprises.

### 10) Security & Side Channels

* Applets **MUST NOT** shell out or perform arbitrary network calls; all llama-orch communication **MUST** go through the approved client.
* Paths **MUST** be treated as data; no path traversal, no implicit expansion except for explicitly supported syntaxes documented in the schema (e.g., `file:~/…` when explicitly allowed).

### 11) Versioning & Compatibility

* Every applet **MUST** carry a semantic **applet version**. Breaking changes **MUST** bump the major version and update schemas.
* Deprecations **SHOULD** include a migration note and a sunset plan.

### 12) Documentation & Examples

* Each applet **MUST** publish: schema, minimal example, and at least one **Blueprint snippet** showing intended composition with neighbors (reader → prompt → invoke → extractor → writer).
* Examples **MUST** reflect M1 constraints where relevant (e.g., no write guardrails).

## Consequences

**Positive:** Predictable composition, fewer applets, no hidden coupling to “seed” concepts, and no vague “catch-all” utilities that make pipelines mushy.
**Negative:** Slightly more upfront rigor (schemas/docs) per applet; stricter inputs mean callers must be explicit.
**Trade-offs:** We bias to small crisp tools with explicit inputs, even if that means callers wire 2–3 tiny steps instead of one “smart” blob.

## Alternatives Considered

* **Highly specialized applets** (e.g., one per known seed file) → unmanageable explosion; rejected.
* **Over-generic applets** (e.g., “convert anything to text”) → ambiguous IO and future surprises; rejected.

## References

* ADR-002 (Blueprints), ADR-003 (Blueprint init CLI expectations).
* Project determinism & artifact-first principles.
