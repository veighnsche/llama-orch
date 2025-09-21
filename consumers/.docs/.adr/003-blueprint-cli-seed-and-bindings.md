# ADR-003: Blueprint Init CLI — Seed Files & Generated Capability Bindings

## Status

Proposed

## Context

We’re standardizing what a newly initialized **Blueprint** (ADR-002) contains and how the **init CLI** generates **compile-time capability bindings** for TS/JS/Rust.
Key principles:

* **Types live in the library** (`llama-orch-toolkit`) and are versioned with it.
* **Generated files are pure data snapshots** (no type definitions), importing the library’s types.
* Snapshot is **static at generation time** (user re-runs CLI to refresh).
* Engines ≠ models: **engines** list backends and supported workloads; **models** are concrete runnable model entries (no `ctx_max` field in our fixed types).
* Pools describe available hardware groups.

## Decision (RFC-2119)

* The init CLI **MUST** create a minimal Blueprint layout:

  * `Spec.md` (seed context)
  * `process.yml` (starter process)
  * Language binding(s): `capabilities.d.ts` + `capabilities.ts` (TS), or `capabilities.js` (JS), or `capabilities.rs` (Rust)
* The CLI **MUST** call the server to obtain:

  * **Capabilities snapshot** (engines + workloads, etc.)
  * **Model catalog snapshot** (model IDs, digests, states)
  * **Pool snapshot** (pool IDs, GPU names)
* Generated bindings **MUST**:

  * Import **types from the toolkit library** (no duplicated type defs).
  * Contain **only constants/data** that conform to those types.
  * Exclude live/volatile fields (queue depth, wait times, etc.).
* If the toolkit library evolves its types, **regeneration MUST be required** for Blueprints to compile (intentionally fail-fast for drift).

## Consequences

**Positive:** Strong compile-time guarantees; single source of truth for types; deterministic snapshots for builds and audits.
**Negative:** Users must re-run the CLI when the server gains new models/pools.
**Trade-offs:** Prefer static determinism over dynamic discovery at runtime.

## Alternatives Considered

* Runtime-only discovery → simpler but loses compile-time guarantees (Rejected).
* JSON manifest only → shifts type work to users (Rejected).

## Preview of Generated Files

> **Note:** The following **do not define types**. They import the fixed types you placed in the library:
>
> ```rust
> pub struct EngineCapability { pub id: &'static str, pub workloads: &'static [&'static str] }
> pub struct ModelInfo { pub id: &'static str, pub digest: &'static str, pub state: &'static str }
> pub struct PoolInfo { pub id: &'static str, pub gpus: &'static [&'static str] }
> pub struct Capabilities {
>   pub engines: &'static [EngineCapability],
>   pub models:  &'static [ModelInfo],
>   pub pools:   &'static [PoolInfo],
> }
> ```
>
> (Exact module path shown below in imports.)

### TypeScript — `capabilities.d.ts`

```ts
// generated file — do not edit
// Types are imported from the toolkit library (WASM/npm package).
import type { Capabilities } from "@llama-orch/toolkit/capabilities";

export declare const CAPABILITIES: Capabilities;
```

### TypeScript — `capabilities.ts`

```ts
// generated file — do not edit
import type {
  Capabilities,
  EngineCapability,
  ModelInfo,
  PoolInfo,
} from "@llama-orch/toolkit/capabilities";

const ENGINES: EngineCapability[] = [
  { id: "llamacpp", workloads: ["completion","embedding","rerank"] },
  { id: "vllm",     workloads: ["completion","embedding","rerank"] },
  // … populated from /v1/capabilities
];

const MODELS: ModelInfo[] = [
  { id: "Meta-Llama-3-8B-Instruct", digest: "sha256:…", state: "ready" },
  { id: "Mistral-7B-Instruct",      digest: "sha256:…", state: "ready" },
  // … populated from catalog snapshot
];

const POOLS: PoolInfo[] = [
  { id: "default", gpus: ["A100-80GB","A100-80GB"] },
  // … populated from pools snapshot
];

export const CAPABILITIES: Capabilities = {
  engines: ENGINES,
  models:  MODELS,
  pools:   POOLS,
};
```

### JavaScript — `capabilities.js`

```js
// generated file — do not edit
// JS has no types; TS users should prefer the .d.ts + .ts pair.
export const CAPABILITIES = {
  engines: [
    { id: "llamacpp", workloads: ["completion","embedding","rerank"] },
    { id: "vllm",     workloads: ["completion","embedding","rerank"] },
  ],
  models: [
    { id: "Meta-Llama-3-8B-Instruct", digest: "sha256:…", state: "ready" },
    { id: "Mistral-7B-Instruct",      digest: "sha256:…", state: "ready" },
  ],
  pools: [
    { id: "default", gpus: ["A100-80GB","A100-80GB"] },
  ],
};
```

### Rust — `capabilities.rs`

```rust
// generated file — do not edit
use llama_orch_toolkit::capabilities::{
    Capabilities, EngineCapability, ModelInfo, PoolInfo,
};

pub const CAPABILITIES: Capabilities = Capabilities {
    engines: &[
        EngineCapability { id: "llamacpp", workloads: &["completion","embedding","rerank"] },
        EngineCapability { id: "vllm",     workloads: &["completion","embedding","rerank"] },
    ],
    models: &[
        ModelInfo { id: "Meta-Llama-3-8B-Instruct", digest: "sha256:…", state: "ready" },
        ModelInfo { id: "Mistral-7B-Instruct",      digest: "sha256:…", state: "ready" },
    ],
    pools: &[
        PoolInfo { id: "default", gpus: &["A100-80GB","A100-80GB"] },
    ],
};
```

## Seed Files (created by CLI)

* `Spec.md` — user-authored context & requirements.
* `process.yml` — starter process referencing toolkit applets.
* Language binding(s) as above in `generated/` (or similar) directory.
* Optional: `.blueprint/manifest.json` with API responses hashed for provenance.

## Artifacts / Proof Bundle

The CLI run **SHOULD** emit a small proof bundle:

* Request/response snapshots for discovery endpoints,
* Hashes of generated files,
* Generation timestamp and toolkit version.

## References

* ADR-002 “Blueprints” (naming & scope).
* Toolkit library capability types (single source of truth).
