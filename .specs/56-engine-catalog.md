# Engine Catalog — Component Specification (root overview)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Purpose & Scope

Provide durable, auditable records of prepared engines (binaries or images) for reproducibility, determinism, and operational traceability. Complements the model catalog by storing engine metadata and canonical artifact paths.

In scope:
- EngineEntry index storage with atomic writes.
- References to build/pull provenance (git ref, image digest) and artifacts.
- Integration with engine-provisioner and pool-managerd.

Out of scope:
- Building engines (delegated to engine-provisioner).
- Runtime supervision (delegated to pool-managerd).

## Data Model

```
EngineEntry {
  id: String,                    // stable engine id (engine:version@digest or similar)
  engine: String,                // e.g., "llamacpp", "vllm", "tgi", "triton"
  version: String,               // semantic or upstream version string
  build_ref: Option<String>,     // git ref/commit or image tag
  digest: Option<String>,        // binary/image digest when available
  build_flags: Option<Vec<String>>, // CMake/CLI flags influencing runtime determinism
  artifacts: Vec<PathBuf>,       // resolved local paths (binary, libs) or image ref
  created_ms: i64,               // unix ms
}
```

## Provided Contracts (summary)

- Read/write API aligned to `catalog-core` helpers (atomic file writes, schema versioning).
- Lookup by `id` or filters (engine, version).

## Consumed Contracts (summary)

- `engine-provisioner` writes EngineEntry after successful ensure/build.
- `pool-managerd` reads fields (engine_version/digest) to report readiness and logs.
- `orchestratord` logs placement decisions including engine metadata.

## Observability & Determinism

- Proof bundles include EngineEntry snapshots for runs.
- Determinism suite captures `engine_version` and `engine_digest` in outputs.

## Security

- No secrets; write paths constrained under the catalog root.

## Testing & Proof Bundles

- Unit: index round-trip, versioning, atomic write under crash simulation.
- Integration: `engine-provisioner` → EngineEntry creation; `pool-managerd` registry propagation.

## Refinement Opportunities

- Signed EngineEntry manifests; SBOM linkage.
- Cache eviction policy and retention windows for old engines.
- Cross-node sharing and deduplication.
