# README.md Audit & Update Plan

**Date**: 2025-10-01  
**Purpose**: Prepare all README.md files for main branch (where only READMEs will remain)  
**Strategy**: Update each README to be comprehensive and self-contained

---

## Context

Main branch will contain **only README.md files** (all other .md files removed by CI).  
Development branch will keep all documentation.  
Therefore, each README.md must be **comprehensive and self-contained**.

---

## README.md Files to Update (67 total)

### Root (1)
- [x] `./README.md` — **DONE** (just updated, removed profile terminology)

### Binaries (4)
- [ ] `./bin/orchestratord/README.md`
- [ ] `./bin/orchestratord/bdd/README.md`
- [ ] `./bin/pool-managerd/README.md`
- [ ] `./bin/pool-managerd/bdd/README.md`

### Core Libraries (6)
- [ ] `./libs/orchestrator-core/README.md`
- [ ] `./libs/orchestrator-core/bdd/README.md`
- [ ] `./libs/catalog-core/README.md`
- [ ] `./libs/catalog-core/bdd/README.md`
- [ ] `./libs/adapter-host/README.md`
- [ ] `./libs/proof-bundle/README.md`

### Multi-Node Libraries (4)
- [ ] `./libs/control-plane/service-registry/README.md`
- [ ] `./libs/gpu-node/handoff-watcher/README.md`
- [ ] `./libs/gpu-node/node-registration/README.md`
- [ ] `./libs/shared/pool-registry-types/README.md`

### Worker Adapters (10)
- [ ] `./libs/worker-adapters/README.md` (overview)
- [ ] `./libs/worker-adapters/adapter-api/README.md`
- [ ] `./libs/worker-adapters/http-util/README.md`
- [ ] `./libs/worker-adapters/llamacpp-http/README.md`
- [ ] `./libs/worker-adapters/vllm-http/README.md`
- [ ] `./libs/worker-adapters/tgi-http/README.md`
- [ ] `./libs/worker-adapters/triton/README.md`
- [ ] `./libs/worker-adapters/openai-http/README.md`
- [ ] `./libs/worker-adapters/mock/README.md`
- [ ] `./libs/worker-adapters/http-util/.proof_bundle/README.md` (template)

### Provisioners (4)
- [ ] `./libs/provisioners/engine-provisioner/README.md`
- [ ] `./libs/provisioners/engine-provisioner/bdd/README.md`
- [ ] `./libs/provisioners/model-provisioner/README.md`
- [ ] `./libs/provisioners/model-provisioner/bdd/README.md`

### Observability (3)
- [ ] `./libs/observability/narration-core/README.md`
- [ ] `./libs/observability/narration-core/bdd/README.md`
- [ ] `./libs/auth-min/README.md`

### Contracts (2)
- [ ] `./contracts/api-types/README.md`
- [ ] `./contracts/config-schema/README.md`

### Test Harness (5)
- [ ] `./test-harness/bdd/README.md`
- [ ] `./test-harness/chaos/README.md`
- [ ] `./test-harness/determinism-suite/README.md`
- [ ] `./test-harness/e2e-haiku/README.md`
- [ ] `./test-harness/metrics-contract/README.md`

### Tools (3)
- [ ] `./tools/openapi-client/README.md`
- [ ] `./tools/spec-extract/README.md`
- [ ] `./tools/readme-index/README.md`
- [ ] `./xtask/README.md`

### Consumers (12)
- [ ] `./consumers/llama-orch-sdk/README.md`
- [ ] `./consumers/llama-orch-utils/README.md`
- [ ] `./consumers/llama-orch-utils/src/fs/file_reader/README.md`
- [ ] `./consumers/llama-orch-utils/src/fs/file_writer/README.md`
- [ ] `./consumers/llama-orch-utils/src/llm/invoke/README.md`
- [ ] `./consumers/llama-orch-utils/src/model/define/README.md`
- [ ] `./consumers/llama-orch-utils/src/orch/response_extractor/README.md`
- [ ] `./consumers/llama-orch-utils/src/params/define/README.md`
- [ ] `./consumers/llama-orch-utils/src/prompt/message/README.md`
- [ ] `./consumers/llama-orch-utils/src/prompt/thread/README.md`
- [ ] `./consumers/.examples/M002-pnpm/README.md`

### Frontend (2)
- [ ] `./frontend/bin/commercial-frontend/README.md`
- [ ] `./frontend/libs/storybook/README.md`
- [ ] `./frontend/bin/commercial-frontend/.docs/sections/README.md` (nested)

### Proof Bundle Templates (10)
- [ ] `./.proof_bundle/README.md`
- [ ] `./.proof_bundle/templates/README.md`
- [ ] `./.proof_bundle/templates/bdd/README.md`
- [ ] `./.proof_bundle/templates/chaos/README.md`
- [ ] `./.proof_bundle/templates/contract/README.md`
- [ ] `./.proof_bundle/templates/determinism/README.md`
- [ ] `./.proof_bundle/templates/e2e-haiku/README.md`
- [ ] `./.proof_bundle/templates/home-profile-smoke/README.md` ⚠️ **RENAME NEEDED**
- [ ] `./.proof_bundle/templates/integration/README.md`
- [ ] `./.proof_bundle/templates/unit/README.md`

### Skip (1)
- ⏭️ `./.pytest_cache/README.md` (generated, skip)

---

## Update Strategy

### Priority Order

1. **P0 - Core Binaries** (4 files)
   - orchestratord, pool-managerd (+ their BDD)
   - Most important for users

2. **P1 - Core Libraries** (6 files)
   - orchestrator-core, catalog-core, adapter-host, proof-bundle
   - Foundation of the system

3. **P2 - Multi-Node Libraries** (4 files)
   - service-registry, handoff-watcher, node-registration, pool-registry-types
   - Critical for distributed deployments

4. **P3 - Worker Adapters** (10 files)
   - All adapter implementations
   - Important for engine integration

5. **P4 - Supporting Libraries** (9 files)
   - Provisioners, observability, auth, contracts

6. **P5 - Test Harness** (5 files)
   - Test infrastructure

7. **P6 - Tools & Consumers** (16 files)
   - Developer tools and SDKs

8. **P7 - Frontend** (3 files)
   - Vue applications

9. **P8 - Templates** (10 files)
   - Proof bundle templates

---

## Standard README Template

Each README should include:

### For Libraries/Crates

```markdown
# [Crate Name]

**Purpose**: One-line description

**Location**: `path/to/crate`

---

## What This Crate Does

Clear explanation of responsibilities.

## Architecture

How it fits into llama-orch.

## Key APIs

Main public interfaces (traits, structs, functions).

## Dependencies

What it depends on and why.

## Usage Example

```rust
// Code example
```

## Testing

How to run tests for this crate.

```bash
cargo test -p crate-name
```

## Related

- Link to specs (if any)
- Link to related crates
```

### For Binaries

```markdown
# [Binary Name]

**Purpose**: One-line description

**Location**: `bin/name`

---

## What This Binary Does

Clear explanation of responsibilities.

## Configuration

Environment variables and config files.

## Running

```bash
cargo run -p binary-name
```

## Architecture

How it uses libraries.

## Testing

How to run tests.

## Related

- Link to specs
- Link to operational docs
```

---

## Terminology to Remove

Replace everywhere:
- `HOME_PROFILE` → describe as "single-machine" or "localhost"
- `CLOUD_PROFILE` → describe as "multi-node" or "distributed"
- "home profile" → "single-machine deployment"
- "cloud profile" → "multi-node deployment"

---

## Special Cases

### 1. Proof Bundle Template: home-profile-smoke
**File**: `.proof_bundle/templates/home-profile-smoke/README.md`  
**Action**: Rename directory to `embedded-mode-smoke` or `localhost-smoke`

### 2. BDD Subcrates
Keep concise - just explain they're test helpers for the parent crate.

### 3. Frontend READMEs
Focus on Vue-specific setup, dev server, build process.

---

## Execution Plan

1. Create template examples for each category
2. Update in priority order (P0 → P8)
3. Verify no profile terminology remains
4. Test that each README is self-contained

---

## Progress Tracking

- **Total**: 67 files
- **Done**: 1 (root README.md)
- **Remaining**: 66
- **Estimated time**: ~10-15 minutes per README = 11-16 hours total

---

## Notes

- Each README must be **self-contained** (no references to other .md files that will be deleted)
- Include **code examples** where helpful
- Include **test commands** for each crate
- Keep **concise but complete**
- Focus on **what developers need to know**
