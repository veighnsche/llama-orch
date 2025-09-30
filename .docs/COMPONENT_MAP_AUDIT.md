# Component Dependency Graph Audit

**Date**: 2025-10-01  
**Status**: OUTDATED - Missing critical components

---

## Executive Summary

You were **absolutely right**. The Component Dependency Graph in README.md (lines 589-693) is **pre-migration** and missing several critical components that were added during the service registry implementation.

## Missing Components

### 1. ❌ Service Registry (libs/control-plane/service-registry)
**Status**: EXISTS in codebase, MISSING from diagram  
**Location**: `libs/control-plane/service-registry/`  
**Purpose**: Node tracking, health management for multi-node deployments  
**Dependencies**: 
- Uses `pool-registry-types`
- Used by `orchestratord`

### 2. ❌ Handoff Watcher (libs/gpu-node/handoff-watcher)
**Status**: EXISTS in codebase, MISSING from diagram  
**Location**: `libs/gpu-node/handoff-watcher/`  
**Purpose**: Watches local handoff files on GPU nodes  
**Dependencies**:
- Uses `pool-registry-types`
- Used by `pool-managerd`

### 3. ❌ Node Registration (libs/gpu-node/node-registration)
**Status**: EXISTS in codebase, MISSING from diagram  
**Location**: `libs/gpu-node/node-registration/`  
**Purpose**: GPU node registration with control plane  
**Dependencies**:
- Uses `pool-registry-types`
- Used by `pool-managerd`

### 4. ❌ Pool Registry Types (libs/shared/pool-registry-types)
**Status**: EXISTS in codebase, MISSING from diagram  
**Location**: `libs/shared/pool-registry-types/`  
**Purpose**: Shared types for node communication  
**Dependencies**:
- Used by service-registry, handoff-watcher, node-registration, orchestratord, pool-managerd

### 5. ❌ Auth-Min (libs/auth-min)
**Status**: EXISTS in codebase, MISSING from diagram  
**Location**: `libs/auth-min/`  
**Purpose**: Bearer token authentication  
**Dependencies**:
- Used by `orchestratord` and `pool-managerd`

### 6. ❌ Narration Core (libs/observability/narration-core)
**Status**: EXISTS in codebase, MISSING from diagram  
**Location**: `libs/observability/narration-core/`  
**Purpose**: Human-readable event narration  
**Dependencies**:
- Used by `orchestratord`

### 7. ❌ Proof Bundle (libs/proof-bundle)
**Status**: EXISTS in codebase, MISSING from diagram  
**Location**: `libs/proof-bundle/`  
**Purpose**: Test artifact standardization  
**Dependencies**:
- Used by test harnesses

### 8. ❌ Consumers (llama-orch-sdk, llama-orch-utils)
**Status**: EXISTS in codebase, MISSING from diagram  
**Location**: `consumers/llama-orch-sdk/`, `consumers/llama-orch-utils/`  
**Purpose**: Client SDKs and utilities  
**Dependencies**:
- SDK uses OpenAPI contracts
- Utils uses SDK

### 9. ❌ BDD Subcrates
**Status**: EXISTS in codebase, MISSING from diagram  
**Multiple locations**:
- `libs/orchestrator-core/bdd`
- `bin/orchestratord/bdd`
- `bin/pool-managerd/bdd`
- `libs/catalog-core/bdd`
- `libs/proof-bundle/bdd`
- `libs/observability/narration-core/bdd`
- `libs/worker-adapters/http-util/bdd`
- `libs/provisioners/engine-provisioner/bdd`
- `libs/provisioners/model-provisioner/bdd`

---

## Components Shown But Outdated

### pool-managerd
**Current diagram**: Shows as library in "Core" subgraph  
**Reality**: It's a **binary** in `bin/pool-managerd/`  
**Fix needed**: Move to Binaries subgraph or clarify it's both binary + lib

---

## Diagram Structure Issues

### Issue 1: No Multi-Node Subgraph
The diagram doesn't have a dedicated subgraph for multi-node components (service-registry, handoff-watcher, node-registration, pool-registry-types).

### Issue 2: No Observability Subgraph
Narration-core and proof-bundle aren't grouped.

### Issue 3: No Auth Subgraph
Auth-min is missing entirely.

### Issue 4: No Consumer Subgraph
SDK and Utils are missing.

---

## Recommended Actions

### Option 1: Update Existing Diagram
Add missing components to the current Component Dependency Graph.

**Pros**: Maintains continuity  
**Cons**: Diagram will become very large and complex

### Option 2: Replace with Binaries and Libraries Map
The "Binaries and Libraries Dependency Map" (lines 239-325) is **more up-to-date** and includes:
- ✅ Service registry
- ✅ Handoff watcher
- ✅ Node registration
- ✅ Pool registry types
- ✅ Auth-min
- ✅ Narration-core

**Recommendation**: **Delete the outdated Component Dependency Graph** and keep only the "Binaries and Libraries Dependency Map" which is current.

### Option 3: Create New Simplified Diagram
Create a new, simpler diagram focused on the two-service architecture:
- orchestratord dependencies
- pool-managerd dependencies
- Shared libraries between them

---

## Comparison: Which Diagram is Better?

### Binaries and Libraries Dependency Map (Line 239)
- ✅ Up-to-date with multi-node components
- ✅ Shows service-registry, handoff-watcher, node-registration
- ✅ Shows auth-min, narration-core
- ✅ Cleaner, more maintainable
- ✅ Focuses on actual binaries (orchestratord, pool-managerd)
- ❌ Doesn't show test harness connections

### Component Dependency Graph (Line 589)
- ❌ Missing 9+ critical components
- ❌ Pre-migration artifact
- ❌ Outdated architecture view
- ✅ Shows test harness connections
- ❌ Too complex and unmaintained

---

## Recommendation

**Delete the Component Dependency Graph** (lines 580-699) entirely.

**Rationale**:
1. It's pre-migration and missing critical components
2. The "Binaries and Libraries Dependency Map" is more accurate
3. Maintaining two large diagrams is error-prone
4. The test harness connections it shows are already documented in the Workspace Map table

**Keep**:
- Single-Machine Deployment diagram (lines 157-186)
- Multi-Machine Deployment diagram (lines 190-235)
- Binaries and Libraries Dependency Map (lines 239-325)
- Workspace Map table (auto-generated, lines 705-840)

---

## Verification Commands

```bash
# List all workspace members
grep -A 100 'members = \[' Cargo.toml

# Find all libs
find libs -maxdepth 2 -name Cargo.toml | sort

# Find all binaries
find bin -maxdepth 2 -name Cargo.toml | sort

# Find all test harnesses
find test-harness -maxdepth 2 -name Cargo.toml | sort
```

---

## Next Steps

1. **Delete** Component Dependency Graph section (lines 580-699)
2. **Keep** Binaries and Libraries Dependency Map (already up-to-date)
3. **Verify** Workspace Map table is current (it's auto-generated)
4. **Update** `.docs/README_MERMAID_STATUS.md` to reflect the deletion
