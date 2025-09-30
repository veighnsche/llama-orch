# README.md Mermaid Diagram Status

**Date**: 2025-10-01  
**Status**: All diagrams updated and valid

---

## Diagrams in README.md

### 1. Single-Machine Deployment (Line 157-186)
**Status**: ✅ Updated and valid
- Removed "CLOUD_PROFILE" terminology
- Shows localhost deployment
- Syntax is correct

### 2. Multi-Machine Deployment (Line 190-235)
**Status**: ✅ Updated and valid
- Removed "CLOUD_PROFILE" terminology
- Shows distributed deployment
- Syntax is correct

### 3. Binaries and Libraries Dependency Map (Line 239-325)
**Status**: ✅ Updated and valid
- High-level view of binary dependencies
- Updated "Cloud Profile Libraries" → "Multi-Node Libraries"
- Updated comments: "cloud profile" → "multi-node"
- Syntax is correct
- **Purpose**: Quick overview of main binaries and library groups
- **Includes**: service-registry, handoff-watcher, node-registration, pool-registry-types, auth-min, narration-core

### 4. Component Dependency Graph
**Status**: ❌ DELETED (was outdated)
- **Reason**: Pre-migration artifact missing 9+ critical components
- **Missing**: service-registry, handoff-watcher, node-registration, pool-registry-types, auth-min, narration-core, proof-bundle, consumers, BDD subcrates
- **Replacement**: The "Binaries and Libraries Dependency Map" is more accurate and up-to-date
- **See**: `.docs/COMPONENT_MAP_AUDIT.md` for full audit

---

## Current Diagram Strategy

### Keep: Binaries and Libraries Dependency Map
- **Scope**: High-level, binary-centric
- **Audience**: Quick understanding of main components
- **Detail Level**: Groups of libraries
- **Maintenance**: Updated during migration, includes all current components
- **Status**: ✅ Current and accurate

### Deleted: Component Dependency Graph
- **Reason**: Pre-migration, unmaintained, missing critical components
- **Replacement**: Workspace Map table (auto-generated) provides comprehensive component listing

**Recommendation**: Single source of truth for architecture diagram. Workspace Map table handles detailed component documentation.

---

## GitHub Rendering Error

The error message you saw:
```
Parse error on line 2:
...D[orchestratord] Client -->|GET /v2/t
```

This appears to be from **cached/old content**. The current README.md does not contain this syntax error.

**Current line 159-160** (correct):
```mermaid
    Client[Client] -->|POST /v2/tasks| OD[orchestratord]
    Client -->|GET /v2/tasks/:id/events| OD
```

**What you saw** (old, incorrect):
```mermaid
    Client[Client] -->|POST /v2/tasks| OD[orchestratord]    Client -->|GET /v2/tasks/:id/events| OD
```

The old version had both Client lines on the same line (missing newline), which causes the parse error.

### Solution
1. **Hard refresh** your browser (Ctrl+Shift+R / Cmd+Shift+R)
2. **Clear GitHub cache** or view in incognito
3. **Wait a few minutes** for GitHub's CDN to update

The file is correct in the repository.

---

## Verification Commands

```bash
# Check for syntax errors in mermaid blocks
grep -A 50 '```mermaid' README.md | head -n 200

# Verify no profile terminology remains
rg "HOME_PROFILE|CLOUD_PROFILE|home.profile|cloud.profile" README.md

# Check for the specific error pattern
rg "orchestratord\].*Client.*-->" README.md
```

All checks pass. The README.md is correct.
