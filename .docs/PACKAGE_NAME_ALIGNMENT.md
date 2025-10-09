# Package Name Alignment — Completed

**Date**: 2025-10-03  
**Status**: ✅ All package names now match directory names

## Problem

Several Cargo.toml files had package names that didn't match their directory names, causing build failures and confusion.

## Changes Made

### queen-rbee-crates

| Directory | Old Package Name | New Package Name |
|-----------|-----------------|------------------|
| `pool-registry/` | `service-registry` | `pool-registry` |
| `scheduling/` | `placement` | `scheduling` |

### pool-managerd-crates

| Directory | Old Package Name | New Package Name |
|-----------|-----------------|------------------|
| `pool-registration-client/` | `node-registration` | `pool-registration-client` |
| `error-ops/` | `error-recovery` | `error-ops` |
| `worker-lifecycle/` | `lifecycle` | `worker-lifecycle` |
| `control-api/` | `pool-managerd-api` | `control-api` |
| `model-catalog/` | `catalog-core` | `model-catalog` |
| `model-catalog/bdd/` | `catalog-core-bdd` | `model-catalog-bdd` |

## Verification

```bash
cargo metadata --format-version 1 --no-deps
# Exit code: 0 ✅
```

All workspace members now load successfully. Package names follow the spec and match their directory names for consistency.

## Principle

**Directory name = Package name** (except for special cases like `observability-narration-core` where the prefix adds context).

This makes the codebase easier to navigate and prevents import confusion.
