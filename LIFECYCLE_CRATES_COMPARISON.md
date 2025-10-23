# Lifecycle Crates Comparison

**Date:** Oct 23, 2025  
**Analysis:** TEAM-276

## Overview

Comparison of three lifecycle crates to assess consistency and identify opportunities for standardization.

## Crate Locations

1. **queen-lifecycle** - `bin/05_rbee_keeper_crates/queen-lifecycle`
2. **hive-lifecycle** - `bin/15_queen_rbee_crates/hive-lifecycle`
3. **worker-lifecycle** - `bin/25_rbee_hive_crates/worker-lifecycle`

## Module Structure Comparison

### queen-lifecycle (11 modules)
```
✅ lib.rs           - Exports and documentation
✅ types.rs         - QueenHandle type
✅ health.rs        - Health checking
✅ ensure.rs        - Ensure queen running
✅ start.rs         - Start operation
✅ stop.rs          - Stop operation
✅ status.rs        - Status operation
✅ rebuild.rs       - Rebuild operation
✅ info.rs          - Build info operation
✅ install.rs       - Install operation
✅ uninstall.rs     - Uninstall operation
```

### hive-lifecycle (14 modules)
```
✅ lib.rs           - Exports and documentation
✅ types.rs         - Request/Response types
✅ validation.rs    - Validation helpers
✅ ssh_helper.rs    - SSH utilities
✅ ssh_test.rs      - SSH connection testing
✅ hive_client.rs   - HTTP client for capabilities
✅ start.rs         - Start operation
✅ stop.rs          - Stop operation
✅ status.rs        - Status operation
✅ list.rs          - List operation
✅ get.rs           - Get operation
✅ install.rs       - Install operation
✅ uninstall.rs     - Uninstall operation
✅ capabilities.rs  - Refresh capabilities
```

### worker-lifecycle (6 modules)
```
✅ lib.rs           - Exports and documentation
✅ types.rs         - Request/Response types
✅ spawn.rs         - Spawn operation
✅ delete.rs        - Delete operation
✅ process_list.rs  - List processes (local ps)
✅ process_get.rs   - Get process (local ps)
```

## Common Operations

| Operation | queen-lifecycle | hive-lifecycle | worker-lifecycle |
|-----------|----------------|----------------|------------------|
| **start/spawn** | ✅ start.rs | ✅ start.rs | ✅ spawn.rs |
| **stop/delete** | ✅ stop.rs | ✅ stop.rs | ✅ delete.rs |
| **status** | ✅ status.rs | ✅ status.rs | ❌ (stateless) |
| **list** | ❌ (singleton) | ✅ list.rs | ✅ process_list.rs |
| **get** | ❌ (singleton) | ✅ get.rs | ✅ process_get.rs |
| **install** | ✅ install.rs | ✅ install.rs | ❌ (catalog) |
| **uninstall** | ✅ uninstall.rs | ✅ uninstall.rs | ❌ (catalog) |

## Naming Consistency

### ✅ Consistent Patterns
- All use `lib.rs` for exports
- All use `types.rs` for type definitions
- All follow `operation.rs` naming (start, stop, status, etc.)

### ⚠️ Inconsistent Patterns

1. **Function Naming**
   - queen: `start_queen()`, `stop_queen()`, `install_queen()`
   - hive: `execute_hive_start()`, `execute_hive_stop()`, `execute_hive_install()`
   - worker: `spawn_worker()`, `delete_worker()`, `list_worker_processes()`

2. **Module Naming**
   - queen: `start.rs`, `stop.rs`, `install.rs`
   - hive: `start.rs`, `stop.rs`, `install.rs` (same ✅)
   - worker: `spawn.rs`, `delete.rs` (different ❌)

3. **Type Naming**
   - queen: Simple types (no request/response structs for most operations)
   - hive: Request/Response pattern (`HiveStartRequest`, `HiveStartResponse`)
   - worker: Config pattern (`WorkerSpawnConfig`, `SpawnResult`)

## Architecture Differences

### queen-lifecycle
- **Purpose**: Manage single queen daemon from CLI
- **Scope**: Local operations only
- **Pattern**: Direct function calls
- **Consumer**: rbee-keeper (CLI)

### hive-lifecycle
- **Purpose**: Manage multiple hive instances from queen
- **Scope**: Remote operations (SSH, HTTP)
- **Pattern**: Request/Response with job_id
- **Consumer**: queen-rbee (orchestrator)

### worker-lifecycle
- **Purpose**: Manage worker processes on hive
- **Scope**: Local process management
- **Pattern**: Config-based spawning
- **Consumer**: rbee-hive (executor)

## Consistency Score

### Structure: 7/10
- ✅ All use modular structure
- ✅ All have lib.rs with exports
- ✅ All have types.rs
- ⚠️ Different number of modules (6-14)
- ⚠️ Different helper modules

### Naming: 5/10
- ✅ Module names mostly consistent
- ❌ Function naming varies significantly
- ❌ Type naming patterns differ
- ❌ Operation names differ (start vs spawn, stop vs delete)

### API Surface: 6/10
- ✅ All export main operations
- ✅ All use Result<T> for errors
- ⚠️ Different parameter patterns
- ⚠️ Different return types

## Recommendations

### 1. **Standardize Function Naming** (High Priority)

**Current:**
```rust
// queen
start_queen(url)
stop_queen(url)

// hive
execute_hive_start(request, config)
execute_hive_stop(request, config)

// worker
spawn_worker(config)
delete_worker(pid, job_id)
```

**Proposed:**
```rust
// Option A: Verb + Noun pattern
start_queen(url)
start_hive(request, config)
spawn_worker(config)

// Option B: Execute pattern (current hive)
execute_queen_start(url)
execute_hive_start(request, config)
execute_worker_spawn(config)

// Option C: Noun + Verb pattern
queen::start(url)
hive::start(request, config)
worker::spawn(config)
```

**Recommendation**: Keep current patterns but document the rationale:
- **queen**: Simple verbs (singleton, CLI-friendly)
- **hive**: Execute pattern (orchestration, job-based)
- **worker**: Action verbs (process management)

### 2. **Standardize Type Patterns** (Medium Priority)

**Proposed:**
```rust
// All operations use Request/Response pattern
pub struct QueenStartRequest { ... }
pub struct QueenStartResponse { ... }

pub struct HiveStartRequest { ... }
pub struct HiveStartResponse { ... }

pub struct WorkerSpawnRequest { ... }
pub struct WorkerSpawnResponse { ... }
```

**Trade-off**: More verbose but more consistent

### 3. **Add Missing Operations** (Low Priority)

**queen-lifecycle:**
- Consider adding `list_queens()` if multi-queen support needed
- Consider adding `get_queen()` for details

**worker-lifecycle:**
- Consider adding `status_worker()` for individual worker status
- Already has process operations (good!)

### 4. **Documentation Consistency** (High Priority)

All crates should have:
- ✅ Module-level documentation
- ✅ Usage examples in lib.rs
- ✅ Function documentation
- ⚠️ Architecture decision records (add to each)

## Current State Assessment

### Strengths ✅
1. **Modular structure** - All crates use clear module separation
2. **Operation coverage** - Each crate covers its domain well
3. **Error handling** - All use Result<T> consistently
4. **Documentation** - All have basic documentation

### Weaknesses ⚠️
1. **Naming inconsistency** - Three different naming patterns
2. **Type patterns** - Three different approaches to types
3. **API surface** - Different parameter patterns
4. **No shared patterns** - Each crate evolved independently

### Opportunities 💡
1. **Create lifecycle-common crate** - Shared types and patterns
2. **Standardize naming** - Document and enforce conventions
3. **Add architecture docs** - Explain why differences exist
4. **Create templates** - For future lifecycle crates

## Conclusion

The three lifecycle crates are **moderately consistent** (6/10 overall):

- ✅ **Structure**: Well-organized, modular, clear separation
- ⚠️ **Naming**: Inconsistent but each has internal consistency
- ⚠️ **Types**: Different patterns but each works for its use case
- ✅ **Functionality**: Complete coverage for each domain

### Recommendation: **Document, Don't Force**

Rather than forcing all crates to be identical, **document the architectural reasons** for differences:

1. **queen-lifecycle** - CLI-friendly, simple functions
2. **hive-lifecycle** - Orchestration-focused, request/response
3. **worker-lifecycle** - Process management, config-based

Create a **LIFECYCLE_ARCHITECTURE.md** that explains:
- Why naming differs (different consumers, different patterns)
- When to use each pattern
- How to extend each crate
- Common anti-patterns to avoid

This preserves the strengths of each crate while providing guidance for future development.

## Action Items

### Immediate (TEAM-276)
- [ ] Create LIFECYCLE_ARCHITECTURE.md explaining patterns
- [ ] Add architecture notes to each crate's lib.rs
- [ ] Document naming conventions in each crate

### Short-term
- [ ] Consider lifecycle-common crate for shared utilities
- [ ] Add more usage examples to each crate
- [ ] Create templates for new lifecycle crates

### Long-term
- [ ] Evaluate if standardization is needed
- [ ] Consider refactoring if patterns prove problematic
- [ ] Monitor for code duplication opportunities
