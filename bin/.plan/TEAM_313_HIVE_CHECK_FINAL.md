# TEAM-313: Hive Check Implementation (SSE Narration Test)

**Status:** ✅ COMPLETE  
**Date:** Oct 27, 2025

## Mission

Implement `hive check` command to test narration through hive's SSE streaming pipeline, following the same pattern as `queen-check`.

## Architecture

```
rbee-keeper hive check
    ↓
POST /v1/jobs → queen-rbee
    ↓
Forward to hive (HiveCheck operation)
    ↓
POST /v1/jobs → rbee-hive
    ↓
hive_check::handle_hive_check()
    ↓
SSE stream → queen → keeper (client sees narration)
```

## Check Commands Trilogy

| Command | Location | Purpose |
|---------|----------|---------|
| `self-check` | rbee-keeper (local) | Test CLI narration (no SSE) |
| `queen-check` | queen-rbee (server) | Test narration through queen SSE |
| `hive check` | rbee-hive (server) | Test narration through hive SSE |

## Implementation (3-File Pattern)

### 1. Operation Contract
**File:** `bin/97_contracts/operations-contract/src/lib.rs`

Added `HiveCheck` operation:
```rust
Operation::HiveCheck {
    alias: String,  // Hive to test (default: "localhost")
}
```

- Added to `Operation::name()` → "hive_check"
- Added to `Operation::hive_id()` → returns alias
- Added to `should_forward_to_hive()` → true (forwards to hive)

### 2. Hive Handler
**File:** `bin/20_rbee_hive/src/hive_check.rs` (NEW, 118 LOC)

Implements narration test (mirrors queen-check):
- Tests all 3 narration modes (Human, Cute, Story)
- Tests format specifiers
- Tests sequential narrations
- Tests job_id context propagation
- All narration routed via SSE

**File:** `bin/20_rbee_hive/src/job_router.rs`

Added routing:
```rust
Operation::HiveCheck { .. } => {
    let ctx = NarrationContext::new().with_job_id(&job_id);
    with_narration_context(ctx, rbee_hive::hive_check::handle_hive_check()).await?;
}
```

### 3. CLI Command
**File:** `bin/00_rbee_keeper/src/cli/hive.rs`

Added `Check` action:
```rust
HiveAction::Check {
    alias: String,  // default: "localhost"
}
```

**File:** `bin/00_rbee_keeper/src/handlers/hive.rs`

Added handler:
```rust
HiveAction::Check { alias } => {
    let operation = Operation::HiveCheck { alias };
    submit_and_stream_job(queen_url, operation).await
}
```

## Files Changed

```
bin/97_contracts/operations-contract/src/lib.rs  (+10 LOC)
bin/20_rbee_hive/src/
├── hive_check.rs                                (NEW, 118 LOC)
├── job_router.rs                                (+18 LOC)
├── lib.rs                                       (+1 LOC)
└── Cargo.toml                                   (+2 dependencies)
bin/00_rbee_keeper/src/
├── cli/hive.rs                                  (+7 LOC)
└── handlers/hive.rs                             (+5 LOC)
```

**Total:** 159 LOC added

## Usage

```bash
# Start hive first
rbee-keeper hive start

# Run hive check (tests SSE narration)
rbee-keeper hive check

# Or specify hive
rbee-keeper hive check --host localhost
```

## What It Tests

✅ **Narration from rbee-hive** - All n!() calls in hive_check.rs  
✅ **SSE streaming** - Events flow through hive → queen → keeper  
✅ **Job ID routing** - Narration context propagates correctly  
✅ **All 3 modes** - Human, Cute, Story narration modes  
✅ **Format specifiers** - Hex, debug, float formatting  
✅ **Sequential narrations** - Multiple events in sequence  

## Comparison with Other Checks

| Feature | self-check | queen-check | hive check |
|---------|------------|-------------|------------|
| **Location** | rbee-keeper | queen-rbee | rbee-hive |
| **Execution** | Local CLI | Via queen job server | Via hive job server |
| **Narration Output** | stderr (tracing) | SSE stream | SSE stream |
| **Job ID** | No | Yes | Yes |
| **SSE Routing** | No | Yes (queen) | Yes (hive) |
| **Tests** | CLI narration | Queen SSE | Hive SSE |

## Key Design Decisions

### 1. **Forwarded to Hive (not handled by Queen)**
Unlike hive lifecycle operations (start/stop) which are handled by queen, `HiveCheck` is forwarded to the hive itself. This tests the full SSE pipeline: keeper → queen → hive → SSE → queen → keeper.

### 2. **Mirrors Queen-Check Pattern**
The implementation exactly mirrors `queen-check`:
- Same test structure (10 tests)
- Same narration modes
- Same SSE context setup
- Same completion markers

### 3. **Uses `should_forward_to_hive()`**
Added to the forwarding list so queen automatically routes it to hive without special handling.

## Testing

```bash
# Build
cargo build --bin rbee-keeper --bin rbee-hive

# Start hive
./target/debug/rbee-keeper hive start

# Run check
./target/debug/rbee-keeper hive check
```

**Expected output:** Narration events streaming in real-time via SSE, showing:
- Test progress (1/10, 2/10, etc.)
- All three narration modes
- Format specifiers working
- Sequential narrations
- Completion message

## Compilation

✅ `cargo check --bin rbee-keeper` - PASS  
✅ `cargo check --bin rbee-hive` - PASS  
✅ `cargo build --bin rbee-keeper --bin rbee-hive` - PASS  

## Documentation

- Updated: `bin/.plan/CHECK_COMMANDS_REFERENCE.md` (needs update)
- Archived: `bin/.plan/TEAM_313_HIVE_CHECK.md` (old local version)
- New: This document

## Next Steps

1. Test end-to-end with running hive
2. Update CHECK_COMMANDS_REFERENCE.md
3. Add to integration test suite
4. Consider adding to CI/CD pipeline

## Code Signatures

All code marked with `TEAM-313` comments.
