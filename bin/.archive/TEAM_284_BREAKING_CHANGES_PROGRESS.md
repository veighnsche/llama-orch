# TEAM-284: Breaking Changes Progress

**Date:** Oct 24, 2025  
**Status:** 🚧 **IN PROGRESS**

## Summary

Making breaking changes to use typed requests throughout the codebase. NO backward compatibility shims!

## Completed

### ✅ 1. operations-contract
- Updated Operation enum to use typed requests
- All 9 operations converted
- Helper methods updated
- **Status:** ✅ Compiles

### ✅ 2. rbee-hive
- Updated all match arms in `job_router.rs`
- Changed from inline fields to typed requests
- Updated import: `rbee_operations` → `operations_contract`
- **Status:** ✅ Compiles

**Files Modified:**
- `bin/20_rbee_hive/src/job_router.rs` (9 match arms updated)

## Remaining

### ⏳ 3. queen-rbee (2 files)

**hive_forwarder.rs:**
- Currently forwards operations to hive
- Needs to handle typed requests
- Estimated: ~10 lines to change

**job_router.rs:**
- Routes hive operations (HiveStart, HiveStop, etc.)
- These operations still use inline fields (not changed yet)
- Estimated: Minimal changes (hive ops unchanged)

### ⏳ 4. rbee-keeper

**main.rs:**
- CLI constructs Operation variants
- Needs to create typed requests
- Estimated: ~20 lines to change

## Breaking Changes Made

### Before
```rust
// Inline fields
Operation::WorkerSpawn {
    hive_id: "localhost".to_string(),
    model: "llama-2-7b".to_string(),
    worker: "cpu".to_string(),
    device: 0,
}
```

### After
```rust
// Typed request
Operation::WorkerSpawn(WorkerSpawnRequest {
    hive_id: "localhost".to_string(),
    model: "llama-2-7b".to_string(),
    worker: "cpu".to_string(),
    device: 0,
})
```

## Match Arm Changes

### Before
```rust
match operation {
    Operation::WorkerSpawn { hive_id, model, worker, device } => {
        // use fields directly
    }
}
```

### After
```rust
match operation {
    Operation::WorkerSpawn(request) => {
        // clone fields from request
        let hive_id = request.hive_id.clone();
        let model = request.model.clone();
        // ...
    }
}
```

## Import Changes

### Before
```rust
use rbee_operations::Operation;
```

### After
```rust
use operations_contract::Operation;
```

## No Shims!

We are NOT creating backward compatibility shims. All code must be updated to use the new typed requests.

## Verification

```bash
✅ cargo check -p operations-contract
✅ cargo check -p rbee-hive
⏳ cargo check -p queen-rbee (needs updates)
⏳ cargo check -p rbee-keeper (needs updates)
```

## Next Steps

1. Update queen-rbee/hive_forwarder.rs
2. Update rbee-keeper/main.rs CLI
3. Verify all packages compile
4. Test end-to-end

## Estimated Completion

- **Completed:** 2/4 packages (50%)
- **Remaining:** ~30 lines to change
- **Time:** ~10 minutes

## Impact

### Lines Changed
- operations-contract: ~70 lines
- rbee-hive: ~20 lines
- queen-rbee: ~10 lines (estimated)
- rbee-keeper: ~20 lines (estimated)
- **Total:** ~120 lines

### Benefits
- ✅ Type safety
- ✅ No duplication
- ✅ Consistent pattern
- ✅ Better error messages
- ✅ Easier to test

## Conclusion

Making good progress! Rbee-hive is complete and compiling. Just need to update queen-rbee and rbee-keeper to finish the breaking changes.

**No backward compatibility - clean break!**
