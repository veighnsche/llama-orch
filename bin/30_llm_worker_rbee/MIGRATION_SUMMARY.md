# llm-worker-rbee Narration Migration Summary

## Overview

The `llm-worker-rbee` crate is currently disabled in the workspace due to using the old narration API. This document summarizes what needs to be migrated.

## Current Status

❌ **DISABLED** - Commented out in `/home/vince/Projects/llama-orch/Cargo.toml`:
```toml
# "bin/30_llm_worker_rbee",     # LLM inference worker daemon - DISABLED: ancient narration API
```

## Migration Scope

### Files Affected: 8 files
### Call Sites: 35 total

| File | Call Sites | Priority |
|------|-----------|----------|
| src/main.rs | 4 | HIGH (startup) |
| src/device.rs | 3 | HIGH (initialization) |
| src/backend/inference.rs | 8 | HIGH (core logic) |
| src/backend/gguf_tokenizer.rs | 3 | MEDIUM |
| src/backend/models/quantized_llama.rs | 12 | MEDIUM |
| src/backend/models/quantized_phi.rs | 2 | LOW |
| src/backend/models/quantized_qwen.rs | 2 | LOW |
| src/narration.rs | 1 | SPECIAL (dual output) |

## Migration Pattern

### Before (Old API):
```rust
narrate(NarrationFields {
    actor: "model-loader",
    action: "gguf_load_start",
    target: path.display().to_string(),
    human: "Loading GGUF model".to_string(),
    ..Default::default()
});
```

### After (New API):
```rust
n!("gguf_load_start")
    .target(path.display().to_string())
    .human("Loading GGUF model")
    .emit();
```

**Key Difference:** The `n!()` macro auto-detects the actor from the crate name.

## Estimated Effort

- **Time:** 1-2 hours
- **Complexity:** LOW (mechanical replacement)
- **Risk:** LOW (compile-time verification)

## Next Steps

1. Review detailed checklist: `NARRATION_MIGRATION_CHECKLIST.md`
2. Migrate files in priority order (HIGH → MEDIUM → LOW)
3. Handle special case in `src/narration.rs`
4. Re-enable in `Cargo.toml`
5. Build and verify

## Benefits After Migration

✅ Workspace builds successfully  
✅ Worker binary available for testing  
✅ Modern narration API (auto-detected actor)  
✅ Consistent with rest of codebase  

## Files to Review

- **Checklist:** `bin/30_llm_worker_rbee/NARRATION_MIGRATION_CHECKLIST.md`
- **Workspace Config:** `Cargo.toml` (line 14)
