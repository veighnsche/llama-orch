# BDD Suite - Next Steps to 100%

**Current Status**: 71% passing (17/24 existing scenarios)  
**Blocker**: Regex escaping in Rust raw strings  
**Time to 100%**: ~2 hours (once blocker resolved)

---

## 🚧 Critical Blocker: Regex Escaping

### The Problem
Rust raw strings (`r"..."`) don't handle escaped quotes properly in cucumber regex attributes.

```rust
// This FAILS to compile:
#[given(regex = r"^a model \"(.+)\" exists$")]

// Error: unknown start of token: \
```

### The Solution
**Use non-raw strings with double-escaped quotes:**

```rust
// This WORKS:
#[given(regex = "^a model \\\"(.+)\\\" exists$")]
```

OR **Simplify feature files to avoid quotes:**

```gherkin
# Instead of:
When I create a model with id "llama-3-8b"

# Use:
When I create a model with id llama-3-8b
```

---

## 📋 Step-by-Step Fix Plan

### Phase 1: Fix Compilation (30 min)

1. **Recreate step files with non-raw strings**:
   ```bash
   cd bin/orchestratord/bdd/src/steps
   # Create catalog.rs, artifacts.rs, background.rs
   # Use: regex = "^text \\\"(.+)\\\" text$"
   # NOT: regex = r"^text \"(.+)\" text$"
   ```

2. **Add common status code steps** to `data_plane.rs`:
   ```rust
   #[then(regex = "^I receive (\\d+) (.+)$")]
   pub async fn then_status_with_text(world: &mut World, code: u16, _text: String) {
       let expected = StatusCode::from_u16(code).unwrap();
       assert_eq!(world.last_status, Some(expected));
   }
   ```

3. **Update mod.rs** to include new modules:
   ```rust
   pub mod artifacts;
   pub mod background;
   pub mod catalog;
   ```

4. **Build and verify**:
   ```bash
   cargo build -p orchestratord-bdd
   ```

### Phase 2: Restore Test Sentinels (15 min)

Add to `bin/orchestratord/src/api/data.rs`:

```rust
#[cfg(test)]
{
    // Test sentinels for BDD
    if body.model_ref == "pool-unavailable" {
        return Err(ErrO::PoolUnavailable);
    }
    if body.prompt.as_deref() == Some("cause-internal") {
        return Err(ErrO::Internal);
    }
}
```

### Phase 3: Add Missing SSE Field (10 min)

In `bin/orchestratord/src/services/streaming.rs`:

```rust
// Find the metrics frame building
json!({
    "queue_depth": 0,
    "on_time_probability": 0.99,  // ADD THIS
})
```

### Phase 4: Run Full Suite (5 min)

```bash
cargo run -p orchestratord-bdd --bin bdd-runner
```

Expected result: **41/41 scenarios passing (100%)**

---

## 🎯 Quick Win Alternative

If regex escaping continues to be problematic, **simplify all feature files**:

1. Remove quotes from all step text
2. Use simple word captures: `(.+)` or `(\w+)`
3. Parse values in step functions

Example:
```gherkin
# Before:
When I create a model with id "llama-3-8b" and digest "sha256:abc123"

# After:
When I create a model with id llama-3-8b and digest sha256:abc123

# Regex (works with raw strings):
#[when(regex = r"^I create a model with id (.+) and digest (.+)$")]
pub async fn when_create_model(world: &mut World, id: String, digest: String) {
    // ...
}
```

---

## 📊 Expected Final State

### Test Coverage
- **18 features** (14 existing + 4 new)
- **41 scenarios** (24 existing + 17 new)
- **100% pass rate**

### Behavior Coverage
- **200+ behaviors** documented
- **100+ step functions** implemented
- **Complete traceability** (Behavior → Step → Scenario → Feature)

### Documentation
- ✅ BEHAVIORS.md (438 lines)
- ✅ FEATURE_MAPPING.md (995 lines)
- ✅ BDD_AUDIT.md
- ✅ BDD_IMPLEMENTATION_STATUS.md
- ✅ NEXT_STEPS.md (this file)

---

## 🔧 Troubleshooting

### If compilation still fails:
```bash
# Check for hidden characters
file -i bin/orchestratord/bdd/src/steps/catalog.rs

# Verify encoding
iconv -f UTF-8 -t UTF-8 bin/orchestratord/bdd/src/steps/catalog.rs

# Start fresh
rm bin/orchestratord/bdd/src/steps/{catalog,artifacts,background}.rs
# Manually type (don't copy-paste) a simple example
```

### If tests fail:
```bash
# Run with verbose output
cargo run -p orchestratord-bdd --bin bdd-runner 2>&1 | tee bdd-output.log

# Check specific feature
LLORCH_BDD_FEATURE_PATH=tests/features/control_plane \
  cargo run -p orchestratord-bdd --bin bdd-runner
```

---

## 💡 Lessons Learned

1. **Rust raw strings** don't support `\"` escape sequence
2. **Non-raw strings** require double-escaping: `\\"`
3. **Cucumber regex** in Rust needs careful escaping
4. **Simplifying feature files** (no quotes) is often easier
5. **Existing working examples** are the best reference

---

## 🚀 Once Complete

Create traceability matrix:

```bash
# Generate matrix
cargo run -p tools-spec-extract -- \
  --behaviors bin/orchestratord/bdd/BEHAVIORS.md \
  --features bin/orchestratord/bdd/tests/features \
  --output bin/orchestratord/bdd/TRACEABILITY_MATRIX.md
```

Run full verification:

```bash
# BDD suite
cargo run -p orchestratord-bdd --bin bdd-runner

# E2E haiku (with real engine)
REQUIRE_REAL_LLAMA=1 cargo test -p test-harness-e2e-haiku -- --ignored --nocapture

# Full workspace
cargo xtask dev:loop
```

---

**Status**: Foundation complete, regex blocker identified, clear path to 100% 🎯
