# GT-057: Test Cleanup and Verification

**Team**: GPT-Gamma 🤖  
**Sprint**: Sprint 9 - Real Inference  
**Size**: XS (1-2 hours)  
**Priority**: P0 (M0 blocker)  
**Spec Ref**: FINE-001-20251005 remediation

---

## Story Description

Remove all stub warnings, rename test back to original name, and submit remediation proof to Testing Team.

---

## Acceptance Criteria

- [ ] Remove all stub warnings from test output
- [ ] Rename test back to `test_haiku_generation_anti_cheat`
- [ ] Remove fine references from test
- [ ] Remove fine references from inference code
- [ ] Run test multiple times to verify different haikus
- [ ] Verify minute word appears in each haiku
- [ ] Submit remediation proof to Testing Team
- [ ] Get Testing Team sign-off

---

## Tasks

### 1. Clean Up Test File

**File**: `tests/haiku_generation_anti_cheat.rs`

**Remove**:
```rust
/// ⚠️  STUB TEST: This test uses hardcoded haiku generation
/// **FINED by Testing Team**: FINE-001-20251005
/// **See**: test-harness/FINES.md
/// ...
#[ignore] // STUB ONLY - not real inference
async fn test_haiku_generation_STUB_PIPELINE_ONLY() {
    eprintln!("⚠️  WARNING: STUB INFERENCE - NOT REAL GPU INFERENCE");
    // ...
}
```

**Replace with**:
```rust
/// M0 Haiku Anti-Cheat Test
/// 
/// Validates real GPU inference by requiring the model to include
/// the current minute (in words) within a generated haiku.
/// 
/// This prevents pre-baked outputs and validates genuine token generation.
#[tokio::test]
#[cfg(feature = "cuda")]
#[ignore] // Run with REQUIRE_REAL_LLAMA=1
async fn test_haiku_generation_anti_cheat() {
    std::env::set_var("REQUIRE_REAL_LLAMA", "1");
    // ... (rest of test unchanged)
}
```

### 2. Clean Up Inference Code

**File**: `cuda/src/inference_impl.cpp`

**Remove**:
```cpp
// ⚠️  WARNING: STUB INFERENCE - NOT REAL GPU INFERENCE
// ⚠️  This is a hardcoded template, not real model inference
// ⚠️  FINED by Testing Team: FINE-001-20251005
// ⚠️  See: test-harness/FINES.md
//
// TODO: Implement real inference (22-31 hours):
// - Phase 1: GGUF weight loading to GPU (9-13h)
// - Phase 2: Tokenizer integration (5-7h)
// - Phase 3: Transformer forward pass (8-11h)

fprintf(stderr, "⚠️  WARNING: STUB INFERENCE - NOT REAL GPU INFERENCE\n");
fprintf(stderr, "⚠️  This test uses a hardcoded template, not real model inference\n");
fprintf(stderr, "⚠️  TODO: Implement real GGUF weight loading and transformer forward pass\n");
fprintf(stderr, "⚠️  FINED: See test-harness/FINES.md #001\n");
```

**Replace with**:
```cpp
// Real GPU inference implementation
// Uses GGUF weights, BPE tokenizer, and transformer forward pass
```

### 3. Verify Multiple Runs

Run test 5 times and verify:
- ✅ Different haiku each time
- ✅ Minute word present in each
- ✅ No stub warnings

```bash
for i in {1..5}; do
  echo "=== Run $i ==="
  REQUIRE_REAL_LLAMA=1 cargo test --test haiku_generation_anti_cheat \
    test_haiku_generation_anti_cheat --features cuda --release \
    -- --ignored --nocapture --test-threads=1 | grep -A 5 "Haiku:"
  sleep 2
done
```

### 4. Submit Remediation Proof

**Create**: `FINE_001_REMEDIATION_PROOF.md`

```markdown
# Fine #001 Remediation Proof

**Fine**: FINE-001-20251005  
**Issued**: 2025-10-05T16:22:45Z  
**Remediated**: 2025-10-15

## Immediate Actions (24h) - ✅ COMPLETE

1. ✅ Warnings added to test output
2. ✅ Test renamed to indicate stub
3. ✅ Documentation updated
4. ✅ Tracking issue created

## Long-term Actions (10 days) - ✅ COMPLETE

5. ✅ Real inference implemented:
   - ✅ GT-051: GGUF config parsing
   - ✅ GT-052: Weight loading to GPU
   - ✅ GT-053: BPE tokenizer
   - ✅ GT-054: Transformer execution
   - ✅ GT-055: LM head
   - ✅ GT-056: Wire inference
   - ✅ GT-057: Test cleanup

6. ✅ Test renamed back to `test_haiku_generation_anti_cheat`

7. ✅ All stub warnings removed

## Verification

### Test Output (5 runs)

Run 1:
```
🎨 M0 Haiku Anti-Cheat Test PASSED
Minute: 17 ("seventeen")
Haiku:
[actual haiku 1]
```

Run 2:
```
🎨 M0 Haiku Anti-Cheat Test PASSED
Minute: 17 ("seventeen")
Haiku:
[actual haiku 2 - DIFFERENT]
```

[... runs 3-5 ...]

### VRAM Usage

Before: 0 MB
After: 400 MB (Qwen model loaded)

### GPU Utilization

`nvidia-smi` shows GPU activity during inference.

## Conclusion

✅ Real GPU inference implemented  
✅ Test passes consistently  
✅ Different haiku each run  
✅ Minute word present  
✅ No stub warnings  

**Fine remediated successfully.**

---

Submitted by: GPT-Gamma 🤖  
Date: 2025-10-15
```

### 5. Get Testing Team Sign-Off

Submit proof to Testing Team for verification and sign-off.

---

## Dependencies

**Upstream**: GT-056 (real inference must work)  
**Downstream**: None (final story)

---

## Definition of Done

- [ ] All warnings removed
- [ ] Test renamed
- [ ] Fine references removed
- [ ] Multiple runs verified
- [ ] Remediation proof created
- [ ] Testing Team sign-off received
- [ ] Fine marked RESOLVED

---

## Estimated Time

**Realistic**: 1-2 hours

---

**Created by**: Project Management Team 📋  
**Assigned to**: GPT-Gamma 🤖  
**Status**: TODO
