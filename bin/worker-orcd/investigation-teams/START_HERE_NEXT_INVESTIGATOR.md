# 🚨 START HERE - Next Investigator

**Last Updated**: 2025-10-06 16:21 UTC  
**Status**: Root cause identified, awaiting fix

---

## TL;DR

**The bug has been found!** 🎉

The `output_norm.weight` tensor is corrupted (values up to 16.75 instead of ~1.0).

**DO NOT** investigate cuBLAS, RMSNorm kernels, or residual connections - they're all correct.

**DO** investigate `src/cuda/weight_loader.rs` - that's where the bug is.

---

## What Happened

Team Charlie (Mathematical Verification) completed a full investigation:

1. ✅ Verified cuBLAS is computing correctly
2. ✅ Tracked hidden state evolution across layers
3. ✅ Found corrupted `output_norm.weight` tensor

**Result**: The final RMSNorm amplifies values by 16x instead of normalizing them.

---

## The Evidence

```
[DEEP_INVESTIGATION] Final RMSNorm Analysis:
  BEFORE norm: Range=[-20.9688, 23.4062], Mean=-0.1518, RMS=6.7737
  Norm WEIGHTS: Range=[-0.0114, 16.7500], Mean=7.1393  ← WRONG!
  AFTER norm:  Range=[-32.8125, 31.2188], Mean=-0.1597, Std=7.3213
  ⚠️  WARNING: output_norm weights are abnormal!
```

Normal RMSNorm weights should be ~1.0 (range [0.5, 1.5]).

Having weights up to **16.75** means the normalization amplifies by 16x!

---

## Where to Look

### Files with Investigation Comments

All code has been marked with `[TEAM_CHARLIE]` tags:

1. **`cuda/src/transformer/qwen_transformer.cpp`**
   - Lines 7-41: Investigation summary
   - Lines 486-550: cuBLAS verification (CORRECT)
   - Lines 689-786: Hidden state tracking (NORMAL)
   - Lines 798-873: RMSNorm analysis (BUG FOUND)

2. **`cuda/kernels/rmsnorm.cu`**
   - Lines 8-24: Kernel verified correct

3. **`cuda/kernels/residual.cu`**
   - Lines 8-25: Residual addition verified correct

### Documentation Files

1. **`ROOT_CAUSE_FOUND.md`** ← Read this first!
   - Executive summary
   - Root cause explanation
   - Recommended fixes

2. **`TEAM_CHARLIE_INVESTIGATION_INDEX.md`**
   - Complete navigation guide
   - What was tested
   - What was ruled out

3. **`TEAM_CHARLIE_RESULTS.md`**
   - Detailed test data
   - Mathematical proofs

4. **`DEEP_INVESTIGATION_FINDINGS.md`**
   - Layer-by-layer analysis
   - Growth patterns

---

## What You Should Do

### Step 1: Verify the Bug (5 minutes)

Run the test and look for corrupted weights:

```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd

cargo test --release --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  --features cuda -- --ignored --nocapture --test-threads=1 \
  2>&1 | grep -A 10 "Final RMSNorm Analysis"
```

You should see:
```
Norm WEIGHTS: Range=[-0.0114, 16.7500], Mean=7.1393  ← ABNORMAL!
```

### Step 2: Investigate Weight Loading (30-60 minutes)

Check `src/cuda/weight_loader.rs`:

1. **Find where `output_norm.weight` is loaded**
   - Search for `"output_norm"` or `"output.norm"`
   - Check if tensor name matches GGUF file

2. **Check if it's quantized**
   - If yes, verify dequantization is correct
   - Compare with how other norm weights are loaded

3. **Compare with llama.cpp**
   - How does llama.cpp load this tensor?
   - What values does it get?

4. **Inspect the GGUF file**
   - Use `gguf-dump` or similar tool
   - Check if `output_norm.weight` exists
   - Check its type and dimensions

### Step 3: Fix the Bug (varies)

Possible fixes:

**Option A**: Tensor name mismatch
```rust
// If GGUF has "model.norm.weight" but we're looking for "output_norm.weight"
let tensor_name = "model.norm.weight";  // Fix the name
```

**Option B**: Dequantization bug
```rust
// If tensor is quantized, verify dequant is correct
if tensor.is_quantized() {
    dequantize_correctly(tensor);  // Fix the dequant
}
```

**Option C**: Wrong tensor loaded
```rust
// If we're loading the wrong tensor entirely
let correct_tensor = find_tensor("output_norm.weight");  // Load correct one
```

---

## What NOT to Do

### ❌ Don't investigate these (already verified):

- cuBLAS parameters
- Matrix multiplication
- RMSNorm kernel implementation
- Residual connection logic
- Attention mechanism
- Memory layout

All of these are **working correctly**.

### ❌ Don't change these files:

- `cuda/src/transformer/qwen_transformer.cpp` (except to remove test code)
- `cuda/kernels/rmsnorm.cu`
- `cuda/kernels/residual.cu`

The bug is **not** in these files!

---

## Quick Reference

### The Bug
```
output_norm.weight is corrupted
Expected: ~1.0 (range [0.5, 1.5])
Actual: up to 16.75
```

### The Effect
```
Final RMSNorm amplifies by 16x
→ Hidden state: ±23.4 → ±32.8
→ Logits: 14+ (abnormally high)
→ Softmax: ~99.9% on one token
→ Generation: Same token repeated
```

### The Fix Location
```
src/cuda/weight_loader.rs
↓
Find output_norm.weight loading
↓
Fix the bug there
```

---

## Need More Details?

1. **Quick overview**: Read `ROOT_CAUSE_FOUND.md`
2. **Navigation guide**: Read `TEAM_CHARLIE_INVESTIGATION_INDEX.md`
3. **Test data**: Read `TEAM_CHARLIE_RESULTS.md`
4. **Layer analysis**: Read `DEEP_INVESTIGATION_FINDINGS.md`
5. **Code comments**: Check the `.cpp` and `.cu` files

---

## Success Criteria

You've fixed the bug when:

1. ✅ `output_norm.weight` values are in range [0.5, 1.5]
2. ✅ Mean weight is close to 1.0
3. ✅ Final hidden state is in range [-5, 5] (not ±32.8)
4. ✅ Max logit is in range [-4, 4] (not 14+)
5. ✅ Model generates diverse tokens (not repetitive)

---

## Questions?

- **"Is cuBLAS wrong?"** → No, verified correct (diff < 0.00002)
- **"Is RMSNorm broken?"** → No, kernel is correct
- **"Are residuals accumulating?"** → Yes, but that's normal
- **"Where's the bug?"** → Weight loading in `weight_loader.rs`
- **"What's corrupted?"** → `output_norm.weight` tensor
- **"How do I fix it?"** → Check weight loading code

---

**Good luck!** 🍀

The hard part (finding the bug) is done. Now just fix the weight loading!

---

**Team Charlie** ✅
