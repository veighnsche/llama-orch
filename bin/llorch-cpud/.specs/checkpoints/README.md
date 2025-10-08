# GPT-2 Implementation Validation Checkpoints

This directory contains 13 detailed validation checkpoints for verifying GPT-2 implementation correctness.

## Quick Start

1. **Start here:** `MASTER_CHECKLIST.md` - Complete overview and tracking
2. **Foundation:** `CHECKPOINT_00_FOUNDATION.md` - HTTP server + project structure
3. **Model checkpoints:** `CHECKPOINT_01_*.md` through `CHECKPOINT_12_*.md`
4. **Usage guide:** `../VALIDATION_CHECKPOINT_USAGE.md`

## File Structure

```
checkpoints/
├── README.md                           # This file
├── MASTER_CHECKLIST.md                 # Complete validation checklist
├── CHECKPOINT_00_FOUNDATION.md         # HTTP server + project structure
├── CHECKPOINT_01_LAYER_NORM.md         # LayerNorm validation
├── CHECKPOINT_02_QKV_PROJECTION.md     # QKV split validation
├── CHECKPOINT_03_KV_CACHE.md           # Cache management validation
├── CHECKPOINT_04_ATTENTION_SCORES.md   # Attention computation validation
├── CHECKPOINT_05_ATTENTION_OUTPUT.md   # Attention output validation
├── CHECKPOINT_06_FFN_OUTPUT.md         # Feedforward network validation
├── CHECKPOINT_07_FIRST_BLOCK.md        # Complete block validation
├── CHECKPOINT_08_FULL_LOGITS.md        # All layers validation
├── CHECKPOINT_09_SELECTED_LOGITS.md    # Logit selection validation
├── CHECKPOINT_10_ARGMAX_SAMPLING.md    # Deterministic sampling validation
├── CHECKPOINT_11_SOFTMAX_PROBS.md      # Stochastic sampling validation
└── CHECKPOINT_12_END_TO_END.md         # Final validation (CRITICAL)
```

## Checkpoint Summary

| # | Name | Component | Tolerance | Critical |
|---|------|-----------|-----------|----------|
| 0 | Foundation Setup | HTTP + Structure | N/A | 🔴 CRITICAL |
| 1 | Layer Normalization | LayerNorm | 1e-5 | ⚠️ HIGH |
| 2 | QKV Projection | Attention Input | 1e-4 | 🔴 CRITICAL |
| 3 | KV Cache State | Cache Management | Exact | 🔴 CRITICAL |
| 4 | Attention Scores | SDPA | 1e-4 | ⚠️ HIGH |
| 5 | Attention Output | Attention Projection | 1e-4 | ⚠️ HIGH |
| 6 | FFN Output | Feedforward Network | 1e-4 | ⚠️ HIGH |
| 7 | First Block Output | Complete Block | 1e-4 | 🟢 VALIDATION |
| 8 | Full Logits | All 24 Layers | 1e-3 | 🟢 VALIDATION |
| 9 | Selected Logits | Last Token Selection | Exact | 🔴 CRITICAL |
| 10 | Argmax Sampling | Deterministic Sampling | Exact | 🔴 CRITICAL |
| 11 | Softmax Probabilities | Stochastic Sampling | 1e-6 | ⚠️ MEDIUM |
| 12 | End-to-End | **FINAL VALIDATION** | Exact | 🟢 FINAL |

## How to Use

### Step 1: Enable Validation in Reference Implementation

**Tinygrad:**
```bash
cd /reference/tinygrad/examples
python gpt2.py --prompt "Hello." --temperature 0 --count 10 --model_size gpt2-medium --validate
```

**Candle/Mistral.rs:**
```bash
export VALIDATE=1
# Run your application
```

### Step 2: Uncomment Checkpoint in Source

Edit the reference implementation and uncomment the specific checkpoint you want to validate.

### Step 3: Run and Compare

1. Run reference implementation → get checkpoint output
2. Run your implementation → get checkpoint output
3. Compare values using tolerance from checkpoint file
4. If match: ✅ proceed to next checkpoint
5. If mismatch: ❌ debug using checkpoint guide

### Step 4: Track Progress

Use `MASTER_CHECKLIST.md` to track which checkpoints have passed.

## Validation Strategy

### Recommended: Sequential
1. Start with Checkpoint 1
2. Fix until it passes
3. Move to Checkpoint 2
4. Repeat through Checkpoint 12

### Alternative: Critical Path
1. Checkpoint 1 (LayerNorm)
2. Checkpoint 2 (QKV)
3. Checkpoint 3 (Cache)
4. Checkpoint 7 (First Block)
5. Checkpoint 12 (End-to-End)

### Alternative: Binary Search
1. Test Checkpoint 7 (middle)
2. If passes → test Checkpoint 12
3. If fails → test Checkpoint 4
4. Narrow down to failing point

## Standard Test Case

All checkpoints use the same test input for consistency:

```
Prompt: "Hello."
Tokens: [15496, 13]
Model: GPT-2 Medium (350M parameters)
Temperature: 0 (deterministic)
Max tokens: 10
Expected output: "Hello. I'm a little late to the party, but"
```

## Each Checkpoint File Contains

1. **Purpose** - What this checkpoint validates
2. **When to Check** - Exact location in pipeline
3. **Validation Checklist** - Step-by-step verification
4. **Reference Locations** - Where to find in tinygrad/Candle/Mistral.rs
5. **Common Failures** - What usually goes wrong
6. **Success Criteria** - How to know it passed
7. **Debug Commands** - How to inspect values
8. **Next Steps** - What to do after pass/fail

## Tolerance Explanation

- **1e-5:** Very tight (LayerNorm, single operations)
- **1e-4:** Standard (most operations)
- **1e-3:** Looser (accumulated through 24 layers)
- **1e-6:** Tight (probability distributions)
- **Exact:** No tolerance (deterministic operations)

## Critical Checkpoints

These checkpoints are most likely to catch errors:

1. **Checkpoint 1** - LayerNorm affects everything
2. **Checkpoint 2** - QKV split is error-prone
3. **Checkpoint 3** - Cache errors compound
4. **Checkpoint 7** - Validates architecture
5. **Checkpoint 12** - Final proof

## Success Criteria

### Minimum
- ✅ Checkpoint 12 passes

### Recommended
- ✅ All checkpoints 1-12 pass
- ✅ All tolerances met

### Production
- ✅ All checkpoints pass
- ✅ Multiple test cases pass
- ✅ Both temp=0 and temp>0 work

## If Checkpoint 12 Fails

Work backwards through checkpoints:

```
Checkpoint 12 fails
  ↓
Check Checkpoint 10 (sampling)
  ↓
Check Checkpoint 9 (logit selection)
  ↓
Check Checkpoint 8 (all layers)
  ↓
Check Checkpoint 7 (first block)
  ↓
Check Checkpoint 1-6 (components)
```

## Reference Implementations

All checkpoints reference three implementations:

1. **Tinygrad** - Python, simple, 255 lines
   - File: `/reference/tinygrad/examples/gpt2.py`
   - Branch: `main`

2. **Candle** - Rust, moderate complexity, 368 lines
   - File: `/reference/candle/candle-transformers/src/models/bigcode.rs`
   - Branch: `orch_log`

3. **Mistral.rs** - Rust, production-grade, ~thousands of lines
   - Files: Multiple modules in `/reference/mistral.rs/mistralrs-core/src/`
   - Branch: `orch_log`

## Additional Resources

- **Main Spec:** `../01_GPT2_PIPELINE_COMPLETE_BEHAVIORS.md`
- **Usage Guide:** `../VALIDATION_CHECKPOINT_USAGE.md`
- **Verification:** `../SPEC_COMPLETENESS_VERIFICATION.md`
- **Framework Comparison:** See Appendix A in main spec

## Tips

1. **Enable one checkpoint at a time** - Easier to compare
2. **Use same test input** - "Hello." with temp=0
3. **Check shapes first** - Wrong shapes = fundamental error
4. **Then check values** - Compare first 5-10 elements
5. **Use tolerances** - Floating point has small differences
6. **Print to file** - For large tensors
7. **Be systematic** - Don't skip checkpoints
8. **Document findings** - Track what you fixed

## Common Patterns

### All Checkpoints Fail
- **Cause:** Model not loaded, wrong weights
- **Fix:** Verify model loading

### Checkpoints 1-3 Fail
- **Cause:** Basic operations broken
- **Fix:** Check tensor ops, shapes

### Checkpoints 4-6 Fail
- **Cause:** Attention or FFN broken
- **Fix:** Check attention, GELU

### Checkpoint 7 Fails
- **Cause:** Architecture wrong
- **Fix:** Check residuals, pre-norm

### Checkpoint 12 Fails (others pass)
- **Cause:** Integration issue
- **Fix:** Check cache, start_pos, loop

## Support

For detailed debugging:
1. Open the specific checkpoint file
2. Follow the validation checklist
3. Check common failures section
4. Use debug commands provided
5. Compare with reference implementation

## Final Note

**If Checkpoint 12 passes with exact match:**
- 🎉 **Your implementation is correct!**
- 🎉 **All components work together!**
- 🎉 **Ready for production use!**

Good luck with your validation! 🚀
