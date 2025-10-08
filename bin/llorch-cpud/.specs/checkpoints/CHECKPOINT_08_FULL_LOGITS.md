# CHECKPOINT 8: Full Logits

**Phase:** 7.2 - LM Head  
**Component:** All 24 layers + final norm + lm_head  
**Tolerance:** 1e-3 (looser due to accumulation)  
**Critical Level:** 🟢 VALIDATION - All layers processed

---

## Purpose

Validate entire forward pass through all 24 transformer blocks. This confirms all layers work correctly.

## When to Check

- **Location:** After LM head, before selecting last token
- **Input:** Final layer norm output from all 24 blocks
- **Timing:** Before logit selection

## Validation Checklist

### ✓ All Blocks Processed
- [ ] 24 transformer blocks executed
- [ ] Each block output fed to next block
- [ ] No blocks skipped
- [ ] Final hidden state shape: `[1, 2, 1024]`

### ✓ Final Layer Norm
- [ ] ln_f applied after all blocks
- [ ] Same epsilon = 1e-5
- [ ] Output shape: `[1, 2, 1024]`

### ✓ LM Head Projection
- [ ] Weight shape: `[1024, 50257]` (vocab_size)
- [ ] NO bias (bias=False)
- [ ] Weight tied with wte.weight
- [ ] Output shape: `[1, 2, 50257]`

### ✓ Value Validation
- [ ] Logits shape: `[1, 2, 50257]`
- [ ] Values in range (typically [-20, 20])
- [ ] Not all same value
- [ ] Max logit index reasonable (< 50257)
- [ ] No NaN/Inf

### ✓ Weight Tying
- [ ] lm_head.weight === wte.weight (same object)
- [ ] Not copied, actually shared
- [ ] Changes to one affect the other

### ✓ Cross-Reference
- [ ] Compare logits[0, 0, :5] with reference
- [ ] Difference within 1e-3 (looser tolerance)

## Reference Locations

**Tinygrad:** `gpt2.py` lines 98-100, 140-144  
**Candle:** `bigcode.rs` lines 407-415  
**Mistral.rs:** Model-specific forward implementations

## Common Failures

- ❌ Not all blocks processed
- ❌ lm_head has bias (should be False)
- ❌ Weights not tied
- ❌ Wrong vocab size

## Success Criteria

- ✅ Shape: `[1, 2, 50257]`
- ✅ All 24 blocks processed
- ✅ Weight tying correct
- ✅ Matches reference within 1e-3
- ✅ **If this passes, all layers work!**

## Note on Tolerance

- Tolerance is 1e-3 (looser than previous checkpoints)
- Reason: Accumulation through 24 layers
- Small errors compound but should stay within 1e-3
- If difference > 1e-3, check earlier checkpoints
