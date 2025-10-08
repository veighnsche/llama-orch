# CHECKPOINT 9: Selected Logits (Last Token)

**Phase:** 7.3 - Logit Selection  
**Component:** Last token selection  
**Tolerance:** Exact  
**Critical Level:** 🔴 CRITICAL - Determines next token

---

## Purpose

Validate correct selection of last token logits. Wrong selection breaks generation.

## When to Check

- **Location:** After selecting last token logits
- **Input:** Full logits from Checkpoint 8
- **Timing:** Before sampling

## Validation Checklist

### ✓ Selection Logic
- [ ] If seq_len > 0: select `logits[:, -1, :]`
- [ ] If seq_len == 0: return ones (edge case)
- [ ] Indexing uses -1 (last position)
- [ ] No off-by-one errors

### ✓ Shape Validation
- [ ] Input: `[1, 2, 50257]` (for prompt)
- [ ] Output: `[1, 50257]` (last token only)
- [ ] Batch dimension preserved
- [ ] Sequence dimension removed

### ✓ Value Validation
- [ ] Selected logits are from position 1 (index -1)
- [ ] Not from position 0
- [ ] Values in range (typically [-20, 20])
- [ ] Argmax gives reasonable token ID

### ✓ Edge Cases
- [ ] Empty sequence handled (seq_len=0)
- [ ] Single token handled (seq_len=1)
- [ ] Multi-token handled (seq_len>1)

### ✓ Cross-Reference
- [ ] Compare selected_logits[0, :5] with reference
- [ ] Must match exactly (no tolerance)
- [ ] Argmax index matches reference

## Reference Locations

**Tinygrad:** `gpt2.py` lines 102-106, 151-155  
**Candle:** `bigcode.rs` lines 412-427  
**Mistral.rs:** Handled in sampling logic

## Common Failures

- ❌ Selecting first token instead of last
- ❌ Not handling -1 index correctly
- ❌ Shape mismatch (not squeezing)
- ❌ Edge case not handled

## Success Criteria

- ✅ Shape: `[1, 50257]`
- ✅ Selected from last position
- ✅ Matches reference exactly
- ✅ Argmax gives same token ID as reference

## Debug Commands

```python
# Tinygrad
print(f"Full logits shape: {logits.shape}")  # [1, 2, 50257]
print(f"Selected shape: {logits[:, -1, :].shape}")  # [1, 50257]
print(f"Argmax: {logits[:, -1, :].argmax(-1).numpy()}")
```

## Next Steps

If this checkpoint **PASSES**:
- ✅ Proceed to Checkpoint 10 (Argmax Sampling)

If this checkpoint **FAILS**:
- ❌ Fix indexing logic
- ❌ Check -1 vs 0 vs 1
- ❌ Verify shape handling
