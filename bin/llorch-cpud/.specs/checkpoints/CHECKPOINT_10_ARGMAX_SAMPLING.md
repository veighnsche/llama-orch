# CHECKPOINT 10: Argmax Sampling (Temperature=0)

**Phase:** 8.1 - Sampling  
**Component:** Deterministic token selection  
**Tolerance:** Exact  
**Critical Level:** 🔴 CRITICAL - Must be deterministic

---

## Purpose

Validate deterministic sampling (temperature=0). Any difference indicates an error.

## When to Check

- **Location:** After argmax on logits
- **Input:** Selected logits from Checkpoint 9
- **Timing:** During token generation with temp=0

## Validation Checklist

### ✓ Temperature Check
- [ ] Temperature < 1e-6 triggers argmax
- [ ] Not using temperature == 0 (floating point)
- [ ] Threshold correct (1e-6, not 0.0)

### ✓ Argmax Operation
- [ ] Applied on last dimension (dim=-1)
- [ ] Returns token ID (integer)
- [ ] Single value per batch item
- [ ] Output shape: `[batch]` or `[batch, 1]`

### ✓ Output Validation
- [ ] Token ID in valid range [0, 50256]
- [ ] Not negative
- [ ] Not >= vocab_size
- [ ] Deterministic (same input → same output)

### ✓ Flattening
- [ ] Output flattened to 1D
- [ ] Shape: `[batch]` (not `[batch, 1]`)
- [ ] Can be converted to list

### ✓ Cross-Reference
- [ ] Token ID matches tinygrad exactly
- [ ] Token ID matches Candle exactly
- [ ] No tolerance - must be identical

## Reference Locations

**Tinygrad:** `gpt2.py` lines 108-109, 157-161  
**Candle:** Application-level  
**Mistral.rs:** `sampler.rs` lines 352-361

## Common Failures

- ❌ Using temperature == 0 instead of < 1e-6
- ❌ Argmax on wrong dimension
- ❌ Not flattening output
- ❌ Returning float instead of int

## Success Criteria

- ✅ Temperature check correct (< 1e-6)
- ✅ Argmax on dim=-1
- ✅ Token ID matches reference exactly
- ✅ Deterministic output
- ✅ **No randomness with temp=0**

## Test Cases

### Test 1: First Token
```
Prompt: "Hello."
Expected token ID: (depends on model, but deterministic)
```

### Test 2: Second Token
```
Prompt: "Hello." + first_token
Expected token ID: (deterministic)
```

### Test 3: Reproducibility
```
Run 1: token_ids = [...]
Run 2: token_ids = [...]  # Must be identical
```

## Debug Commands

```python
# Tinygrad
print(f"Temperature: {temperature}")  # Should be 0
print(f"Logits shape: {logits.shape}")  # [1, 50257]
print(f"Argmax result: {logits.argmax(-1).numpy()}")  # Single token ID
print(f"Max logit value: {logits.max().numpy()}")
```

## Next Steps

If this checkpoint **PASSES**:
- ✅ Proceed to Checkpoint 11 (Softmax Probabilities)
- ✅ Or skip to Checkpoint 12 if only testing temp=0

If this checkpoint **FAILS**:
- ❌ Check temperature threshold
- ❌ Verify argmax dimension
- ❌ Ensure deterministic behavior
