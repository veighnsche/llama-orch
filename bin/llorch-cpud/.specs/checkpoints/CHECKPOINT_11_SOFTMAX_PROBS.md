# CHECKPOINT 11: Softmax Probabilities (Temperature>0)

**Phase:** 8.2 - Sampling  
**Component:** Probability distribution  
**Tolerance:** 1e-6  
**Critical Level:** ⚠️ MEDIUM - Validates sampling correctness

---

## Purpose

Validate probability distribution for stochastic sampling. Token sampling will differ, but distribution must match.

## When to Check

- **Location:** After temperature scaling and softmax
- **Input:** Selected logits from Checkpoint 9
- **Timing:** When temperature > 0

## Validation Checklist

### ✓ Temperature Scaling
- [ ] Temperature >= 1e-6 triggers softmax path
- [ ] Logits divided by temperature
- [ ] Lower temp → sharper distribution
- [ ] Higher temp → flatter distribution

### ✓ Softmax Computation
- [ ] Applied on last dimension (dim=-1)
- [ ] Formula: `exp(x) / sum(exp(x))`
- [ ] Numerically stable (subtract max)
- [ ] Output shape same as input: `[1, 50257]`

### ✓ Probability Validation
- [ ] All probabilities >= 0
- [ ] All probabilities <= 1
- [ ] Sum of probabilities = 1.0 (within 1e-6)
- [ ] Max probability < 1.0 (not degenerate)
- [ ] Min probability >= 0.0

### ✓ Distribution Properties
- [ ] Top-k probabilities reasonable
- [ ] Not all equal (unless temp very high)
- [ ] Not single 1.0 (unless temp very low)
- [ ] Smooth distribution

### ✓ Cross-Reference
- [ ] Compare probability distribution with reference
- [ ] Top 5 probabilities match within 1e-6
- [ ] Top 5 indices match exactly
- [ ] Sum matches 1.0 within 1e-6

## Reference Locations

**Tinygrad:** `gpt2.py` lines 110-111, 163-169  
**Candle:** Application-level  
**Mistral.rs:** `sampler.rs` (comprehensive sampling)

## Common Failures

- ❌ Probabilities don't sum to 1.0
- ❌ Softmax on wrong dimension
- ❌ Numerical instability (overflow)
- ❌ Not dividing by temperature

## Success Criteria

- ✅ Probabilities sum to 1.0 (±1e-6)
- ✅ All values in [0, 1]
- ✅ Distribution matches reference (±1e-6)
- ✅ Top-k indices match reference

## Temperature Effects

### Temperature = 0.5 (Sharp)
- Most probability mass on top token
- Very peaked distribution
- Nearly deterministic

### Temperature = 1.0 (Normal)
- Balanced distribution
- Original softmax output
- Standard sampling

### Temperature = 2.0 (Flat)
- More uniform distribution
- More randomness
- Exploratory sampling

## Debug Commands

```python
# Tinygrad
probs = (logits / temperature).softmax()
print(f"Probability sum: {probs.sum().numpy()}")  # Should be 1.0
print(f"Max probability: {probs.max().numpy()}")
print(f"Top 5 probs: {probs.topk(5)[0].numpy()}")
print(f"Top 5 indices: {probs.topk(5)[1].numpy()}")
```

## Note on Sampling

- **Probability distribution must match**
- **Actual sampled token will differ** (stochastic)
- Only validate distribution, not sampled token
- For deterministic validation, use temperature=0

## Next Steps

If this checkpoint **PASSES**:
- ✅ Proceed to Checkpoint 12 (End-to-End)

If this checkpoint **FAILS**:
- ❌ Check softmax implementation
- ❌ Verify temperature scaling
- ❌ Ensure numerical stability
