# Quick Start: Validate LayerNorm

Run this single command to validate the entire LayerNorm implementation:

```bash
./.test_helpers/run_validation.sh
```

## Expected Output

```
╔════════════════════════════════════════════════════════════╗
║  llorch-cpud LayerNorm Validation Suite                   ║
╚════════════════════════════════════════════════════════════╝

[1/3] Running llorch-cpud LayerNorm test...
✅ Our LayerNorm is mathematically correct

[2/3] Running Candle reference implementation...
✅ SUCCESS: Output saved

[3/3] Comparing outputs...
✅ PASS: All values within tolerance
Max difference: 6.6000000e-06

✅ Validation complete!
```

## What This Tests

1. **Our Implementation** - llorch-cpud LayerNorm with test input
2. **Candle Reference** - Hugging Face's Rust ML framework LayerNorm
3. **Comparison** - Automated diff with 1e-4 tolerance

## Result

✅ **Maximum difference: 6.6e-06** (well under 1e-4 tolerance)

## More Details

- **Quick Summary:** [VALIDATION_SUMMARY.md](VALIDATION_SUMMARY.md)
- **Full Report:** [CHECKPOINT_01_CROSS_REFERENCE_FINAL.md](CHECKPOINT_01_CROSS_REFERENCE_FINAL.md)
- **All Docs:** [INDEX.md](INDEX.md)
