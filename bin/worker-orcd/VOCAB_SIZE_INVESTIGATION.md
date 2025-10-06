# Vocab Size Investigation - 2025-10-06

## Summary

The bug report claimed the issue was a vocab size mismatch (lm_head=[896,151643] vs vocab_size=151936), but investigation reveals:

### Actual Facts

1. **output.weight tensor dimensions**: [896, 151936] (NOT 151643 as claimed)
2. **llama.cpp n_vocab**: 151936 (confirmed from logs)
3. **Garbage values found**: Yes, at positions 137131 and 44394 with values ~14-15
4. **Real logits range**: -4 to +4
5. **lm_head weights at problematic positions**: Normal values (±0.01-0.03)

### Observed Behavior

**Prefill phase (argmax #0-4)**:
- Finds token_id=137131 with value ~14.3-14.7
- Logits[0:10] are changing normally

**Generation phase (argmax #5-14)**:
- Finds token_id=44394 with value ~14.4-15.1  
- Model generates token 44394 ("coholic") 100 times in a row
- Logits[0:10] continue to change, but argmax always finds the same garbage value

### Root Cause

The issue is NOT a vocab size mismatch. The lm_head tensor IS [896, 151936] as expected.

The real issue is that **certain positions in the logits buffer contain garbage values** that are much higher than the legitimate logits. These garbage values win the argmax, causing the model to always select the same token.

### Possible Causes

1. **GEMM output stride mismatch**: The cuBLAS GEMM might not be writing to all positions correctly
2. **Uninitialized memory**: Despite initialization to -INFINITY, some positions are being overwritten with garbage
3. **Memory corruption**: Something is writing garbage values to specific logits positions
4. **lm_head tensor loading issue**: The tensor might not be loaded correctly despite weights looking normal

### Next Steps

1. Add debug output to check logits values at positions 137131 and 44394 after GEMM
2. Verify the GEMM is actually writing to all vocab_size positions
3. Check if the issue is consistent (same positions always have garbage) or random
4. Compare with llama.cpp's lm_head projection implementation

### Test Command

```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd
cargo test --release --test haiku_generation_anti_cheat test_haiku_generation_stub_pipeline_only \
  --features cuda -- --ignored --nocapture --test-threads=1
```

### Debug Output Added

- `cuda/kernels/sampling_wrapper.cu`: Extended argmax debug to 15 calls
- `cuda/src/transformer/qwen_transformer.cpp`: Added lm_head weight sampling at problematic positions

