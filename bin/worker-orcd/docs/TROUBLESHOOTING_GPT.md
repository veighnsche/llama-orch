# GPT Troubleshooting Guide

**Team**: GPT-Gamma  
**Version**: M0  
**Last Updated**: 2025-10-05

---

## Common Issues

### Model Loading

#### Issue: Model file not found
**Symptoms**: Error loading model, file not found  
**Cause**: Incorrect model path  
**Solution**:
```bash
# Verify model file exists
ls -lh models/gpt-oss-20b-mxfp4.gguf

# Check permissions
chmod 644 models/gpt-oss-20b-mxfp4.gguf
```

#### Issue: GGUF parsing error
**Symptoms**: Failed to parse GGUF metadata  
**Cause**: Corrupted or incompatible GGUF file  
**Solution**:
1. Verify file hash against known-good value
2. Re-download model from trusted source
3. Check GGUF version compatibility

#### Issue: Architecture not detected
**Symptoms**: Unknown architecture, adapter not found  
**Cause**: GGUF metadata missing or incorrect  
**Solution**:
```bash
# Inspect GGUF metadata
gguf-dump models/gpt-oss-20b-mxfp4.gguf | grep architecture

# Expected: "gpt2" or "gpt"
```

---

### VRAM Issues

#### Issue: Out of memory (OOM)
**Symptoms**: cudaErrorMemoryAllocation, worker crash  
**Cause**: Insufficient VRAM for model  
**Solution**:
1. Check available VRAM:
   ```bash
   nvidia-smi
   ```
2. Reduce batch size or sequence length
3. Use smaller model or more aggressive quantization
4. Clear VRAM:
   ```bash
   # Kill other GPU processes
   nvidia-smi | grep python | awk '{print $5}' | xargs kill -9
   ```

#### Issue: VRAM usage higher than expected
**Symptoms**: Model uses more VRAM than documented  
**Cause**: KV cache, activations, or fragmentation  
**Solution**:
1. Check VRAM breakdown:
   ```cpp
   size_t vram = adapter.get_vram_usage();
   ```
2. Reduce max_seq_len to decrease KV cache
3. Check for memory leaks with cuda-memcheck

#### Issue: VRAM fragmentation
**Symptoms**: Allocation fails despite sufficient free memory  
**Cause**: Memory fragmentation  
**Solution**:
1. Restart worker to clear fragmentation
2. Allocate large buffers first
3. Use memory pools for small allocations

---

### Inference Issues

#### Issue: Generation produces garbage
**Symptoms**: Incoherent output, random tokens  
**Cause**: Model not loaded correctly, weights corrupted  
**Solution**:
1. Verify model provenance (hash check)
2. Re-load model
3. Check MXFP4 dequantization accuracy
4. Run regression tests

#### Issue: Slow inference
**Symptoms**: Tokens/sec below target  
**Cause**: CPU bottleneck, inefficient kernels  
**Solution**:
1. Profile with nsys:
   ```bash
   nsys profile --stats=true ./worker-orcd
   ```
2. Check GPU utilization:
   ```bash
   nvidia-smi dmon
   ```
3. Verify CUDA kernels are being used (not CPU fallback)

#### Issue: Inconsistent output
**Symptoms**: Different output for same prompt  
**Cause**: Non-deterministic sampling, temperature > 0  
**Solution**:
1. Use temperature=0 for deterministic output
2. Set fixed seed for reproducibility
3. Check for race conditions in KV cache

---

### MXFP4 Issues

#### Issue: Accuracy degradation
**Symptoms**: Output quality worse than expected  
**Cause**: MXFP4 quantization error, numerical instability  
**Solution**:
1. Run regression tests:
   ```bash
   ./test_mxfp4_regression
   ```
2. Compare with FP16 baseline
3. Check dequantization kernel correctness

#### Issue: MXFP4 slower than Q4_K_M
**Symptoms**: MXFP4 performance below Q4_K_M  
**Cause**: Dequantization overhead  
**Solution**:
1. Use persistent buffer optimization
2. Profile dequantization kernel
3. Check for repeated dequantization

#### Issue: Numerical instability
**Symptoms**: NaN or Inf values in output  
**Cause**: FP8 scale overflow, FP4 precision limits  
**Solution**:
1. Check FP8 scale values
2. Validate input data range
3. Add numerical stability checks

---

### UTF-8 Issues

#### Issue: Broken multibyte characters
**Symptoms**: � characters in output, encoding errors  
**Cause**: UTF-8 boundary split during streaming  
**Solution**:
1. Use UTF-8 streaming buffer
2. Validate character boundaries
3. Buffer incomplete sequences

#### Issue: Emoji rendering issues
**Symptoms**: Emoji appear as multiple characters  
**Cause**: 4-byte UTF-8 sequence handling  
**Solution**:
1. Ensure full 4-byte sequence buffered
2. Test with emoji test suite
3. Validate UTF-8 encoding

---

### Performance Issues

#### Issue: High latency
**Symptoms**: First token latency > 100ms  
**Cause**: Large prefill, inefficient attention  
**Solution**:
1. Reduce prompt length
2. Optimize attention kernel
3. Use Flash Attention

#### Issue: Low throughput
**Symptoms**: Tokens/sec < 20  
**Cause**: Small batch size, GPU underutilization  
**Solution**:
1. Increase batch size
2. Use continuous batching
3. Profile GPU utilization

---

## Debugging Tools

### CUDA Debugging

```bash
# Check for CUDA errors
cuda-memcheck ./worker-orcd

# Profile with Nsight Systems
nsys profile --stats=true ./worker-orcd

# Profile kernel with Nsight Compute
ncu --set full ./worker-orcd
```

### Logging

```bash
# Enable debug logging
RUST_LOG=debug ./worker-orcd

# Enable CUDA kernel logging
CUDA_LAUNCH_BLOCKING=1 ./worker-orcd
```

### Testing

```bash
# Run integration tests
cargo test --test gpt_integration

# Run regression tests
./test_mxfp4_regression

# Run VRAM boundary tests
./test_24gb_vram_boundary
```

---

## Error Codes

| Code | Meaning | Solution |
|------|---------|----------|
| E001 | Model file not found | Check path |
| E002 | GGUF parse error | Verify file |
| E003 | Architecture not detected | Check metadata |
| E004 | OOM during loading | Reduce model size |
| E005 | OOM during inference | Reduce batch/seq_len |
| E006 | CUDA kernel error | Check GPU driver |
| E007 | Invalid token ID | Clamp to vocab size |
| E008 | UTF-8 encoding error | Validate input |

---

## Getting Help

### Before Reporting

1. Check this troubleshooting guide
2. Search existing issues
3. Run diagnostic tests
4. Collect logs and error messages

### Reporting Issues

Include:
- Error message (full stack trace)
- Model name and quantization format
- Hardware specs (GPU, VRAM)
- Steps to reproduce
- Logs (RUST_LOG=debug)

### Contact

- GitHub Issues: https://github.com/your-org/llama-orch
- Team: GPT-Gamma
- Docs: `/docs/GPT_ARCHITECTURE.md`

---

Crafted by GPT-Gamma 🤖
