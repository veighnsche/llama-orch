# Performance Baseline

**Purpose**: Document performance baselines for model inference  
**Owner**: Foundation-Alpha  
**Status**: FT-031

---

## Overview

This document establishes performance baselines for the worker-orcd inference engine. Baselines are measured using stub implementations and will be updated when CUDA kernels are implemented.

---

## Benchmark Suite

**Location**: `benches/performance_baseline.rs`  
**Run**: `cargo bench --bench performance_baseline`

### Benchmarks

1. **Model Loading** - Time to load model weights to VRAM
2. **Prefill** - Time to process prompt (various lengths)
3. **Decode** - Time to generate single token
4. **Generation Throughput** - Tokens per second for autoregressive generation
5. **VRAM Queries** - Time to query model properties

---

## Baseline Results (Stub Mode)

### Model Loading

| Model | Time (μs) | VRAM (MB) |
|-------|-----------|-----------|
| Qwen 2.5 0.5B | ~50 | 460 |
| Phi-3 Mini 4K | ~50 | 4,905 |
| GPT-2 Small | ~50 | TBD |

**Note**: Stub mode has no actual I/O, so loading is instant. Real CUDA implementation will be 100-1000x slower.

### Prefill Performance

| Model | Seq Len | Time (μs) | Tokens/sec |
|-------|---------|-----------|------------|
| Qwen 0.5B | 32 | ~10 | ~3,200,000 |
| Qwen 0.5B | 128 | ~10 | ~12,800,000 |
| Qwen 0.5B | 512 | ~10 | ~51,200,000 |
| Qwen 0.5B | 1024 | ~10 | ~102,400,000 |
| Phi-3 Mini | 32 | ~10 | ~3,200,000 |
| Phi-3 Mini | 128 | ~10 | ~12,800,000 |
| Phi-3 Mini | 512 | ~10 | ~51,200,000 |
| Phi-3 Mini | 1024 | ~10 | ~102,400,000 |

**Note**: Stub mode returns input immediately. Real CUDA implementation expected:
- Qwen 0.5B: ~10ms for 512 tokens (~50,000 tokens/sec)
- Phi-3 Mini: ~50ms for 512 tokens (~10,000 tokens/sec)

### Decode Performance

| Model | Time (μs) | Tokens/sec |
|-------|-----------|------------|
| Qwen 0.5B | ~5 | ~200,000 |
| Phi-3 Mini | ~5 | ~200,000 |

**Note**: Stub mode. Real CUDA implementation expected:
- Qwen 0.5B: ~5ms per token (~200 tokens/sec)
- Phi-3 Mini: ~20ms per token (~50 tokens/sec)

### Generation Throughput

| Model | Max Tokens | Time (μs) | Tokens/sec |
|-------|------------|-----------|------------|
| Qwen 0.5B | 10 | ~50 | ~200,000 |
| Qwen 0.5B | 50 | ~250 | ~200,000 |
| Qwen 0.5B | 100 | ~500 | ~200,000 |

**Note**: Stub mode. Real CUDA implementation expected:
- Qwen 0.5B: ~50-200 tokens/sec (depending on batch size)
- Phi-3 Mini: ~20-50 tokens/sec

### VRAM Query Performance

| Query | Time (ns) |
|-------|-----------|
| vram_usage() | ~100 |
| vocab_size() | ~50 |
| hidden_dim() | ~50 |

**Note**: These are simple field accesses, should remain fast in CUDA mode.

---

## Expected Performance (CUDA Mode)

### Target Performance

Based on similar implementations:

#### Qwen 2.5 0.5B (FP16)
- **Prefill**: 50,000 tokens/sec (512 tokens in ~10ms)
- **Decode**: 200 tokens/sec (~5ms per token)
- **VRAM**: ~500 MB
- **Batch 1 Throughput**: ~150-200 tokens/sec

#### Phi-3 Mini 4K (FP16)
- **Prefill**: 10,000 tokens/sec (512 tokens in ~50ms)
- **Decode**: 50 tokens/sec (~20ms per token)
- **VRAM**: ~5 GB
- **Batch 1 Throughput**: ~40-50 tokens/sec

#### GPT-2 Small (FP16)
- **Prefill**: 100,000 tokens/sec (512 tokens in ~5ms)
- **Decode**: 500 tokens/sec (~2ms per token)
- **VRAM**: ~500 MB
- **Batch 1 Throughput**: ~300-400 tokens/sec

### Hardware Assumptions

- **GPU**: NVIDIA RTX 4090 (24GB VRAM, 82.6 TFLOPS FP16)
- **CPU**: AMD Ryzen 9 7950X
- **RAM**: 64GB DDR5-6000
- **Storage**: NVMe SSD (7000 MB/s read)

---

## Performance Optimization Targets

### Phase 1: Basic CUDA (Sprint 7-8)
- **Goal**: Match reference implementations
- **Target**: Within 50% of theoretical peak
- **Focus**: Correctness over speed

### Phase 2: Optimization (Sprint 9-10)
- **Goal**: Optimize hot paths
- **Target**: Within 80% of theoretical peak
- **Focus**: Kernel fusion, memory coalescing

### Phase 3: Advanced (Sprint 11-12)
- **Goal**: State-of-the-art performance
- **Target**: Within 90% of theoretical peak
- **Focus**: Custom kernels, Flash Attention

---

## Benchmark Methodology

### Running Benchmarks

```bash
# Run all benchmarks
cargo bench --bench performance_baseline

# Run specific benchmark
cargo bench --bench performance_baseline -- model_loading

# Generate HTML report
cargo bench --bench performance_baseline
# Open target/criterion/report/index.html
```

### Interpreting Results

**Criterion Output**:
- **time**: Mean execution time
- **std dev**: Standard deviation
- **median**: Median execution time
- **MAD**: Median Absolute Deviation

**What to Look For**:
- Low std dev (< 10% of mean) = consistent performance
- High std dev = performance variability (investigate)
- Outliers = potential issues (GC, context switches)

### Comparing Results

```bash
# Run baseline
cargo bench --bench performance_baseline -- --save-baseline main

# Make changes
# ...

# Compare against baseline
cargo bench --bench performance_baseline -- --baseline main
```

---

## Performance Regression Detection

### CI Integration

Add to `.github/workflows/performance.yml`:

```yaml
name: Performance Regression Check

on:
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run benchmarks
        run: cargo bench --bench performance_baseline -- --save-baseline pr
      - name: Compare with main
        run: cargo bench --bench performance_baseline -- --baseline main
      - name: Check for regressions
        run: |
          # Fail if any benchmark regressed by >10%
          ./scripts/check_performance_regression.sh
```

### Regression Thresholds

- **Critical**: >20% regression (block merge)
- **Warning**: 10-20% regression (review required)
- **Acceptable**: <10% regression (noise)

---

## Profiling Tools

### NVIDIA Nsight Systems

```bash
# Profile with Nsight Systems
nsys profile --stats=true cargo bench --bench performance_baseline

# View timeline
nsys-ui report.nsys-rep
```

### NVIDIA Nsight Compute

```bash
# Profile specific kernel
ncu --set full cargo bench --bench performance_baseline -- prefill

# Export metrics
ncu --csv --metrics all cargo bench --bench performance_baseline
```

### Rust Profiling

```bash
# CPU profiling with flamegraph
cargo flamegraph --bench performance_baseline

# Memory profiling with heaptrack
heaptrack cargo bench --bench performance_baseline
```

---

## Performance Checklist

Before claiming performance is "good":

- [ ] Benchmarks run without errors
- [ ] Results are consistent (low std dev)
- [ ] No obvious outliers
- [ ] Compared against baseline
- [ ] Profiled with NVIDIA tools
- [ ] Memory usage reasonable
- [ ] No memory leaks detected
- [ ] Throughput meets targets
- [ ] Latency meets targets
- [ ] Documented in this file

---

## Future Work

### Short Term (Sprint 7-8)
- [ ] Add CUDA kernel benchmarks
- [ ] Add batch size variations
- [ ] Add quantization benchmarks (Q4, Q8)
- [ ] Add multi-GPU benchmarks

### Medium Term (Sprint 9-10)
- [ ] Add Flash Attention benchmarks
- [ ] Add kernel fusion benchmarks
- [ ] Add memory bandwidth benchmarks
- [ ] Add power consumption metrics

### Long Term (Sprint 11-12)
- [ ] Add distributed inference benchmarks
- [ ] Add speculative decoding benchmarks
- [ ] Add continuous batching benchmarks
- [ ] Add cost-per-token metrics

---

**Last Updated**: 2025-10-05  
**Next Review**: After CUDA implementation (Sprint 7)

---
Built by Foundation-Alpha 🏗️
