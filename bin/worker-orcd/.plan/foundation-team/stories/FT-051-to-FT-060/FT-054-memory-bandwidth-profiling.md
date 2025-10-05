# FT-054: Memory Bandwidth Profiling
**Team**: Foundation-Alpha  
**Sprint**: Sprint 6 - FP16 Optimization (Post-M0)  
**Size**: S (1 day)  
**Priority**: Medium (Post-M0)  
**Spec Ref**: M0-W-1430
---
## ⚠️ Prerequisites
**Requires M0 completion:**
- Working inference pipeline
- Functional CUDA kernels to profile
- Performance baseline established
---
## Story Description
Implement memory bandwidth profiling infrastructure to measure FP16 optimization impact. Track bandwidth usage for GEMM, attention, and KV cache operations. Generate performance reports comparing FP32 vs FP16.
---
## Acceptance Criteria
- [ ] CUDA event-based timing for all kernels
- [ ] Memory bandwidth calculation (bytes transferred / time)
- [ ] Profiling for GEMM operations
- [ ] Profiling for attention kernels
- [ ] Profiling for KV cache operations
- [ ] CSV/JSON output for benchmark results
- [ ] Visualization script (Python/gnuplot)
- [ ] Integration with  system
---
## Dependencies
**Upstream**: FT-053 (KV cache FP16, Day 5)  
**Downstream**: FT-055 (Fused kernel optimization)
---
## Technical Details
### Implementation Plan
**Phase 1: Profiling Infrastructure** (Day 1)
```cpp
// cuda/profiling/bandwidth_profiler.cu
#include <cuda_runtime.h>
#include <stdio.h>
/**
 * Bandwidth profiler for CUDA kernels
 * 
 * Measures kernel execution time and calculates memory bandwidth.
 */
struct BandwidthProfiler {
    cudaEvent_t start;
    cudaEvent_t stop;
    const char* kernel_name;
    size_t bytes_read;
    size_t bytes_written;
    BandwidthProfiler(const char* name) : kernel_name(name) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        bytes_read = 0;
        bytes_written = 0;
    }
    ~BandwidthProfiler() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void start_profile() {
        cudaEventRecord(start);
    }
    void stop_profile() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    }
    float get_elapsed_ms() {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
    float get_bandwidth_gb_s() {
        float ms = get_elapsed_ms();
        if (ms == 0.0f) return 0.0f;
        size_t total_bytes = bytes_read + bytes_written;
        float seconds = ms / 1000.0f;
        float gb_per_s = (total_bytes / (1024.0f * 1024.0f * 1024.0f)) / seconds;
        return gb_per_s;
    }
    void set_bytes(size_t read, size_t written) {
        bytes_read = read;
        bytes_written = written;
    }
    void print_report() {
        printf("=== Bandwidth Profile: %s ===\n", kernel_name);
        printf("Elapsed time: %.3f ms\n", get_elapsed_ms());
        printf("Bytes read: %zu (%.2f MB)\n", bytes_read, 
               bytes_read / (1024.0f * 1024.0f));
        printf("Bytes written: %zu (%.2f MB)\n", bytes_written,
               bytes_written / (1024.0f * 1024.0f));
        printf("Bandwidth: %.2f GB/s\n", get_bandwidth_gb_s());
        printf("=============================\n\n");
    }
};
/**
 * Profile GEMM operation
 */
void profile_gemm_bandwidth(
    int M, int N, int K,
    bool use_fp16
) {
    BandwidthProfiler profiler(use_fp16 ? "GEMM FP16" : "GEMM FP32");
    // Calculate bytes transferred
    size_t element_size = use_fp16 ? 2 : 4;
    size_t bytes_read = (M * K + K * N) * element_size;  // A + B
    size_t bytes_written = M * N * element_size;  // C
    profiler.set_bytes(bytes_read, bytes_written);
    // Profile kernel (implementation depends on FT-051)
    profiler.start_profile();
    // ... launch GEMM kernel ...
    profiler.stop_profile();
    profiler.print_report();
}
/**
 * Profile attention operation
 */
void profile_attention_bandwidth(
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    bool use_fp16
) {
    BandwidthProfiler profiler(use_fp16 ? "Attention FP16" : "Attention FP32");
    size_t element_size = use_fp16 ? 2 : 4;
    // Q, K, V reads
    size_t qkv_size = batch_size * seq_len * num_heads * head_dim * element_size;
    size_t bytes_read = 3 * qkv_size;  // Q + K + V
    // Attention scores (intermediate)
    size_t scores_size = batch_size * num_heads * seq_len * seq_len * element_size;
    bytes_read += scores_size;  // Read scores for softmax
    // Output write
    size_t bytes_written = qkv_size;  // Output same size as Q
    profiler.set_bytes(bytes_read, bytes_written);
    profiler.start_profile();
    // ... launch attention kernel ...
    profiler.stop_profile();
    profiler.print_report();
}
/**
 * Profile KV cache operation
 */
void profile_kv_cache_bandwidth(
    int batch_size,
    int seq_len,
    int num_kv_heads,
    int head_dim,
    bool use_fp16
) {
    BandwidthProfiler profiler(use_fp16 ? "KV Cache FP16" : "KV Cache FP32");
    size_t element_size = use_fp16 ? 2 : 4;
    size_t kv_size = batch_size * seq_len * num_kv_heads * head_dim * element_size;
    // Read K, V
    size_t bytes_read = 2 * kv_size;
    // Write to cache
    size_t bytes_written = 2 * kv_size;
    profiler.set_bytes(bytes_read, bytes_written);
    profiler.start_profile();
    // ... launch KV cache kernel ...
    profiler.stop_profile();
    profiler.print_report();
}
extern "C" {
/**
 * Run full bandwidth benchmark suite
 * 
 * Compares FP32 vs FP16 for all operations.
 */
int cuda_benchmark_bandwidth(const char* output_csv) {
    FILE* fp = fopen(output_csv, "w");
    if (!fp) {
        fprintf(stderr, "Failed to open output file: %s\n", output_csv);
        return -1;
    }
    // CSV header
    fprintf(fp, "operation,precision,M,N,K,time_ms,bandwidth_gb_s,bytes_read,bytes_written\n");
    // Benchmark configurations
    struct Config {
        const char* name;
        int M, N, K;
    };
    Config configs[] = {
        {"decode_gemm", 1, 896, 896},
        {"prefill_gemm_small", 32, 896, 896},
        {"prefill_gemm_medium", 128, 896, 896},
        {"ffn_gemm", 128, 4864, 896},
    };
    for (const auto& cfg : configs) {
        // FP32
        profile_gemm_bandwidth(cfg.M, cfg.N, cfg.K, false);
        // Write to CSV...
        // FP16
        profile_gemm_bandwidth(cfg.M, cfg.N, cfg.K, true);
        // Write to CSV...
    }
    fclose(fp);
    return 0;
}
} // extern "C"
```
**Phase 2: Visualization Script** (Day 1)
```python
#!/usr/bin/env python3
# cuda/profiling/visualize_bandwidth.py
import pandas as pd
import matplotlib.pyplot as plt
import sys
def plot_bandwidth_comparison(csv_path):
    df = pd.read_csv(csv_path)
    # Group by operation
    operations = df['operation'].unique()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('FP32 vs FP16 Memory Bandwidth Comparison', fontsize=16)
    for idx, op in enumerate(operations):
        ax = axes[idx // 2, idx % 2]
        op_data = df[df['operation'] == op]
        fp32_data = op_data[op_data['precision'] == 'FP32']
        fp16_data = op_data[op_data['precision'] == 'FP16']
        x = range(len(fp32_data))
        width = 0.35
        ax.bar([i - width/2 for i in x], fp32_data['bandwidth_gb_s'], 
               width, label='FP32', color='blue', alpha=0.7)
        ax.bar([i + width/2 for i in x], fp16_data['bandwidth_gb_s'],
               width, label='FP16', color='green', alpha=0.7)
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Bandwidth (GB/s)')
        ax.set_title(op)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('bandwidth_comparison.png', dpi=300)
    print(f"Saved bandwidth_comparison.png")
    # Print summary statistics
    print("\n=== Bandwidth Summary ===")
    for op in operations:
        op_data = df[df['operation'] == op]
        fp32_avg = op_data[op_data['precision'] == 'FP32']['bandwidth_gb_s'].mean()
        fp16_avg = op_data[op_data['precision'] == 'FP16']['bandwidth_gb_s'].mean()
        speedup = fp16_avg / fp32_avg if fp32_avg > 0 else 0
        print(f"{op}:")
        print(f"  FP32: {fp32_avg:.2f} GB/s")
        print(f"  FP16: {fp16_avg:.2f} GB/s")
        print(f"  Speedup: {speedup:.2f}x")
        print()
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: visualize_bandwidth.py <benchmark.csv>")
        sys.exit(1)
    plot_bandwidth_comparison(sys.argv[1])
```
**Phase 3: Integration with Proof Bundle** (Day 1)
```rust
// src/profiling/bandwidth.rs
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
pub struct BandwidthReport {
    pub operation: String,
    pub precision: String,
    pub time_ms: f32,
    pub bandwidth_gb_s: f32,
    pub bytes_read: usize,
    pub bytes_written: usize,
}
impl BandwidthReport {
    pub fn save_to_proof_bundle(&self, run_id: &str) -> std::io::Result<()> {
        let proof_dir = PathBuf::from(".test-results/bandwidth")
            .join(run_id);
        std::fs::create_dir_all(&proof_dir)?;
        let report_path = proof_dir.join("bandwidth_report.json");
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(report_path, json)?;
        Ok(())
    }
}
```
---
## Files to Create/Modify
**Create**:
- `cuda/profiling/bandwidth_profiler.cu` - Profiling infrastructure
- `cuda/profiling/bandwidth_profiler.h` - Header
- `cuda/profiling/visualize_bandwidth.py` - Visualization script
- `src/profiling/bandwidth.rs` - Rust integration
**Modify**:
- `cuda/kernels/gemm.cu` - Add profiling hooks
- `cuda/kernels/gqa_attention.cu` - Add profiling hooks
- `cuda/kernels/kv_cache.cu` - Add profiling hooks
---
## Testing Strategy
### Unit Tests (3 tests)
1. **test_profiler_timing** - Validate timing accuracy
2. **test_bandwidth_calculation** - Validate bandwidth formula
3. **test_csv_output** - Validate CSV format
### Integration Tests (2 tests)
1. **test_full_benchmark_suite** - Run all benchmarks
2. **test_proof_bundle_integration** - Save to 
---
## Expected Results
### Memory Bandwidth Comparison
| Operation | FP32 (GB/s) | FP16 (GB/s) | Improvement |
|-----------|-------------|-------------|-------------|
| Decode GEMM | 150 | 250 | 1.67x |
| Prefill GEMM | 300 | 450 | 1.5x |
| Attention | 200 | 320 | 1.6x |
| KV Cache | 180 | 300 | 1.67x |
**Note**: Actual numbers depend on GPU (A100, RTX 4090, etc.)
---
## Definition of Done
- [ ] All acceptance criteria met
- [ ] Profiling infrastructure implemented
- [ ] Benchmarks run successfully
- [ ] Visualization script working
- [ ] CSV output validated
- [ ] Documentation updated
- [ ] Code reviewed
- [ ] Story marked complete
---
**Status**: 📋 Ready  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-05
---
## References
- CUDA profiling guide: https://docs.nvidia.com/cuda/profiler-users-guide/
- Memory bandwidth optimization: https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
---
Built by Foundation-Alpha 🏗️
