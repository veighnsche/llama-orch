# VRAM Debugging Guide

**Purpose**: Guide for debugging VRAM allocation and usage issues  
**Audience**: All teams working with GPU memory  
**Owner**: Foundation-Alpha

---

## Overview

VRAM (Video RAM) management is critical for model inference. This guide helps you:
1. **Understand VRAM calculations**
2. **Debug allocation failures**
3. **Optimize VRAM usage**
4. **Validate VRAM enforcement**

---

## VRAM Calculation Formulas

### Model Weights

```rust
// Formula: weights_bytes = num_parameters * bytes_per_parameter
//
// For FP16 models:
// bytes_per_parameter = 2 (16 bits)
//
// For quantized models (Q4_0, Q4_1, Q8_0):
// bytes_per_parameter varies by quantization format

fn calculate_weight_vram(config: &ModelConfig) -> usize {
    let embedding_params = config.vocab_size * config.hidden_dim;
    let layer_params = calculate_layer_params(config);
    let total_params = embedding_params + (layer_params * config.num_layers);
    
    // FP16: 2 bytes per parameter
    total_params * 2
}

fn calculate_layer_params(config: &ModelConfig) -> usize {
    // Attention weights
    let qkv_params = 3 * config.hidden_dim * config.hidden_dim;
    let attn_out_params = config.hidden_dim * config.hidden_dim;
    
    // FFN weights
    let ffn_up_params = config.hidden_dim * config.ffn_dim;
    let ffn_down_params = config.ffn_dim * config.hidden_dim;
    
    // LayerNorm weights
    let ln_params = 2 * config.hidden_dim;
    
    qkv_params + attn_out_params + ffn_up_params + ffn_down_params + ln_params
}
```

### KV Cache

```rust
// Formula: kv_cache_bytes = 2 * num_layers * num_kv_heads * head_dim * max_seq_len * batch_size * 2
//
// Breakdown:
// - 2: for K and V
// - num_layers: cache for each layer
// - num_kv_heads: number of KV heads (GQA)
// - head_dim: dimension per head
// - max_seq_len: maximum sequence length
// - batch_size: number of sequences
// - 2: FP16 (2 bytes per value)

fn calculate_kv_cache_vram(config: &ModelConfig, max_seq_len: usize, batch_size: usize) -> usize {
    2 * config.num_layers 
      * config.num_kv_heads 
      * config.head_dim 
      * max_seq_len 
      * batch_size 
      * 2  // FP16
}
```

### Activation Memory

```rust
// Formula: activation_bytes = batch_size * seq_len * hidden_dim * 2 * num_intermediate_tensors
//
// Intermediate tensors vary by architecture:
// - Attention: Q, K, V, scores, context
// - FFN: up projection, activation, down projection
//
// Conservative estimate: 10 intermediate tensors

fn calculate_activation_vram(config: &ModelConfig, seq_len: usize, batch_size: usize) -> usize {
    let num_intermediate = 10;
    batch_size * seq_len * config.hidden_dim * 2 * num_intermediate
}
```

### Total VRAM

```rust
fn calculate_total_vram(
    config: &ModelConfig,
    max_seq_len: usize,
    batch_size: usize,
) -> usize {
    let weights = calculate_weight_vram(config);
    let kv_cache = calculate_kv_cache_vram(config, max_seq_len, batch_size);
    let activations = calculate_activation_vram(config, max_seq_len, batch_size);
    
    // Add 10% overhead for CUDA internals
    let total = weights + kv_cache + activations;
    (total as f64 * 1.1) as usize
}
```

---

## Example Calculations

### Qwen 2.5 0.5B

```
Configuration:
- vocab_size: 151,936
- hidden_dim: 896
- num_layers: 24
- num_q_heads: 14
- num_kv_heads: 2 (GQA)
- head_dim: 64
- ffn_dim: 4,864

Weights:
- Embedding: 151,936 * 896 * 2 = 272 MB
- Per layer: ~3.5 MB
- Total layers: 24 * 3.5 MB = 84 MB
- Total weights: 272 + 84 = 356 MB

KV Cache (max_seq_len=2048, batch=1):
- 2 * 24 * 2 * 64 * 2048 * 1 * 2 = 25 MB

Activations (seq_len=2048, batch=1):
- 1 * 2048 * 896 * 2 * 10 = 37 MB

Total: 356 + 25 + 37 = 418 MB
With overhead: 418 * 1.1 = 460 MB
```

### Phi-3 Mini 4K

```
Configuration:
- vocab_size: 32,064
- hidden_dim: 3,072
- num_layers: 32
- num_q_heads: 32
- num_kv_heads: 32 (MHA)
- head_dim: 96
- ffn_dim: 8,192

Weights:
- Embedding: 32,064 * 3,072 * 2 = 197 MB
- Per layer: ~75 MB
- Total layers: 32 * 75 MB = 2,400 MB
- Total weights: 197 + 2,400 = 2,597 MB

KV Cache (max_seq_len=4096, batch=1):
- 2 * 32 * 32 * 96 * 4096 * 1 * 2 = 1,610 MB

Activations (seq_len=4096, batch=1):
- 1 * 4096 * 3,072 * 2 * 10 = 252 MB

Total: 2,597 + 1,610 + 252 = 4,459 MB
With overhead: 4,459 * 1.1 = 4,905 MB (~4.9 GB)
```

---

## Debugging VRAM Issues

### Issue 1: Allocation Failure

**Symptom**: `CudaError::AllocationFailed`

**Diagnosis**:
```bash
# Check available VRAM
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits

# Check VRAM usage
nvidia-smi dmon -s m -c 1
```

**Common Causes**:
1. **Model too large**: Weights exceed available VRAM
2. **KV cache too large**: max_seq_len * batch_size too big
3. **Fragmentation**: Many small allocations
4. **Memory leak**: Previous allocations not freed

**Solutions**:
```rust
// Solution 1: Reduce max_seq_len
let config = ModelConfig {
    max_seq_len: 1024,  // Instead of 4096
    ..default
};

// Solution 2: Use quantization
let config = ModelConfig {
    quantization: Quantization::Q4_0,  // 4-bit instead of FP16
    ..default
};

// Solution 3: Check for leaks
// Run with CUDA-MEMCHECK
// $ cuda-memcheck ./target/debug/worker-orcd
```

### Issue 2: VRAM Usage Higher Than Expected

**Symptom**: Model uses more VRAM than calculated

**Diagnosis**:
```rust
// Log detailed VRAM breakdown
tracing::info!(
    weights_mb = weights_bytes / (1024 * 1024),
    kv_cache_mb = kv_cache_bytes / (1024 * 1024),
    activations_mb = activation_bytes / (1024 * 1024),
    total_mb = total_bytes / (1024 * 1024),
    "VRAM breakdown"
);

// Query actual VRAM usage
let (free, total) = cuda_context.get_vram_info()?;
let used = total - free;
tracing::info!(
    used_mb = used / (1024 * 1024),
    expected_mb = total_bytes / (1024 * 1024),
    diff_mb = (used as i64 - total_bytes as i64) / (1024 * 1024),
    "VRAM usage comparison"
);
```

**Common Causes**:
1. **CUDA overhead**: cuBLAS, cuDNN allocations
2. **Alignment padding**: GPU memory alignment requirements
3. **Temporary buffers**: Intermediate computation buffers
4. **Multiple contexts**: Multiple CUDA contexts on same GPU

**Solutions**:
```rust
// Solution 1: Account for overhead
let total_with_overhead = (calculated_bytes as f64 * 1.2) as usize;  // 20% overhead

// Solution 2: Profile with NVIDIA tools
// $ nsys profile ./target/debug/worker-orcd
// $ ncu --target-processes all ./target/debug/worker-orcd

// Solution 3: Use memory pool
// Reuse allocations instead of allocating/freeing repeatedly
```

### Issue 3: VRAM Leak

**Symptom**: VRAM usage grows over time

**Diagnosis**:
```bash
# Monitor VRAM over time
watch -n 1 nvidia-smi

# Run with leak detection
cuda-memcheck --leak-check full ./target/debug/worker-orcd

# Run with Valgrind (for host memory)
valgrind --leak-check=full --show-leak-kinds=all ./target/debug/worker-orcd
```

**Common Causes**:
1. **Missing Drop**: CUDA pointers not freed
2. **Circular references**: Rc/Arc cycles
3. **Static allocations**: Never-freed global state
4. **Error path leaks**: Allocation before error, no cleanup

**Solutions**:
```rust
// Solution 1: Implement Drop correctly
impl Drop for SafeCudaPtr {
    fn drop(&mut self) {
        if self.ptr != 0 {
            unsafe { cudaFree(self.ptr as *mut c_void); }
        }
    }
}

// Solution 2: Use RAII pattern
struct VramGuard {
    ptr: SafeCudaPtr,
}

impl Drop for VramGuard {
    fn drop(&mut self) {
        // Guaranteed cleanup
    }
}

// Solution 3: Add leak tests
#[test]
fn test_no_vram_leak() {
    let initial_free = get_free_vram();
    
    for _ in 0..1000 {
        let model = load_model();
        drop(model);
    }
    
    let final_free = get_free_vram();
    assert_eq!(initial_free, final_free, "VRAM leaked");
}
```

### Issue 4: Fragmentation

**Symptom**: Allocation fails despite enough total free VRAM

**Diagnosis**:
```rust
// Check fragmentation
let (free, total) = cuda_context.get_vram_info()?;
let used = total - free;

tracing::warn!(
    free_mb = free / (1024 * 1024),
    used_mb = used / (1024 * 1024),
    allocation_mb = size / (1024 * 1024),
    "Allocation failed despite free VRAM - possible fragmentation"
);
```

**Common Causes**:
1. **Many small allocations**: Interleaved with large allocations
2. **Long-lived allocations**: Block large contiguous regions
3. **No memory pool**: Each allocation is independent

**Solutions**:
```rust
// Solution 1: Use memory pool
struct VramPool {
    blocks: Vec<SafeCudaPtr>,
}

impl VramPool {
    fn allocate(&mut self, size: usize) -> Option<SafeCudaPtr> {
        // Reuse existing block if available
        // Otherwise allocate new
    }
}

// Solution 2: Allocate in order
// Allocate all long-lived memory first
// Then allocate short-lived memory

// Solution 3: Reset context
// Periodically reset CUDA context to defragment
```

---

## VRAM Optimization Strategies

### Strategy 1: Quantization

```rust
// FP16 -> Q8_0: 2x reduction
// FP16 -> Q4_0: 4x reduction

let fp16_size = num_params * 2;
let q8_size = num_params * 1;
let q4_size = num_params * 0.5;

println!("FP16: {} MB", fp16_size / (1024 * 1024));
println!("Q8_0: {} MB", q8_size / (1024 * 1024));
println!("Q4_0: {} MB", q4_size / (1024 * 1024));
```

### Strategy 2: KV Cache Optimization

```rust
// Reduce max_seq_len
let config = ModelConfig {
    max_seq_len: 2048,  // Instead of 4096
    ..default
};

// Use GQA instead of MHA
let config = ModelConfig {
    num_kv_heads: 2,  // Instead of 32
    ..default
};

// Dynamic KV cache growth
// Start small, grow as needed
```

### Strategy 3: Batch Size Tuning

```rust
// VRAM scales linearly with batch size
// Find optimal batch size for throughput vs VRAM

fn find_optimal_batch_size(config: &ModelConfig, vram_limit: usize) -> usize {
    let mut batch_size = 1;
    
    loop {
        let vram_needed = calculate_total_vram(config, config.max_seq_len, batch_size);
        
        if vram_needed > vram_limit {
            return batch_size - 1;
        }
        
        batch_size += 1;
    }
}
```

### Strategy 4: Gradient Checkpointing

```rust
// Trade compute for memory
// Recompute activations instead of storing them
// Reduces activation memory by ~50%

let activation_vram_with_checkpointing = calculate_activation_vram(config, seq_len, batch_size) / 2;
```

---

## Validation Tools

### Tool 1: VRAM Calculator

```rust
// In src/vram_calculator.rs
pub fn calculate_and_log_vram(config: &ModelConfig, max_seq_len: usize, batch_size: usize) {
    let weights = calculate_weight_vram(config);
    let kv_cache = calculate_kv_cache_vram(config, max_seq_len, batch_size);
    let activations = calculate_activation_vram(config, max_seq_len, batch_size);
    let total = weights + kv_cache + activations;
    let with_overhead = (total as f64 * 1.1) as usize;
    
    println!("VRAM Breakdown:");
    println!("  Weights:     {:>8} MB", weights / (1024 * 1024));
    println!("  KV Cache:    {:>8} MB", kv_cache / (1024 * 1024));
    println!("  Activations: {:>8} MB", activations / (1024 * 1024));
    println!("  Subtotal:    {:>8} MB", total / (1024 * 1024));
    println!("  Overhead:    {:>8} MB", (with_overhead - total) / (1024 * 1024));
    println!("  Total:       {:>8} MB", with_overhead / (1024 * 1024));
}
```

### Tool 2: VRAM Monitor

```bash
#!/bin/bash
# monitor_vram.sh

while true; do
    clear
    date
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total --format=csv,noheader
    sleep 1
done
```

### Tool 3: VRAM Pressure Test

```rust
#[test]
fn test_vram_pressure() {
    let config = ModelConfig::default();
    let mut models = Vec::new();
    
    // Load models until VRAM exhausted
    loop {
        match load_model(&config) {
            Ok(model) => {
                models.push(model);
                println!("Loaded {} models", models.len());
            }
            Err(CudaError::AllocationFailed(_)) => {
                println!("VRAM exhausted after {} models", models.len());
                break;
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }
    
    // Verify cleanup
    drop(models);
    let (free, _) = get_vram_info().unwrap();
    println!("Free VRAM after cleanup: {} MB", free / (1024 * 1024));
}
```

---

## Best Practices

### DO ✅

1. **Calculate before allocating**: Know how much VRAM you need
2. **Log VRAM usage**: Track allocations and deallocations
3. **Test with limited VRAM**: Validate behavior under pressure
4. **Use RAII**: Ensure cleanup with Drop
5. **Profile regularly**: Use NVIDIA tools to understand usage

### DON'T ❌

1. **Don't guess VRAM needs**: Calculate precisely
2. **Don't ignore allocation failures**: Handle gracefully
3. **Don't leak memory**: Implement Drop correctly
4. **Don't over-allocate**: Use only what you need
5. **Don't skip validation**: Test VRAM calculations

---

## Troubleshooting Checklist

When debugging VRAM issues:

- [ ] Calculate expected VRAM usage
- [ ] Query actual VRAM usage with nvidia-smi
- [ ] Compare expected vs actual
- [ ] Check for memory leaks with cuda-memcheck
- [ ] Profile with NVIDIA Nsight
- [ ] Verify Drop implementations
- [ ] Test with VRAM pressure tests
- [ ] Log detailed VRAM breakdown
- [ ] Check for fragmentation
- [ ] Validate cleanup in error paths

---

## References

- [CUDA Memory Management](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-management)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [CUDA-MEMCHECK](https://docs.nvidia.com/cuda/cuda-memcheck/)

---

**Last Updated**: 2025-10-05  
**Maintainer**: Foundation-Alpha  
**Status**: Complete

---
Built by Foundation-Alpha 🏗️
