# Troubleshooting Guide

**Component**: Support & Debugging  
**Audience**: Developers

---

## Common Issues

### Model Loading

#### Issue: Model fails to load

**Symptoms**:
```
Error: WeightMappingError: Tensor not found: "blk.0.attn_q.weight"
```

**Causes**:
1. GGUF file corrupted or incomplete
2. Wrong model architecture
3. Unsupported GGUF version

**Solutions**:
```bash
# Verify file integrity
sha256sum qwen2.5-0.5b.gguf

# Check file size
ls -lh qwen2.5-0.5b.gguf
# Expected: ~1.3 GB for Qwen, ~7.5 GB for Phi-3

# Verify GGUF version
hexdump -C qwen2.5-0.5b.gguf | head -n 1
# Should start with: 47 47 55 46 (GGUF magic)
```

#### Issue: Out of VRAM

**Symptoms**:
```
Error: AllocationFailed: 1300000000 bytes
CUDA error: out of memory
```

**Solutions**:
```bash
# Check available VRAM
nvidia-smi

# Free VRAM
# 1. Close other GPU applications
# 2. Unload other models
# 3. Use smaller model (Qwen instead of Phi-3)

# Monitor VRAM usage
watch -n 1 nvidia-smi
```

**VRAM Requirements**:
- Qwen2.5-0.5B: ~1.5 GB (model + KV cache)
- Phi-3-mini-4k: ~8 GB (model + KV cache)

---

### Tokenization

#### Issue: Tokenizer fails to load

**Symptoms**:
```
Error: GGUFError: Metadata key not found: "tokenizer.ggml.tokens"
```

**Causes**:
1. GGUF file missing tokenizer metadata
2. Incompatible GGUF version

**Solutions**:
```rust
// Verify metadata exists
let mmap = MmapFile::open("model.gguf")?;
let metadata = parse_gguf_metadata(&mmap)?;

// Check for tokenizer keys
assert!(metadata.contains_key("tokenizer.ggml.tokens"));
assert!(metadata.contains_key("tokenizer.ggml.merges"));
```

#### Issue: Broken UTF-8 in output

**Symptoms**:
```
Output: "Hello, world! �"
```

**Causes**:
1. Not using `StreamingDecoder`
2. Token boundary splits UTF-8 character

**Solutions**:
```rust
// Use StreamingDecoder for streaming
let mut streaming = StreamingDecoder::new(decoder);

for token_id in generated_ids {
    let partial = streaming.decode_token(token_id);
    print!("{}", partial);
}

let remaining = streaming.flush();
print!("{}", remaining);
```

---

### Generation

#### Issue: Slow generation

**Symptoms**:
- <5 tokens/sec (Qwen)
- <3 tokens/sec (Phi-3)

**Causes**:
1. CPU fallback (CUDA not available)
2. GPU underutilized
3. Memory bandwidth bottleneck

**Solutions**:
```bash
# Check CUDA availability
nvidia-smi

# Check GPU utilization
nvidia-smi dmon -s u
# Should show >80% GPU utilization during decode

# Profile kernels
nvprof ./worker-orcd

# Check for CPU fallback
CUDA_LAUNCH_BLOCKING=1 ./worker-orcd
```

**Expected Performance**:
- Qwen: ~10 tokens/sec
- Phi-3: ~6-7 tokens/sec

#### Issue: Gibberish output

**Symptoms**:
```
Output: "asdfkjhasdkfh asdkfh askdfh"
```

**Causes**:
1. Wrong tokenizer for model
2. Temperature too high
3. Model weights corrupted

**Solutions**:
```rust
// Verify tokenizer matches model
let encoder = BPEEncoder::from_gguf("qwen2.5-0.5b.gguf")?;
let decoder = BPEDecoder::from_gguf("qwen2.5-0.5b.gguf")?;
// Must use same GGUF file

// Lower temperature
let config = AdapterForwardConfig {
    temperature: 0.7,  // Try 0.5-0.7
    // ...
};

// Verify model checksum
sha256sum qwen2.5-0.5b.gguf
```

#### Issue: Repeated tokens

**Symptoms**:
```
Output: "Hello hello hello hello hello..."
```

**Causes**:
1. Temperature too low
2. Sampling bug
3. KV cache not updating

**Solutions**:
```rust
// Increase temperature
let config = AdapterForwardConfig {
    temperature: 0.8,  // Try 0.7-1.0
    // ...
};

// Verify KV cache updates
for i in 0..max_tokens {
    config.cache_len = input_ids.len() + i;  // Must increment
    let next_token = adapter.decode(current_token, &config)?;
}
```

---

### Reproducibility

#### Issue: Non-deterministic output

**Symptoms**:
```
Run 1: "Hello, world! How are you?"
Run 2: "Hello, world! What's up?"
```

**Causes**:
1. Different seeds
2. Non-deterministic CUDA operations
3. Uninitialized memory

**Solutions**:
```rust
// Use fixed seed
let config = AdapterForwardConfig {
    seed: 42,  // Same seed for all runs
    // ...
};

// Verify reproducibility
let output1 = adapter.generate(&input_ids, 20, &config)?;
let output2 = adapter.generate(&input_ids, 20, &config)?;
assert_eq!(output1, output2);
```

---

### Context Length

#### Issue: Context length exceeded

**Symptoms**:
```
Error: ContextLengthExceeded { length: 5000, max: 4096 }
```

**Causes**:
1. Prompt too long
2. Generated too many tokens

**Solutions**:
```rust
// Check prompt length
let input_ids = encoder.encode(prompt)?;
if input_ids.len() > 4096 {
    eprintln!("Prompt too long: {} tokens", input_ids.len());
    // Truncate or use sliding window
}

// Limit generation
let max_tokens = 4096 - input_ids.len();
let output_ids = adapter.generate(&input_ids, max_tokens, &config)?;
```

**Context Limits**:
- Qwen: 32768 tokens
- Phi-3: 4096 tokens

---

### Memory Leaks

#### Issue: VRAM usage grows over time

**Symptoms**:
```bash
# Initial: 1.5 GB
# After 100 generations: 3.0 GB
```

**Causes**:
1. KV cache not cleared
2. Temporary buffers not freed
3. CUDA allocator fragmentation

**Solutions**:
```rust
// Monitor VRAM
let initial_vram = get_vram_usage();

for i in 0..100 {
    adapter.generate(&input_ids, 20, &config)?;
    
    if i % 10 == 0 {
        let current_vram = get_vram_usage();
        println!("Iteration {}: {} MB", i, current_vram / (1024 * 1024));
    }
}

let final_vram = get_vram_usage();
let leak = final_vram - initial_vram;
assert!(leak < 100_000_000, "Memory leak: {} MB", leak / (1024 * 1024));
```

---

## Debugging Tools

### Enable Logging

```rust
// Add to Cargo.toml
[dependencies]
tracing = "0.1"
tracing-subscriber = "0.3"

// In main.rs
fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();
    
    // Your code
}
```

### CUDA Debugging

```bash
# Enable CUDA error checking
export CUDA_LAUNCH_BLOCKING=1

# Enable CUDA debugging
export CUDA_VISIBLE_DEVICES=0

# Profile with nvprof
nvprof --print-gpu-trace ./worker-orcd

# Profile with Nsight Systems
nsys profile --trace=cuda,nvtx ./worker-orcd
```

### Memory Profiling

```bash
# Monitor VRAM
watch -n 1 nvidia-smi

# Detailed memory info
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# CUDA memory profiling
cuda-memcheck ./worker-orcd
```

---

## Error Messages

### AdapterError

| Error | Cause | Solution |
|-------|-------|----------|
| `ModelNotLoaded` | Model not initialized | Call `load_to_vram()` first |
| `InvalidModelType` | Unsupported model | Use Qwen or Phi-3 |
| `ForwardPassFailed` | CUDA kernel error | Check CUDA installation |
| `UnsupportedOperation` | Feature not implemented | Use supported model type |

### TokenizerError

| Error | Cause | Solution |
|-------|-------|----------|
| `TokenNotFound` | Unknown token | Check vocabulary |
| `IdNotFound` | Invalid token ID | Verify ID range |
| `InvalidUtf8` | Broken UTF-8 | Use `StreamingDecoder` |
| `GGUFError` | GGUF parsing failed | Verify file integrity |

### WeightLoadingError

| Error | Cause | Solution |
|-------|-------|----------|
| `AllocationFailed` | Out of VRAM | Free VRAM or use smaller model |
| `TransferFailed` | H2D copy failed | Check CUDA installation |
| `MappingError` | Tensor not found | Verify GGUF file |

---

## Performance Tuning

### Optimize Latency

```rust
// Use lower precision (future)
let config = ModelConfig {
    dtype: DataType::FP16,  // or INT8
    // ...
};

// Reduce context length
let max_context = 2048;  // Instead of 32768

// Use smaller model
let model = QwenWeightLoader::load_to_vram(...)?;  // Instead of Phi-3
```

### Optimize Throughput

```rust
// Batch requests (future)
let config = AdapterForwardConfig {
    batch_size: 4,  // Process 4 prompts simultaneously
    // ...
};

// Use larger model for quality
let model = Phi3WeightLoader::load_to_vram(...)?;
```

---

## Getting Help

### Information to Provide

When reporting issues, include:

1. **System info**:
   ```bash
   nvidia-smi
   nvcc --version
   rustc --version
   ```

2. **Model info**:
   ```rust
   println!("Model: {:?}", adapter.model_type());
   println!("VRAM: {} MB", adapter.vram_usage()? / (1024 * 1024));
   ```

3. **Error message**: Full error with stack trace

4. **Minimal reproduction**: Smallest code that reproduces issue

### Resources

- **Documentation**: `/home/vince/Projects/llama-orch/bin/worker-orcd/.docs/llama/`
- **Spec**: `bin/.specs/01_M0_worker_orcd.md`
- **Tests**: `tests/llama_integration_suite.rs`

---

**Status**: Complete  
**Coverage**: 15+ common issues  
**Last Updated**: 2025-10-05
