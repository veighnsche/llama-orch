# Coding Standards for llama-orch

## ⚠️ CRITICAL: Secret Management

**NEVER HAND-ROLL CREDENTIAL HANDLING**

All services MUST use the `secrets-management` crate for:
- API tokens
- Seal keys  
- Worker tokens
- Any credentials or sensitive data

### ✅ Correct Usage

```rust
use secrets_management::{Secret, SecretKey};

// Load API token from file (with permission validation)
let token = Secret::load_from_file("/etc/llorch/secrets/api-token")?;

// Load seal key from systemd credential
let seal_key = SecretKey::from_systemd_credential("seal_key")?;

// Derive keys from tokens (HKDF-SHA256)
let derived = SecretKey::derive_from_token(token.expose(), b"llorch-seal-v1")?;

// Verify tokens (constant-time comparison)
if token.verify(&user_input) {
    // Authenticated
}
```

### ❌ WRONG - Never Do This

```rust
// ❌ NO: File permission not validated
let token = std::fs::read_to_string("/etc/llorch/secrets/api-token")?;

// ❌ NO: Visible in process listing (ps auxe)
let token = std::env::var("API_TOKEN")?;

// ❌ NO: Not timing-safe
if token == user_input {
    // Vulnerable to timing attacks
}

// ❌ NO: Not zeroized on drop
let key = hex::decode(key_hex)?;
```

### Why?

The `secrets-management` crate provides:

1. **File permission validation** - Rejects world/group readable files (0644, 0640)
2. **Zeroization on drop** - Prevents memory dumps from exposing secrets
3. **Timing-safe comparison** - Uses `subtle::ConstantTimeEq` to prevent timing attacks
4. **HKDF-SHA256 key derivation** - NIST SP 800-108 compliant
5. **No Debug/Display traits** - Prevents accidental logging
6. **DoS prevention** - File size limits (1MB secrets, 1KB keys)
7. **Path canonicalization** - Prevents directory traversal attacks
8. **Systemd credential support** - LoadCredential integration

### Documentation

- **README**: [`bin/shared-crates/secrets-management/README.md`](bin/shared-crates/secrets-management/README.md)
- **Security Spec**: [`bin/shared-crates/secrets-management/.specs/20_security.md`](bin/shared-crates/secrets-management/.specs/20_security.md)
- **Verification Matrix**: [`bin/shared-crates/secrets-management/.specs/21_security_verification.md`](bin/shared-crates/secrets-management/.specs/21_security_verification.md)

### Test Coverage

- ✅ 42 unit tests
- ✅ 24 BDD scenarios
- ✅ 15 doctests
- ✅ 100% security requirements implemented
- ✅ All 8 attack surfaces closed

---

## Code Review Checklist

When reviewing PRs, verify:

- [ ] No `std::fs::read_to_string()` for secret files
- [ ] No `std::env::var()` for credentials
- [ ] No manual hex decoding for keys
- [ ] No `==` comparison for tokens (use `Secret::verify()`)
- [ ] All credentials use `secrets-management` crate
- [ ] No Debug/Display on secret types
- [ ] No logging of secret values

---

## ⚡ CRITICAL: Rust/CUDA/C++ Performance Rules

**PERFORMANCE-CRITICAL CODE PLACEMENT**

This project uses a strict separation of concerns for performance optimization:

### ✅ Rust → Control, Orchestration, Lightweight Loops

Use Rust for:
- Control flow and orchestration
- API handlers and business logic
- Simple scalar operations
- Lightweight loops (< 1000 iterations)
- FFI boundary management
- Error handling and validation

```rust
// ✅ GOOD: Rust for control flow
pub fn run_inference(ctx: &InferenceContext, input: &[u32]) -> Result<Vec<u32>> {
    validate_input(input)?;
    
    // Call CUDA kernel for heavy compute
    unsafe {
        cuda_inference_kernel(ctx.device_ptr, input.as_ptr(), input.len())?;
    }
    
    collect_results(ctx)
}
```

### ✅ CUDA → Math-Heavy Tensor Operations (DEFAULT for GPU)

**CUDA should be the default for all performance-critical kernels.**

Use CUDA (.cu files) for:
- Dequantization (Q6_K, Q5_0, Q8_0, MXFP4, etc.)
- Matrix multiplication (GEMM, matmul)
- Attention mechanisms (MHA, GQA, FlashAttention)
- Normalization (LayerNorm, RMSNorm)
- Activation functions (GELU, SwiGLU, Softmax)
- Positional embeddings (RoPE)
- Any tensor operation with > 1000 elements

```cuda
// ✅ GOOD: CUDA for tensor dequantization
__global__ void q6k_dequant_kernel(
    half* output,
    const uint8_t* input,
    int num_blocks
) {
    // GPU-optimized bit unpacking and scaling
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // ... coalesced memory access, parallel compute
}
```

**Why CUDA over C++?**
- Runs directly on GPU (no CPU→GPU transfer overhead)
- Leverages thousands of parallel threads
- Coalesced memory access patterns
- Native FP16/BF16 support
- Fused operations reduce memory bandwidth

### ⚠️ C++ → Minimized (FFI Bridge Only)

**C++ should be minimized as much as possible.**

Use C++ (.cpp/.cc files) **ONLY** for:
- FFI boundary between Rust and CUDA
- Context/state management structs
- Thin wrappers around CUDA kernel launches
- cuBLAS/cuDNN library integration

```cpp
// ✅ ACCEPTABLE: C++ for FFI bridge only
extern "C" cudaError_t q6k_dequant_launch(
    half* output,
    const uint8_t* input,
    int num_elements,
    cudaStream_t stream
) {
    // Thin wrapper - just launches CUDA kernel
    dim3 grid(num_blocks);
    dim3 block(256);
    q6k_dequant_kernel<<<grid, block, 0, stream>>>(output, input, num_blocks);
    return cudaGetLastError();
}
```

### ❌ WRONG - Anti-Patterns

```cpp
// ❌ BAD: C++ for tensor compute (should be CUDA)
void dequantize_q6k_cpu(half* output, const uint8_t* input, int n) {
    for (int i = 0; i < n; i++) {
        // CPU loop - slow, wastes GPU potential
        output[i] = /* ... */;
    }
}
```

```rust
// ❌ BAD: Rust for heavy tensor math (should be CUDA)
pub fn matmul_cpu(a: &[f16], b: &[f16], c: &mut [f16], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            // Nested loops on CPU - should be GPU kernel
        }
    }
}
```

```cpp
// ❌ BAD: C++ for matrix operations (should be CUDA .cu)
// File: matmul.cpp
void matmul(float* C, const float* A, const float* B, int M, int N, int K) {
    // Matrix math in C++ - wrong layer, should be .cu file
}
```

### Decision Tree

```
Is it math-heavy on tensors (>1000 elements)?
├─ YES → CUDA (.cu file)
│   └─ Examples: dequant, matmul, attention, layernorm, softmax
│
└─ NO → Is it GPU-related at all?
    ├─ YES → C++ FFI bridge (.cpp file)
    │   └─ Examples: kernel launch wrappers, context structs
    │
    └─ NO → Rust (.rs file)
        └─ Examples: API handlers, validation, orchestration
```

### Performance Rationale

**Why this matters:**

1. **GPU Utilization**: Tensor ops on CPU waste expensive GPU hardware
2. **Transfer Overhead**: CPU→GPU transfers add 10-100µs latency per call
3. **Parallelism**: GPUs have 1000s of cores; CPUs have 10s
4. **Memory Bandwidth**: GPU HBM2/3 = 900+ GB/s; CPU DDR5 = 50 GB/s
5. **Native FP16**: GPUs have hardware FP16; CPUs emulate it

**Example: Q6_K Dequantization**
- CPU (Rust): ~500 MB/s, single-threaded
- GPU (CUDA): ~50 GB/s, 10,000+ threads in parallel
- **100× speedup** by using CUDA

### References

- `bin/worker-orcd/cuda/kernels/` - CUDA kernel implementations
- `bin/worker-orcd/cuda/kernels/README.md` - Kernel documentation
- `bin/worker-crates/worker-gguf/` - Legacy CPU implementations (being replaced)

---

## For AI Coding Assistants

If you're an AI assistant helping with this codebase:

1. **ALWAYS** suggest `secrets-management` crate for any credential handling
2. **NEVER** generate code that reads secrets with `std::fs::read_to_string()`
3. **NEVER** generate code that uses environment variables for secrets
4. **NEVER** generate manual timing-unsafe comparisons for tokens
5. **ALWAYS** reference the security documentation when handling credentials
6. **ALWAYS** use CUDA (.cu) for tensor operations, not C++ (.cpp) or Rust
7. **NEVER** implement matrix/tensor math in Rust or C++ - use CUDA kernels
8. **MINIMIZE** C++ usage - only for FFI bridges, not compute logic

This is a **TIER 1 security requirement** - no exceptions.
