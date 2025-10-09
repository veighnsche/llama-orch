# Test Implementation Guide
**Date:** 2025-10-07T12:53Z  
**Purpose:** Guide for implementing comprehensive tests to address €800 in fines  
**Status:** 🚧 TESTS CREATED (Implementation pending)

---

## Overview

I've created comprehensive test suites to address the €800 in TEAM_PEAR fines for insufficient testing. The tests are currently marked `#[ignore]` because they require infrastructure that may not be fully implemented yet.

**Test Files Created:**
1. `tests/tokenization_verification.rs` - Addresses €500 Phase 1 fines
2. `tests/cublas_comprehensive_verification.rs` - Addresses €300 Phase 2 fines

---

## Phase 1: Tokenization Tests (€500)

### File: `tests/tokenization_verification.rs`

#### Test 1: `test_chat_template_special_tokens` (€150)
**Addresses:** Test bypass fine - previous test used `use_chat_template=false`

**Status:** ✅ Ready to run (may crash if special tokens cause issues)

**How to run:**
```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd
cargo test --test tokenization_verification test_chat_template_special_tokens --ignored -- --nocapture
```

**What it does:**
- Starts worker with FP16 model
- Sends prompt that will trigger chat template
- Special tokens 151644 and 151645 should be processed
- Test passes if it completes without crashing

**Expected outcome:**
- If special tokens work: Test passes ✅
- If special tokens crash: Test fails, revealing the actual bug ❌

**Implementation needed:** None - test is ready to run

---

#### Test 2: `test_verify_special_token_ids` (€100)
**Addresses:** Hardcoded magic numbers without source verification

**Status:** ⚠️ Requires tokenizer introspection API

**What it needs:**
```rust
// Need to expose tokenizer vocab from GGUF file
pub fn get_tokenizer_vocab(model_path: &str, start_token: u32, end_token: u32) -> Vec<(u32, String)>
```

**Implementation approach:**
1. Load GGUF file
2. Parse tokenizer section
3. Extract vocab for tokens 151640-151650
4. Return list of (token_id, token_string) pairs

**Expected verification:**
- Token 151643 exists
- Token 151644 = "<|im_start|>"
- Token 151645 = "<|im_end|>"

**Where to implement:**
- `bin/worker-crates/worker-gguf/src/parser.rs` - Add vocab extraction
- Expose through public API

---

#### Test 3: `test_dump_embeddings_from_vram` (€200)
**Addresses:** Unverified embeddings (only in comments, never dumped from VRAM)

**Status:** ⚠️ Requires CUDA memory introspection API

**What it needs:**
```rust
// Need to dump tensor data from GPU memory
pub fn dump_tensor_from_vram(
    model: &CudaModel,
    tensor_name: &str,
    start_idx: usize,
    count: usize
) -> Vec<f16>
```

**Implementation approach:**
1. Get GPU pointer for `token_embd.weight`
2. Use `cudaMemcpy` to copy data from device to host
3. Extract values for tokens 151643-151645
4. Return as Vec<f16>

**Expected verification:**
- Token 151643: [0.0031, 0.0067, 0.0078, ...] (first 10 values)
- Token 151644: [0.0014, -0.0084, 0.0073, ...]
- Token 151645: [0.0029, -0.0117, 0.0049, ...]
- Values are NOT all zeros
- Values are NOT NaN/Inf

**Where to implement:**
- `bin/worker-orcd/src/cuda/model.rs` - Add memory dump method
- Use existing `cuda_memcpy_device_to_host` if available

---

#### Test 4: `test_create_llamacpp_reference` (€50)
**Addresses:** Non-existent reference file cited

**Status:** ⚠️ Requires llama.cpp installed

**What it needs:**
- llama.cpp binary in PATH or at known location
- Script to run llama.cpp and capture output

**Implementation approach:**
```bash
#!/bin/bash
# Create llama.cpp reference output

MODEL="/home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-fp16.gguf"
OUTPUT=".archive/llama_cpp_debug.log"

mkdir -p .archive

./reference/llama.cpp/main \
  -m "$MODEL" \
  -p "Write a haiku about GPU computing" \
  -n 50 \
  --log-disable \
  --verbose-prompt \
  > "$OUTPUT" 2>&1

echo "✅ Reference output saved to $OUTPUT"
```

**Expected output:**
- File `.archive/llama_cpp_debug.log` created
- Contains llama.cpp's tokenization and generation output
- Can be used for comparison with our implementation

**Where to implement:**
- Create script at `bin/worker-orcd/scripts/create_llamacpp_reference.sh`
- Call from test using `std::process::Command`

---

## Phase 2: cuBLAS Tests (€300)

### File: `tests/cublas_comprehensive_verification.rs`

#### Test 1: `test_q_projection_comprehensive` (€100)
**Addresses:** Incomplete verification (0.11% coverage - only Q[0])

**Status:** ⚠️ Requires manual verification infrastructure

**What it needs:**
```rust
// Need to run manual FP32 calculation and compare with cuBLAS
pub fn verify_q_projection(
    model: &CudaModel,
    token_idx: usize,
    layer_idx: usize,
    elements_to_check: &[usize]
) -> Vec<(usize, f32, f32, f32)> // (idx, manual, cublas, diff)
```

**Implementation approach:**
1. Load model to GPU
2. Run Q projection for specified token and layer
3. For each element in `elements_to_check`:
   - Calculate expected value using manual FP32 matmul
   - Read actual value from cuBLAS output
   - Compare with tolerance ±0.001
4. Return results

**Elements to check:**
- Q[0, 100, 200, 300, 400, 500, 600, 700, 800] (9 elements = 1% coverage)
- Across tokens 0, 1, 2 (3 tokens)
- Total: 27 verifications = 3% coverage (30x better than 0.11%)

**Where to implement:**
- `bin/worker-orcd/tests/manual_verification.rs` - Helper functions
- Use existing `tests/verify_manual_q0.rs` as template

---

#### Tests 2-8: K, V, Attn Output, FFN Gate/Up/Down, LM Head
**Addresses:** These projections were NOT verified at all

**Status:** ⚠️ Same infrastructure as Test 1

**Implementation:** Same approach as Q projection, but for different matmuls

**Coverage target:** 3% per matmul (27 elements across 3 tokens)

---

#### Test 9: `test_cublas_parameter_comparison` (€100)
**Addresses:** Unproven difference (no side-by-side parameter comparison)

**Status:** ⚠️ Requires cuBLAS call introspection

**What it needs:**
```rust
// Need to log cuBLAS parameters for each call
pub struct CublasCallParams {
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub lda: i32,
    pub ldb: i32,
    pub ldc: i32,
    pub transa: cublasOperation_t,
    pub transb: cublasOperation_t,
    pub alpha: f32,
    pub beta: f32,
    pub compute_type: cublasComputeType_t,
}

pub fn get_cublas_params(matmul_name: &str) -> CublasCallParams
```

**Implementation approach:**
1. Add logging to each cuBLAS call in `qwen_transformer.cpp`
2. Capture parameters for all 8 matmuls
3. Compare against documented parameters from other teams
4. Create side-by-side comparison table

**Where to implement:**
- `cuda/src/transformer/qwen_transformer.cpp` - Add parameter logging
- Create comparison table in test output

---

#### Test 10: `test_cublas_multi_layer_verification` (€100)
**Addresses:** Only layer 0 was verified

**Status:** ⚠️ Same infrastructure as Test 1

**Implementation:** Run verification for layers 0, 1, 2 (not just layer 0)

---

## Implementation Priority

### High Priority (Can implement now)
1. ✅ **test_chat_template_special_tokens** - Ready to run
2. **test_create_llamacpp_reference** - Just needs bash script

### Medium Priority (Requires new APIs)
3. **test_verify_special_token_ids** - Tokenizer introspection
4. **test_dump_embeddings_from_vram** - CUDA memory dump

### Low Priority (Requires significant infrastructure)
5. **All cuBLAS comprehensive tests** - Manual verification framework

---

## Running the Tests

### Run all tokenization tests:
```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd
cargo test --test tokenization_verification --ignored -- --nocapture
```

### Run all cuBLAS tests:
```bash
cargo test --test cublas_comprehensive_verification --ignored -- --nocapture
```

### Run specific test:
```bash
cargo test --test tokenization_verification test_chat_template_special_tokens --ignored -- --nocapture
```

---

## Expected Outcomes

### If Tests Pass ✅
- Tokenization with chat template works correctly
- Special tokens don't cause crashes
- Embeddings are valid
- cuBLAS operations are mathematically correct

### If Tests Fail ❌
- Reveals actual bugs that were masked by insufficient testing
- Provides specific failure points for debugging
- Shows where the garbage token bug actually is

---

## Test Coverage Goals

### Phase 1: Tokenization
- **Before:** 0% (test bypassed special tokens)
- **After:** 100% (full chat template path tested)

### Phase 2: cuBLAS
- **Before:** 0.11% (only Q[0] verified)
- **After:** 3% per matmul (27 elements × 8 matmuls = 216 verifications)
- **Improvement:** 30x better coverage

---

## Next Steps

1. **Immediate:** Run `test_chat_template_special_tokens` to see if special tokens work
2. **Short-term:** Implement tokenizer vocab introspection
3. **Medium-term:** Implement CUDA memory dump for embeddings
4. **Long-term:** Build manual verification framework for cuBLAS

---

## References

**Fines:**
- `test-harness/TEAM_PEAR_VERIFICATION.md` - Phase 1 & 2 fines
- `bin/worker-orcd/investigation-teams/TEAM_PEAR/FINES_LEDGER.csv` - Complete ledger

**Existing Tests:**
- `tests/haiku_generation_anti_cheat.rs` - Current haiku test (bypasses chat template)
- `tests/verify_manual_q0.rs` - Manual Q[0] verification (0.11% coverage)

**Code to Fix:**
- `src/inference/cuda_backend.rs:231` - `use_chat_template = false` (should be true)
- `cuda/src/transformer/qwen_transformer.cpp` - cuBLAS calls to verify

---

**Status:** Tests created, implementation guide complete  
**Next:** Run `test_chat_template_special_tokens` to test special token handling  
**By:** Testing Team 🔍
