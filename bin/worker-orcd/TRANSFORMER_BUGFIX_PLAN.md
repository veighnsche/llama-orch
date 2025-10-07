# Transformer Inference Bugfix Plan - Reference Comparison Strategy

**Date**: 2025-10-07  
**Status**: ACTIVE  
**Priority**: CRITICAL  
**Bug**: Transformer generates garbage tokens despite working infrastructure

---

## Executive Summary

**Problem**: Worker-orcd produces garbage tokens (mojibake, repetitive tokens, wrong language) despite:
- ✅ Infrastructure works (test runs without crashes)
- ✅ Tokenization verified
- ✅ Matrix layout fixed
- ✅ KV cache verified
- ✅ Sampling logic correct

**Root Cause**: Transformer forward pass has numerical bugs in C++/CUDA implementation

**Solution**: Systematic comparison with proven reference implementations

**⚠️ RECOMMENDED ALTERNATIVE**: See `GPT2_FIRST_STRATEGY.md` for a simpler approach - debug with GPT-2 (simpler architecture) first, then expand to Qwen2. This plan focuses on direct Qwen2 debugging.

---

## Reference Selection Analysis

### Available References in `/home/vince/Projects/llama-orch/reference/`

| Reference | Language | Qwen2 Support | Quality | Recommendation |
|-----------|----------|---------------|---------|----------------|
| **mistral.rs** | Rust | ✅ Full | ⭐⭐⭐⭐⭐ | **PRIMARY** |
| **candle** | Rust | ✅ Full | ⭐⭐⭐⭐⭐ | **PRIMARY** |
| vllm | Python | ✅ Full | ⭐⭐⭐⭐ | Secondary |
| text-generation-inference | Python/Rust | ✅ Full | ⭐⭐⭐⭐ | Secondary |
| llama.cpp | C++ | ❌ Excluded | ⭐⭐⭐ | **FORBIDDEN** |
| llamafile | C++ | ❌ Excluded | ⭐⭐⭐ | Not suitable |
| drama_llama | Unknown | ❓ Unknown | ⭐⭐ | Not suitable |
| tinygrad | Python | ❓ Unknown | ⭐⭐ | Not suitable |
| flash-attention | CUDA | ❌ Kernels only | ⭐⭐⭐⭐ | Not applicable |

### Why NOT llama.cpp?

**From spec**: `/reference/README.md` line 16-26:
> "We started with that and everything went messed up"

**Reasons**:
1. Already used as initial reference - led to current bugs
2. C++ codebase - harder to trace logic
3. Complex macro-heavy code
4. `/NO_LLAMA_CPP.md` policy

### Why mistral.rs and candle?

**mistral.rs** (`/reference/mistral.rs/mistralrs-core/src/models/qwen2.rs`):
- ✅ **Rust-native** - Clean, readable, type-safe
- ✅ **Production-grade** - Used in real deployments
- ✅ **Qwen2 support** - Full implementation with GQA, RoPE, RMSNorm
- ✅ **Quantization support** - Handles Q4_K_M like we do
- ✅ **Well-documented** - Clear variable names, good structure
- ✅ **Active maintenance** - Recent commits, good practices

**candle** (`/reference/candle/candle-transformers/src/models/qwen3.rs`):
- ✅ **Rust ML framework** - HuggingFace official Rust framework
- ✅ **Clean abstractions** - Easy to understand tensor operations
- ✅ **Qwen3 implementation** - Similar to Qwen2.5 architecture
- ✅ **Educational quality** - Designed for learning
- ✅ **Minimal dependencies** - Pure Rust, easy to trace

**Both are Rust** → Easy to read, understand, and compare with our Rust layer

---

## Current Bug Status

### What Works ✅
- HTTP server and SSE streaming
- Model loading (GGUF parsing, VRAM allocation)
- Tokenization (encode/decode)
- Sampling (temperature, argmax)
- KV cache infrastructure
- Matrix layout (row-major vs column-major fixed)
- cuBLAS operations (dimensions correct)

### What's Broken ❌
- **Transformer forward pass produces garbage logits**
- Symptoms:
  - Mojibake: `è®«æŁ¥æī¾`, `ĠLích`, `ĠKw`
  - Repetitive tokens: Same token 10+ times
  - Wrong language: Chinese/Thai/Korean tokens
  - Code tokens: `FileWriter`, `strcasecmp`, `Operator`
  - High token IDs: 119578, 109547, 120042 (near vocab limit)

### Investigation History (From `haiku_generation_anti_cheat.rs`)

**Teams that investigated**:
1. **TEAM_CHAIR** - Fixed infrastructure (chat template bypass)
2. **TEAM_GREEN** - Identified logits corruption (not sampling)
3. **TEAM_HOTEL** - Fixed cuBLAS dimensions
4. **TEAM_SEA** - Verified sampling correct
5. **TEAM_WATER** - Verified KV cache correct
6. **TEAM_CHARLIE** - Verified RMSNorm correct
7. **TEAM_THIMBLE** - Tested transpose flags (no change)
8. **TEAM_TOP_HAT** - Tested compute types (no change)
9. **TEAM_BATTLESHIP** - Investigated downstream wiring
10. **TEAM_RACE_CAR** - Investigated FFN down projection
11. **TEAM_PAPER_CUTTER** - Last block FFN parity
12. **TEAM_PLOTTER** - Attention output projection parity

**Current state**: All standard hypotheses eliminated. Bug is deeper.

### Debugging Comments in Code

**File**: `/home/vince/Projects/llama-orch/bin/worker-orcd/cuda/src/transformer/qwen_transformer.cpp`

**Lines 1-200**: Multiple team investigations with guard macros:
- `THIMBLE_PRETRANSPOSE_EXPERIMENT` - Transpose testing
- `BATTLESHIP_CANARIES` - Buffer integrity checks
- `BATTLESHIP_ATTN_PROJ_AUDIT` - Attention projection audit
- `RACECAR_FFN_TRACE` - FFN parity logging
- `PAPER_CUTTER_LAST_BLOCK_TRACE` - Last block FFN trace
- `PLOTTER_WO_TRACE` - Attention output projection trace

**Problem**: Too many guard macros, too much debugging code, no clear path forward

---

## The Plan: Systematic Reference Comparison

### Phase 1: Reference Code Study (2-3 hours)

**Goal**: Understand how mistral.rs and candle implement Qwen2 transformer

**Tasks**:

1. **Study mistral.rs Qwen2 implementation**
   - File: `/reference/mistral.rs/mistralrs-core/src/models/qwen2.rs`
   - Focus areas:
     - Attention mechanism (lines 61-250)
     - MLP/FFN structure (lines 250-350)
     - Layer structure and residual connections
     - RoPE application
     - RMSNorm placement
     - Tensor shapes and reshaping logic

2. **Study candle Qwen3 implementation**
   - File: `/reference/candle/candle-transformers/src/models/qwen3.rs`
   - Focus areas:
     - `Qwen3Attention::forward()` (lines 114-200)
     - `Qwen3MLP::forward()` (lines 85-91)
     - `Qwen3RotaryEmbedding::apply()` (lines 55-64)
     - Tensor dimension flow

3. **Document key differences**
   - Create comparison table: mistral.rs vs candle vs our implementation
   - Identify potential bugs in our code

**Deliverable**: `REFERENCE_COMPARISON_NOTES.md`

---

### Phase 2: Critical Path Identification (1 hour)

**Goal**: Identify the 3-5 most likely bug locations

**Method**: Compare our implementation against both references

**Focus Areas** (in priority order):

1. **Attention Output Projection** (`W_o`)
   - **Why**: TEAM_PLOTTER investigating this
   - **Check**: 
     - Tensor shapes before/after projection
     - Head concatenation order
     - Transpose flags in cuBLAS call
     - lda/ldb/ldc stride parameters

2. **FFN Down Projection** (`ffn_down`)
   - **Why**: TEAM_RACE_CAR and TEAM_PAPER_CUTTER investigating this
   - **Check**:
     - Weight loading (is `ffn_down` actually loaded?)
     - cuBLAS parameters (M, N, K, lda, ldb, ldc)
     - Transpose flags
     - Input/output shapes

3. **RoPE Application**
   - **Why**: RoPE bugs cause position-independent outputs
   - **Check**:
     - Frequency calculation
     - Sin/cos application order
     - Dimension splitting (real/imaginary parts)
     - Position offset handling

4. **Residual Connections**
   - **Why**: TEAM_BATTLESHIP investigating bypass
   - **Check**:
     - Addition order (pre-norm vs post-norm)
     - Buffer aliasing
     - In-place vs out-of-place operations

5. **RMSNorm Placement**
   - **Why**: Norm placement affects gradient flow
   - **Check**:
     - Pre-attention norm vs post-attention norm
     - Pre-FFN norm vs post-FFN norm
     - Epsilon value

**Deliverable**: `CRITICAL_BUGS_HYPOTHESIS.md` with top 3 suspects

---

### Phase 3: Targeted Fixes (2-4 hours per bug)

**Goal**: Fix bugs one at a time, verify with test

**Process for each bug**:

1. **Isolate the bug**
   - Add minimal logging (first 8 values, min/max/mean)
   - Compare with reference implementation output
   - Identify exact divergence point

2. **Implement fix**
   - Make minimal, surgical change
   - Follow reference implementation exactly
   - Document change in commit message

3. **Verify fix**
   - Run haiku test: `cargo test --test haiku_generation_anti_cheat -- --ignored`
   - Check output quality
   - Verify minute word appears in output

4. **Clean up**
   - Remove debugging code
   - Remove guard macros
   - Simplify implementation

**Success Criteria**:
- ✅ Haiku test passes
- ✅ Minute word found in output
- ✅ No mojibake or repetitive tokens
- ✅ Coherent English text

---

### Phase 4: Comprehensive Verification (1 hour)

**Goal**: Ensure fix works across different scenarios

**Tests**:

1. **Basic inference test**
   ```bash
   cargo test --test haiku_generation_anti_cheat -- --ignored
   ```

2. **Multiple prompts**
   - Short prompt (10 tokens)
   - Medium prompt (50 tokens)
   - Long prompt (100 tokens)

3. **Different temperatures**
   - Temperature 0.0 (greedy)
   - Temperature 0.7 (default)
   - Temperature 1.5 (creative)

4. **Reproducibility**
   - Same seed → same output (2 runs)

**Deliverable**: Test results showing all scenarios pass

---

## Comparison Methodology

### Overview: Three-Way Comparison Strategy

We have **C++/CUDA implementation** but **Rust/Python references**. Strategy:

1. **Understand reference logic** (Rust/Python) - What should happen
2. **Map to our C++/CUDA** - What we're actually doing  
3. **Compare intermediate values** - Where do we diverge

**Available references**:
- **Rust**: mistral.rs, candle (clean, readable)
- **Python**: vllm, text-generation-inference (high-level logic)
- **C++/CUDA**: We can examine CUDA kernels in references for low-level details

### How to Compare Implementations

**Step 1: Map our code to reference code**

Our code structure:
```
qwen_transformer.cpp
├── forward() - Main entry point
├── attention() - Attention mechanism
│   ├── Q/K/V projections (cuBLAS)
│   ├── RoPE application
│   ├── Attention scores (softmax)
│   └── Output projection (W_o)
└── ffn() - Feed-forward network
    ├── Gate projection
    ├── Up projection
    ├── SwiGLU activation
    └── Down projection
```

mistral.rs structure:
```rust
qwen2.rs
├── Qwen2Model::forward()
├── Qwen2DecoderLayer::forward()
│   ├── Attention::forward()
│   │   ├── q_proj, k_proj, v_proj
│   │   ├── RotaryEmbedding::forward()
│   │   ├── Sdpa::run_attention()
│   │   └── o_proj
│   └── Mlp::forward()
│       ├── gate_proj
│       ├── up_proj
│       └── down_proj
```

**Step 2: Compare tensor shapes at each stage**

Example for attention:
```
Stage                  | mistral.rs shape      | Our shape          | Match?
-----------------------|-----------------------|--------------------|-------
Input                  | [B, L, H]             | [B, L, H]          | ✅
After Q projection     | [B, L, num_heads*D]   | [B, L, num_heads*D]| ✅
After reshape          | [B, num_heads, L, D]  | [B, num_heads, L, D]| ?
After RoPE             | [B, num_heads, L, D]  | [B, num_heads, L, D]| ?
After attention        | [B, num_heads, L, D]  | [B, num_heads, L, D]| ?
After concat           | [B, L, num_heads*D]   | [B, L, num_heads*D]| ?
After O projection     | [B, L, H]             | [B, L, H]          | ?
```

**Step 3: Compare cuBLAS parameters**

Example for Q projection:
```
Parameter    | mistral.rs (via candle) | Our cuBLAS call | Match?
-------------|-------------------------|-----------------|-------
M            | seq_len                 | seq_len         | ?
N            | num_heads * head_dim    | num_heads * D   | ?
K            | hidden_size             | hidden_size     | ?
lda          | hidden_size             | ?               | ?
ldb          | hidden_size             | ?               | ?
ldc          | num_heads * head_dim    | ?               | ?
opA          | CUBLAS_OP_N             | ?               | ?
opB          | CUBLAS_OP_T             | ?               | ?
```

**Step 4: Compare numerical values**

Add logging to dump first 8 values at each stage:
```cpp
// Our code
fprintf(stderr, "[OUR] After Q proj: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n",
        q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]);
```

Run mistral.rs with same model and prompt, capture output:
```bash
# Run mistral.rs with debug logging
RUST_LOG=debug mistralrs-server --model qwen2.5-0.5b-instruct
```

Compare values - should match within tolerance (1e-3 for FP16)

---

## Key Files to Examine

### Our Implementation

**Main transformer file**:
- `/home/vince/Projects/llama-orch/bin/worker-orcd/cuda/src/transformer/qwen_transformer.cpp`
  - Lines 1-200: Debugging comments and guard macros
  - Lines 200+: Actual implementation (need to read)

**Weight loader**:
- `/home/vince/Projects/llama-orch/bin/worker-orcd/cuda/src/cuda/weight_loader.cpp`
  - Verify all weights loaded correctly

**Test file**:
- `/home/vince/Projects/llama-orch/bin/worker-orcd/tests/haiku_generation_anti_cheat.rs`
  - Lines 49-72: Current status and progress
  - Lines 248-309: Investigation history

### Reference Implementations

**mistral.rs**:
- `/reference/mistral.rs/mistralrs-core/src/models/qwen2.rs` (775 lines)
  - Lines 48-133: `Attention` struct and implementation
  - Lines 135-250: `Attention::forward()` method
  - Lines 252-350: `Mlp` struct and implementation

**candle**:
- `/reference/candle/candle-transformers/src/models/qwen3.rs` (390 lines)
  - Lines 30-64: `Qwen3RotaryEmbedding` implementation
  - Lines 66-91: `Qwen3MLP` implementation
  - Lines 93-200: `Qwen3Attention` implementation

**vllm** (Python, for high-level logic):
- `/reference/vllm/vllm/model_executor/models/qwen2.py` (544 lines)
  - Lines 74-108: `Qwen2MLP` implementation
  - Lines 111-200: `Qwen2Attention` implementation

---

## Expected Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1: Reference Study | 2-3 hours | Comparison notes |
| Phase 2: Bug Identification | 1 hour | Top 3 hypotheses |
| Phase 3: Fix Implementation | 2-4 hours per bug | Working transformer |
| Phase 4: Verification | 1 hour | Test results |
| **Total** | **8-16 hours** | **Bug fixed** |

---

## Success Criteria

**Minimum Success** (M0 requirement):
- ✅ Haiku test passes
- ✅ Minute word found in output
- ✅ No crashes or errors
- ✅ Coherent text output

**Full Success**:
- ✅ All above
- ✅ No mojibake
- ✅ No repetitive tokens
- ✅ Correct language (English)
- ✅ Contextually appropriate output
- ✅ Reproducible with same seed

---

## Risk Mitigation

**Risk 1: Multiple bugs compound**
- **Mitigation**: Fix one bug at a time, verify after each fix

**Risk 2: Reference implementations differ from GGUF format**
- **Mitigation**: Use multiple references (mistral.rs + candle + vllm)

**Risk 3: Bug is in CUDA kernels, not C++ logic**
- **Mitigation**: Compare intermediate values, not just final output

**Risk 4: Time estimate too optimistic**
- **Mitigation**: Focus on top 3 bugs first, defer others if needed

---

## Next Steps

**Immediate actions**:

1. **Read this plan** - Understand the strategy
2. **Start Phase 1** - Study mistral.rs Qwen2 implementation
3. **Take notes** - Document findings in `REFERENCE_COMPARISON_NOTES.md`
4. **Identify bugs** - Create `CRITICAL_BUGS_HYPOTHESIS.md`
5. **Fix bugs** - One at a time, verify after each

**Questions to answer**:
- What does mistral.rs do differently in attention?
- What does candle do differently in FFN?
- Where do our tensor shapes diverge?
- What are the correct cuBLAS parameters?

---

## Appendix: Why This Approach Will Work

**Evidence from investigation history**:

1. **llama.cpp produces perfect output** with same model
   - Proves: Model file is correct
   - Proves: Bug is in our code, not model

2. **Infrastructure works** (no crashes, tokens flow correctly)
   - Proves: FFI boundary correct
   - Proves: Sampling correct
   - Proves: Tokenization correct

3. **Matrix layout fixed** (Q values in correct range)
   - Proves: cuBLAS dimensions correct
   - Proves: Row-major vs column-major handled

4. **KV cache verified** (positions increment correctly)
   - Proves: Cache infrastructure correct
   - Proves: Position tracking correct

**Conclusion**: Bug is in transformer logic (attention or FFN), not infrastructure

**Why reference comparison works**:
- mistral.rs and candle are **proven implementations**
- Both produce **correct output** with Qwen2 models
- Both are **Rust** - easy to read and understand
- Both are **well-documented** - clear variable names
- Both are **actively maintained** - recent best practices

**Strategy**: Find where our implementation diverges from references, fix the divergence

---

---

## Appendix A: Step-by-Step Debugging Instructions

### Detailed Comparison Process

This section provides **concrete, actionable steps** for comparing our C++/CUDA implementation with Rust/Python references.

---

### Step 1: Set Up Comparison Environment (15 minutes)

**Goal**: Prepare tools and references for side-by-side comparison

**Actions**:

1. **Open reference implementations in editor**
   ```bash
   # mistral.rs Qwen2
   code /home/vince/Projects/llama-orch/reference/mistral.rs/mistralrs-core/src/models/qwen2.rs
   
   # candle Qwen3
   code /home/vince/Projects/llama-orch/reference/candle/candle-transformers/src/models/qwen3.rs
   
   # vllm Qwen2 (Python)
   code /home/vince/Projects/llama-orch/reference/vllm/vllm/model_executor/models/qwen2.py
   ```

2. **Open our implementation**
   ```bash
   code /home/vince/Projects/llama-orch/bin/worker-orcd/cuda/src/transformer/qwen_transformer.cpp
   ```

3. **Create comparison notes file**
   ```bash
   touch /home/vince/Projects/llama-orch/bin/worker-orcd/COMPARISON_NOTES.md
   ```

4. **Prepare test model**
   ```bash
   # Verify test model exists
   ls -lh /home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-fp16.gguf
   ```

---

### Step 2: Understand Reference Implementation (30-60 minutes)

**Goal**: Understand what the correct implementation should do

**Focus on mistral.rs** (most similar to our use case):

#### 2.1: Read Attention Implementation

**File**: `/reference/mistral.rs/mistralrs-core/src/models/qwen2.rs` lines 48-250

**Key sections to understand**:

```rust
// Line 48-59: Attention struct definition
struct Attention {
    q_proj: Arc<dyn QuantMethod>,  // Q projection
    k_proj: Arc<dyn QuantMethod>,  // K projection
    v_proj: Arc<dyn QuantMethod>,  // V projection
    o_proj: Arc<dyn QuantMethod>,  // Output projection
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    // ...
}

// Line 135-250: Attention::forward() method
fn forward(&self, xs: &Tensor, ...) -> Result<Tensor> {
    // 1. Q/K/V projections (lines 145-159)
    let mut q = MatMul.qmethod_matmul(&xs, &*self.q_proj)?;
    let mut k = MatMul.qmethod_matmul(&xs, &*self.k_proj)?;
    let mut v = MatMul.qmethod_matmul(&xs, &*self.v_proj)?;
    
    // 2. Reshape to [B, H, L, D] (lines 161-177)
    let q = q.reshape((b_sz, q_len, self.num_heads, self.head_dim))?
             .transpose(1, 2)?;
    let k = k.reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
             .transpose(1, 2)?;
    let v = v.reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
             .transpose(1, 2)?;
    
    // 3. Apply RoPE (line 179)
    let (q, k) = self.rotary_emb.forward(&q, &k, seqlen_offsets)?;
    
    // 4. Attention computation (lines 181-220)
    let mut attn_output = paged_attn.forward(&q, &k, &v, ...)?;
    
    // 5. Reshape back to [B, L, H*D] (lines 222-230)
    let attn_output = attn_output.transpose(1, 2)?
                                  .reshape((b_sz, q_len, hidden_sz))?;
    
    // 6. Output projection (lines 232-240)
    let mut attn_output = MatMul.qmethod_matmul(&attn_output, &*self.o_proj)?;
    
    return Ok(attn_output);
}
```

**Document in COMPARISON_NOTES.md**:
```markdown
## mistral.rs Attention Flow

1. Input: [B, L, H] (batch, seq_len, hidden_size)
2. Q/K/V projections: [B, L, H] → [B, L, num_heads*head_dim]
3. Reshape: [B, L, num_heads*head_dim] → [B, num_heads, L, head_dim]
4. RoPE: Apply rotation to Q and K
5. Attention: Scaled dot-product with softmax
6. Reshape: [B, num_heads, L, head_dim] → [B, L, num_heads*head_dim]
7. Output projection: [B, L, num_heads*head_dim] → [B, L, H]

Key observations:
- GQA: num_kv_heads (2) < num_heads (14), need to repeat K/V
- Transpose: (1, 2) means swap seq_len and num_heads dimensions
- RoPE: Applied after reshape, before attention
```

#### 2.2: Read MLP/FFN Implementation

**File**: `/reference/mistral.rs/mistralrs-core/src/models/qwen2.rs` lines 252-350

```rust
// MLP structure (simplified)
struct Mlp {
    gate_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
}

fn forward(&self, xs: &Tensor) -> Result<Tensor> {
    // 1. Gate projection
    let gate = MatMul.qmethod_matmul(xs, &*self.gate_proj)?;
    
    // 2. Up projection
    let up = MatMul.qmethod_matmul(xs, &*self.up_proj)?;
    
    // 3. SwiGLU: silu(gate) * up
    let gate = candle_nn::ops::silu(&gate)?;
    let xs = (gate * up)?;
    
    // 4. Down projection
    let xs = MatMul.qmethod_matmul(&xs, &*self.down_proj)?;
    
    return Ok(xs);
}
```

**Document in COMPARISON_NOTES.md**:
```markdown
## mistral.rs FFN Flow

1. Input: [B, L, H]
2. Gate projection: [B, L, H] → [B, L, intermediate_size]
3. Up projection: [B, L, H] → [B, L, intermediate_size]
4. SwiGLU: silu(gate) * up (element-wise)
5. Down projection: [B, L, intermediate_size] → [B, L, H]

Key observations:
- Two parallel projections (gate and up)
- SwiGLU = SiLU activation + element-wise multiply
- Down projection brings back to hidden_size
```

---

### Step 3: Map Our Implementation (30 minutes)

**Goal**: Understand what our C++/CUDA code is doing

**File**: `/home/vince/Projects/llama-orch/bin/worker-orcd/cuda/src/transformer/qwen_transformer.cpp`

#### 3.1: Find Our Attention Implementation

**Search for attention function**:
```bash
grep -n "attention" qwen_transformer.cpp | head -20
```

**Read the attention code** and document:

```markdown
## Our Attention Implementation

File: qwen_transformer.cpp, lines XXX-YYY

Current flow:
1. Input: [B, L, H]
2. Q/K/V projections: cuBLAS calls
   - Q: cublasSgemm(handle, opA=?, opB=?, M=?, N=?, K=?, ...)
   - K: cublasSgemm(handle, opA=?, opB=?, M=?, N=?, K=?, ...)
   - V: cublasSgemm(handle, opA=?, opB=?, M=?, N=?, K=?, ...)
3. Reshape: ??? (need to verify)
4. RoPE: ??? (need to verify)
5. Attention: ??? (need to verify)
6. Output projection: ??? (need to verify)

Questions:
- [ ] Are cuBLAS parameters correct?
- [ ] Is reshape logic correct?
- [ ] Is RoPE applied correctly?
- [ ] Is head concatenation correct?
```

#### 3.2: Find Our FFN Implementation

**Search for FFN/MLP function**:
```bash
grep -n "ffn\|mlp" qwen_transformer.cpp | head -20
```

**Document**:
```markdown
## Our FFN Implementation

File: qwen_transformer.cpp, lines XXX-YYY

Current flow:
1. Gate projection: cuBLAS call
2. Up projection: cuBLAS call
3. SwiGLU: ??? (need to verify)
4. Down projection: cuBLAS call

Questions:
- [ ] Are all three weights loaded?
- [ ] Is SwiGLU implemented correctly?
- [ ] Are cuBLAS parameters correct?
```

---

### Step 4: Compare cuBLAS Parameters (1-2 hours)

**Goal**: Verify our cuBLAS calls match reference matrix operations

#### 4.1: Understanding cuBLAS for Row-Major Data

**Key concept**: cuBLAS assumes **column-major** (Fortran), but we use **row-major** (C/C++).

**Row-major to column-major trick**:
```
Row-major: C = A @ B^T
Column-major equivalent: C^T = B @ A^T

So for cuBLAS:
- Swap A and B
- Swap opA and opB
- Swap M and N
```

#### 4.2: Q Projection Example

**Reference (mistral.rs)**:
```rust
// Q = X @ W_q^T
// X: [batch*seq, hidden_size]
// W_q: [num_heads*head_dim, hidden_size]
// Q: [batch*seq, num_heads*head_dim]
let q = MatMul.qmethod_matmul(&xs, &*self.q_proj)?;
```

**Our cuBLAS call** (should be):
```cpp
// For row-major data:
// C = A @ B^T
// A: [M, K] = [batch*seq, hidden_size]
// B: [N, K] = [num_heads*head_dim, hidden_size]
// C: [M, N] = [batch*seq, num_heads*head_dim]

cublasSgemm(
    handle,
    CUBLAS_OP_N,  // opA: A not transposed
    CUBLAS_OP_T,  // opB: B transposed (W_q^T)
    M,            // M = batch * seq_len
    N,            // N = num_heads * head_dim
    K,            // K = hidden_size
    &alpha,
    A,            // A = input [M, K]
    K,            // lda = K (leading dim of A in row-major)
    B,            // B = W_q [N, K]
    K,            // ldb = K (leading dim of B in row-major)
    &beta,
    C,            // C = output [M, N]
    N             // ldc = N (leading dim of C in row-major)
);
```

**Checklist for our code**:
```markdown
## Q Projection Verification

- [ ] M = batch * seq_len (correct?)
- [ ] N = num_heads * head_dim (correct?)
- [ ] K = hidden_size (correct?)
- [ ] opA = CUBLAS_OP_N (correct?)
- [ ] opB = CUBLAS_OP_T (correct?)
- [ ] lda = K (correct?)
- [ ] ldb = K (correct?)
- [ ] ldc = N (correct?)
- [ ] Weight pointer correct?
- [ ] Input pointer correct?
- [ ] Output pointer correct?
```

**Action**: Go through our code and fill in this checklist for:
- [ ] Q projection
- [ ] K projection
- [ ] V projection
- [ ] O projection (attention output)
- [ ] Gate projection (FFN)
- [ ] Up projection (FFN)
- [ ] Down projection (FFN)

---

### Step 5: Add Logging and Compare Values (2-3 hours)

**Goal**: Find where our values diverge from reference

#### 5.1: Add Logging to Our Code

**Strategy**: Log first 8 values + statistics (min/max/mean) at each stage

**Template**:
```cpp
void log_tensor(const char* name, float* d_tensor, int size, int layer_idx, int token_idx) {
    if (layer_idx != 0 || token_idx != 0) return;  // Only log layer 0, token 0
    
    float h_values[8];
    cudaMemcpy(h_values, d_tensor, 8 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compute statistics
    float h_all[size];
    cudaMemcpy(h_all, d_tensor, size * sizeof(float), cudaMemcpyDeviceToHost);
    float min_val = h_all[0], max_val = h_all[0], sum = 0;
    for (int i = 0; i < size; i++) {
        min_val = fmin(min_val, h_all[i]);
        max_val = fmax(max_val, h_all[i]);
        sum += h_all[i];
    }
    float mean = sum / size;
    
    fprintf(stderr, "[%s] First8: [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f] "
                    "Stats: min=%.6f, max=%.6f, mean=%.6f\n",
            name, h_values[0], h_values[1], h_values[2], h_values[3],
            h_values[4], h_values[5], h_values[6], h_values[7],
            min_val, max_val, mean);
}
```

**Add logging at key points**:
```cpp
void transformer_forward(...) {
    // 1. After input embedding
    log_tensor("INPUT", d_input, hidden_size, layer_idx, 0);
    
    // 2. After Q projection
    log_tensor("Q_PROJ", d_q, num_heads * head_dim, layer_idx, 0);
    
    // 3. After K projection
    log_tensor("K_PROJ", d_k, num_kv_heads * head_dim, layer_idx, 0);
    
    // 4. After V projection
    log_tensor("V_PROJ", d_v, num_kv_heads * head_dim, layer_idx, 0);
    
    // 5. After RoPE on Q
    log_tensor("Q_ROPE", d_q, num_heads * head_dim, layer_idx, 0);
    
    // 6. After RoPE on K
    log_tensor("K_ROPE", d_k, num_kv_heads * head_dim, layer_idx, 0);
    
    // 7. After attention (before O projection)
    log_tensor("ATTN_OUT", d_attn, num_heads * head_dim, layer_idx, 0);
    
    // 8. After O projection
    log_tensor("O_PROJ", d_o, hidden_size, layer_idx, 0);
    
    // 9. After residual 1
    log_tensor("RESIDUAL_1", d_hidden, hidden_size, layer_idx, 0);
    
    // 10. After gate projection
    log_tensor("GATE_PROJ", d_gate, intermediate_size, layer_idx, 0);
    
    // 11. After up projection
    log_tensor("UP_PROJ", d_up, intermediate_size, layer_idx, 0);
    
    // 12. After SwiGLU
    log_tensor("SWIGLU", d_swiglu, intermediate_size, layer_idx, 0);
    
    // 13. After down projection
    log_tensor("DOWN_PROJ", d_down, hidden_size, layer_idx, 0);
    
    // 14. After residual 2
    log_tensor("RESIDUAL_2", d_hidden, hidden_size, layer_idx, 0);
}
```

#### 5.2: Run Test and Capture Output

```bash
cd /home/vince/Projects/llama-orch
cargo test --test haiku_generation_anti_cheat -- --ignored 2>&1 | tee debug_output.log
```

#### 5.3: Add Logging to Reference (Optional)

If you want to compare exact values, add similar logging to candle:

```rust
// In candle Qwen3 implementation
let q = self.q_proj.forward(&xs)?;
println!("[Q_PROJ] First8: {:?}", &q.to_vec1::<f32>()?[..8]);
```

Run candle with same model and prompt, compare outputs.

#### 5.4: Compare Values

**Create comparison table**:
```markdown
## Value Comparison (Layer 0, Token 0)

| Stage | Our Values | Reference Values | Difference | Status |
|-------|-----------|------------------|------------|--------|
| INPUT | [0.123, 0.234, ...] | [0.123, 0.234, ...] | <1e-6 | ✅ |
| Q_PROJ | [0.456, 0.567, ...] | [0.456, 0.567, ...] | <1e-6 | ✅ |
| K_PROJ | [0.678, 0.789, ...] | [0.678, 0.789, ...] | <1e-6 | ✅ |
| Q_ROPE | [0.890, 0.901, ...] | [0.123, 0.234, ...] | >0.5 | ❌ BUG! |

**First divergence**: Q_ROPE stage
**Hypothesis**: RoPE implementation incorrect
```

---

### Step 6: Fix Bugs One at a Time (2-4 hours per bug)

**Goal**: Fix identified bugs systematically

#### 6.1: Bug Fix Template

For each bug:

**1. Isolate the bug**
```markdown
## Bug: RoPE Implementation Incorrect

**Evidence**:
- Our Q_ROPE values: [0.890, 0.901, ...]
- Reference values: [0.123, 0.234, ...]
- Divergence: >0.5 (significant)

**Location**: qwen_transformer.cpp, lines XXX-YYY

**Current implementation**:
```cpp
// Our RoPE code
void apply_rope(float* q, float* k, int pos, ...) {
    // ... current code ...
}
```

**Reference implementation** (candle):
```rust
pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
    let cos = self.cos.narrow(0, offset, seq_len)?;
    let sin = self.sin.narrow(0, offset, seq_len)?;
    let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
    let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
    Ok((q_embed, k_embed))
}
```

**Hypothesis**: We're not computing cos/sin correctly
```

**2. Implement fix**
```cpp
// Fixed RoPE implementation
void apply_rope(float* q, float* k, int pos, ...) {
    // Match reference implementation exactly
    // ...
}
```

**3. Test fix**
```bash
cargo test --test haiku_generation_anti_cheat -- --ignored 2>&1 | tee debug_output_fixed.log
```

**4. Verify fix**
```markdown
## Fix Verification

**After fix**:
- Our Q_ROPE values: [0.123, 0.234, ...]
- Reference values: [0.123, 0.234, ...]
- Difference: <1e-6 ✅

**Next divergence**: O_PROJ stage (continue debugging)
```

**5. Commit fix**
```bash
git add cuda/src/transformer/qwen_transformer.cpp
git commit -m "fix(transformer): correct RoPE frequency calculation

- Match candle implementation exactly
- Fix cos/sin computation
- Verify values match reference within 1e-6 tolerance

Fixes garbage token generation by ensuring correct positional encoding."
```

#### 6.2: Common Bug Patterns and Fixes

**Bug Pattern 1: Wrong cuBLAS transpose flag**
```cpp
// WRONG
cublasSgemm(..., CUBLAS_OP_N, CUBLAS_OP_N, ...);  // Both not transposed

// RIGHT
cublasSgemm(..., CUBLAS_OP_N, CUBLAS_OP_T, ...);  // Weight transposed
```

**Bug Pattern 2: Wrong leading dimension**
```cpp
// WRONG (assumes column-major)
lda = M;  // Wrong for row-major

// RIGHT (row-major)
lda = K;  // Leading dimension is number of columns
```

**Bug Pattern 3: Head concatenation order**
```cpp
// WRONG
for (int h = 0; h < num_heads; h++) {
    for (int d = 0; d < head_dim; d++) {
        out[h * head_dim + d] = heads[h][d];  // Wrong stride
    }
}

// RIGHT
// [B, H, L, D] → [B, L, H*D]
for (int b = 0; b < batch; b++) {
    for (int l = 0; l < seq_len; l++) {
        for (int h = 0; h < num_heads; h++) {
            for (int d = 0; d < head_dim; d++) {
                out[b][l][h * head_dim + d] = heads[b][h][l][d];
            }
        }
    }
}
```

**Bug Pattern 4: GQA K/V repetition missing**
```cpp
// WRONG (treats GQA like MHA)
// num_heads = 14, num_kv_heads = 2
// K/V have shape [B, 2, L, D] but need to repeat to [B, 14, L, D]

// RIGHT
for (int h = 0; h < num_heads; h++) {
    int kv_head = h / (num_heads / num_kv_heads);  // Repeat K/V
    // Use K[kv_head] and V[kv_head]
}
```

---

### Step 7: Verify Complete Fix (1 hour)

**Goal**: Ensure all bugs are fixed and output is correct

#### 7.1: Run Full Test Suite

```bash
# Haiku test
cargo test --test haiku_generation_anti_cheat -- --ignored

# Other integration tests
cargo test --test qwen_integration -- --ignored
cargo test --test qwen_real_inference_test -- --ignored
```

#### 7.2: Manual Verification

```bash
# Start worker
./target/debug/worker-orcd \
    --worker-id test-worker \
    --model /home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-fp16.gguf \
    --gpu-device 0 \
    --port 8080

# Test inference
curl -X POST http://localhost:8080/execute \
    -H "Content-Type: application/json" \
    -d '{
        "job_id": "test-1",
        "prompt": "Write a haiku about debugging:",
        "max_tokens": 50,
        "temperature": 0.7
    }'
```

**Expected output**: Coherent English haiku, no mojibake, no repetitive tokens

#### 7.3: Document Success

```markdown
## Fix Complete ✅

**Bugs fixed**:
1. RoPE frequency calculation (qwen_transformer.cpp:XXX)
2. O projection transpose flag (qwen_transformer.cpp:YYY)
3. FFN down projection lda parameter (qwen_transformer.cpp:ZZZ)

**Test results**:
- ✅ Haiku test passes
- ✅ Minute word found in output
- ✅ No mojibake
- ✅ No repetitive tokens
- ✅ Coherent English text

**Example output**:
```
Code flows like streams
Debugging finds the right path
Tokens now make sense
```

**Commits**:
- abc1234: fix(transformer): correct RoPE frequency calculation
- def5678: fix(transformer): fix attention output projection transpose
- ghi9012: fix(transformer): correct FFN down projection parameters
```

---

## Document History

- **2025-10-07**: Initial plan created based on investigation history and reference analysis
- **2025-10-07**: Added GPT-2 strategy reference and detailed debugging instructions
