# FOUNDATION RESET: Starting Over with the Right Model
**Date:** 2025-10-08  
**From:** TEAM-007 (Architecture Review)  
**To:** Next Implementation Team  
**Status:** 🔄 **STRATEGIC PIVOT**

---

## Executive Summary

After comprehensive analysis, we're **starting over with a better foundation**. The original GPT-2 plan was educational but commercially limited. We're pivoting to **Llama-2 7B GGUF FP16** as our foundation model.

**This is not a failure - this is strategic course correction before significant investment.**

---

## Why We're Starting Over

### The Original Plan (GPT-2)
**What we planned:**
- Model: GPT-2 base/medium (124M/350M params)
- Format: PyTorch FP32 `.pt` files
- Purpose: Educational validation
- Architecture: Outdated (2019)

**Why it's limited:**
- ❌ Outdated architecture (absolute positions, LayerNorm, GELU)
- ❌ Only works for GPT-2 (dead-end)
- ❌ Not production-ready (toy models)
- ❌ Wrong format (FP32 too large for real deployment)
- ❌ No commercial viability

### The New Foundation (Llama-2 7B)
**What we're doing now:**
- Model: Llama-2 7B
- Format: GGUF FP16
- Purpose: Production-ready validation
- Architecture: Modern (2023+)

**Why it's better:**
- ✅ Modern architecture (RoPE, RMSNorm, SwiGLU)
- ✅ Works for 50+ models (Llama-3, Mistral, Qwen, CodeLlama)
- ✅ Production-ready (real deployments)
- ✅ Right format (GGUF = industry standard)
- ✅ Commercial viability from day 1

---

## What This Means

### What We Keep ✅
1. **Validation methodology** - 13 checkpoints still valid
2. **Testing rigor** - Proof bundle, determinism, multi-reference
3. **Architecture principles** - Worker isolation, HTTP server, streaming
4. **Development process** - Spec→Contract→Tests→Code
5. **Quality standards** - No shortcuts, proper validation

### What Changes 🔄
1. **Model:** GPT-2 → Llama-2 7B
2. **Format:** PyTorch FP32 → GGUF FP16
3. **Architecture spec:** Update to Llama-2 components
4. **Reference:** tinygrad/PyTorch → llama.cpp + PyTorch
5. **Validation:** Single reference → Multi-reference (llama.cpp + PyTorch)

### What We Abandon ❌
1. GPT-2 specific code (if any exists)
2. PyTorch `.pt` weight loading
3. GPT-2 architecture components (absolute positions, old LayerNorm)
4. Single-reference validation approach

---

## The New Foundation Model

### Llama-2 7B GGUF FP16

**Download:**
```bash
cd /home/vince/Projects/llama-orch
./.docs/testing/download_llama2_7b_fp16.sh
```

**Specifications:**
- **Size:** ~13.5 GB (FP16)
- **Parameters:** 7 billion
- **Architecture:** Llama-2 (RoPE + RMSNorm + SwiGLU)
- **Format:** GGUF (native quantization support)
- **Fits on:** RTX 3060 (12GB), MacBook Pro (16GB), Any homelab GPU

**Why this specific model:**

1. **Perfect Size Balance**
   - Not too small (toy model)
   - Not too large (can't fit on consumer hardware)
   - Just right (7B = sweet spot)

2. **Modern Architecture**
   - RoPE (Rotary Position Embeddings) - better than absolute
   - RMSNorm (Root Mean Square Norm) - simpler than LayerNorm
   - SwiGLU (Swish-Gated Linear Unit) - better than GELU
   - GQA-ready (for Llama-3 upgrade path)

3. **Maximum Reusability**
   - Same architecture as Llama-3 8B/70B
   - Same architecture as Mistral 7B
   - Same architecture as Qwen 7B/14B
   - Same architecture as CodeLlama
   - **One implementation → 50+ models**

4. **Commercial Viability**
   - Actually used in production
   - Fits consumer hardware
   - GGUF = industry standard
   - Quantization-ready (Q4/Q8 for deployment)

---

## Required Reading (CRITICAL)

### 1. Project Rules
**File:** `/home/vince/Projects/llama-orch/.windsurf/rules/llorch-cpud-rules.md`

**Key points:**
- Always add team signatures to code
- No background testing (blocking only)
- No CLI piping into interactive tools
- Update existing docs, don't create duplicates

### 2. Testing Standards
**File:** `/home/vince/Projects/llama-orch/.windsurf/rules/destructive-actions.md`

**Key points:**
- We're pre-v2, allowed to be destructive for cleanup
- No dangling files, no dead code
- Clean up aggressively

### 3. Rust Standards
**File:** `/home/vince/Projects/llama-orch/.windsurf/rules/rust-rules.md`

**Key points:**
- (Read this file for Rust-specific conventions)

### 4. Original Validation Plan
**File:** `/home/vince/Projects/llama-orch/bin/llorch-cpud/.specs/checkpoints/README.md`

**Key points:**
- 13 validation checkpoints
- Sequential validation strategy
- Tolerance levels per checkpoint
- Multi-reference validation approach

### 5. Proof Bundle Standard
**Memory:** Proof bundle system for test artifacts
- Output to `.proof_bundle/<type>/<run_id>/`
- Respect `LLORCH_RUN_ID` and `LLORCH_PROOF_DIR`
- Auto-generated headers per PB-1012

---

## Updated Architecture Specifications

### Core Components (Llama-2)

#### 1. RoPE (Rotary Position Embeddings)
**Replaces:** Absolute position embeddings (GPT-2)
```rust
// Apply rotary embeddings to Q and K
fn apply_rope(q: &Tensor, k: &Tensor, positions: &[usize]) -> (Tensor, Tensor)
```

#### 2. RMSNorm (Root Mean Square Normalization)
**Replaces:** LayerNorm (GPT-2)
```rust
// Simpler normalization: x / rms(x) * weight
fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Tensor
```

#### 3. SwiGLU (Swish-Gated Linear Unit)
**Replaces:** GELU (GPT-2)
```rust
// FFN with gating: swish(gate) * up
fn swiglu_ffn(x: &Tensor, gate: &Tensor, up: &Tensor, down: &Tensor) -> Tensor
```

#### 4. GQA (Grouped Query Attention) - Optional
**Note:** Llama-2 7B uses standard MHA, but architecture supports GQA for Llama-3
```rust
// n_heads_kv < n_heads_q (grouped)
// Llama-2 7B: n_heads_kv = n_heads_q = 32 (standard MHA)
// Llama-3 8B: n_heads_kv = 8, n_heads_q = 32 (GQA)
```

### Model Configuration (Llama-2 7B)
```rust
pub struct LlamaConfig {
    pub vocab_size: usize,      // 32000
    pub hidden_size: usize,      // 4096
    pub num_layers: usize,       // 32
    pub num_heads: usize,        // 32
    pub num_kv_heads: usize,     // 32 (same as num_heads for Llama-2)
    pub intermediate_size: usize, // 11008
    pub max_position_embeddings: usize, // 4096
    pub rms_norm_eps: f32,       // 1e-5
    pub rope_theta: f32,         // 10000.0
}
```

---

## Validation Strategy Update

### Multi-Reference Validation

**Primary Reference:** llama.cpp (C++ implementation)
- Extract checkpoints using Team 006's tool
- GGUF native, proven correct
- Production-grade implementation

**Secondary Reference:** PyTorch (if needed)
- Convert GGUF → PyTorch for validation
- Use transformers library
- Sanity check only

**Our Implementation:** llorch-cpud (Rust)
- Pure Rust from scratch
- GGUF loading
- Compare against both references

### Checkpoint Mapping (Updated)

| # | Checkpoint | Llama-2 Component | GPT-2 Equivalent |
|---|------------|-------------------|------------------|
| 1 | RMSNorm Output | RMSNorm | LayerNorm |
| 2 | QKV Projection | Linear (no bias) | Linear (with bias) |
| 3 | After RoPE | RoPE applied | Position added |
| 4 | Attention Scores | Scaled dot-product | Same |
| 5 | Attention Output | Linear projection | Same |
| 6 | FFN Output | SwiGLU | GELU |
| 7 | First Block | Complete block | Same concept |
| 8 | Full Logits | All 32 layers | All 12 layers |
| 9-12 | Sampling | Same | Same |

---

## Implementation Roadmap

### Phase 1: GGUF Loading (Week 1)
**Goal:** Load Llama-2 7B GGUF and extract weights

**Tasks:**
1. Implement GGUF parser (format spec)
2. Extract model metadata
3. Load FP16 weights into memory
4. Verify weight shapes match config

**Validation:**
- Compare weight shapes with llama.cpp
- Verify metadata parsing
- Check memory layout

### Phase 2: Core Components (Week 2)
**Goal:** Implement Llama-2 specific components

**Tasks:**
1. Implement RMSNorm
2. Implement RoPE
3. Implement SwiGLU FFN
4. Implement attention (standard MHA)

**Validation:**
- Unit test each component
- Compare outputs with llama.cpp checkpoints

### Phase 3: Full Inference (Week 3)
**Goal:** Complete forward pass

**Tasks:**
1. Wire up all components
2. Implement KV cache
3. Add sampling (greedy + temperature)
4. Add streaming

**Validation:**
- Checkpoint 1-6 validation
- End-to-end generation test

### Phase 4: Production Ready (Week 4)
**Goal:** Optimize and deploy

**Tasks:**
1. Add quantization support (Q4/Q8)
2. Optimize memory usage
3. Add HTTP server integration
4. Performance benchmarks

**Validation:**
- All 13 checkpoints pass
- Performance meets targets
- Memory usage acceptable

---

## What Team 006 Built (Still Useful!)

**Checkpoint Extractor Tool:**
- Location: `bin/llorch-cpud/tools/checkpoint-extractor/`
- Purpose: Extract checkpoints from llama.cpp inference
- Status: ✅ Compiled and working
- **Use for:** Extracting reference checkpoints from Llama-2 7B

**How to use:**
```bash
cd bin/llorch-cpud/tools/checkpoint-extractor
./build.sh  # Already built by Team 007

# Extract Llama-2 checkpoints
./build/llorch-checkpoint-extractor \
  /.test-models/llama2-7b/llama-2-7b.fp16.gguf \
  "Hello" \
  /tmp/llama2_reference_checkpoints
```

**This gives us reference checkpoints to validate against!**

---

## Specifications to Update

### 1. Main Behavioral Spec
**File:** `01_GPT2_PIPELINE_COMPLETE_BEHAVIORS.md`
**Action:** Create `02_LLAMA2_PIPELINE_COMPLETE_BEHAVIORS.md`

**Changes needed:**
- Update model config (7B params, 32 layers)
- Replace LayerNorm → RMSNorm
- Replace absolute positions → RoPE
- Replace GELU → SwiGLU
- Update all tensor shapes
- Update checkpoint definitions

### 2. Checkpoint Specs
**Files:** `checkpoints/CHECKPOINT_01_*.md` through `CHECKPOINT_12_*.md`
**Action:** Update each checkpoint for Llama-2 architecture

**Key changes:**
- Checkpoint 1: LayerNorm → RMSNorm
- Checkpoint 3: Add RoPE application
- Checkpoint 6: GELU → SwiGLU
- Update all expected shapes

### 3. Implementation Roadmap
**File:** `IMPLEMENTATION_ROADMAP.md`
**Action:** Update for Llama-2 components

**Changes:**
- Remove GPT-2 specific items
- Add GGUF loading
- Add RoPE implementation
- Add RMSNorm implementation
- Add SwiGLU implementation

---

## Success Criteria

### Immediate (Week 1)
- [ ] Llama-2 7B FP16 GGUF downloaded
- [ ] GGUF parser implemented
- [ ] Weights loaded correctly
- [ ] Metadata verified

### Short-term (Week 2-3)
- [ ] All Llama-2 components implemented
- [ ] Checkpoints 1-6 passing
- [ ] Basic inference working
- [ ] Greedy sampling correct

### Long-term (Week 4+)
- [ ] All 13 checkpoints passing
- [ ] Temperature sampling working
- [ ] HTTP server integrated
- [ ] Production-ready

---

## Key Decisions Made

### 1. Model Choice: Llama-2 7B ✅
**Rationale:**
- Modern architecture
- Perfect size for validation
- Maximum reusability
- Commercial viability

### 2. Format: GGUF FP16 ✅
**Rationale:**
- Industry standard
- Native quantization support
- Proven format
- Future-proof

### 3. Reference: llama.cpp Primary ✅
**Rationale:**
- C++ implementation (different from Rust)
- GGUF native
- Production-grade
- Checkpoint extractor already built

### 4. Validation: Multi-Reference ✅
**Rationale:**
- Higher confidence
- Catch implementation-specific bugs
- Industry best practice

---

## What NOT to Do

### ❌ Don't Copy Code
- Study references, understand behavior
- Write your own implementation
- Use specs, not source code

### ❌ Don't Skip Validation
- Every component must be validated
- Use checkpoints religiously
- No "it probably works"

### ❌ Don't Optimize Prematurely
- Correctness first
- Performance second
- Get it working, then make it fast

### ❌ Don't Create Duplicate Docs
- Update existing specs
- One source of truth
- Clean up as you go

---

## Resources

### Model Download
```bash
./.docs/testing/download_llama2_7b_fp16.sh
```

### Checkpoint Extractor
```bash
cd bin/llorch-cpud/tools/checkpoint-extractor
./build.sh
```

### Reference Implementations
- **llama.cpp:** `/home/vince/Projects/llama-orch/reference/llama.cpp`
- **Candle:** `/home/vince/Projects/llama-orch/reference/candle`
- **Mistral.rs:** `/home/vince/Projects/llama-orch/reference/mistral.rs`

### Specifications
- **Main spec:** `01_GPT2_PIPELINE_COMPLETE_BEHAVIORS.md` (needs update)
- **Checkpoints:** `checkpoints/CHECKPOINT_*.md` (needs update)
- **Roadmap:** `IMPLEMENTATION_ROADMAP.md` (needs update)

---

## Next Steps for Implementation Team

### Step 1: Read Everything (1 day)
1. ✅ Read this handoff completely
2. ✅ Read all rules files (`.windsurf/rules/*.md`)
3. ✅ Read checkpoint README (`checkpoints/README.md`)
4. ✅ Study Llama-2 architecture (papers, blogs)

### Step 2: Download Model (30 min)
```bash
./.docs/testing/download_llama2_7b_fp16.sh
```

### Step 3: Extract Reference Checkpoints (1 hour)
```bash
cd bin/llorch-cpud/tools/checkpoint-extractor
./build.sh
./build/llorch-checkpoint-extractor \
  ../../.test-models/llama2-7b/llama-2-7b.fp16.gguf \
  "Hello" \
  /tmp/llama2_ref
```

### Step 4: Update Specifications (2-3 days)
1. Create `02_LLAMA2_PIPELINE_COMPLETE_BEHAVIORS.md`
2. Update all checkpoint specs for Llama-2
3. Update implementation roadmap
4. Get specs reviewed

### Step 5: Start Implementation (Week 2+)
1. GGUF parser
2. Weight loading
3. Core components (RMSNorm, RoPE, SwiGLU)
4. Attention + FFN
5. Full inference
6. Validation

---

## Questions & Clarifications

**Q: Why abandon GPT-2 work?**
A: GPT-2 is a dead-end. Llama-2 gives us a foundation for 50+ models. Better to pivot now than after significant investment.

**Q: Is the checkpoint extractor still useful?**
A: YES! It extracts Llama-2 checkpoints from llama.cpp, which we use as reference.

**Q: Do we lose validation work?**
A: No. The 13-checkpoint methodology is the same. Just different architecture components.

**Q: What about PyTorch reference?**
A: llama.cpp is primary. PyTorch is secondary/optional for additional validation.

**Q: When do we add quantization?**
A: After FP16 validation passes. Get correctness first, then optimize.

---

## Sign-off

**Prepared by:** TEAM-007 (Architecture Review)  
**Date:** 2025-10-08  
**Status:** Ready for implementation

**This is the right foundation. Let's build it properly.**

---

*"Start with the right foundation, and everything else becomes easier."*  
— TEAM-007, Strategic Architecture Division

**END HANDOFF**
