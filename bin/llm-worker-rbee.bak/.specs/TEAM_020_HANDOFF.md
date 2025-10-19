# TEAM-020 Handoff: Multi-Architecture Model Support & Testing

**Date:** 2025-10-09  
**From:** TEAM-019  
**To:** TEAM-020  
**Status:** 🚀 Ready for multi-model validation

---

## Executive Summary

TEAM-019 fixed the Metal/CUDA inference bug (cache recreation workaround). All three backends (CPU, CUDA, Metal) now work correctly with **TinyLlama**. However, we only tested one model architecture.

**Your mission:** Validate that our multi-model architecture system actually works across different model types (Llama, Mistral, Phi, Qwen) on all backends.

---

## What TEAM-019 Completed ✅

### 1. Fixed Metal/CUDA Inference Bug
- **Root cause:** Candle's mask broadcasting bug with KV cache growth
- **Solution:** Recreate cache at position=0 (workaround)
- **Status:** All backends working with TinyLlama
- **Documentation:** `.specs/METAL_CUDA_INFERENCE_BUG_REPORT.md`

### 2. Normalized Debugging Commands
- Added `debug-inference` command to homelab CLI
- Added `logs` command to view worker logs
- Updated README with troubleshooting section
- All debugging is now reproducible via CLI

### 3. Reverted F16 Dtype Bug
- TEAM-018 incorrectly used F16 for Metal
- Reverted to F32 for all backends (consistent with other models)

---

## What's Left TODO 🚧

### Priority 1: Multi-Architecture Testing ⚠️ CRITICAL

**Problem:** We have model loaders for 4 architectures but only tested Llama.

**Current model support (claimed):**
- ✅ Llama (tested on all backends)
- ❓ Mistral (code exists, never tested)
- ❓ Phi (code exists, never tested)
- ❓ Qwen (code exists, never tested)

**Files to review:**
- `src/backend/models/llama.rs` - ✅ Working
- `src/backend/models/mistral.rs` - ⚠️ Untested
- `src/backend/models/phi.rs` - ⚠️ Untested
- `src/backend/models/qwen.rs` - ⚠️ Untested
- `src/backend/models/mod.rs` - Model factory (auto-detection)

**Risk:** If Mistral/Phi/Qwen have the same cache bug, they'll fail on Metal/CUDA.

### Priority 2: Create Multi-Model Test Suite

**Requirements:**
1. Download test models for each architecture (small versions):
   - Llama: TinyLlama 1.1B ✅ (already have)
   - Mistral: Mistral-7B-Instruct-v0.2 (or smaller variant)
   - Phi: Phi-2 (2.7B)
   - Qwen: Qwen2-0.5B-Instruct

2. Create test script that validates:
   - Model loads successfully
   - Warmup completes
   - Inference generates tokens
   - Output is coherent text (not garbage)

3. Test matrix:
   - 4 architectures × 3 backends = 12 test cases
   - Run on: CPU (local), Metal (mac.home.arpa), CUDA (workstation.home.arpa)

### Priority 3: Use Our Candle Fork with Proper Fix

**IMPORTANT:** We're NOT fixing models individually. We're fixing Candle itself.

**Strategy:**
1. Create branch in `reference/candle/` with mask broadcasting fix
2. Update `llm-worker-rbee` to use our Candle fork
3. Remove TEAM-019's workaround (cache recreation)
4. Test all models on all backends

**Why this approach:**
- ✅ Fixes root cause, not symptoms
- ✅ All models benefit automatically
- ✅ No per-model workarounds needed
- ✅ Proper solution we can upstream later

**See:** `.specs/CANDLE_UPSTREAM_OPPORTUNITIES.md` for implementation details

**Your tasks:**
- [ ] Create `llorch/metal-bugfixes` branch in `reference/candle/`
- [ ] Apply mask fix from candle-vllm
- [ ] Update `Cargo.toml` to use fork
- [ ] Remove workaround from `llama.rs`
- [ ] Test all models work with fork

### Priority 4: Document Model Support Matrix

Create `docs/MODEL_SUPPORT.md` with:
- Supported architectures
- Tested models per architecture
- Backend compatibility (CPU/CUDA/Metal)
- Known issues per model type
- Download instructions for test models

---

## Recommended Approach

### Week 1: Candle Fork Setup

**Day 1-2: Create Fork Branch**
- [ ] Create `llorch/metal-bugfixes` branch in `reference/candle/`
- [ ] Study mask fix in `reference/candle-vllm/src/openai/models/layers/mask.rs`
- [ ] Apply fix to `candle-transformers/src/models/llama.rs`
- [ ] Update Cache::mask() signature
- [ ] Add tests for mask broadcasting

**Day 3: Integrate Fork**
- [ ] Update `llm-worker-rbee/Cargo.toml` to use fork
- [ ] Remove TEAM-019 workaround from `llama.rs`
- [ ] Rebuild all backends
- [ ] Verify compilation

**Day 4-5: Initial Testing**
- [ ] Test Llama on CPU (should still work)
- [ ] Test Llama on Metal (should work without workaround)
- [ ] Test Llama on CUDA (should work without workaround)
- [ ] Debug any issues

### Week 2: Model Downloads & Multi-Model Setup

**Day 1-2: Model Research**
- [ ] Research small models for each architecture
- [ ] Check model availability (HuggingFace, GGUF format)
- [ ] Verify model sizes (prefer <3GB for fast testing)
- [ ] Document model URLs and formats

**Day 3-4: Download Scripts**
- [ ] Create `download_mistral.sh` (similar to `download_tinyllama.sh`)
- [ ] Create `download_phi.sh`
- [ ] Create `download_qwen.sh`
- [ ] Test downloads on all homelab machines

**Day 5: Test Infrastructure**
- [ ] Create `test-multi-model.sh` script
- [ ] Add to homelab CLI: `llorch-remote <host> <backend> test-models`
- [ ] Define success criteria (tokens generated, no errors)

### Week 3: Multi-Model Testing with Fork

**Day 1-2: CPU Backend Testing**
- [ ] Test Mistral on CPU with fork
- [ ] Test Phi on CPU with fork
- [ ] Test Qwen on CPU with fork
- [ ] Document any failures

**Day 3: Metal Backend Testing**
- [ ] Test all models on Metal (mac.home.arpa) with fork
- [ ] Verify no broadcasting errors
- [ ] Verify all models work
- [ ] Benchmark performance

**Day 4: CUDA Backend Testing**
- [ ] Test all models on CUDA (workstation.home.arpa) with fork
- [ ] Verify no broadcasting errors
- [ ] Verify all models work
- [ ] Benchmark performance

**Day 5: Documentation & Cleanup**
- [ ] Create MODEL_SUPPORT.md
- [ ] Update README with model support info
- [ ] Document fork usage in README
- [ ] Add test results to handoff document

### Week 4: Validation & Production Readiness

**Day 1-2: Performance Benchmarks**
- [ ] Compare fork vs. workaround performance
- [ ] Test with long contexts (>1000 tokens)
- [ ] Measure memory usage
- [ ] Document improvements

**Day 3-4: Stability Testing**
- [ ] Run extended inference tests
- [ ] Test edge cases (very long sequences)
- [ ] Test rapid request sequences
- [ ] Verify no memory leaks

**Day 5: Handoff Preparation**
- [ ] Document fork branch usage
- [ ] Update TEAM_021_HANDOFF.md
- [ ] List any remaining issues
- [ ] Recommend when to upstream

---

## Test Script Template

Create `.docs/testing/test_multi_model.sh`:

```bash
#!/usr/bin/env bash
# Multi-model architecture test
# Tests all supported architectures on current backend

set -euo pipefail

BACKEND="${1:-cpu}"
MODELS_DIR="${2:-.test-models}"

echo "Testing multi-model support on $BACKEND backend"
echo "================================================"

# Test each architecture
for MODEL in llama mistral phi qwen; do
    echo ""
    echo "Testing $MODEL..."
    
    # Check if model exists
    if [[ ! -d "$MODELS_DIR/$MODEL" ]]; then
        echo "⚠️  Model not found: $MODELS_DIR/$MODEL (skipping)"
        continue
    fi
    
    # Run inference test
    if ./target/release/llorch-${BACKEND}-candled \
        --worker-id test-$MODEL \
        --model "$MODELS_DIR/$MODEL" \
        --port 9876 \
        --callback-url http://localhost:9999 &
    then
        WORKER_PID=$!
        sleep 5
        
        # Test inference
        RESPONSE=$(curl -s -X POST http://localhost:9876/execute \
            -H "Content-Type: application/json" \
            -d '{"job_id":"test","prompt":"Hello","max_tokens":10,"seed":42}')
        
        kill $WORKER_PID 2>/dev/null || true
        
        if echo "$RESPONSE" | grep -q '"t":'; then
            echo "✅ $MODEL works on $BACKEND"
        else
            echo "❌ $MODEL failed on $BACKEND"
            echo "Response: $RESPONSE"
        fi
    else
        echo "❌ $MODEL worker failed to start on $BACKEND"
    fi
done
```

---

## Known Issues & Gotchas

### 1. Cache Bug May Affect All Models

**Symptom:** `cannot broadcast [N, N] to [1, H, N, M]` error on Metal/CUDA

**Fix:** Apply TEAM-019's cache recreation workaround:
```rust
if position == 0 {
    self.cache = Cache::new(true, DType::F32, &self.config, device)?;
}
```

**Location:** Each model's `forward()` method in `src/backend/models/*.rs`

### 2. Different Cache Types

- **Llama:** Uses `candle_transformers::models::llama::Cache`
- **Mistral:** May use different cache structure
- **Phi:** May not use position-based cache
- **Qwen:** May use different cache structure

**Action:** Check each model's Candle implementation before applying fix.

### 3. Model Format Compatibility

- **SafeTensors:** Preferred format (all models should support)
- **GGUF:** Some models only available in GGUF
- **Quantization:** May need different handling per model

### 4. Tokenizer Differences

Each architecture may have different tokenizer:
- Llama: SentencePiece
- Mistral: SentencePiece
- Phi: GPT-2 tokenizer
- Qwen: Custom tokenizer

**Action:** Verify tokenizer loading works for each model type.

---

## Success Criteria

Your work is complete when:

- [ ] **4 model architectures tested** on all 3 backends (12 test cases)
- [ ] **Test script created** and integrated into homelab CLI
- [ ] **Download scripts created** for all test models
- [ ] **All models work** on CPU, Metal, and CUDA
- [ ] **Cache bug fixed** for any failing models
- [ ] **Documentation created** (MODEL_SUPPORT.md)
- [ ] **Test results documented** in handoff

---

## Resources

### Internal Documentation
- `.specs/METAL_CUDA_INFERENCE_BUG_REPORT.md` - Cache bug details
- `src/backend/models/mod.rs` - Model factory and auto-detection
- `.docs/testing/download_tinyllama.sh` - Example download script
- `scripts/homelab/llorch-remote` - CLI tool for remote testing

### Model Sources
- **HuggingFace:** https://huggingface.co/models
- **GGUF Models:** https://huggingface.co/models?library=gguf
- **Candle Examples:** `reference/candle/candle-examples/`

### Candle Model Implementations
- `reference/candle/candle-transformers/src/models/llama.rs`
- `reference/candle/candle-transformers/src/models/mistral.rs`
- `reference/candle/candle-transformers/src/models/phi.rs`
- `reference/candle/candle-transformers/src/models/qwen2.rs`

---

## Questions for TEAM-020

Before you start:

1. **Model selection:** Which specific models should we use for testing?
   - Prefer small models (<3GB) for fast iteration
   - Must be available in SafeTensors or GGUF format

2. **Test coverage:** Should we test quantized models (Q4, Q8)?
   - May have different behavior than F32 models

3. **Performance benchmarks:** Should we measure tokens/sec per model?
   - Would help identify performance regressions

4. **CI integration:** Should multi-model tests run in CI?
   - Would require downloading models in CI (slow)

---

## Handoff Checklist

- [x] Metal/CUDA inference bug fixed (cache workaround)
- [x] All backends tested with TinyLlama
- [x] Debug commands normalized in CLI
- [x] Technical debt documented
- [x] Bug report completed
- [ ] Multi-architecture testing (TEAM-020)
- [ ] Model support matrix documented (TEAM-020)
- [ ] Test suite created (TEAM-020)

---

**Handoff completed:** 2025-10-09  
**From:** TEAM-019  
**To:** TEAM-020  
**Status:** ✅ Ready for multi-model validation  
**Next action:** TEAM-020 to test Mistral, Phi, and Qwen on all backends

---

**Signed:**  
TEAM-019  
2025-10-09T12:08:37+02:00
