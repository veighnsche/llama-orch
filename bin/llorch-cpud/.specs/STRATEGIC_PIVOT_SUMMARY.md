# Strategic Pivot: GPT-2 → Llama-2 7B
**Date:** 2025-10-08  
**Decision:** Foundation model change  
**Status:** ✅ **APPROVED**

---

## TL;DR

**We're starting over with Llama-2 7B GGUF FP16 instead of GPT-2.**

This is the right call. Better foundation = better future.

---

## The Change

| Aspect | Before (GPT-2) | After (Llama-2 7B) |
|--------|----------------|-------------------|
| **Model** | GPT-2 base/medium | Llama-2 7B |
| **Size** | 124M/350M params | 7B params |
| **Format** | PyTorch FP32 | GGUF FP16 |
| **Architecture** | 2019 (outdated) | 2023 (modern) |
| **Reusability** | GPT-2 only | 50+ models |
| **Commercial** | Toy models | Production-ready |

---

## Why This is Better

### ✅ Modern Architecture
- **RoPE** instead of absolute positions
- **RMSNorm** instead of LayerNorm  
- **SwiGLU** instead of GELU
- Industry standard (2023+)

### ✅ Maximum Reusability
One implementation works for:
- Llama-2 7B/13B/70B
- Llama-3 8B/70B
- Mistral 7B
- Qwen 7B/14B
- CodeLlama
- 50+ other models

### ✅ Commercial Viability
- Fits consumer GPUs (RTX 3060, MacBook)
- GGUF = industry standard format
- Actually used in production
- Quantization-ready

### ✅ Right Size
- Not too small (toy model)
- Not too large (won't fit)
- Perfect for validation (7B params)

---

## What We Keep

✅ **Validation methodology** - 13 checkpoints  
✅ **Testing rigor** - Multi-reference, proof bundles  
✅ **Quality standards** - No shortcuts  
✅ **Development process** - Spec→Contract→Tests→Code  
✅ **Team 006's checkpoint extractor** - Works for Llama-2!

---

## What Changes

🔄 **Model:** GPT-2 → Llama-2 7B  
🔄 **Format:** PyTorch → GGUF  
🔄 **Architecture specs:** Update components  
🔄 **Reference:** tinygrad → llama.cpp (primary)

---

## Action Items

### Immediate
1. ✅ Download Llama-2 7B FP16 GGUF
   ```bash
   ./.docs/testing/download_llama2_7b_fp16.sh
   ```

2. ✅ Extract reference checkpoints
   ```bash
   cd bin/llorch-cpud/tools/checkpoint-extractor
   ./build/llorch-checkpoint-extractor \
     /.test-models/llama2-7b/llama-2-7b.fp16.gguf \
     "Hello" /tmp/llama2_ref
   ```

### Next Week
3. Update specifications for Llama-2
4. Implement GGUF parser
5. Implement Llama-2 components

---

## Key Documents

📄 **Full Handoff:** `FOUNDATION_RESET_HANDOFF.md`  
📄 **Download Script:** `/.docs/testing/download_llama2_7b_fp16.sh`  
📄 **Checkpoint Tool:** `tools/checkpoint-extractor/` (already built)

---

## Bottom Line

**This is not a setback - this is strategic positioning.**

Starting with Llama-2 gives us:
- Modern foundation
- Commercial viability
- Maximum reusability
- Production readiness

**The right foundation makes everything else easier.**

---

*Approved by: TEAM-007 Architecture Review*  
*Date: 2025-10-08*
