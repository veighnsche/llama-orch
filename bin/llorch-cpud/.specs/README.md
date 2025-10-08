# llorch-cpud Specifications

This directory contains the complete behavioral specifications for the llorch-cpud CPU inference engine.

## 🚨 IMPORTANT: Strategic Pivot (2025-10-08)

**We've changed our foundation model from GPT-2 to Llama-2 7B.**

📄 **READ FIRST:** `FOUNDATION_RESET_HANDOFF.md` - Complete handoff explaining the change  
📄 **Quick Summary:** `STRATEGIC_PIVOT_SUMMARY.md` - Why this is better

**New Foundation Model:** Llama-2 7B GGUF FP16  
**Download:** `/.docs/testing/download_llama2_7b_fp16.sh`

---

## Documents

### 🔴 Critical Reading (Start Here)
- **`FOUNDATION_RESET_HANDOFF.md`** - Strategic pivot explanation and new roadmap
- **`STRATEGIC_PIVOT_SUMMARY.md`** - Quick summary of the change
- **Rules:** `/.windsurf/rules/llorch-cpud-rules.md` - Team signatures, testing standards

### Process Documentation
- **`SPEC_DEVELOPMENT_PROCESS.md`** - The 5-phase development process for llorch-cpud
  - Defines how we study references, create specs, and validate implementations
  - Emphasizes NO code copying, only learning from references

### Behavioral Specifications

#### Current (Being Updated)
- **`01_GPT2_PIPELINE_COMPLETE_BEHAVIORS.md`** - ⚠️ OUTDATED - GPT-2 specific
  - **568 lines** of comprehensive behavioral specifications
  - **Being replaced with Llama-2 spec**
  - Keep for reference only

#### Coming Soon
- **`02_LLAMA2_PIPELINE_COMPLETE_BEHAVIORS.md`** - 🚧 TO BE CREATED
  - Llama-2 7B architecture
  - RoPE, RMSNorm, SwiGLU components
  - GGUF format support
  - Based on llama.cpp reference

## What's Been Completed

✅ Studied complete tinygrad GPT-2 implementation (255 lines)  
✅ Documented all 10 phases of inference pipeline  
✅ Specified exact tensor shapes throughout  
✅ Defined MUST/SHOULD/COULD requirements  
✅ Included validation test cases  

## Phases Documented

1. **Model Initialization and Weight Loading** - Parameters, weight loading, component initialization
2. **Input Processing and Embeddings** - Tokenization, token embeddings, position embeddings
3. **Attention Mask Creation** - Causal masking for autoregressive generation
4. **Transformer Blocks** - 12-layer iteration and pre-norm architecture
5. **Attention Mechanism** - Layer norm, QKV projection, KV cache, scaled dot-product, output projection
6. **Feedforward Network** - Two-layer MLP with GELU activation
7. **Final Layer Norm and LM Head** - Final normalization and vocabulary projection
8. **Sampling and Token Generation** - Temperature=0 argmax and temperature>0 sampling
9. **Autoregressive Generation Loop** - Multi-step generation with state tracking
10. **Validation and Testing** - Deterministic output validation

## Next Steps (Per SPEC_DEVELOPMENT_PROCESS.md)

**Current Status:** Phase 1-2 complete (Study + Spec Creation)

**Remaining:**
- Phase 3: Hand specs to developers
- Phase 4: Developers implement from specs (NO code copying)
- Phase 5: Validate each component against tinygrad reference with logging

## Key Principles

- **Educational focus** - Using tinygrad as primary reference (simplest, most readable)
- **Original implementation** - Developers write their own code from specs
- **Step-by-step validation** - Test each component before moving forward
- **Comprehensive logging** - Enable validation at every checkpoint
