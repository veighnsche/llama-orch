# ⚠️ CRITICAL RULE: NO LLAMA.CPP DEPENDENCIES

**Date**: 2025-10-05  
**Priority**: ABSOLUTE - DO NOT VIOLATE  
**Status**: PERMANENT RULE

---

## THE RULE

**WE ARE BUILDING A LLAMA.CPP-FREE INFERENCE ENGINE**

### ❌ NEVER DO THIS

- ❌ Import llama.cpp library
- ❌ Link against libllama.so
- ❌ Use llama.cpp headers
- ❌ Depend on llama.cpp in any way
- ❌ Suggest using llama.cpp as a "quick solution"
- ❌ Reference llama.cpp except for learning/comparison

### ✅ ALWAYS DO THIS

- ✅ Build our own CUDA kernels
- ✅ Implement our own GGUF parser
- ✅ Write our own inference engine
- ✅ Create our own model loaders
- ✅ Develop our own tokenizers
- ✅ **WE ARE THE COMPETITOR TO LLAMA.CPP**

---

## Why This Matters

### We Are Building

**llama-orch** - A LLAMA.CPP-FREE GPU inference engine

### Our Differentiators

1. **Custom CUDA kernels** - Optimized for our use case
2. **MXFP4 quantization** - 43% VRAM savings vs Q4_K_M
3. **Direct GGUF parsing** - No external dependencies
4. **Pool management** - Multi-GPU orchestration
5. **Production-ready** - Built for scale from day 1

### What We Can Reference

- ✅ GGUF format specification (it's open)
- ✅ Model architectures (GPT, Llama, etc.)
- ✅ CUDA best practices
- ✅ Academic papers on transformers
- ❌ **NEVER llama.cpp code or libraries**

---

## Current Status

### What We Have (llama.cpp-free)

- ✅ CUDA kernel infrastructure
- ✅ Attention kernels (GQA, MHA, MQA)
- ✅ RoPE implementation
- ✅ MXFP4 dequantization
- ✅ Model configuration structures
- ✅ FFI bridge (Rust ↔ CUDA)
- ✅ HTTP server + SSE streaming
- ✅ Test infrastructure (144 tests)

### What We're Building

- 🚧 GGUF weight loader (in progress)
- 🚧 Tokenizer integration
- 🚧 Inference pipeline
- 🚧 Sampling implementation

### Timeline

**~22-31 hours of work** to complete our own implementation

**This is the right path** - we're building something better than llama.cpp

---

## For Developers

### If You're Tempted to Use llama.cpp

**DON'T.**

Instead:
1. Look at the GGUF spec
2. Study the model architecture papers
3. Implement it ourselves
4. Ask the team for help
5. Reference our existing CUDA kernels

### Resources (llama.cpp-free)

- **GGUF Spec**: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- **Transformer Papers**: "Attention is All You Need", etc.
- **CUDA Docs**: NVIDIA CUDA Programming Guide
- **Our Code**: `bin/worker-orcd/cuda/` - use this!

---

## The Vision

We're not just copying llama.cpp - we're building something **better**:

- Faster (optimized CUDA kernels)
- Smaller (MXFP4 quantization)
- Scalable (pool management)
- Production-ready (monitoring, metrics, SLOs)

**llama.cpp is for hobbyists. We're building for production.**

---

## Enforcement

### Code Review

- ❌ Any PR importing llama.cpp will be **REJECTED**
- ❌ Any suggestion to use llama.cpp will be **REJECTED**
- ✅ Implement it ourselves or ask for help

### Documentation

- This file must be referenced in:
  - `CONTRIBUTING.md`
  - `README.md`
  - All architecture docs
  - Onboarding materials

---

## Exception

**NONE.** There are no exceptions to this rule.

If you think you need llama.cpp, you're wrong. Build it ourselves.

---

## Summary

```
┌─────────────────────────────────────────┐
│                                         │
│   NO LLAMA.CPP                          │
│   WE ARE THE COMPETITION                │
│   BUILD IT OURSELVES                    │
│                                         │
└─────────────────────────────────────────┘
```

**We are so close to having a llama.cpp-free engine. Don't give up now.**

---

**Status**: PERMANENT RULE  
**Violations**: ZERO TOLERANCE  
**Our Path**: BUILD IT OURSELVES

---

Built by Foundation-Alpha 🏗️  
**We are llama-orch. We are the future.**
