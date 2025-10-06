# Reference Directory

**Purpose**: Code inspection and learning ONLY. Not for building or linking.

## ⚠️ CRITICAL: Read-Only Reference

This directory contains external codebases for **reference purposes only**:

- 🔍 **Code inspection** - Understanding implementation approaches
- 🐛 **Debugging** - Comparing our output to reference implementations
- 📚 **Learning** - Studying algorithms and patterns
- ❌ **NOT for building, linking, or depending on**

## Contents

### `/reference/llama.cpp/`

Git submodule of [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)

**Use cases**:
- Compare GGUF parsing logic when debugging weight loading
- Study attention kernel implementations
- Understand tokenizer integration patterns
- Reference sampling algorithms

**Forbidden**:
- ❌ Building llama.cpp
- ❌ Linking to llama.cpp libraries
- ❌ Including llama.cpp headers in our code
- ❌ Running llama.cpp binaries in production
- ❌ Copy-pasting without understanding

## Build System Exclusion

The `/reference/` directory is **explicitly excluded** from all builds:

- Not in Cargo workspace
- Not in CMakeLists.txt
- Not in CI/CD pipelines
- Build artifacts are gitignored

## The Line We Walk

```
┌─────────────────────────────────────────────────────┐
│  ✅ Read reference code to understand algorithms    │
│  ❌ Use reference code as a dependency              │
│                                                     │
│  We learn from them. We don't depend on them.      │
└─────────────────────────────────────────────────────┘
```

## Our Philosophy

**We are llama-orch** - a llama.cpp competitor building something better:

- Custom CUDA kernels optimized for our use case
- MXFP4 quantization (43% VRAM savings)
- Production-ready from day 1
- Multi-GPU pool management

**We implement everything ourselves.** This directory exists to help us learn faster, not to shortcut the work.

---

See: `/NO_LLAMA_CPP.md` and `/bin/worker-orcd/NO_LLAMA_CPP_RULE.md` for full policy.
