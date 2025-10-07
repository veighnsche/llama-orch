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

Git submodule of [veighnsche/llama.cpp](https://github.com/veighnsche/llama.cpp) (fork of ggerganov/llama.cpp)

**Use cases**:
- Compare GGUF parsing logic when debugging weight loading
- Study attention kernel implementations
- Understand tokenizer integration patterns
- Reference sampling algorithms
- Custom orchestrator logging integration (orch_log branch)

### `/reference/vllm/`

Git submodule of [veighnsche/vllm](https://github.com/veighnsche/vllm) (fork of vllm-project/vllm)

**Use cases**:
- Study PagedAttention and KV cache management
- Reference continuous batching implementations
- Understand tensor parallelism patterns
- Compare scheduling algorithms

### `/reference/llamafile/`

Git submodule of [veighnsche/llamafile](https://github.com/veighnsche/llamafile) (fork of Mozilla-Ocho/llamafile)

**Use cases**:
- Study single-file deployment patterns
- Reference cross-platform binary packaging
- Understand embedded model distribution
- Compare lightweight inference approaches

### `/reference/mistral.rs/`

Git submodule of [veighnsche/mistral.rs](https://github.com/veighnsche/mistral.rs) (fork of EricLBuehler/mistral.rs)

**Use cases**:
- Study Rust-native inference implementations
- Reference Mistral model architecture patterns
- Compare quantization strategies in Rust
- Understand pipeline parallelism approaches

### `/reference/drama_llama/`

Git submodule of [veighnsche/drama_llama](https://github.com/veighnsche/drama_llama)

**Use cases**:
- Study experimental inference patterns
- Reference alternative architecture approaches
- Understand novel optimization techniques

### `/reference/candle/`

Git submodule of [veighnsche/candle](https://github.com/veighnsche/candle) (fork of huggingface/candle)

**Use cases**:
- Study Rust ML framework design patterns
- Reference tensor operations and kernels
- Understand safetensors integration
- Compare CUDA/Metal backend implementations

### `/reference/text-generation-inference/`

Git submodule of [veighnsche/text-generation-inference](https://github.com/veighnsche/text-generation-inference) (fork of huggingface/text-generation-inference)

**Use cases**:
- Study production inference server architecture
- Reference gRPC/HTTP API patterns
- Understand model loading and caching strategies
- Compare batching and scheduling implementations

### `/reference/tinygrad/`

Git submodule of [veighnsche/tinygrad](https://github.com/veighnsche/tinygrad) (fork of tinygrad/tinygrad)

**Use cases**:
- Study minimal tensor framework design
- Reference kernel fusion patterns
- Understand JIT compilation approaches
- Compare backend abstraction layers

### `/reference/flash-attention/`

Git submodule of [veighnsche/flash-attention](https://github.com/veighnsche/flash-attention) (fork of Dao-AILab/flash-attention)

**Use cases**:
- Study memory-efficient attention implementations
- Reference CUDA kernel optimization techniques
- Understand IO-aware algorithm design
- Compare attention variants (Flash-2, Flash-3)

**Universal Forbidden Actions**:
- ❌ Building any reference code
- ❌ Linking to reference libraries
- ❌ Including reference headers in our code
- ❌ Running reference binaries in production
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
