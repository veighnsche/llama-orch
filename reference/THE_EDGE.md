# The Edge We Walk

**Date**: 2025-10-06  
**Status**: Policy Updated - Walking the Edge

---

## The Razor's Edge

We are positioned **exactly at the edge** between learning and dependency:

```
                    THE EDGE
                       │
    ✅ ALLOWED         │         ❌ FORBIDDEN
                       │
  Read their code      │      Build their code
  Study algorithms     │      Link their libs
  Compare approaches   │      Import headers
  Learn patterns       │      Copy-paste blindly
  Debug by reference   │      Depend on them
                       │
         WE ARE HERE ──┤
                       │
```

## What This Means

### We Can Do

1. **Open their source files** in our editor
2. **Read their implementations** to understand algorithms
3. **Compare their approach** to ours when debugging
4. **Study their kernel patterns** to learn CUDA best practices
5. **Reference their logic** when our output differs

### We Cannot Do

1. **Build llama.cpp** in our workspace
2. **Link to libllama.so** in our binaries
3. **Include their headers** in our code
4. **Run their binaries** in production
5. **Copy-paste** without understanding
6. **Add them to Cargo.toml** or CMakeLists.txt

## The Philosophy

```rust
// ❌ WRONG - Dependency
extern crate llama_cpp;
use llama_cpp::Model;

// ✅ RIGHT - Learning
// Studied llama.cpp's attention implementation in:
// reference/llama.cpp/ggml-cuda.cu:flash_attn_ext()
// 
// Our implementation below uses similar approach but:
// - Optimized for our tensor layout
// - Integrated with our MXFP4 quantization
// - Custom memory management
fn our_attention_kernel() {
    // Our implementation
}
```

## Real Example: Debugging Garbage Output

**Current Issue**: Test `haiku_generation_anti_cheat.rs` produces garbage tokens

### ❌ Wrong Approach
```bash
# Link to llama.cpp
cargo add llama-cpp-rs
```

### ✅ Right Approach
```bash
# Read their weight loading logic
cd reference/llama.cpp
grep -A 50 "llm_load_tensors" llama.cpp

# Compare to ours
cd ../../bin/worker-orcd
grep -A 50 "load_weights_to_gpu" src/cuda/weight_loader.rs

# Identify the difference
# Implement our fix
# Test our fix
```

## Why This Edge Exists

1. **Learning is faster** when you can see working code
2. **Debugging is easier** when you can compare implementations
3. **Quality is higher** when you understand the problem space
4. **Independence is critical** - we must own our stack

## The Test

Ask yourself:

> "If llama.cpp disappeared tomorrow, would our code still work?"

- ✅ **YES** - We're on the right side of the edge
- ❌ **NO** - We've crossed the line

## Our Commitment

We are **llama-orch** - the llama.cpp competitor.

- We **learn** from them
- We **don't depend** on them
- We **build better** than them

## The Line in Practice

| Scenario | Allowed? | Why |
|----------|----------|-----|
| Read `llama.cpp` to understand Q4_K dequant | ✅ | Learning |
| Copy their dequant kernel verbatim | ❌ | Not understanding |
| Study their attention pattern, implement ours | ✅ | Learning + implementing |
| Link to libllama.so | ❌ | Dependency |
| Compare their output to ours for debugging | ✅ | Debugging |
| Use their binary in production | ❌ | Dependency |
| Reference their GGUF parsing logic | ✅ | Learning |
| Import their GGUF parser | ❌ | Dependency |

## Success Metrics

We're succeeding when:

1. ✅ We can explain **why** their approach works
2. ✅ We implement **our own version** that's better
3. ✅ Our code **doesn't import** their code
4. ✅ We **learn faster** by having the reference
5. ✅ We remain **independent** of their releases

## The Vision

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  llama.cpp: The reference implementation            │
│  llama-orch: The production implementation          │
│                                                     │
│  We study them. We compete with them.               │
│  We don't depend on them.                           │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

**This is the edge. We walk it carefully.**

We are close enough to learn.  
Far enough to remain independent.

**This is llama-orch. This is our way.**
