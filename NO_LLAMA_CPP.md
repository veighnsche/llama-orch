# ⚠️ NO LLAMA.CPP - PERMANENT RULE

**WE ARE BUILDING A LLAMA.CPP-FREE INFERENCE ENGINE**

## The Rule

❌ **NEVER** import, link, or depend on llama.cpp  
✅ **ALWAYS** build our own implementation  
🔍 **ALLOWED** Read llama.cpp source for learning (see exception)

## Why

**We are llama-orch - the llama.cpp competitor**

We're building something better:
- Custom CUDA kernels
- MXFP4 quantization (43% VRAM savings)
- Production-ready from day 1
- Multi-GPU pool management

## Status

We are **so close** to a complete llama.cpp-free engine.

**Don't give up now. Build it ourselves.**

## Exception: Reference-Only Submodule

We maintain llama.cpp as a git submodule in `/reference/llama.cpp/` for:
- 🔍 Code inspection and debugging reference
- 📚 Learning implementation patterns
- ❌ **NOT** for building, linking, or depending on

**The line**: We read their code. We don't use their code.

---

See: `bin/worker-orcd/NO_LLAMA_CPP_RULE.md` for full details.
