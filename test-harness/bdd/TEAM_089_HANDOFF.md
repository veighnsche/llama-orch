# TEAM-089 HANDOFF - Fix GGUF Metadata Issue

**From:** TEAM-088  
**Date:** 2025-10-11  
**Status:** 🔴 BLOCKED - GGUF files missing required metadata, preventing all inference

---

## Mission

**Fix the GGUF metadata issue so workers can load models and perform inference.**

---

## Current State

### ✅ What Works (TEAM-088 Delivered)

**Comprehensive narration plumbing is now in place:**
- ✅ All services log to stdout/stderr (visible debugging)
- ✅ 11 narration events in GGUF loading pipeline
- ✅ Human-friendly log format by default
- ✅ Fast failure mode: `RBEE_NO_RETRY=1`
- ✅ Silent mode: `RBEE_SILENT=1`
- ✅ Complete error context with all metadata keys listed

**You can now see EXACTLY what's happening:**
```bash
RBEE_NO_RETRY=1 ./target/release/rbee infer --node localhost --model "tinyllama" --prompt "test" --max-tokens 50
```

### ❌ What's Broken

**Worker crashes during model loading:**
```
INFO actor="model-loader" action="gguf_metadata_missing" target=llama.vocab_size
     human="Missing llama.vocab_size in GGUF metadata"
     error_kind="missing_metadata_field"

ERROR GGUF metadata missing required field
      required_key="llama.vocab_size"
      available_keys=["tokenizer.chat_template", "llama.attention.layer_norm_rms_epsilon", 
                      "tokenizer.ggml.bos_token_id", "llama.context_length", 
                      "tokenizer.ggml.eos_token_id", "tokenizer.ggml.merges", 
                      "tokenizer.ggml.tokens", "llama.feed_forward_length", 
                      "general.file_type", "general.architecture", "general.name", 
                      "llama.attention.head_count", "tokenizer.ggml.model", 
                      "llama.embedding_length", "llama.rope.dimension_count", 
                      "llama.block_count", "tokenizer.ggml.scores", 
                      "tokenizer.ggml.token_type", "tokenizer.ggml.unknown_token_id", 
                      "general.quantization_version", "llama.attention.head_count_kv", 
                      "llama.rope.freq_base", "tokenizer.ggml.padding_token_id"]
```

**Root Cause:** The GGUF files in `.test-models/` are missing the `llama.vocab_size` metadata field that our loader requires.

---

## The Problem (Clearly Visible Now!)

**File:** `bin/llm-worker-rbee/src/backend/models/quantized_llama.rs`  
**Line:** ~105-134

```rust
// Extract metadata
let vocab_size = content
    .metadata
    .get("llama.vocab_size")
    .and_then(|v| v.to_u32().ok())
    .with_context(|| {
        // TEAM-088: Narrate missing vocab_size with helpful context
        let available_keys: Vec<String> = content.metadata.keys().map(|k| k.to_string()).collect();
        narrate(NarrationFields {
            actor: "model-loader",
            action: "gguf_metadata_missing",
            target: "llama.vocab_size".to_string(),
            human: "Missing llama.vocab_size in GGUF metadata".to_string(),
            cute: Some("Oh no! The GGUF file is missing vocab_size! 😟🔍".to_string()),
            error_kind: Some("missing_metadata_field".to_string()),
            ..Default::default()
        });
        
        format!(
            "Missing llama.vocab_size in GGUF metadata. Available keys: [{}]. \
             This GGUF file may be incomplete or corrupted. Try downloading a fresh copy from HuggingFace.",
            available_keys.join(", ")
        )
    })?
    as usize;
```

**The issue:** We require `llama.vocab_size` but the GGUF files don't have it. However, we DO have `tokenizer.ggml.tokens` which is an array of all tokens!

---

## Solution Options

### Option A: Derive vocab_size from tokenizer.ggml.tokens (RECOMMENDED)

**Rationale:** The vocab size is just the length of the tokens array. This is a safe fallback.

**Implementation:**
```rust
// Extract metadata - with fallback to tokenizer.ggml.tokens length
let vocab_size = content
    .metadata
    .get("llama.vocab_size")
    .and_then(|v| v.to_u32().ok())
    .or_else(|| {
        // TEAM-089: Fallback - derive vocab_size from tokenizer.ggml.tokens array
        content.metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| {
                if let Some(arr) = v.as_array() {
                    narrate(NarrationFields {
                        actor: "model-loader",
                        action: "gguf_vocab_size_derived",
                        target: "tokenizer.ggml.tokens".to_string(),
                        human: format!("Derived vocab_size={} from tokenizer.ggml.tokens array", arr.len()),
                        cute: Some(format!("Found {} tokens in the tokenizer! Using that as vocab_size! 🔢", arr.len())),
                        ..Default::default()
                    });
                    Some(arr.len() as u32)
                } else {
                    None
                }
            })
    })
    .with_context(|| {
        // TEAM-089: Enhanced error - show we tried both methods
        let available_keys: Vec<String> = content.metadata.keys().map(|k| k.to_string()).collect();
        narrate(NarrationFields {
            actor: "model-loader",
            action: "gguf_vocab_size_missing",
            target: "llama.vocab_size".to_string(),
            human: "Cannot determine vocab_size: missing llama.vocab_size and tokenizer.ggml.tokens".to_string(),
            cute: Some("Oh no! Can't find vocab_size anywhere! 😟🔍".to_string()),
            error_kind: Some("missing_vocab_size".to_string()),
            ..Default::default()
        });
        
        format!(
            "Cannot determine vocab_size. Missing both 'llama.vocab_size' and 'tokenizer.ggml.tokens'. \
             Available keys: [{}]. This GGUF file may be corrupted.",
            available_keys.join(", ")
        )
    })?
    as usize;
```

**Add narration constant:**
```rust
// In bin/llm-worker-rbee/src/narration.rs
pub const ACTION_GGUF_VOCAB_SIZE_DERIVED: &str = "gguf_vocab_size_derived";
```

### Option B: Download Valid GGUF Files

**Replace the broken test files with properly formatted ones from HuggingFace:**

```bash
# Download known-good GGUF from HuggingFace
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -O .test-models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Verify it has the required metadata
# (You'll need a GGUF inspection tool for this)
```

### Option C: Implement SafeTensors Support

**The catalog expects SafeTensors format anyway:**
- Worker already supports SafeTensors for non-quantized models
- Catalog has `.test-models/qwen-safetensors/model.safetensors`
- This is the "correct" long-term solution per the architecture

---

## Testing Your Fix

### Test Command (with narration!)
```bash
RBEE_NO_RETRY=1 ./target/release/rbee infer \
  --node localhost \
  --model "tinyllama" \
  --prompt "Why is the sky blue?" \
  --max-tokens 50
```

### Expected Output (Success)
```
✓ queen-rbee started successfully
✓ rbee-hive started successfully
✅ Worker spawned successfully
INFO actor="model-loader" action="gguf_load_start" - Loading GGUF model
INFO actor="model-loader" action="gguf_file_opened" - GGUF file opened
INFO actor="model-loader" action="gguf_inspect_metadata" - Inspecting GGUF metadata (23 keys found)
INFO actor="model-loader" action="gguf_vocab_size_derived" - Derived vocab_size=32000 from tokenizer.ggml.tokens array
INFO actor="model-loader" action="gguf_metadata_loaded" - GGUF metadata: vocab=32000, eos=2, tensors=291
INFO actor="model-loader" action="gguf_load_weights" - Loading 291 tensors from GGUF
INFO actor="model-loader" action="gguf_load_complete" - GGUF model loaded successfully
✅ Worker ready
Tokens: The sky appears blue because...
```

---

## Files to Modify

### Primary Fix
- **`bin/llm-worker-rbee/src/backend/models/quantized_llama.rs`** (lines 105-134)
  - Add fallback to derive vocab_size from `tokenizer.ggml.tokens` array
  - Add narration for derived vocab_size
  - Update error message to mention both methods tried

### Narration Constants
- **`bin/llm-worker-rbee/src/narration.rs`**
  - Add `ACTION_GGUF_VOCAB_SIZE_DERIVED` constant

### Verification
- **Test with actual GGUF file** to ensure it loads
- **Verify inference works** end-to-end
- **Check narration** shows the derivation happening

---

## Environment Variables (TEAM-088 Added)

**For development:**
```bash
RBEE_NO_RETRY=1          # Fail fast without retries
LLORCH_LOG_FORMAT=json   # JSON output for machine parsing (default: pretty)
RBEE_SILENT=1            # Suppress all subprocess logs
```

**For production:**
```bash
LLORCH_LOG_FORMAT=json   # Machine-readable logs for parsing
```

---

## Debugging Tools (TEAM-088 Delivered)

**All narration is now visible!** You can see:
1. Which service is doing what (actor/action)
2. Step-by-step progress through GGUF loading
3. All available metadata keys when errors occur
4. Error classification (error_kind)
5. Actionable error messages (human field)
6. Cute explanations for stress relief (cute field)

**Example narration event:**
```
INFO actor="model-loader" action="gguf_metadata_keys" target=.test-models/tinyllama/...
     human="GGUF has 23 metadata keys"
     cute="Metadata keys: tokenizer.chat_template, llama.attention.layer_norm_rms_epsilon, ..."
```

---

## Success Criteria

- [ ] Worker loads GGUF model successfully
- [ ] Narration shows vocab_size derivation (if using Option A)
- [ ] Worker starts HTTP server on port 8081
- [ ] Worker calls back to rbee-hive with ready state
- [ ] Inference request completes successfully
- [ ] Tokens are generated and streamed back
- [ ] No timeout errors
- [ ] All narration events visible in logs

---

## Context from TEAM-088

**What we fixed:**
1. ✅ Made all logs visible (no more `/dev/null`)
2. ✅ Added comprehensive GGUF loading narration (11 events)
3. ✅ Added fast-fail mode (`RBEE_NO_RETRY=1`)
4. ✅ Added human-friendly log format (pretty by default)
5. ✅ Listed all available metadata keys on error

**What we found:**
- The GGUF files are missing `llama.vocab_size` metadata
- But they DO have `tokenizer.ggml.tokens` (array of all tokens)
- Vocab size = length of tokens array (simple fix!)

**Files we modified:**
- `bin/rbee-keeper/src/commands/infer.rs` - Added `RBEE_NO_RETRY`
- `bin/rbee-keeper/src/queen_lifecycle.rs` - Inherit stdout/stderr
- `bin/queen-rbee/src/http/inference.rs` - Inherit stdout/stderr for rbee-hive
- `bin/rbee-hive/src/http/workers.rs` - Inherit stdout/stderr for workers
- `bin/llm-worker-rbee/src/backend/models/quantized_llama.rs` - 11 narration events
- `bin/llm-worker-rbee/src/main.rs` - Error handling + log format control
- `bin/llm-worker-rbee/src/narration.rs` - 13 new action constants

---

## Quick Start

```bash
# 1. Make your fix (Option A recommended)
# Edit: bin/llm-worker-rbee/src/backend/models/quantized_llama.rs

# 2. Rebuild
cargo build --release --bin llm-worker-rbee

# 3. Test with narration visible
RBEE_NO_RETRY=1 ./target/release/rbee infer \
  --node localhost \
  --model "tinyllama" \
  --prompt "Why is the sky blue?" \
  --max-tokens 50

# 4. Look for this in the output:
#    INFO actor="model-loader" action="gguf_vocab_size_derived"
#    INFO actor="model-loader" action="gguf_load_complete"
#    ✅ Worker ready
#    Tokens: The sky appears blue because...
```

---

**Created by:** TEAM-088  
**Date:** 2025-10-11  
**Time:** 21:28  
**Next Team:** TEAM-089  
**Priority:** P0 - Blocks all inference functionality

---

**Good luck! The narration will guide you. You can see everything now.** 🔍✨

— TEAM-088 (Narration Plumbing Specialists)
