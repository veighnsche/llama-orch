# Team BLUE - Fix Implementation

**Date:** 2025-10-06 21:08 UTC  
**Status:** 🔧 **IMPLEMENTING FIX**

---

## Root Cause

Special tokens `<|im_start|>` and `<|im_end|>` are being split by BPE into multiple tokens:
- `<|im_start|>` → `<` + `|` + `im` + `_start` + `|` + `>` (6 tokens!)
- Should be: **single token ID** (e.g., 151644)

Model was trained with these as atomic tokens, so when we feed split tokens, it doesn't recognize the chat format.

---

## Fix Strategy

Modify `bin/worker-crates/worker-tokenizer/src/encoder.rs`:

1. Before running BPE, scan text for special tokens
2. If special token found, look it up directly in vocabulary
3. Only run BPE on non-special text segments

---

## Implementation

Replace the `encode()` function in `encoder.rs` with:

```rust
/// Encode text to token IDs
pub fn encode(&self, text: &str) -> Result<Vec<u32>, EncodeError> {
    if text.is_empty() {
        return Ok(vec![]);
    }

    // [TEAM BLUE] 2025-10-06T21:07Z - CRITICAL FIX
    // Handle special tokens atomically before BPE
    let mut token_ids = Vec::new();
    let mut remaining = text;
    
    while !remaining.is_empty() {
        let mut found_special = false;
        
        // Check for Qwen2.5 ChatML special tokens
        for special in &["<|im_start|>", "<|im_end|>", "
