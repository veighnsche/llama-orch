# CHECKPOINT 7: First Block Output

**Phase:** 4.2 - Transformer Blocks  
**Component:** Complete TransformerBlock  
**File:** `src/layers/transformer.rs`  
**Imports:** Internal crate imports only (LayerNorm, Attention, FFN, KVCache)  
**Tolerance:** 1e-4  
**Critical Level:** 🟢 VALIDATION - Entire block architecture  
**Prerequisites:** ✅ Checkpoint 6 (FFN) passed

---

## Purpose

Validate complete transformer block: attention + FFN + residuals + layer norms. If this passes, architecture is correct.

**Why This Matters:**
- **MAJOR MILESTONE:** Validates entire transformer architecture
- Combines all components: LayerNorm, Attention, FFN, residuals
- Pre-norm architecture (LayerNorm before sublayer)
- If this passes, remaining 23 blocks will likely work
- Validates residual connections don't break
- Confirms architecture matches reference implementations

## When to Check

- **Location:** After complete first transformer block
- **Input:** Embeddings from Phase 2
- **Timing:** Week 3, Day 3-4 (after Checkpoint 6 passes)
- **Before:** Implementing full model (Checkpoint 8)

## Validation Checklist

### ✓ Block Structure (Pre-Norm)
- [ ] Layer norm 1 applied BEFORE attention
- [ ] Residual 1: `h = x + attention(ln1(x))`
- [ ] Layer norm 2 applied BEFORE FFN
- [ ] Residual 2: `h = h + ffn(ln2(h))`
- [ ] Output contiguous: `.contiguous()`

### ✓ Data Flow
- [ ] Input: embeddings `[1, 2, 1024]`
- [ ] After attention + residual: `[1, 2, 1024]`
- [ ] After FFN + residual: `[1, 2, 1024]`
- [ ] Output: `[1, 2, 1024]` (contiguous)

### ✓ Residual Connections
- [ ] First residual adds attention output to original input
- [ ] Second residual adds FFN output to intermediate result
- [ ] No residual scaling (just addition)
- [ ] Values accumulated correctly

### ✓ Memory Layout
- [ ] Output is contiguous (`.contiguous()` called)
- [ ] No memory fragmentation
- [ ] Ready for next block

### ✓ Value Validation
- [ ] Output values in reasonable range (typically [-3, 3])
- [ ] Not identical to input (block changed values)
- [ ] No NaN/Inf
- [ ] Variance preserved (not collapsed)

### ✓ Cross-Reference
- [ ] Compare output[0, 0, :5] with tinygrad
- [ ] Compare output[0, 0, :5] with Candle
- [ ] Difference within 1e-4

## Reference Locations

**Tinygrad:** `gpt2.py` lines 65-67, 101-107  
**Candle:** `bigcode.rs` lines 323-346  
**Mistral.rs:** Model-specific block implementations

## Common Failures

- ❌ Post-norm instead of pre-norm
- ❌ Residuals not added
- ❌ Output not contiguous
- ❌ Wrong layer norm order

## Success Criteria

- ✅ Pre-norm architecture correct
- ✅ Both residuals applied
- ✅ Output contiguous
- ✅ Matches reference within 1e-4
- ✅ **If this passes, architecture is correct!**

## Next Steps

If this checkpoint **PASSES**:
- ✅ Architecture is correct
- ✅ Remaining blocks will likely work
- ✅ Proceed to Checkpoint 8 (Full Logits)

If this checkpoint **FAILS**:
- ❌ Review block structure
- ❌ Check residual connections
- ❌ Verify layer norm order

---

## Implementation File

**File:** `src/layers/transformer.rs`

**Imports:**
```rust
use crate::layers::{LayerNorm, Attention, FFN};
use crate::cache::KVCache;
use ndarray::Array2;
// Internal imports only - NO worker-crates
```

**Structure:**
```rust
pub struct TransformerBlock {
    ln_1: LayerNorm,
    attn: Attention,
    ln_2: LayerNorm,
    ffn: FFN,
}

impl TransformerBlock {
    pub fn new(ln_1: LayerNorm, attn: Attention, ln_2: LayerNorm, ffn: FFN) -> Self {
        Self { ln_1, attn, ln_2, ffn }
    }
    
    pub fn forward(&mut self, x: &Array2<f32>, cache: &mut KVCache, start_pos: usize) -> Array2<f32> {
        // Pre-norm architecture:
        // 1. h = x + attention(ln_1(x))
        // 2. h = h + ffn(ln_2(h))
        // 3. return h.contiguous()
    }
}
```

**Key Points:**
- ✅ Single-threaded (no rayon, no parallel)
- ✅ Pure ndarray operations
- ✅ NO worker-crates imports (only internal crate imports)
- ✅ Pre-norm architecture (LayerNorm before sublayer)
- ✅ Two residual connections
- ✅ Output is contiguous

---

## Implementation Steps

### Step 1: Create File
```bash
touch src/layers/transformer.rs
```

### Step 2: Implement Transformer Block
```rust
// src/layers/transformer.rs
use crate::layers::{LayerNorm, Attention, FFN};
use crate::cache::KVCache;
use ndarray::Array2;

pub struct TransformerBlock {
    ln_1: LayerNorm,
    attn: Attention,
    ln_2: LayerNorm,
    ffn: FFN,
}

impl TransformerBlock {
    pub fn new(ln_1: LayerNorm, attn: Attention, ln_2: LayerNorm, ffn: FFN) -> Self {
        Self { ln_1, attn, ln_2, ffn }
    }
    
    pub fn forward(&mut self, x: &Array2<f32>, cache: &mut KVCache, start_pos: usize) -> Array2<f32> {
        // First sublayer: attention with residual
        let normalized = self.ln_1.forward(x);
        let attn_output = self.attn.forward(&normalized, cache, start_pos);
        let h = x + &attn_output;
        
        // Second sublayer: FFN with residual
        let normalized = self.ln_2.forward(&h);
        let ffn_output = self.ffn.forward(&normalized);
        let output = &h + &ffn_output;
        
        // Ensure contiguous
        output.as_standard_layout().to_owned()
    }
}
```

### Step 3: Write Test
```rust
// tests/checkpoint_07_first_block.rs
#[test]
fn checkpoint_07_matches_reference() {
    // Load reference
    let reference = load_reference("checkpoint_07_block.npy");
    
    // Create transformer block
    let mut block = TransformerBlock::new(ln_1, attn, ln_2, ffn);
    let mut cache = KVCache::new(2048, 16, 64);
    
    // Run our implementation
    let output = block.forward(&input, &mut cache, 0);
    
    // Compare
    assert_tensors_close(&output, &reference, 1e-4);
}
```

### Step 4: Validate
```bash
cargo test checkpoint_07
```

---

## Integration with Overall System

**Where This Fits:**
```
Checkpoint 0: HTTP Server ✅
    ↓
Checkpoint 1: LayerNorm ✅
    ↓
Checkpoint 2: QKV Projection ✅
    ↓
Checkpoint 3: KV Cache ✅
    ↓
Checkpoint 4: Attention Scores ✅
    ↓
Checkpoint 5: Attention Output ✅
    ↓
Checkpoint 6: FFN Output ✅
    ↓
Checkpoint 7: Transformer Block ← YOU ARE HERE
    ↓
Checkpoint 8: Full Logits
    ↓
...
```

**Files Involved:**
- `src/layers/transformer.rs` - Implementation
- `tests/checkpoint_07_first_block.rs` - Validation
- `src/layers/mod.rs` - Export TransformerBlock

**Dependencies:**
- **Depends on:** All previous checkpoints (1-6)
- **Used by:** Checkpoint 8 (Full Model - stacks 24 blocks)

**No HTTP Server Changes Needed:**
- HTTP server from Checkpoint 0 still works
- This is pure model implementation
- No changes to main.rs or backend

**IMPORTANT:**
- This is a **major milestone**
- If this passes, architecture is validated
- Remaining 23 blocks use same structure
- Validates pre-norm architecture
- Validates residual connections work

---

## Next Steps

If this checkpoint **PASSES**:
- ✅ **ARCHITECTURE IS CORRECT!**
- ✅ All components work together
- ✅ Pre-norm architecture validated
- ✅ Residual connections work
- ✅ Proceed to Checkpoint 8 (Full Logits)
- ✅ Remaining 23 blocks will use same structure
- ✅ **Major confidence boost!**

If this checkpoint **FAILS**:
- ❌ Fix transformer block before proceeding
- ❌ Do not continue - architecture is wrong
- ❌ Debug: Check layer norm order, residual connections, contiguous output
- ❌ Verify pre-norm (LayerNorm before sublayer, not after)
- ❌ Compare intermediate values (after ln_1, after attn, after residual_1, after ln_2, after ffn, after residual_2)
- ❌ Review reference implementations for architecture

---

## Notes

- Pre-norm architecture: LayerNorm BEFORE sublayer (not after)
- Two residual connections: one for attention, one for FFN
- Output must be contiguous (`.contiguous()` or `.as_standard_layout()`)
- This validates entire transformer architecture
- **MAJOR MILESTONE** - if this passes, architecture is correct
- Remaining 23 blocks use identical structure
