# CHECKPOINT 3: KV Cache State

**Phase:** 5.3 - Attention Mechanism  
**Component:** KV Cache Management  
**File:** `src/cache/kv_cache.rs`  
**Imports:** ndarray only (NO worker-crates)  
**Tolerance:** Exact  
**Critical Level:** 🔴 CRITICAL - Generation breaks after first token  
**Prerequisites:** ✅ Checkpoint 2 (QKV Projection) passed

---

## Purpose

Validate that KV cache is correctly initialized, updated, and retrieved. Cache errors compound over generation.

**Why This Matters:**
- KV cache enables autoregressive generation (reuses past keys/values)
- Used by all 24 transformer blocks
- Errors compound: wrong cache at token 1 breaks all subsequent tokens
- Cache is top-level module (signals engineering investment area)
- Future optimization target (paged attention, quantization)

## When to Check

- **Location:** After first token generation (start_pos=1)
- **Input:** K, V from Checkpoint 2
- **Timing:** Week 2, Day 2 (after Checkpoint 2 passes)
- **Before:** Implementing attention scores (Checkpoint 4)

## Validation Checklist

### ✓ Cache Initialization
- [ ] Cache created on first use
- [ ] Shape: `[2, batch, MAX_CONTEXT, n_heads, head_dim]`
- [ ] First dim: 0=keys, 1=values
- [ ] Initialized with zeros
- [ ] Contiguous memory layout (`.contiguous()`)
- [ ] Realized/allocated (tinygrad `.realize()`)

### ✓ Cache Update
- [ ] Correct slice indexing: `[start_pos:start_pos+seqlen]`
- [ ] K stored at cache[0]
- [ ] V stored at cache[1]
- [ ] Assignment successful
- [ ] Memory contiguous after update

### ✓ Cache Retrieval
- [ ] Retrieved K shape: `[batch, start_pos+seqlen, n_heads, head_dim]`
- [ ] Retrieved V shape: `[batch, start_pos+seqlen, n_heads, head_dim]`
- [ ] Contains all previous tokens
- [ ] No data corruption

### ✓ Cross-Reference (Real GPT-2 Validation)
- [ ] Load REAL GPT-2 weights from HuggingFace
- [ ] Use REAL embeddings from "Hello." tokens [15496, 13]
- [ ] Compare cached K/V with HuggingFace transformers reference
- [ ] Cache state exact match (no tolerance for cache)
- [ ] Run negative tests: wrong cache indexing should fail
- [ ] Run determinism test: bit-exact across runs

## Reference Locations

**Tinygrad:** `gpt2.py` lines 34-48  
**Candle:** `bigcode.rs` lines 223-248  
**Mistral.rs:** `kv_cache/mod.rs` lines 69-115

## Common Failures

- ❌ Wrong indexing (off-by-one)
- ❌ Not contiguous
- ❌ Cache not initialized
- ❌ Wrong retrieval slice

## Success Criteria

- ✅ Cache contains exactly 1 token after first generation
- ✅ K/V values match reference exactly
- ✅ No NaN/Inf in cache

---

## Implementation File

**File:** `src/cache/kv_cache.rs`

**Imports:**
```rust
use ndarray::Array3;
// NO worker-crates imports - pure implementation
```

**Structure:**
```rust
pub struct KVCache {
    cache: Option<Array3<f32>>,  // [2, max_seq, n_heads, head_dim]
    max_seq_len: usize,
    n_heads: usize,
    head_dim: usize,
}

impl KVCache {
    pub fn new(max_seq_len: usize, n_heads: usize, head_dim: usize) -> Self {
        Self {
            cache: None,
            max_seq_len,
            n_heads,
            head_dim,
        }
    }
    
    pub fn update(&mut self, k: &Array3<f32>, v: &Array3<f32>, start_pos: usize) {
        // Initialize cache on first use
        // Update cache at position start_pos
    }
    
    pub fn get(&self, end_pos: usize) -> (Array3<f32>, Array3<f32>) {
        // Retrieve K and V up to end_pos
        // Returns: (keys, values) each [batch, seq, n_heads, head_dim]
    }
}
```

**Key Points:**
- ✅ Single-threaded (no rayon, no parallel)
- ✅ Pure ndarray operations
- ✅ NO worker-crates imports
- ✅ Top-level module (NOT in layers/attention/)
- ✅ Simple implementation for MVP
- ✅ Room to grow (paged attention later)

**Why Top-Level:**
- Used by all 24 attention layers
- Future optimization target (paged attention, quantization)
- Signals engineering investment area
- See KV_CACHE_MODULE_ANALYSIS.md for details

---

## Implementation Steps

### Step 1: Create File Structure
```bash
mkdir -p src/cache
touch src/cache/kv_cache.rs
touch src/cache/mod.rs
```

### Step 2: Implement KV Cache
```rust
// src/cache/kv_cache.rs
use ndarray::{Array3, Array4, s};

pub struct KVCache {
    cache: Option<Array4<f32>>,  // [2, batch, max_seq, n_heads, head_dim]
    max_seq_len: usize,
    n_heads: usize,
    head_dim: usize,
}

impl KVCache {
    pub fn new(max_seq_len: usize, n_heads: usize, head_dim: usize) -> Self {
        Self {
            cache: None,
            max_seq_len,
            n_heads,
            head_dim,
        }
    }
    
    pub fn update(&mut self, k: &Array3<f32>, v: &Array3<f32>, start_pos: usize) {
        let batch = k.shape()[0];
        let seq_len = k.shape()[1];
        
        // Initialize cache on first use
        if self.cache.is_none() {
            self.cache = Some(Array4::zeros((2, batch, self.max_seq_len, self.n_heads, self.head_dim)));
        }
        
        // Update cache
        let cache = self.cache.as_mut().unwrap();
        cache.slice_mut(s![0, .., start_pos..start_pos+seq_len, .., ..]).assign(k);
        cache.slice_mut(s![1, .., start_pos..start_pos+seq_len, .., ..]).assign(v);
    }
    
    pub fn get(&self, end_pos: usize) -> (Array3<f32>, Array3<f32>) {
        let cache = self.cache.as_ref().unwrap();
        let k = cache.slice(s![0, .., ..end_pos, .., ..]).to_owned();
        let v = cache.slice(s![1, .., ..end_pos, .., ..]).to_owned();
        (k, v)
    }
}
```

### Step 3: Write Tests (Positive + Negative)

**Positive Test:**
```rust
// tests/real_gpt2_checkpoint_03.rs
#[test]
fn test_checkpoint_03_real_gpt2() {
    let dir = weights_dir();
    
    // Load REAL K, V from Checkpoint 2
    let k: Array3<f32> = load_npy(dir.join("checkpoint_02_k.npy"));
    let v: Array3<f32> = load_npy(dir.join("checkpoint_02_v.npy"));
    
    // Create cache and update
    let mut cache = KVCache::new(2048, 12, 64);  // GPT-2 base: 12 heads
    cache.update(&k, &v, 0);
    
    // Retrieve
    let (cached_k, cached_v) = cache.get(2);
    
    // Compare with original (should be exact)
    assert_tensors_exact(&cached_k, &k);
    assert_tensors_exact(&cached_v, &v);
    
    println!("✅ PASS: KV cache stores and retrieves correctly");
}
```

**Negative Test:**
```rust
#[test]
#[should_panic]
fn test_wrong_cache_indexing_fails() {
    // Wrong start_pos should produce wrong output
    cache.update(&k, &v, 1);  // WRONG: should be 0
    let (cached_k, _) = cache.get(2);
    assert_tensors_exact(&cached_k, &k);  // Should fail
}
```

### Step 4: Validate with Real GPT-2
```bash
# Positive test
cargo test --test real_gpt2_checkpoint_03 -- --nocapture

# Negative tests
cargo test --test proof_negative_checkpoint_03 -- --nocapture
```

**Expected:**
- Positive test: ✅ PASS (exact match)
- Negative tests: ❌ All should panic/fail

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
Checkpoint 3: KV Cache ← YOU ARE HERE
    ↓
Checkpoint 4: Attention Scores
    ↓
Checkpoint 5: Attention Output
    ↓
...
```

**Files Involved:**
- `src/cache/kv_cache.rs` - Implementation
- `tests/checkpoint_03_kv_cache.rs` - Validation
- `src/cache/mod.rs` - Export KVCache

**Dependencies:**
- **Depends on:** Checkpoint 2 (QKV Projection - provides K, V)
- **Used by:** Checkpoint 4 (Attention Scores - uses cached K, V)

**No HTTP Server Changes Needed:**
- HTTP server from Checkpoint 0 still works
- This is pure model implementation
- No changes to main.rs or backend

**Important Note:**
- Cache is **top-level** (src/cache/), NOT in layers/attention/
- This signals it's a future optimization target
- All 24 attention layers share the same cache structure
- See KV_CACHE_MODULE_ANALYSIS.md for architectural reasoning

---

## Next Steps

If this checkpoint **PASSES**:
- ✅ KV cache is correct
- ✅ Cache initialization works
- ✅ Cache update works
- ✅ Cache retrieval works
- ✅ Proceed to Checkpoint 4 (Attention Scores)
- ✅ Ready for autoregressive generation

If this checkpoint **FAILS**:
- ❌ Fix KV cache before proceeding
- ❌ Do not continue - generation will break after first token
- ❌ Debug: Check initialization, update indexing, retrieval slicing
- ❌ Verify cache is contiguous and properly allocated
- ❌ Compare cache state with reference at each step

---

## Notes

- Cache is top-level module (src/cache/), not in layers/attention/
- Simple implementation for MVP
- Room to grow: paged attention, quantization, multi-layer sharing
- Cache errors compound over generation
- Must be exact - no tolerance for numerical errors
- See KV_CACHE_MODULE_ANALYSIS.md for why cache is top-level
