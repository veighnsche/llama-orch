# Instructions for Updating Checkpoints 2-7

**To:** Checkpoint Updater  
**From:** TEAM CASCADE  
**Date:** 2025-10-08  
**Subject:** Update Checkpoints 2-7 with Learnings from Checkpoint 0

---

## Overview

We've completed comprehensive updates to **CHECKPOINT_00_FOUNDATION.md** and **CHECKPOINT_01_LAYER_NORM.md** with critical learnings about project structure, imports, single-threaded architecture, and system integration.

**Your task:** Apply these same learnings to **Checkpoints 2-7** to maintain consistency across all documentation.

---

## What We Learned from Checkpoint 0

### 1. **Import Clarity is Critical**
Every file must clearly show which worker-crates it imports:
- `main.rs` → worker-http, worker-common
- `backend/cpu_backend.rs` → worker-http, worker-common, worker-tokenizer
- `model/gpt2.rs` → worker-models, worker-common
- **All layers** → NO worker-crates (pure implementation)

### 2. **Single-Threaded Architecture**
- Must use `tokio::main(flavor = "current_thread")`
- No rayon, no parallel processing
- Sequential request processing
- 10-30% faster than multi-threaded

### 3. **Project Structure Context**
- llorch-cpud is a **worker daemon**, not a standalone server
- Part of larger system (orchestrated by pool-managerd)
- Reuses worker-crates for infrastructure
- Focus on model implementation

### 4. **Checkpoint Flow Visualization**
Show where each checkpoint fits in the overall flow:
```
Checkpoint 0: HTTP Server ✅
    ↓
Checkpoint 1: LayerNorm ✅
    ↓
Checkpoint 2: QKV ← Next
    ↓
...
```

### 5. **Implementation Steps**
Each checkpoint needs step-by-step guide:
- Step 1: Create file
- Step 2: Implement component
- Step 3: Write test
- Step 4: Validate

---

## Checkpoints to Update

### ✅ CHECKPOINT_00_FOUNDATION.md - COMPLETE
### ✅ CHECKPOINT_01_LAYER_NORM.md - COMPLETE
### ⬜ CHECKPOINT_02_QKV_PROJECTION.md - **YOUR TASK**
### ⬜ CHECKPOINT_03_KV_CACHE.md - **YOUR TASK**
### ⬜ CHECKPOINT_04_ATTENTION_SCORES.md - **YOUR TASK**
### ⬜ CHECKPOINT_05_ATTENTION_OUTPUT.md - **YOUR TASK**
### ⬜ CHECKPOINT_06_FFN_OUTPUT.md - **YOUR TASK**
### ⬜ CHECKPOINT_07_FIRST_BLOCK.md - **YOUR TASK**

---

## Standard Template to Apply

For each checkpoint, add these sections:

### Section 1: Enhanced Header

```markdown
# CHECKPOINT X: [Component Name]

**Phase:** [Phase]
**Component:** [Component]
**File:** `src/[path]/[file].rs`
**Imports:** [worker-crates or ndarray only]
**Tolerance:** [tolerance]
**Critical Level:** [level]
**Prerequisites:** ✅ Checkpoint X-1 passed

---

## Purpose

Validate that [component] is correct. This is [why it matters].

**Why This Matters:**
- [Impact on system]
- [Error propagation]
- [Validation benefit]

## When to Check

- **Location:** [Where in code]
- **Input:** [What input]
- **Timing:** Week X, Day Y (after Checkpoint X-1 passes)
- **Before:** Implementing [next component] (Checkpoint X+1)
```

### Section 2: Implementation File

```markdown
## Implementation File

**File:** `src/[path]/[file].rs`

**Imports:**
```rust
use ndarray::{Array2, Array3};
// NO worker-crates imports - pure implementation
// OR: List specific worker-crates if needed
```

**Structure:**
```rust
pub struct [Component] {
    // Fields
}

impl [Component] {
    pub fn new(...) -> Self {
        // Constructor
    }
    
    pub fn forward(&self, ...) -> ... {
        // Implementation
    }
}
```

**Key Points:**
- ✅ Single-threaded (no rayon, no parallel)
- ✅ Pure ndarray operations (or specify worker-crates)
- ✅ No worker-crates imports (unless needed)
- ✅ Simple, focused implementation
```

### Section 3: Implementation Steps

```markdown
## Implementation Steps

### Step 1: Create File Structure
```bash
touch src/[path]/[file].rs
```

### Step 2: Implement [Component]
```rust
// src/[path]/[file].rs
// Show code example
```

### Step 3: Write Test
```rust
// tests/checkpoint_0X_[name].rs
#[test]
fn checkpoint_0X_matches_reference() {
    // Load reference output
    let reference = load_reference("checkpoint_0X.npy");
    
    // Run our implementation
    let component = [Component]::new(...);
    let output = component.forward(&input);
    
    // Compare
    assert_tensors_close(&output, &reference, [tolerance]);
}
```

### Step 4: Validate
```bash
cargo test checkpoint_0X
```
```

### Section 4: Integration Context

```markdown
## Integration with Overall System

**Where This Fits:**
```
Checkpoint 0: HTTP Server ✅
    ↓
Checkpoint 1: LayerNorm ✅
    ↓
Checkpoint X: [Component] ← YOU ARE HERE
    ↓
Checkpoint X+1: [Next Component]
    ↓
...
```

**Files Involved:**
- `src/[path]/[file].rs` - Implementation
- `tests/checkpoint_0X_[name].rs` - Validation
- `src/[path]/mod.rs` - Export [Component]

**Dependencies:**
- **Depends on:** Checkpoint X-1 ([previous component])
- **Used by:** Checkpoint X+1 ([next component])

**No HTTP Server Changes Needed:**
- HTTP server from Checkpoint 0 still works
- This is pure model implementation
- No changes to main.rs or backend
```

### Section 5: Next Steps

```markdown
## Next Steps

If this checkpoint **PASSES**:
- ✅ [Component] is correct
- ✅ [What's validated]
- ✅ Proceed to Checkpoint X+1 ([next component])
- ✅ [Confidence statement]

If this checkpoint **FAILS**:
- ❌ Fix [component] implementation before proceeding
- ❌ Do not continue - errors will compound
- ❌ Debug: Check [specific things to check]
- ❌ Compare intermediate values with reference
```

---

## Specific Instructions for Each Checkpoint

### CHECKPOINT_02_QKV_PROJECTION.md

**File:** `src/layers/attention/qkv.rs`

**Imports:**
```rust
use ndarray::{Array2, Array3};
// NO worker-crates imports
```

**Key Points:**
- Part of attention module (see ATTENTION_MODULE_STRUCTURE.md)
- Splits combined QKV weights into Q, K, V
- Input: [batch, seq, dim]
- Output: (Q, K, V) each [batch, seq, n_heads, head_dim]

**Prerequisites:** Checkpoint 1 (LayerNorm) must pass

**Next:** Checkpoint 3 (KV Cache)

---

### CHECKPOINT_03_KV_CACHE.md

**File:** `src/cache/kv_cache.rs`

**Imports:**
```rust
use ndarray::Array3;
// NO worker-crates imports
```

**Key Points:**
- **Top-level cache module** (see KV_CACHE_MODULE_ANALYSIS.md)
- NOT in layers/attention/ - it's in src/cache/
- Simple implementation for MVP
- Room to grow (paged attention later)
- Manages K and V tensors across generation steps

**Prerequisites:** Checkpoint 2 (QKV) must pass

**Next:** Checkpoint 4 (Attention Scores)

**IMPORTANT:** Emphasize that cache is top-level because:
- Used by all attention layers (24 layers)
- Future optimization target (paged attention)
- Signals engineering investment area

---

### CHECKPOINT_04_ATTENTION_SCORES.md

**File:** `src/layers/attention/scores.rs`

**Imports:**
```rust
use ndarray::{Array3, Array4};
// NO worker-crates imports
```

**Key Points:**
- Part of attention module
- Computes scaled dot-product attention
- Q @ K^T / sqrt(head_dim)
- Applies causal mask
- Applies softmax

**Prerequisites:** Checkpoint 3 (Cache) must pass

**Next:** Checkpoint 5 (Attention Output)

---

### CHECKPOINT_05_ATTENTION_OUTPUT.md

**File:** `src/layers/attention/output.rs`

**Imports:**
```rust
use ndarray::{Array2, Array3, Array4};
// NO worker-crates imports
```

**Key Points:**
- Part of attention module
- Applies attention weights to values
- Projects output back to model dimension
- Final step of attention mechanism

**Prerequisites:** Checkpoint 4 (Scores) must pass

**Next:** Checkpoint 6 (FFN)

---

### CHECKPOINT_06_FFN_OUTPUT.md

**File:** `src/layers/ffn.rs`

**Imports:**
```rust
use ndarray::Array2;
// NO worker-crates imports
```

**Key Points:**
- Feedforward network (MLP)
- Two linear layers with GELU activation
- up_proj (c_fc): dim → 4*dim
- down_proj (c_proj): 4*dim → dim

**Prerequisites:** Checkpoint 5 (Attention Output) must pass

**Next:** Checkpoint 7 (Transformer Block)

---

### CHECKPOINT_07_FIRST_BLOCK.md

**File:** `src/layers/transformer.rs`

**Imports:**
```rust
use crate::layers::{LayerNorm, Attention, FFN};
use crate::cache::KVCache;
use ndarray::Array2;
// Internal imports only - NO worker-crates
```

**Key Points:**
- Combines all components (LayerNorm, Attention, FFN)
- Pre-norm architecture (LayerNorm before sublayer)
- Residual connections
- **This validates entire architecture!**

**Prerequisites:** Checkpoint 6 (FFN) must pass

**Next:** Checkpoint 8 (Full Logits)

**IMPORTANT:** This is a major milestone - validates all layers work together!

---

## Reference Documents

As you update each checkpoint, reference these documents:

1. **CHECKPOINT_00_FOUNDATION.md** - Template and structure
2. **CHECKPOINT_01_LAYER_NORM.md** - Example of updated checkpoint
3. **ATTENTION_MODULE_STRUCTURE.md** - How attention is split
4. **KV_CACHE_MODULE_ANALYSIS.md** - Why cache is top-level
5. **SINGLE_THREADED_ARCHITECTURE.md** - Single-threaded requirement
6. **WORKER_CRATES_IMPORT_MAP.md** - Import guidelines
7. **PROJECT_STRUCTURE_COMPARISON.md** - Why our structure differs
8. **SYSTEM_ALIGNMENT_CHECK.md** - System requirements

---

## Quality Checklist

For each checkpoint you update, verify:

### ✓ Header Section
- [ ] File path specified
- [ ] Imports clearly listed
- [ ] Prerequisites noted
- [ ] Tolerance specified

### ✓ Purpose Section
- [ ] Why this matters explained
- [ ] Error propagation noted
- [ ] Timing specified (Week X, Day Y)

### ✓ Implementation File Section
- [ ] File structure shown
- [ ] Imports listed (worker-crates or ndarray)
- [ ] Code example provided
- [ ] Key points listed (single-threaded, pure implementation)

### ✓ Implementation Steps Section
- [ ] Step 1: Create file
- [ ] Step 2: Implement component
- [ ] Step 3: Write test
- [ ] Step 4: Validate

### ✓ Integration Section
- [ ] "Where This Fits" diagram
- [ ] Files involved listed
- [ ] Dependencies noted
- [ ] "No HTTP Server Changes" clarified

### ✓ Next Steps Section
- [ ] Pass criteria
- [ ] Fail criteria
- [ ] Debug guidance

---

## Common Patterns to Apply

### Pattern 1: Import Clarity

**Always show imports at the top:**
```markdown
**Imports:**
```rust
use ndarray::{Array2, Array3};
// NO worker-crates imports - pure implementation
```
```

### Pattern 2: Single-Threaded Emphasis

**Always mention in Key Points:**
```markdown
**Key Points:**
- ✅ Single-threaded (no rayon, no parallel)
- ✅ Pure ndarray operations
- ✅ No worker-crates imports
```

### Pattern 3: Prerequisites

**Always link to previous checkpoint:**
```markdown
**Prerequisites:** ✅ Checkpoint X-1 passed

## Next Steps

If this checkpoint **FAILS**:
- ❌ Fix before proceeding
- ❌ Do not continue - errors will compound
```

### Pattern 4: Integration Context

**Always show the flow:**
```markdown
## Integration with Overall System

**Where This Fits:**
```
Checkpoint X-1: [Previous] ✅
    ↓
Checkpoint X: [Current] ← YOU ARE HERE
    ↓
Checkpoint X+1: [Next]
```
```

---

## Estimated Time

- **Per checkpoint:** 30-45 minutes
- **Total for 6 checkpoints:** 3-4.5 hours

---

## Deliverables

When complete, all checkpoints should have:

1. ✅ Enhanced header with file path and imports
2. ✅ Implementation file section with code examples
3. ✅ Step-by-step implementation guide
4. ✅ Integration context with flow diagram
5. ✅ Next steps with pass/fail criteria
6. ✅ Consistent formatting and structure

---

## Questions?

If you have questions while updating:

1. **Check CHECKPOINT_01** - It's the complete example
2. **Check reference docs** - Listed above
3. **Follow the template** - Consistency is key
4. **Ask if unsure** - Better to clarify than guess

---

## Final Notes

### Remember:

- **Consistency is critical** - All checkpoints should follow same structure
- **Import clarity matters** - Always show which worker-crates are used
- **Single-threaded is key** - Emphasize in every checkpoint
- **Integration context helps** - Show where each checkpoint fits
- **Prerequisites prevent errors** - Always link to previous checkpoint

### The Goal:

Create a **cohesive set of checkpoints** that guide developers through implementing llorch-cpud with:
- Clear structure
- Consistent formatting
- Implementation guidance
- Validation steps
- System context

---

## Performance Measurement Scaffolding

**Note from User:** We might need extra scaffolding just for measuring performance and results.

### Considerations for Future Implementation:

**Performance Measurement Tools:**
- Checkpoint timing (measure each component's execution time)
- Memory profiling (track allocations per checkpoint)
- Throughput metrics (tokens/second)
- Latency tracking (per-token generation time)

**Result Validation Tools:**
- Numerical accuracy tracking (track drift from reference)
- Statistical analysis (mean, variance, max error per checkpoint)
- Regression detection (compare against baseline)
- Visual diff tools (for tensor comparisons)

**Scaffolding Structure:**
```rust
// Example: Performance measurement wrapper
#[cfg(feature = "benchmark")]
pub struct CheckpointProfiler {
    timings: HashMap<String, Duration>,
    memory: HashMap<String, usize>,
}

impl CheckpointProfiler {
    pub fn measure<F, R>(&mut self, checkpoint: &str, f: F) -> R
    where F: FnOnce() -> R {
        let start = Instant::now();
        let result = f();
        self.timings.insert(checkpoint.to_string(), start.elapsed());
        result
    }
}
```

**Integration Points:**
- Add `#[cfg(feature = "benchmark")]` sections to checkpoints
- Create `benches/` directory for criterion benchmarks
- Add `--features benchmark` flag for performance testing
- Document performance baselines in checkpoint files

**This is optional and can be added later** - focus on correctness first, performance measurement second.

---

**Good luck! You've got this! 🚀**

---

Built by TEAM CASCADE 🌊
