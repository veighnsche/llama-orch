# 🎨 TEAM PICASSO - Numeric Parity Logging System

**Created:** 2025-10-07T15:38Z  
**Purpose:** Enable systematic comparison between llama.cpp (ground truth) and our CUDA engine  
**Status:** ✅ IMPLEMENTED AND TESTED

---

## 📚 Table of Contents

1. [Overview](#overview)
2. [Why This Exists](#why-this-exists)
3. [Quick Start](#quick-start)
4. [Architecture](#architecture)
5. [Usage Examples](#usage-examples)
6. [Troubleshooting](#troubleshooting)
7. [Future Enhancements](#future-enhancements)

---

## Overview

The Numeric Parity Logging System allows us to capture intermediate computation results from both llama.cpp and our CUDA engine, then compare them to identify where our implementation diverges.

### Key Features

- ✅ **Zero-cost when disabled** - No runtime overhead without the feature flag
- ✅ **Append-only JSONL** - Easy to parse, stream-friendly format
- ✅ **Automatic flush** - Writes to disk at program exit (no manual cleanup)
- ✅ **Silent operation** - No stdout/stderr pollution during generation
- ✅ **Safe numeric handling** - Gracefully handles inf/nan values
- ✅ **Thread-safe** - Can be called from multiple threads (Rust only)

---

## Why This Exists

### The Problem

From TEAM PICASSO's investigation (2025-10-07):

```
llama.cpp (CUBLAS_OP_T) → "Powerful cores, CUDA threads dance, GPU shines." ✅
Our code  (CUBLAS_OP_T) → "erne)initĠstatusĹ[ofvoluciÃ³n..." ❌
```

**Same model. Same cuBLAS parameters. Different results.**

This proves the bug is NOT in cuBLAS operation type - it's in:
- Weight loading/dequantization
- Matrix dimension interpretation  
- Memory layout assumptions
- Or some other subsystem

### The Solution

By logging intermediate values at matching checkpoints in both implementations, we can:

1. **Identify the first divergence point** - Binary search through layers
2. **Compare numeric outputs** - See exact value differences
3. **Validate fixes** - Confirm changes bring us closer to ground truth
4. **Prevent regressions** - Automated parity tests

---

## Quick Start

### Prerequisites

- llama.cpp built with ORCH_LOGGING (already configured)
- worker-orcd with `orch_logging` feature
- FP16 model file (qwen2.5-0.5b-instruct-fp16.gguf)

### Step 1: Generate llama.cpp Ground Truth

```bash
cd reference/llama.cpp

# Clean any old logs
rm -f /tmp/llama_hidden_states.jsonl

# Run with logging enabled
ORCH_LOG_FILE=/tmp/llama_hidden_states.jsonl \
ORCH_LOG_TEAM="llama.cpp" \
ORCH_LOG_VALUES=10 \
./build/bin/llama-cli \
  -m ../../.test-models/qwen/qwen2.5-0.5b-instruct-fp16.gguf \
  -p "Write a haiku about GPU computing" \
  -n 10 \
  --temp 0.7 \
  --top-k 0 \
  --top-p 1.0 \
  -no-cnv \
  </dev/null > /tmp/llama_output.log 2>&1

# Verify output
wc -l /tmp/llama_hidden_states.jsonl
head -1 /tmp/llama_hidden_states.jsonl | python3 -m json.tool
```

**Expected output:**
```
14 /tmp/llama_hidden_states.jsonl
{
    "checkpoint": "logits",
    "team": "llama.cpp",
    "token_idx": 4,
    "dtype": "f32",
    "shape": "[151936]",
    "values": [1.23, 4.56, 7.89, ...]
}
```

### Step 2: Generate Our Engine Output

```bash
cd bin/worker-orcd

# Clean any old logs
rm -f /tmp/our_hidden_states.jsonl

# Run with logging enabled
ORCH_LOG_FILE=/tmp/our_hidden_states.jsonl \
ORCH_LOG_TEAM="worker-orcd" \
ORCH_LOG_VALUES=10 \
REQUIRE_REAL_LLAMA=1 \
cargo test --test haiku_generation_anti_cheat \
  --features cuda,orch_logging \
  --release \
  -- --ignored --nocapture --test-threads=1

# Verify output
wc -l /tmp/our_hidden_states.jsonl
head -1 /tmp/our_hidden_states.jsonl | python3 -m json.tool
```

### Step 3: Compare Outputs

```bash
# Quick visual comparison
echo "=== llama.cpp first entry ==="
head -1 /tmp/llama_hidden_states.jsonl | python3 -m json.tool | head -15

echo "=== our engine first entry ==="
head -1 /tmp/our_hidden_states.jsonl | python3 -m json.tool | head -15

# Check if checkpoints align
echo "=== llama.cpp checkpoints ==="
cat /tmp/llama_hidden_states.jsonl | jq -r '.checkpoint' | sort | uniq -c

echo "=== our engine checkpoints ==="
cat /tmp/our_hidden_states.jsonl | jq -r '.checkpoint' | sort | uniq -c
```

---

## Architecture

### File Structure

```
reference/llama.cpp/
├── orch_log.hpp                    # C++ header-only logger
└── tools/main/
    ├── main.cpp                    # Logging calls at line 679-700
    └── CMakeLists.txt              # ORCH_LOGGING option (line 6-10)

bin/worker-orcd/
├── src/
│   ├── orch_log.rs                 # Rust logger implementation
│   └── lib.rs                      # Module declaration (line 12-14)
├── Cargo.toml                      # orch_logging feature (line 48-53)
└── investigation-teams/
    ├── PARITY_COMPARISON_SPEC.md   # Comparison methodology
    └── PARITY_LOGGING_README.md    # This file
```

### Data Flow

```
┌─────────────────┐         ┌──────────────────┐
│  llama.cpp      │         │  worker-orcd     │
│  (C++)          │         │  (Rust)          │
└────────┬────────┘         └────────┬─────────┘
         │                           │
         │ ORCH_LOG_JSON_TOKEN       │ orch_log!
         ▼                           ▼
┌─────────────────┐         ┌──────────────────┐
│ orch_log::      │         │ orch_log::       │
│ Logger          │         │ OrchLogger       │
│ (singleton)     │         │ (Mutex)          │
└────────┬────────┘         └────────┬─────────┘
         │                           │
         │ atexit()                  │ Drop
         ▼                           ▼
┌─────────────────────────────────────────────┐
│  llama_hidden_states.jsonl                  │
│  our_hidden_states.jsonl                    │
│  (append-only JSONL files)                  │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  compare_parity.py (future)                 │
│  - Parse JSONL                              │
│  - Align by checkpoint + token_idx          │
│  - Compute metrics (max_diff, mean_diff)    │
│  - Generate report                          │
└─────────────────────────────────────────────┘
```

### JSONL Schema

```json
{
  "checkpoint": "string",     // e.g., "logits", "embedding", "layer_0_output"
  "team": "string",           // "llama.cpp" or "worker-orcd"
  "token_idx": integer,       // Position in sequence (0-indexed)
  "dtype": "string",          // "f32" or "f16"
  "shape": "string",          // e.g., "[151936]", "[896]"
  "values": [float, ...]      // First N values (default: 10)
}
```

---

## Usage Examples

### Example 1: Add Logging to a New Checkpoint (C++)

```cpp
// In reference/llama.cpp/tools/main/main.cpp or similar

#ifdef ORCH_LOGGING
{
    // Get the values you want to log
    float* hidden_states = /* ... */;
    int hidden_dim = 896;
    
    // Format the shape string
    char shape_buf[64];
    snprintf(shape_buf, sizeof(shape_buf), "[%d]", hidden_dim);
    
    // Log it
    ORCH_LOG_JSON_TOKEN("layer_0_output", hidden_states, hidden_dim, "f32", shape_buf, n_past);
}
#endif
```

### Example 2: Add Logging to a New Checkpoint (Rust)

```rust
// In bin/worker-orcd/src/inference/cuda_backend.rs or similar

#[cfg(feature = "orch_logging")]
{
    // Convert GPU data to CPU f32 vec
    let mut hidden_states_f32 = vec![0.0f32; 896];
    unsafe {
        cudaMemcpy(
            hidden_states_f32.as_mut_ptr() as *mut c_void,
            gpu_ptr,
            896 * std::mem::size_of::<f32>(),
            cudaMemcpyDeviceToHost,
        );
    }
    
    // Log it (macro auto-infers shape)
    orch_log!("layer_0_output", &hidden_states_f32, token_idx);
    
    // Or with explicit shape
    orch_log!("layer_0_output", &hidden_states_f32, token_idx, "[896]");
}
```

### Example 3: Manual Flush Before Crash Point

```rust
// If you suspect a crash is coming, flush logs early
#[cfg(feature = "orch_logging")]
{
    use crate::orch_log::flush_logs;
    flush_logs();  // Write everything to disk NOW
}

// ... potentially crashy code ...
```

---

## Troubleshooting

### Problem: No JSONL file created

**Symptoms:**
```bash
$ ls /tmp/llama_hidden_states.jsonl
ls: cannot access '/tmp/llama_hidden_states.jsonl': No such file or directory
```

**Solutions:**
1. Check that `ORCH_LOG_FILE` env var is set:
   ```bash
   echo $ORCH_LOG_FILE  # Should print the file path
   ```

2. For llama.cpp, verify ORCH_LOGGING is enabled:
   ```bash
   cd reference/llama.cpp
   grep -r "ORCH_LOGGING" build/
   # Should show compile definitions
   ```

3. For worker-orcd, verify feature is enabled:
   ```bash
   cargo test --features cuda,orch_logging --release -- --version
   # Should compile without errors
   ```

### Problem: JSONL file is empty

**Symptoms:**
```bash
$ wc -l /tmp/llama_hidden_states.jsonl
0 /tmp/llama_hidden_states.jsonl
```

**Solutions:**
1. Check that logging calls are actually being executed:
   - Add `fprintf(stderr, "[DEBUG] About to log\n");` before ORCH_LOG_JSON
   - Verify the code path is reached

2. Check that program ran to completion (flush happens at exit):
   ```bash
   # Check exit code
   echo $?  # Should be 0
   ```

3. For Rust, check that Drop was called:
   ```rust
   // Add explicit flush before exit
   #[cfg(feature = "orch_logging")]
   crate::orch_log::flush_logs();
   ```

### Problem: Invalid JSON in JSONL file

**Symptoms:**
```bash
$ head -1 /tmp/llama_hidden_states.jsonl | python3 -m json.tool
Extra data: line 2 column 1 (char 263)
```

**Solutions:**
1. Check for inf/nan values (should be handled automatically, but verify):
   ```bash
   grep -E "inf|nan|Infinity" /tmp/llama_hidden_states.jsonl
   ```

2. Verify each line is a complete JSON object:
   ```bash
   cat /tmp/llama_hidden_states.jsonl | while read line; do
       echo "$line" | python3 -m json.tool > /dev/null || echo "Invalid: $line"
   done
   ```

### Problem: Checkpoints don't align between implementations

**Symptoms:**
```bash
$ diff <(cat llama.jsonl | jq -r '.checkpoint' | sort) \
       <(cat ours.jsonl | jq -r '.checkpoint' | sort)
< embedding
> layer_0_output
```

**Solutions:**
1. Use EXACT same checkpoint names in both implementations
2. Check that both are logging at the same points in the pipeline
3. Verify token_idx alignment (should match for same position)

---

## Future Enhancements

### Priority 1: Automated Comparison Script

**File:** `tools/compare_parity.py`

```python
#!/usr/bin/env python3
"""
Compare numeric parity between llama.cpp and worker-orcd.
Usage: python3 compare_parity.py llama.jsonl ours.jsonl
"""

import json
import sys

def compare_parity(llama_file, ours_file):
    # Parse both files
    llama_entries = [json.loads(line) for line in open(llama_file)]
    ours_entries = [json.loads(line) for line in open(ours_file)]
    
    # Align by checkpoint + token_idx
    # Compute max_diff, mean_diff, rel_error
    # Generate report
    
    pass

if __name__ == "__main__":
    compare_parity(sys.argv[1], sys.argv[2])
```

### Priority 2: Intermediate Layer Logging

Add logging hooks at strategic points:
- After each transformer layer (0, 5, 10, 15, 20, 23)
- After RMSNorm operations
- After attention aggregation
- After FFN blocks

This enables binary search to find the first diverging layer.

### Priority 3: Attention Internals Logging

Log Q, K, V, attention scores, and attention output separately:
```cpp
ORCH_LOG_JSON_TOKEN("layer_0_q", q_ptr, q_dim, "f32", "[896]", token_idx);
ORCH_LOG_JSON_TOKEN("layer_0_k", k_ptr, kv_dim, "f32", "[128]", token_idx);
ORCH_LOG_JSON_TOKEN("layer_0_v", v_ptr, kv_dim, "f32", "[128]", token_idx);
ORCH_LOG_JSON_TOKEN("layer_0_attn_scores", scores_ptr, seq_len, "f32", "[1]", token_idx);
```

### Priority 4: Visualization Tools

Generate plots showing:
- Value distribution histograms
- Difference heatmaps
- Divergence timeline (which layer diverges first)

---

## Related Documents

- **TEAM_PICASSO_CUBLAS_RESOLUTION.md** - Why we need parity logging
- **PARITY_COMPARISON_SPEC.md** - Detailed comparison methodology
- **TEAM_PICASSO_CHRONICLE.md** - Investigation log

---

## Support

If you have questions or issues:

1. Check this README first
2. Review the PARITY_COMPARISON_SPEC.md
3. Look at existing usage in:
   - `reference/llama.cpp/tools/main/main.cpp:679-700`
   - `bin/worker-orcd/src/orch_log.rs` (examples in comments)
4. Ask in the investigation team channel

---

**TEAM PICASSO**  
*"When experts disagree, we test everything."*

**Last Updated:** 2025-10-07T15:38Z
