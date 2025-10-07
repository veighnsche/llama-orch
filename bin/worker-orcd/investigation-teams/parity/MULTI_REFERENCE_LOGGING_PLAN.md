# Multi-Reference Logging Plan - TEAM PICASSO

**Date:** 2025-10-07T21:35Z  
**Goal:** Add ORCH logging to ALL viable reference implementations for comprehensive parity testing

---

## 🎯 Mission

**Add the same logits logging infrastructure to multiple reference implementations to:**
1. ✅ Compare our implementation against MULTIPLE ground truths
2. ✅ Identify which reference is most accurate (PyTorch baseline)
3. ✅ Find common patterns vs outliers
4. ✅ Validate our implementation is correct
5. ✅ Build comprehensive parity test suite

---

## 📋 Reference Analysis

### ✅ CAN Add Logging (Inference Engines)

| Reference | Language | Inference | Logging Feasibility | Priority |
|-----------|----------|-----------|---------------------|----------|
| **llama.cpp** | C++ | ✅ Yes | ✅ **DONE** | ✅ Complete |
| **vllm** | Python/C++ | ✅ Yes | ✅ **HIGH** | 🔥 Priority 1 |
| **mistral.rs** | Rust | ✅ Yes | ✅ **HIGH** | 🔥 Priority 2 |
| **candle** | Rust | ✅ Yes | ✅ **MEDIUM** | ⏭️ Priority 3 |
| **text-generation-inference** | Python/Rust | ✅ Yes | ✅ **MEDIUM** | ⏭️ Priority 4 |
| **llamafile** | C++ | ✅ Yes | ✅ **LOW** | ⏭️ Priority 5 |

### ❌ CANNOT Add Logging (Not Inference Engines)

| Reference | Reason |
|-----------|--------|
| **drama_llama** | Experimental, unclear if it does inference |
| **tinygrad** | Framework, not inference engine |
| **flash-attention** | Kernel library, not full inference |

---

## 🔥 Priority 1: vllm (Python/C++)

### Why Priority 1?
- ✅ **Industry standard** for production inference
- ✅ **Python-based** - Easy to add logging
- ✅ **PagedAttention** - Different architecture from llama.cpp
- ✅ **Continuous batching** - Real production patterns
- ✅ **Well-documented** - Easy to understand

### Implementation Plan

**Location:** `/reference/vllm/`

**Step 1: Find Logits Output Point**
```bash
# Search for logits computation
grep -r "logits\|forward.*output" vllm/model_executor/ --include="*.py"
grep -r "sample\|generate" vllm/engine/ --include="*.py"
```

**Step 2: Create Python Logger**
```python
# /reference/vllm/vllm/orch_log.py

import os
import json
import numpy as np
from typing import Optional

class OrchLogger:
    def __init__(self):
        self.log_file = os.getenv('ORCH_LOG_FILE')
        self.team = os.getenv('ORCH_LOG_TEAM', 'vllm')
        self.max_values = int(os.getenv('ORCH_LOG_VALUES', '10'))
        self.enabled = self.log_file is not None
        self.entries = []
    
    def log_logits(self, logits: np.ndarray, token_idx: int):
        """Log logits for a token"""
        if not self.enabled:
            return
        
        # Convert to numpy if needed
        if hasattr(logits, 'cpu'):
            logits = logits.cpu().numpy()
        
        # Take first N values
        values = logits.flatten()[:self.max_values].tolist()
        
        entry = {
            'checkpoint': 'logits',
            'team': self.team,
            'token_idx': token_idx,
            'dtype': 'f32',
            'shape': f'[{len(logits.flatten())}]',
            'values': values
        }
        
        self.entries.append(entry)
    
    def flush(self):
        """Write all entries to file"""
        if not self.enabled or not self.entries:
            return
        
        with open(self.log_file, 'a') as f:
            for entry in self.entries:
                f.write(json.dumps(entry) + '\n')
        
        self.entries.clear()

# Global instance
_logger = OrchLogger()

def log_logits(logits, token_idx):
    _logger.log_logits(logits, token_idx)

def flush():
    _logger.flush()
```

**Step 3: Hook Into Inference Loop**
```python
# Find the main inference loop in vllm/engine/llm_engine.py or similar

# Add at the top
from vllm.orch_log import log_logits, flush
import atexit
atexit.register(flush)

# In the generation loop, after logits are computed:
if os.getenv('ORCH_LOG_FILE'):
    log_logits(logits, token_idx)
```

**Step 4: Test Script**
```bash
#!/bin/bash
# /reference/vllm/test_orch_logging.sh

export ORCH_LOG_FILE=/tmp/vllm_logits.jsonl
export ORCH_LOG_TEAM=vllm
export ORCH_LOG_VALUES=10

python3 << 'EOF'
from vllm import LLM, SamplingParams

llm = LLM(model="openai-community/gpt2")
sampling_params = SamplingParams(temperature=0.0, max_tokens=15)

prompts = ["GPU haiku with word fifty-one: "]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
EOF

echo "✅ Logged to $ORCH_LOG_FILE"
```

**Effort:** 2-4 hours  
**Complexity:** Low (Python is easy)  
**Value:** HIGH (industry standard comparison)

---

## 🔥 Priority 2: mistral.rs (Rust)

### Why Priority 2?
- ✅ **Rust-native** - Same language as worker-orcd
- ✅ **Mistral architecture** - Different from Llama
- ✅ **Quantization focus** - Good for comparing quant strategies
- ✅ **Similar to our stack** - Direct comparison

### Implementation Plan

**Location:** `/reference/mistral.rs/`

**Step 1: Find Logits Output**
```bash
grep -r "logits\|forward" mistralrs-core/src/ --include="*.rs"
grep -r "generate\|sample" mistralrs-core/src/ --include="*.rs"
```

**Step 2: Create Rust Logger**
```rust
// /reference/mistral.rs/mistralrs-core/src/orch_log.rs

use std::env;
use std::fs::OpenOptions;
use std::io::Write;
use serde_json::json;

pub struct OrchLogger {
    log_file: Option<String>,
    team: String,
    max_values: usize,
    entries: Vec<serde_json::Value>,
}

impl OrchLogger {
    pub fn new() -> Self {
        Self {
            log_file: env::var("ORCH_LOG_FILE").ok(),
            team: env::var("ORCH_LOG_TEAM").unwrap_or_else(|_| "mistral.rs".to_string()),
            max_values: env::var("ORCH_LOG_VALUES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10),
            entries: Vec::new(),
        }
    }
    
    pub fn log_logits(&mut self, logits: &[f32], token_idx: usize) {
        if self.log_file.is_none() {
            return;
        }
        
        let values: Vec<f32> = logits.iter().take(self.max_values).copied().collect();
        
        let entry = json!({
            "checkpoint": "logits",
            "team": self.team,
            "token_idx": token_idx,
            "dtype": "f32",
            "shape": format!("[{}]", logits.len()),
            "values": values,
        });
        
        self.entries.push(entry);
    }
    
    pub fn flush(&mut self) {
        if let Some(ref path) = self.log_file {
            if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
                for entry in &self.entries {
                    if let Ok(line) = serde_json::to_string(entry) {
                        let _ = writeln!(file, "{}", line);
                    }
                }
            }
            self.entries.clear();
        }
    }
}

impl Drop for OrchLogger {
    fn drop(&mut self) {
        self.flush();
    }
}

// Global logger
use std::sync::Mutex;
use once_cell::sync::Lazy;

static LOGGER: Lazy<Mutex<OrchLogger>> = Lazy::new(|| {
    Mutex::new(OrchLogger::new())
});

pub fn log_logits(logits: &[f32], token_idx: usize) {
    if let Ok(mut logger) = LOGGER.lock() {
        logger.log_logits(logits, token_idx);
    }
}
```

**Step 3: Hook Into Inference**
```rust
// In the main inference loop
#[cfg(feature = "orch_logging")]
{
    crate::orch_log::log_logits(&logits, token_idx);
}
```

**Step 4: Add Feature Flag**
```toml
# Cargo.toml
[features]
orch_logging = []
```

**Effort:** 3-5 hours  
**Complexity:** Medium (Rust + finding right hook point)  
**Value:** HIGH (Rust comparison, different architecture)

---

## ⏭️ Priority 3: candle (Rust Framework)

### Why Priority 3?
- ✅ **Rust ML framework** - Good for understanding tensor ops
- ✅ **Safetensors** - Different weight loading
- ✅ **Multiple backends** - CUDA/Metal/CPU comparison
- ⚠️ **Framework, not engine** - Need to build inference on top

### Implementation Plan

**Challenge:** Candle is a framework, not an inference engine. We'd need to:
1. Build a simple inference script using candle
2. Add logging to that script
3. Compare tensor operations

**Effort:** 6-10 hours  
**Complexity:** High (need to build inference first)  
**Value:** MEDIUM (more about tensor ops than full inference)

**Recommendation:** Do this AFTER vllm and mistral.rs

---

## ⏭️ Priority 4: text-generation-inference (Python/Rust)

### Why Priority 4?
- ✅ **Production server** - Real-world patterns
- ✅ **HuggingFace official** - Industry standard
- ✅ **gRPC/HTTP** - Different interface
- ⚠️ **Complex architecture** - Harder to instrument

### Implementation Plan

**Similar to vllm but more complex:**
- Python frontend + Rust backend
- Need to instrument both sides
- gRPC adds complexity

**Effort:** 8-12 hours  
**Complexity:** High (distributed architecture)  
**Value:** MEDIUM (similar to vllm but more complex)

**Recommendation:** Do this AFTER vllm (similar but easier)

---

## ⏭️ Priority 5: llamafile (C++)

### Why Priority 5?
- ✅ **Single-file deployment** - Interesting approach
- ✅ **Cross-platform** - Good for portability testing
- ⚠️ **Based on llama.cpp** - Likely similar results
- ⚠️ **Packaging focus** - Not much different from llama.cpp

### Implementation Plan

**Likely very similar to llama.cpp:**
- Same C++ approach
- Same GGUF format
- Probably same logits output

**Effort:** 2-4 hours  
**Complexity:** Low (copy llama.cpp approach)  
**Value:** LOW (likely same as llama.cpp)

**Recommendation:** SKIP unless we need cross-platform validation

---

## 📊 Recommended Execution Order

### Phase 1: Quick Wins (Week 1)
1. ✅ **llama.cpp** - DONE!
2. 🔥 **vllm** - 2-4 hours, HIGH value
3. 🔥 **mistral.rs** - 3-5 hours, HIGH value

**Deliverable:** 3 reference implementations with logging

### Phase 2: Deep Dive (Week 2)
4. ⏭️ **candle** - 6-10 hours, build inference + logging
5. ⏭️ **text-generation-inference** - 8-12 hours, complex but valuable

**Deliverable:** 5 reference implementations

### Phase 3: Optional (If Needed)
6. ⏭️ **llamafile** - 2-4 hours, low priority
7. ⏭️ **PyTorch baseline** - Direct HuggingFace transformers

**Deliverable:** Complete reference suite

---

## 🎯 Success Criteria

### For Each Reference

**Must Have:**
1. ✅ Logging infrastructure in place
2. ✅ Test script that runs same prompt
3. ✅ JSONL output compatible with our analyzer
4. ✅ Documentation in reference README

**Nice to Have:**
1. ✅ Layer-by-layer checkpoints
2. ✅ Multiple model support
3. ✅ Automated comparison script

### Overall Success

**We can:**
1. ✅ Run same prompt on 3+ references
2. ✅ Compare logits across all implementations
3. ✅ Identify outliers (which implementation is wrong?)
4. ✅ Validate worker-orcd against multiple ground truths
5. ✅ Build confidence in our implementation

---

## 🛠️ Shared Infrastructure

### Reusable Components

**1. Analyzer Tool** (Already exists!)
```bash
# Works with ANY JSONL from any reference
./analyze_logits.py /tmp/vllm_logits.jsonl
./analyze_logits.py /tmp/mistralrs_logits.jsonl --compare /tmp/our_logits.jsonl
```

**2. Test Harness**
```bash
#!/bin/bash
# test_all_references.sh

PROMPT="GPU haiku with word fifty-one: "
N_TOKENS=15

# Run all references
./test_llama_cpp.sh "$PROMPT" $N_TOKENS
./test_vllm.sh "$PROMPT" $N_TOKENS
./test_mistralrs.sh "$PROMPT" $N_TOKENS
./test_worker_orcd.sh "$PROMPT" $N_TOKENS

# Compare all
python3 compare_all.py \
  /tmp/llama_cpp.jsonl \
  /tmp/vllm.jsonl \
  /tmp/mistralrs.jsonl \
  /tmp/our.jsonl
```

**3. Comparison Dashboard**
```python
# compare_all.py
# Load all JSONL files
# Show side-by-side comparison
# Highlight outliers
# Generate report
```

---

## 📋 Implementation Checklist

### For Each Reference

- [ ] **Step 1:** Clone/update submodule
- [ ] **Step 2:** Find logits output point
- [ ] **Step 3:** Implement logger (language-specific)
- [ ] **Step 4:** Hook into inference loop
- [ ] **Step 5:** Create test script
- [ ] **Step 6:** Verify output format
- [ ] **Step 7:** Test with analyze_logits.py
- [ ] **Step 8:** Document in reference README
- [ ] **Step 9:** Add to test_all_references.sh

---

## 🎨 TEAM PICASSO Execution Plan

### Week 1: Foundation
- ✅ llama.cpp (DONE)
- 🔥 vllm (2-4 hours)
- 🔥 mistral.rs (3-5 hours)
- ✅ Build comparison infrastructure

### Week 2: Expansion
- ⏭️ candle (6-10 hours)
- ⏭️ text-generation-inference (8-12 hours)
- ✅ Automated testing suite

### Week 3: Validation
- ✅ Run comprehensive parity tests
- ✅ Identify and fix any issues
- ✅ Document findings
- ✅ Build confidence in worker-orcd

---

## 🎯 Expected Outcomes

### What We'll Learn

1. **Which reference is most accurate?**
   - Compare all against PyTorch ground truth
   - Identify outliers

2. **Where do implementations diverge?**
   - Different optimizations
   - Different precision choices
   - Different architectures

3. **Is worker-orcd correct?**
   - Validate against multiple references
   - Build confidence
   - Find and fix bugs

4. **What's acceptable parity?**
   - Define tolerance levels
   - Understand precision trade-offs
   - Document best practices

---

## 🚀 Next Steps

### Immediate (This Week)
1. ✅ **Approve this plan**
2. 🔥 **Start with vllm** (highest value, easiest)
3. 🔥 **Then mistral.rs** (Rust comparison)

### Short Term (Next Week)
4. ⏭️ **Add candle** (if time permits)
5. ⏭️ **Build comparison dashboard**

### Long Term (Future)
6. ⏭️ **Automated CI testing** (run on every commit)
7. ⏭️ **Parity regression tests** (catch drift)

---

**TEAM PICASSO** 🎨  
**Mission:** Multi-reference parity validation  
**Status:** Plan ready for execution  
**Priority:** vllm → mistral.rs → candle → TGI
