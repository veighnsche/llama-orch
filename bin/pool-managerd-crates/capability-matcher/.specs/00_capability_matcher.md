# Capability Matcher SPEC — Preflight Model Compatibility (CMATCH-8xxx)

**Status**: Draft  
**Applies to**: `bin/pool-managerd-crates/capability-matcher/`  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Scope

### Purpose

The `capability-matcher` crate validates model compatibility with GPU capabilities BEFORE spawning a worker. It prevents worker startup failures due to incompatible model requirements.

**Why it exists:**
- Pool manager must validate model will work on target GPU before spawning worker
- Prevents wasted worker spawn attempts (fast-fail validation)
- Avoids VRAM allocation for incompatible models

**What it does:**
- Parse Model Capability Descriptor (MCD) from model metadata
- Query Engine Capability Profile (ECP) from GPU/CUDA driver
- Validate MCD ⊆ ECP (model requirements fit GPU capabilities)
- Return compatibility result (pass/fail with reason)

**What it does NOT do:**
- ❌ Allocate VRAM (worker does this after spawn)
- ❌ Load models (worker does this)
- ❌ Execute inference (worker does this)
- ❌ Make placement decisions (orchestratord does this)

---

## 1. Core Responsibilities

### [CMATCH-8001] Preflight Validation
The crate MUST validate model compatibility BEFORE worker spawn.

### [CMATCH-8002] MCD Parsing
The crate MUST parse Model Capability Descriptors from model metadata.

### [CMATCH-8003] ECP Querying
The crate MUST query GPU capabilities to build Engine Capability Profile.

### [CMATCH-8004] Compatibility Check
The crate MUST verify MCD requirements are subset of ECP capabilities.

---

## 2. Model Capability Descriptor (MCD)

### [CMATCH-8010] MCD Structure
Model metadata MUST include:
```json
{
  "model_id": "meta-llama/Llama-3.1-8B",
  "positional": "rope_llama",
  "attention": "gqa",
  "quant": ["q4_0", "q8_0"],
  "context_max": 8192,
  "vocab_size": 128256
}
```

### [CMATCH-8011] MCD Sources
The crate MUST extract MCD from:
- Embedded in model file (GGUF metadata)
- Sidecar JSON file (`.mcd.json`)
- Model catalog entry

---

## 3. Engine Capability Profile (ECP)

### [CMATCH-8020] ECP Structure
GPU capabilities:
```json
{
  "worker_id": "gpu-0-worker",
  "supports_positional": ["rope_llama", "rope_neox", "alibi"],
  "supports_attention": ["mha", "gqa", "mqa"],
  "supports_quant": ["q4_0", "q8_0", "fp16"],
  "max_context": 16384,
  "vram_bytes": 24000000000
}
```

### [CMATCH-8021] ECP Sources
The crate MUST build ECP from:
- CUDA compute capability
- VRAM capacity (from `gpu-inventory`)
- Inference engine capabilities (hardcoded for M0)

---

## 4. Compatibility Validation

### [CMATCH-8030] Validation Rules
The crate MUST validate:
1. **Positional encoding**: `mcd.positional ∈ ecp.supports_positional`
2. **Attention mechanism**: `mcd.attention ∈ ecp.supports_attention`
3. **Quantization**: `∃q ∈ mcd.quant: q ∈ ecp.supports_quant`
4. **Context size**: `mcd.context_max ≤ ecp.max_context`

### [CMATCH-8031] Error Reporting
On incompatibility, the crate MUST return detailed error:
```rust
CapabilityError::IncompatiblePositional {
    required: "rope_neox",
    supported: vec!["rope_llama", "alibi"]
}
```

---

## 5. Integration with Pool Manager

### [CMATCH-8040] Workflow
Pool manager workflow:
1. Orchestratord requests: "Start worker for model X on GPU 0"
2. Pool manager loads model metadata → extract MCD
3. Pool manager queries GPU 0 → build ECP
4. Pool manager calls `CapabilityMatcher::check_compatibility(&mcd, &ecp)`
5. If pass → spawn worker
6. If fail → return error to orchestratord

---

## 6. Error Handling

### [CMATCH-8050] Error Types
```rust
pub enum CapabilityError {
    IncompatiblePositional { required: String, supported: Vec<String> },
    IncompatibleAttention { required: String, supported: Vec<String> },
    IncompatibleQuant { required: Vec<String>, supported: Vec<String> },
    ContextTooLarge { required: usize, max: usize },
    InvalidMcd(String),
    InvalidEcp(String),
}
```

---

## 7. Dependencies

### [CMATCH-8060] Required Crates
```toml
[dependencies]
serde = { workspace = true, features = ["derive"] }
serde_json = { workspace = true }
thiserror = { workspace = true }
tracing = { workspace = true }
```

---

## 8. Traceability

**Code**: `bin/pool-managerd-crates/capability-matcher/src/`  
**Tests**: `bin/pool-managerd-crates/capability-matcher/tests/`  
**Parent**: `bin/pool-managerd/.specs/00_pool-managerd.md`  
**Used by**: `pool-managerd`  
**Spec IDs**: CMATCH-8001 to CMATCH-8060

---

**End of Specification**
