# Capability Matcher SPEC — MCD/ECP Compatibility Checking (WORKER-4xxx)

**Status**: Draft  
**Applies to**: `bin/worker-orcd-crates/capability-matcher/`  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Scope

This crate implements Model Capability Descriptor (MCD) and Engine Capability Profile (ECP) matching to ensure models are only loaded on compatible workers.

**Parent spec**: `bin/worker-orcd/.specs/00_worker-orcd.md`

---

## 1. Model Capability Descriptor (MCD)

- [WORKER-4600] Each model artifact MUST include or reference a Model Capability Descriptor (MCD).
- [WORKER-4601] MCD MUST include: `model_id`, `positional` (RoPE variant), `attention` (MHA/GQA/MQA), `quant` formats, `context_max`, `vocab_size`.
- [WORKER-4602] MCD MAY be embedded in GGUF metadata or provided as sidecar JSON file.
- [WORKER-4603] Workers MUST parse and validate MCD before accepting model in Plan endpoint.

### 1.1 MCD Schema

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

**Fields**:
- `model_id` — Unique model identifier
- `positional` — Positional encoding: `rope_llama`, `rope_neox`, `alibi`, etc.
- `attention` — Attention mechanism: `mha`, `gqa`, `mqa`
- `quant` — Supported quantization formats (array)
- `context_max` — Maximum context length
- `vocab_size` — Vocabulary size

---

## 2. Engine Capability Profile (ECP)

- [WORKER-4610] Each worker MUST advertise an Engine Capability Profile (ECP) at startup.
- [WORKER-4611] ECP MUST include: `worker_id`, `supports_positional`, `supports_attention`, `supports_quant`, `max_context`, `vram_bytes`.
- [WORKER-4612] ECP MUST be reported to `pool-managerd` via readiness callback.
- [WORKER-4613] Workers MUST update ECP if capabilities change (e.g., after kernel upgrade).

### 2.1 ECP Schema

```json
{
  "worker_id": "worker-gpu-0",
  "supports_positional": ["rope_llama", "rope_neox", "alibi"],
  "supports_attention": ["mha", "gqa", "mqa"],
  "supports_quant": ["q4_0", "q5_1", "q8_0"],
  "max_context": 16384,
  "vram_bytes": 24000000000
}
```

---

## 3. Capability Matching

- [WORKER-4620] Workers MUST implement `MCD ⊆ ECP` checker in Plan endpoint.
- [WORKER-4621] Workers MUST reject models if MCD requires capabilities not in ECP.
- [WORKER-4622] Workers MUST return detailed error message on incompatibility: `"Model requires rope_llama, worker only supports rope_neox"`.
- [WORKER-4623] Compatibility checks MUST be deterministic and explicit (no silent fallbacks).

### 3.1 Matching Rules

**Positional encoding**:
- MCD.positional MUST be in ECP.supports_positional

**Attention mechanism**:
- MCD.attention MUST be in ECP.supports_attention

**Quantization**:
- At least one format in MCD.quant MUST be in ECP.supports_quant

**Context length**:
- MCD.context_max MUST be <= ECP.max_context

---

## 4. API

### 4.1 Check Compatibility

```rust
pub fn check_compatibility(
    &self,
    mcd: &ModelCapabilityDescriptor,
    ecp: &EngineCapabilityProfile,
) -> Result<()>
```

- Returns `Ok(())` if compatible
- Returns `Err(CapabilityError::*)` with detailed reason if incompatible

### 4.2 Parse MCD/ECP

```rust
pub fn parse_mcd(&self, json: &str) -> Result<ModelCapabilityDescriptor>
pub fn parse_ecp(&self, json: &str) -> Result<EngineCapabilityProfile>
```

- Parse from JSON string
- Validate schema
- Return typed structs

---

## 5. Error Types

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

## 6. Security Properties

- **TIER 2 Clippy configuration** (high-importance)
- Deny: unwrap, expect, panic, todo
- Explicit compatibility (no silent fallbacks)
- Deterministic checks
- Clear error messages

---

## 7. Dependencies

**Crates used**:
- `serde` — JSON serialization
- `serde_json` — JSON parsing
- `thiserror` — Error types
- `tracing` — Logging

---

## 8. Traceability

**Code**: `bin/worker-orcd-crates/capability-matcher/src/lib.rs`  
**Tests**: `bin/worker-orcd-crates/capability-matcher/tests/`  
**Parent**: `bin/worker-orcd/.specs/00_worker-orcd.md` §7
