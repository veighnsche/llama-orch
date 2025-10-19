# TEAM-130 PRODUCT CODE FIX

**Date:** 2025-10-19  
**Mission:** Add missing error codes to product code to align with BDD tests

---

## 🔧 CHANGES MADE

### File: `bin/llm-worker-rbee/src/common/error.rs`

**Added 2 new error types to `WorkerError` enum:**

```rust
/// TEAM-130: Insufficient resources (RAM, CPU, etc.)
#[error("Insufficient resources: {0}")]
InsufficientResources(String),

/// TEAM-130: Insufficient VRAM for model
#[error("Insufficient VRAM: {0}")]
InsufficientVram(String),
```

**Updated error code mapping:**
- `InsufficientResources` → `"INSUFFICIENT_RESOURCES"`
- `InsufficientVram` → `"INSUFFICIENT_VRAM"`

**Updated retriability:**
- Both errors are retriable (orchestrator can retry on different node)

**Updated HTTP status codes:**
- Both return `503 Service Unavailable`

**Added comprehensive tests:**
- `test_insufficient_resources_error()` - Properties test
- `test_insufficient_vram_error()` - Properties test
- `test_into_response_insufficient_vram()` - HTTP response test
- `test_into_response_insufficient_resources()` - HTTP response test
- Updated `test_error_code_stability()` - Stability verification
- Updated `test_error_message_formatting()` - Message formatting

---

## ✅ VERIFICATION

### Compilation
```bash
cargo check --manifest-path bin/llm-worker-rbee/Cargo.toml
```
**Result:** ✅ SUCCESS (0 errors, 1 warning - unrelated unused import)

### Error Code Alignment

**BDD Tests (errors.rs):**
- Uses `INSUFFICIENT_RESOURCES` ✅
- Uses `INSUFFICIENT_VRAM` ✅
- Returns 503 Service Unavailable ✅
- Errors are retriable ✅

**Product Code (error.rs):**
- Implements `INSUFFICIENT_RESOURCES` ✅
- Implements `INSUFFICIENT_VRAM` ✅
- Returns 503 Service Unavailable ✅
- Errors are retriable ✅

**Status:** 100% aligned ✅

---

## 📊 IMPACT

### Before TEAM-130
- BDD tests used error codes that didn't exist in product
- Tests were "test-first" but product was incomplete

### After TEAM-130
- Product code has all error codes used by BDD tests
- Full alignment between tests and implementation
- Ready for real integration testing

---

## 🎯 USAGE EXAMPLE

### Worker Spawn with Insufficient VRAM

```rust
// In worker spawn logic
let available_vram = get_available_vram(device)?;
let required_vram = model.vram_requirement();

if available_vram < required_vram {
    return Err(WorkerError::InsufficientVram(
        format!("required: {}GB, available: {}GB", 
            required_vram / 1_000_000_000,
            available_vram / 1_000_000_000)
    ));
}
```

**HTTP Response:**
```json
{
  "code": "INSUFFICIENT_VRAM",
  "message": "Insufficient VRAM: required: 8GB, available: 2GB",
  "retriable": true
}
```

**Status Code:** `503 Service Unavailable`

### Worker Spawn with Insufficient Resources

```rust
// In worker spawn logic
let available_ram = get_available_ram()?;
let required_ram = model.ram_requirement();

if available_ram < required_ram {
    return Err(WorkerError::InsufficientResources(
        format!("not enough RAM: need {}MB, have {}MB",
            required_ram / 1_000_000,
            available_ram / 1_000_000)
    ));
}
```

**HTTP Response:**
```json
{
  "code": "INSUFFICIENT_RESOURCES",
  "message": "Insufficient resources: not enough RAM: need 8192MB, have 2048MB",
  "retriable": true
}
```

**Status Code:** `503 Service Unavailable`

---

## 🔍 ERROR CODE CATALOG

### Complete List of Worker Error Codes

| Error Code | HTTP Status | Retriable | Use Case |
|-----------|-------------|-----------|----------|
| `CUDA_ERROR` | 500 | ✅ Yes | CUDA/GPU failures |
| `INVALID_REQUEST` | 400 | ❌ No | Bad request data |
| `INFERENCE_TIMEOUT` | 408 | ✅ Yes | Request timeout |
| `WORKER_UNHEALTHY` | 503 | ❌ No | Worker not ready |
| `INTERNAL` | 500 | ✅ Yes | Internal errors |
| `INSUFFICIENT_RESOURCES` | 503 | ✅ Yes | RAM/CPU exhausted |
| `INSUFFICIENT_VRAM` | 503 | ✅ Yes | GPU VRAM exhausted |

---

## 📝 NOTES

### Why These Errors Are Retriable

**Insufficient Resources:**
- Different nodes may have different resource availability
- Orchestrator can retry on a node with more RAM/CPU
- Temporary condition (resources may free up)

**Insufficient VRAM:**
- Different nodes may have different GPU configurations
- Orchestrator can retry on a node with larger GPU
- May succeed with different model quantization

### Error Code Stability

All error codes follow UPPER_SNAKE_CASE convention and are **stable** (won't change).
This is critical for API compatibility and client error handling.

---

## ✅ TEAM-130 VERIFICATION CHECKLIST

- [x] Added `InsufficientResources` error type
- [x] Added `InsufficientVram` error type
- [x] Updated `code()` method with new error codes
- [x] Updated `is_retriable()` to mark both as retriable
- [x] Updated `status_code()` to return 503 for both
- [x] Added error code stability tests
- [x] Added message formatting tests
- [x] Added HTTP response structure tests
- [x] Verified compilation succeeds
- [x] Aligned with BDD test expectations

---

**TEAM-130: Product code now fully aligned with BDD tests. All error codes implemented. ✅**
