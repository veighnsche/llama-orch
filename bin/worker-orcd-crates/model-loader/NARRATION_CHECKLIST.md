# 🎀 Narration Checklist — model-loader

**Crate**: `model-loader`  
**Purpose**: Security-first GGUF model validation and loading  
**Narration Status**: ⬜ **NOT YET IMPLEMENTED**  
**Priority**: HIGH (observability for security-critical operations)

---

## 📋 Overview

This checklist guides the implementation of **narration events** for model-loader using the `observability-narration-core` crate. Model-loader performs security-critical validation, so every operation should emit human-readable narration for debugging and incident investigation.

---

## 🎯 Why Narration Matters for model-loader

### Security Operations Need Stories

Model-loader performs **security-critical validation**:
- Path traversal prevention
- Hash verification (integrity checks)
- GGUF format validation (bounds checking)
- Size limit enforcement

When something goes wrong, developers need to know:
- **What** was being loaded
- **Why** it failed
- **Where** the failure occurred
- **How** to fix it

Narration provides this context in **human-readable form**.

### Sibling Relationship: narration-core + audit-logging

**narration-core** (us):
- Human-readable debugging stories
- Developer-friendly error context
- Cute children's book mode! 🎀

**audit-logging** (our serious sibling):
- Legally defensible compliance records
- Formal, machine-readable events
- Immutable audit trail 🔒

**Both are necessary!** We help developers debug. They help auditors investigate.

---

## ✅ Narration Events to Implement

### 1. Model Load Started

**When**: `load_and_validate()` begins  
**Actor**: `model-loader`  
**Action**: `load_start`  
**Target**: Model path (sanitized)

**Human**:
```
"Loading model from /var/lib/llorch/models/llama-7b.gguf (max size: 100 GB)"
```

**Cute**:
```
"Looking for model at /var/lib/llorch/models/llama-7b.gguf! Let's load it up! 📦✨"
```

**Fields**:
- `correlation_id` — Request tracking
- `worker_id` — Worker identifier
- `device` — GPU device (if known)

---

### 2. Path Validation Success

**When**: Path canonicalization succeeds  
**Actor**: `model-loader`  
**Action**: `path_validated`  
**Target**: Canonical path

**Human**:
```
"Validated path: /var/lib/llorch/models/llama-7b.gguf (within allowed root)"
```

**Cute**:
```
"Found the model at /var/lib/llorch/models/llama-7b.gguf! Path looks safe! ✅🗂️"
```

**Fields**:
- `correlation_id`
- `worker_id`

---

### 3. Path Validation Failed (Path Traversal)

**When**: Path traversal detected  
**Actor**: `model-loader`  
**Action**: `path_validation_failed`  
**Target**: Attempted path (sanitized)

**Human**:
```
"Path validation failed: path traversal detected in '../../../etc/passwd'"
```

**Cute**:
```
"Whoa! That path looks suspicious (../../../etc/passwd)! Nice try, but no! 🛑🔍"
```

**Fields**:
- `correlation_id`
- `worker_id`
- `error_kind`: `"path_traversal"`

---

### 4. File Size Check

**When**: File size validated  
**Actor**: `model-loader`  
**Action**: `size_checked`  
**Target**: Model path

**Human**:
```
"File size: 7.5 GB (within 100 GB limit)"
```

**Cute**:
```
"Model is 7.5 GB — fits perfectly within our 100 GB limit! 📏✨"
```

**Fields**:
- `correlation_id`
- `worker_id`

---

### 5. File Too Large

**When**: File exceeds max_size  
**Actor**: `model-loader`  
**Action**: `size_check_failed`  
**Target**: Model path

**Human**:
```
"File too large: 150 GB (max: 100 GB)"
```

**Cute**:
```
"Oh no! Model is 150 GB but we can only handle 100 GB! Too big! 😟📦"
```

**Fields**:
- `correlation_id`
- `worker_id`
- `error_kind`: `"file_too_large"`

---

### 6. Hash Verification Started

**When**: Hash verification begins  
**Actor**: `model-loader`  
**Action**: `hash_verify_start`  
**Target**: Model path

**Human**:
```
"Verifying SHA-256 hash for llama-7b.gguf"
```

**Cute**:
```
"Checking llama-7b.gguf's fingerprint to make sure it's authentic! 🔍🔐"
```

**Fields**:
- `correlation_id`
- `worker_id`

---

### 7. Hash Verification Success

**When**: Hash matches expected  
**Actor**: `model-loader`  
**Action**: `hash_verified`  
**Target**: Model path

**Human**:
```
"Hash verified: abc123... (SHA-256 match)"
```

**Cute**:
```
"Perfect! llama-7b.gguf's fingerprint matches! All authentic! ✅✨"
```

**Fields**:
- `correlation_id`
- `worker_id`
- `duration_ms` — Verification time

---

### 8. Hash Verification Failed

**When**: Hash mismatch detected  
**Actor**: `model-loader`  
**Action**: `hash_verification_failed`  
**Target**: Model path

**Human**:
```
"Hash mismatch: expected abc123..., got def456... (integrity violation)"
```

**Cute**:
```
"Uh oh! llama-7b.gguf's fingerprint doesn't match! Expected one thing, got another! 😟❌"
```

**Fields**:
- `correlation_id`
- `worker_id`
- `error_kind`: `"hash_mismatch"`

---

### 9. GGUF Validation Started

**When**: GGUF format validation begins  
**Actor**: `model-loader`  
**Action**: `gguf_validate_start`  
**Target**: Model path

**Human**:
```
"Validating GGUF format for llama-7b.gguf"
```

**Cute**:
```
"Checking if llama-7b.gguf is a valid GGUF file! 📋✨"
```

**Fields**:
- `correlation_id`
- `worker_id`

---

### 10. GGUF Validation Success

**When**: GGUF format is valid  
**Actor**: `model-loader`  
**Action**: `gguf_validated`  
**Target**: Model path

**Human**:
```
"GGUF format validated: version 3, 1024 tensors, 50 metadata KV pairs"
```

**Cute**:
```
"llama-7b.gguf is a perfect GGUF file (v3, 1024 tensors)! Ready to load! 🎉📦"
```

**Fields**:
- `correlation_id`
- `worker_id`
- `duration_ms` — Validation time

---

### 11. GGUF Validation Failed (Invalid Magic)

**When**: GGUF magic number invalid  
**Actor**: `model-loader`  
**Action**: `gguf_validation_failed`  
**Target**: Model path

**Human**:
```
"Invalid GGUF magic number: expected 0x46554747, got 0x00000000"
```

**Cute**:
```
"Hmm, llama-7b.gguf doesn't look like a GGUF file! Wrong magic number! 🤔❌"
```

**Fields**:
- `correlation_id`
- `worker_id`
- `error_kind`: `"invalid_magic"`

---

### 12. GGUF Validation Failed (Bounds Check)

**When**: GGUF bounds check fails  
**Actor**: `model-loader`  
**Action**: `gguf_validation_failed`  
**Target**: Model path

**Human**:
```
"GGUF bounds check failed: tensor count 999999999 exceeds MAX_TENSORS (1000000)"
```

**Cute**:
```
"Whoa! llama-7b.gguf claims to have 999999999 tensors! That's way too many! 😟📊"
```

**Fields**:
- `correlation_id`
- `worker_id`
- `error_kind`: `"bounds_check_failed"`

---

### 13. Model Load Complete

**When**: `load_and_validate()` succeeds  
**Actor**: `model-loader`  
**Action**: `load_complete`  
**Target**: Model path

**Human**:
```
"Model loaded: llama-7b.gguf (7.5 GB, validated, ready for VRAM)"
```

**Cute**:
```
"Hooray! llama-7b.gguf is loaded and validated! Ready to go to VRAM! 🎉🚀"
```

**Fields**:
- `correlation_id`
- `worker_id`
- `duration_ms` — Total load time

---

### 14. Bytes Validation (Memory Mode)

**When**: `validate_bytes()` called  
**Actor**: `model-loader`  
**Action**: `bytes_validated`  
**Target**: "memory" (no path)

**Human**:
```
"Validated model bytes: 7.5 GB, hash verified, GGUF format valid"
```

**Cute**:
```
"Checked the model bytes — 7.5 GB of perfect GGUF data! All good! ✅✨"
```

**Fields**:
- `correlation_id`
- `worker_id`

---

## 🛠️ Implementation Steps

### Step 1: Add Dependency

**File**: `Cargo.toml`

```toml
[dependencies]
observability-narration-core = { path = "../../shared-crates/narration-core" }
```

### Step 2: Create Narration Module

**File**: `src/narration/mod.rs`

```rust
//! Narration module for model-loader
//!
//! Provides structured, human-readable narration for all model loading operations.

pub mod events;

pub use events::*;
```

**File**: `src/narration/events.rs`

```rust
//! Narration events for model loading operations

use observability_narration_core::{narrate, NarrationFields};

/// Narrate model load start
pub fn narrate_load_start(
    model_path: &str,
    max_size_gb: f64,
    worker_id: Option<&str>,
    correlation_id: Option<&str>,
) {
    narrate(NarrationFields {
        actor: "model-loader",
        action: "load_start",
        target: model_path.to_string(),
        human: format!(
            "Loading model from {} (max size: {:.1} GB)",
            model_path, max_size_gb
        ),
        cute: Some(format!(
            "Looking for model at {}! Let's load it up! 📦✨",
            model_path
        )),
        worker_id: worker_id.map(|s| s.to_string()),
        correlation_id: correlation_id.map(|s| s.to_string()),
        ..Default::default()
    });
}

// ... (implement all 14 functions)
```

### Step 3: Export Module

**File**: `src/lib.rs`

```rust
// Narration (observability)
pub mod narration;
```

### Step 4: Integrate into loader.rs

**File**: `src/loader.rs`

```rust
use crate::narration;

impl ModelLoader {
    pub fn load_and_validate(&self, request: LoadRequest) -> Result<Vec<u8>> {
        let start = std::time::Instant::now();
        
        // Narrate start
        narration::narrate_load_start(
            request.model_path.to_str().unwrap_or("unknown"),
            request.max_size as f64 / 1_000_000_000.0,
            None, // TODO: Add worker_id
            None, // TODO: Add correlation_id
        );
        
        // ... validation logic ...
        
        // Narrate completion
        let duration_ms = start.elapsed().as_millis() as u64;
        narration::narrate_load_complete(
            request.model_path.to_str().unwrap_or("unknown"),
            file_size as f64 / 1_000_000_000.0,
            duration_ms,
            None,
            None,
        );
        
        Ok(model_bytes)
    }
}
```

---

## 🧪 Testing Narration

### BDD Test Example

**File**: `bdd/tests/features/narration.feature`

```gherkin
Feature: Model Loader Narration

  Scenario: Successful model load emits narration
    Given a valid GGUF model at "/tmp/test.gguf"
    When I load the model with hash verification
    Then narration includes "load_start"
    And narration includes "hash_verified"
    And narration includes "gguf_validated"
    And narration includes "load_complete"
    And cute narration includes "Hooray!"

  Scenario: Hash mismatch emits error narration
    Given a GGUF model at "/tmp/test.gguf"
    When I load with wrong expected hash
    Then narration includes "hash_verification_failed"
    And cute narration includes "Uh oh!"
    And error_kind is "hash_mismatch"
```

### Unit Test Example

```rust
#[test]
fn test_load_emits_narration() {
    use observability_narration_core::CaptureAdapter;
    
    let capture = CaptureAdapter::install();
    
    let loader = ModelLoader::new();
    let request = LoadRequest {
        model_path: &PathBuf::from("/tmp/test.gguf"),
        expected_hash: Some("abc123"),
        max_size: 1_000_000,
    };
    
    let _ = loader.load_and_validate(request);
    
    // Verify narration
    capture.assert_includes("load_start");
    capture.assert_field("actor", "model-loader");
    capture.assert_cute_present();
}
```

---

## 📊 Narration Coverage Checklist

### Core Operations
- [ ] `load_and_validate()` — Start, success, all failure modes
- [ ] `validate_bytes()` — Memory mode validation
- [ ] Path validation — Success, traversal detection
- [ ] Hash verification — Success, mismatch
- [ ] GGUF validation — Success, invalid magic, bounds check

### Error Cases
- [ ] Path traversal detected
- [ ] File too large
- [ ] Hash mismatch
- [ ] Invalid GGUF magic
- [ ] GGUF bounds check failed
- [ ] File not found
- [ ] Permission denied

### Performance Tracking
- [ ] Load duration (`duration_ms`)
- [ ] Hash verification time
- [ ] GGUF validation time

### Correlation
- [ ] `correlation_id` propagated from worker-api
- [ ] `worker_id` included in all events

---

## 🎯 Success Criteria

### ✅ Implementation Complete When:

1. **All 14 narration functions implemented** in `src/narration/events.rs`
2. **Integrated into loader.rs** — All operations emit narration
3. **BDD tests written** — Narration assertions in feature files
4. **Cute mode enabled** — All events have cute children's book narration
5. **Correlation IDs propagated** — Request tracking works end-to-end
6. **Error context clear** — Developers can debug from narration alone

### ✅ Quality Checks:

- [ ] No secrets in narration (paths sanitized, no tokens)
- [ ] All human fields ≤100 characters (ORCH-3305)
- [ ] All cute fields ≤150 characters
- [ ] Emojis used appropriately (at least one per cute field)
- [ ] Error narrations include actionable context
- [ ] Success narrations celebrate completion 🎉

---

## 🤝 Sibling Crate Coordination

### With audit-logging (our serious sibling)

**They emit** (for compliance):
```json
{
  "event_type": "ModelLoadFailed",
  "reason": "hash_mismatch",
  "severity": "HIGH"
}
```

**We emit** (for debugging):
```json
{
  "actor": "model-loader",
  "action": "hash_verification_failed",
  "human": "Hash mismatch: expected abc123..., got def456...",
  "cute": "Uh oh! Fingerprint doesn't match! 😟❌"
}
```

**Both are necessary!** They help auditors investigate. We help developers debug.

### With vram-residency (downstream consumer)

vram-residency receives **validated bytes** from us. They trust our validation and emit their own narration:

**Our narration**:
```
"Model loaded: llama-7b.gguf (7.5 GB, validated, ready for VRAM)"
```

**Their narration**:
```
"Tucked llama-7b safely into GPU0's warm 7.5 GB nest! Sweet dreams! 🛏️✨"
```

**Perfect handoff!** We validate, they seal. Both narrate their part of the story.

---

## 📝 Notes

### Security Considerations

1. **Path Sanitization**: Always sanitize paths before narration (use `input-validation`)
2. **No Sensitive Data**: Never log full hashes (use first 6 chars: `abc123...`)
3. **Error Context**: Include enough context for debugging, not exploitation
4. **Correlation IDs**: Essential for tracking requests across services

### Performance Considerations

1. **Non-Blocking**: Narration is synchronous but fast (< 1ms)
2. **Buffered**: narration-core buffers events internally
3. **Minimal Overhead**: String formatting only when narration enabled

### Future Enhancements

- [ ] Add metadata extraction narration (post-M0)
- [ ] Add async variant narration (post-M0)
- [ ] Add signature verification narration (post-M0)
- [ ] Add narration filtering by log level

---

## 🎀 Cute Mode Guidelines

### Whimsical Metaphors for model-loader

- **Model file** → "precious cargo", "package", "treasure"
- **Loading** → "fetching", "bringing in", "picking up"
- **Validation** → "checking", "inspecting", "making sure"
- **Hash** → "fingerprint", "signature", "unique mark"
- **GGUF** → "format", "structure", "blueprint"
- **Success** → "perfect!", "all good!", "ready to go!"
- **Failure** → "oh no!", "uh oh!", "something's not right!"

### Emoji Usage

- 📦 (package) — for loading/files
- ✅ (checkmark) — for validation success
- ❌ (X) — for validation failure
- 🔍 (magnifying glass) — for checking/verification
- 🔐 (lock) — for hash/security
- 📋 (clipboard) — for GGUF validation
- 🎉 (party) — for completion
- 😟 (worried) — for errors
- ✨ (sparkles) — for success/magic

---

## 🏆 Completion Status

- [ ] **Dependency added** (`observability-narration-core`)
- [ ] **Narration module created** (`src/narration/`)
- [ ] **14 functions implemented** (all events covered)
- [ ] **Integrated into loader.rs** (all operations narrate)
- [ ] **BDD tests written** (narration assertions)
- [ ] **Cute mode enabled** (whimsical storytelling)
- [ ] **Documentation updated** (README mentions narration)
- [ ] **Examples added** (narration usage examples)

---

**Status**: ⬜ **PENDING IMPLEMENTATION**  
**Priority**: HIGH (security-critical operations need observability)  
**Estimated Effort**: 2-3 hours (14 functions + integration + tests)  
**Maintainer**: model-loader team (with narration-core guidance)

---

> **"If you can't narrate the load, you can't debug the failure!"** 🎀📦

---

**Version**: 0.0.0 (checklist created, implementation pending)  
**License**: GPL-3.0-or-later  
**Sibling Crates**: narration-core (cute stories), audit-logging (serious compliance)
