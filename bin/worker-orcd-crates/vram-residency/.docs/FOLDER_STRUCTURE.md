# VRAM Residency - Folder Structure

**Purpose**: Document the organized folder structure for vram-residency crate  
**Status**: ✅ Complete - All stub files created  
**Last Updated**: 2025-10-02

---

## Overview

The source code is organized into **6 high-level behavioral modules**, each containing related functionality:

```
src/
├── lib.rs                      # Public API and module exports
├── error.rs                    # Error types (VramError, Result)
│
├── types/                      # Core type definitions
│   ├── mod.rs
│   ├── sealed_shard.rs         # SealedShard type
│   └── vram_config.rs          # VramConfig type
│
├── seal/                       # Cryptographic sealing operations
│   ├── mod.rs
│   ├── signature.rs            # HMAC-SHA256 seal signatures
│   ├── digest.rs               # SHA-256 digest computation
│   └── key_derivation.rs       # Seal key derivation (HKDF-SHA256)
│
├── allocator/                  # VRAM allocation and management
│   ├── mod.rs
│   ├── vram_manager.rs         # Main VramManager API
│   ├── mock_allocator.rs       # Mock VRAM (for testing)
│   ├── cuda_allocator.rs       # Real CUDA VRAM allocation
│   └── capacity.rs             # Capacity tracking
│
├── policy/                     # VRAM-only policy enforcement
│   ├── mod.rs
│   ├── enforcement.rs          # Policy enforcement logic
│   └── validation.rs           # Device property validation
│
├── validation/                 # Input validation
│   ├── mod.rs
│   ├── shard_id.rs             # Shard ID validation
│   ├── gpu_device.rs           # GPU device validation
│   └── model_size.rs           # Model size validation
│
└── audit/                      # Audit logging integration
    ├── mod.rs
    └── events.rs               # Audit event emission
```

---

## Module Responsibilities

### 1. **types/** - Core Type Definitions
**Purpose**: All core types used throughout the crate

**Files**:
- `sealed_shard.rs` - `SealedShard` type with private VRAM pointer
- `vram_config.rs` - `VramConfig` for VramManager initialization

**Key Types**:
- `SealedShard` - Cryptographically sealed model shard
- `VramConfig` - Configuration for VRAM manager

---

### 2. **seal/** - Cryptographic Sealing
**Purpose**: HMAC-SHA256 seal signature computation and verification

**Files**:
- `signature.rs` - HMAC-SHA256 signature computation/verification
- `digest.rs` - SHA-256 digest computation
- `key_derivation.rs` - Seal key derivation via HKDF-SHA256

**Security**:
- TIER 1 security requirements
- Timing-safe verification (via `subtle` crate)
- Integration with `secrets-management` crate

---

### 3. **allocator/** - VRAM Allocation
**Purpose**: VRAM allocation, deallocation, and capacity tracking

**Files**:
- `vram_manager.rs` - Main public API for VRAM operations
- `mock_allocator.rs` - Mock VRAM for testing (no GPU required)
- `cuda_allocator.rs` - Real CUDA VRAM allocation (via FFI)
- `capacity.rs` - Capacity tracking and enforcement

**Features**:
- Automatic GPU detection (via `gpu-info`)
- Mock fallback for testing
- Production mode (fail fast if no GPU)

---

### 4. **policy/** - VRAM-Only Policy
**Purpose**: Enforce VRAM-only inference policy

**Files**:
- `enforcement.rs` - Policy enforcement at initialization
- `validation.rs` - Device property validation

**Requirements**:
- Disable unified memory (UMA)
- Disable zero-copy
- Disable pinned host memory
- Fail fast on policy violations

---

### 5. **validation/** - Input Validation
**Purpose**: Validate all inputs to prevent injection attacks

**Files**:
- `shard_id.rs` - Shard ID validation (no path traversal, null bytes)
- `gpu_device.rs` - GPU device index validation
- `model_size.rs` - Model size validation (prevent DoS)

**Integration**:
- Uses `input-validation` crate
- Prevents path traversal, null bytes, control characters

---

### 6. **audit/** - Audit Logging
**Purpose**: Emit audit events for all VRAM operations

**Files**:
- `events.rs` - Audit event emission functions

**Events**:
- `VramSealed` - Model sealed in VRAM
- `SealVerified` - Seal verification succeeded
- `SealVerificationFailed` - Seal verification failed (CRITICAL)
- `VramDeallocated` - VRAM deallocated
- `PolicyViolation` - VRAM-only policy violated (CRITICAL)

**Integration**:
- Uses `audit-logging` crate
- All security-critical operations logged

---

## Public API Exports

From `lib.rs`:

```rust
// Core types
pub use error::{Result, VramError};
pub use types::{SealedShard, VramConfig};

// Main API
pub use allocator::VramManager;

// Seal functions (convenience)
pub use seal::{compute_digest, compute_signature, verify_signature};
```

---

## Implementation Status

| Module | Files | Status |
|--------|-------|--------|
| **types/** | 2 | ✅ Implemented |
| **error.rs** | 1 | ✅ Implemented |
| **seal/** | 3 | 🚧 Stubs (TODO) |
| **allocator/** | 4 | 🚧 Partial (VramManager basic impl) |
| **policy/** | 2 | 🚧 Stubs (TODO) |
| **validation/** | 3 | 🚧 Stubs (TODO) |
| **audit/** | 1 | 🚧 Stubs (TODO) |

**Total**: 16 files created (7 modules + lib.rs + error.rs)

---

## Next Steps

### Phase 1: Core Security (P0)
1. Implement `seal/signature.rs` (HMAC-SHA256)
2. Implement `seal/key_derivation.rs` (integrate secrets-management)
3. Add `validation/` implementations (integrate input-validation)
4. Add `audit/events.rs` implementations (integrate audit-logging)

### Phase 2: CUDA Integration (P1)
5. Implement `allocator/cuda_allocator.rs` (CUDA FFI)
6. Implement `policy/enforcement.rs` (UMA detection)
7. Implement `seal/digest.rs` VRAM re-verification

### Phase 3: Testing (P1)
8. Unit tests for all modules
9. BDD tests (already created)
10. Property tests
11. Security tests

---

## Design Principles

1. **Behavior-Driven Organization** - Each folder groups related behaviors
2. **Clear Separation of Concerns** - Each module has a single responsibility
3. **Security-First** - Security-critical code isolated in `seal/` and `policy/`
4. **Testability** - Mock implementations for testing without GPU
5. **Integration Points** - Clear integration with shared crates

---

## Dependencies

### Shared Crates (TODO: Add to Cargo.toml)
- `secrets-management` - Seal key management
- `input-validation` - Input sanitization
- `audit-logging` - Security audit trail
- `gpu-info` - GPU detection

### External Crates
- `sha2` - SHA-256 digests ✅
- `hmac` - HMAC-SHA256 signatures (TODO)
- `subtle` - Timing-safe comparison (via secrets-management)
- `thiserror` - Error types ✅
- `tracing` - Logging ✅

---

**Status**: ✅ Folder structure complete with all stub files  
**Compiles**: ✅ Yes (with `#[allow(clippy::todo)]`)  
**Ready for**: Implementation of TODO items
