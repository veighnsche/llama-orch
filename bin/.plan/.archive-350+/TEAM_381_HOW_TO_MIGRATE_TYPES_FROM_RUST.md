# TEAM-381: How to Migrate TypeScript Types from Rust

**Date:** 2025-11-01  
**Status:** ✅ COMPLETE - ALL TYPES NOW AUTO-GENERATED

## 🎯 The Goal

**Single source of truth: Define types ONCE in Rust, auto-generate for TypeScript.**

NO MORE MANUAL TYPESCRIPT TYPE DEFINITIONS!

## ✅ What We Accomplished

Migrated ALL types from manual TypeScript to auto-generated from Rust:
- ✅ `ProcessStats` - from `hive-contract/telemetry.rs`
- ✅ `HiveInfo` - from `hive-contract/types.rs`
- ✅ `HiveDevice` - from `hive-contract/heartbeat.rs`
- ✅ `HiveTelemetry` - from `hive-contract/telemetry.rs`
- ✅ `QueenHeartbeat` - from `hive-contract/telemetry.rs`
- ✅ `HeartbeatSnapshot` - from `hive-contract/telemetry.rs`

## 📋 Step-by-Step Guide

### Step 1: Add Type to Contract Crate (Rust)

**Location:** `bin/97_contracts/hive-contract/src/telemetry.rs` (or appropriate module)

```rust
use serde::{Deserialize, Serialize};

// TEAM-381: Optional WASM support for TypeScript type generation
#[cfg(feature = "wasm")]
use tsify::Tsify;

/// Your type documentation here
/// 
/// TEAM-381: This type is auto-generated for TypeScript via tsify.
/// DO NOT manually define this type in TypeScript - it will be generated automatically.
/// Import from SDK: `import type { YourType } from '@rbee/queen-rbee-sdk'`
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub struct YourType {
    /// Field documentation
    pub field_name: String,
    pub number_field: u32,
    pub optional_field: Option<String>,
}
```

**Key points:**
- ✅ Use `#[cfg(feature = "wasm")]` for conditional WASM support
- ✅ Use `#[cfg_attr(feature = "wasm", derive(Tsify))]` to derive Tsify only when wasm feature is enabled
- ✅ Use `#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]` for WASM bindings
- ✅ Add clear documentation that this is auto-generated

### Step 2: Re-export from Contract Crate

**Location:** `bin/97_contracts/hive-contract/src/lib.rs`

```rust
/// Telemetry types (TEAM-381: moved from rbee-hive-monitor)
pub mod telemetry;

// Re-export main types for convenience
pub use telemetry::{YourType, OtherType};
```

### Step 3: Verify Contract Crate is Pure (WASM-Compatible)

**Check:** `bin/97_contracts/hive-contract/Cargo.toml`

```toml
[dependencies]
# Shared contract types
shared-contract = { path = "../shared-contract" }

# Serialization (WASM-compatible)
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Time handling (WASM-compatible)
chrono = { version = "0.4", features = ["serde"] }

# TEAM-381: Optional WASM support for TypeScript type generation
tsify = { version = "0.4", optional = true }
wasm-bindgen = { version = "0.2", optional = true }

[features]
# TEAM-381: Enable WASM bindings and TypeScript type generation (pure types only)
wasm = ["tsify", "wasm-bindgen"]
```

**CRITICAL:** Contract crates must have NO non-WASM dependencies!
- ❌ NO `tokio`, `mio`, `async-std` (runtime dependencies)
- ❌ NO `heartbeat-registry` (unless optional)
- ✅ ONLY `serde`, `chrono`, `tsify`, `wasm-bindgen`

### Step 4: Add to SDK Cargo.toml

**Location:** `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/Cargo.toml`

```toml
[dependencies]
# TEAM-381: Enable wasm features for TypeScript type generation
# hive-contract is a pure contract crate (types only, no runtime deps)
hive-contract = { path = "../../../../97_contracts/hive-contract", features = ["wasm"] }
```

### Step 5: Re-export from SDK lib.rs

**Location:** `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/src/lib.rs`

```rust
// TEAM-381: Re-export Rust types for TypeScript generation
// hive-contract is a pure contract crate (types only, WASM-compatible)
pub use hive_contract::{
    YourType,
    OtherType,
    // ... all types you want to export
};
```

### Step 6: Force Type Generation with Dummy Functions

**Location:** `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/src/types.rs`

```rust
//! Type exports for TypeScript generation
//!
//! TEAM-381: This module exists solely to force Tsify types to be generated in the .d.ts file
//! By using the types in WASM functions, they get included in the TypeScript definitions

use hive_contract::YourType;
use wasm_bindgen::prelude::*;

/// TEAM-381: Dummy function to force YourType into TypeScript definitions
/// This function will never be called - it exists only to make wasm-bindgen generate the type
#[wasm_bindgen]
pub fn __export_your_type(data: YourType) -> YourType {
    data
}
```

**WHY?** Tsify generates types, but they only appear in `.d.ts` if they're used in a `#[wasm_bindgen]` function!

### Step 7: Add Module to lib.rs

**Location:** `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/src/lib.rs`

```rust
// TEAM-381: Type exports for TypeScript generation
mod types;
```

### Step 8: Build the SDK

```bash
cd bin/10_queen_rbee/ui/packages/queen-rbee-sdk
pnpm build
```

This will:
1. Compile Rust to WASM
2. Generate TypeScript definitions in `pkg/bundler/queen_rbee_sdk.d.ts`
3. Your types will appear in the `.d.ts` file!

### Step 9: Verify Types Were Generated

```bash
grep "interface YourType" bin/10_queen_rbee/ui/packages/queen-rbee-sdk/pkg/bundler/queen_rbee_sdk.d.ts
```

You should see:
```typescript
export interface YourType {
    field_name: string;
    number_field: number;
    optional_field: string | null;
}
```

### Step 10: Import in TypeScript

**Location:** `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/src/index.ts`

```typescript
// TEAM-381: ✅ ALL TYPES AUTO-GENERATED FROM RUST! 
export type { 
  QueenClient, 
  HeartbeatMonitor, 
  // ... other types
  // TEAM-381: ✅ AUTO-GENERATED FROM RUST! 
  // Source: bin/97_contracts/hive-contract/src/telemetry.rs
  YourType,
  OtherType,
} from './pkg/bundler/queen_rbee_sdk'
```

### Step 11: Delete Manual TypeScript Definitions

**CRITICAL:** Remove ALL manual type definitions!

❌ **DELETE THIS:**
```typescript
export interface YourType {
  field_name: string
  number_field: number
  optional_field: string | null
}
```

✅ **KEEP THIS:**
```typescript
// Import from Rust-generated types
export type { YourType } from './pkg/bundler/queen_rbee_sdk'
```

## 🚨 Common Mistakes to Avoid

### ❌ Mistake 1: Non-WASM Dependencies in Contract Crate

```toml
# ❌ WRONG - This will break WASM compilation!
[dependencies]
tokio = "1.0"  # Runtime dependency, not WASM-compatible
```

**Fix:** Make runtime dependencies optional or move types to pure contract crate.

### ❌ Mistake 2: Forgetting the Dummy Function

If you don't add the dummy function in `types.rs`, the type won't appear in `.d.ts`!

```rust
// ❌ WRONG - Type won't be generated
pub use hive_contract::YourType;

// ✅ RIGHT - Add dummy function
#[wasm_bindgen]
pub fn __export_your_type(data: YourType) -> YourType {
    data
}
```

### ❌ Mistake 3: Keeping Manual TypeScript Definitions

```typescript
// ❌ WRONG - Duplicate definition!
export interface YourType { ... }  // Manual
export type { YourType } from './pkg/bundler/queen_rbee_sdk'  // From Rust

// ✅ RIGHT - Only import from Rust
export type { YourType } from './pkg/bundler/queen_rbee_sdk'
```

### ❌ Mistake 4: Wrong Import Path

```typescript
// ❌ WRONG - Old path
export type { YourType } from './pkg/bundler/rbee_sdk'

// ✅ RIGHT - Correct path (matches crate name)
export type { YourType } from './pkg/bundler/queen_rbee_sdk'
```

## 📁 File Structure

```
bin/
├── 97_contracts/
│   └── hive-contract/
│       ├── src/
│       │   ├── lib.rs              # Re-export types
│       │   ├── telemetry.rs        # Define types with Tsify
│       │   ├── types.rs            # More types
│       │   └── heartbeat.rs        # More types
│       └── Cargo.toml              # Pure deps only!
│
└── 10_queen_rbee/
    └── ui/
        └── packages/
            └── queen-rbee-sdk/
                ├── src/
                │   ├── lib.rs      # Re-export from hive-contract
                │   ├── types.rs    # Dummy functions for generation
                │   └── index.ts    # Import generated types
                ├── Cargo.toml      # Add hive-contract with wasm feature
                └── pkg/
                    └── bundler/
                        └── queen_rbee_sdk.d.ts  # Generated types!
```

## 🎯 Quick Checklist

When migrating a type from TypeScript to Rust:

- [ ] 1. Add type to contract crate with `#[cfg_attr(feature = "wasm", derive(Tsify))]`
- [ ] 2. Re-export from contract crate `lib.rs`
- [ ] 3. Verify contract crate has NO non-WASM dependencies
- [ ] 4. Add contract crate to SDK `Cargo.toml` with `features = ["wasm"]`
- [ ] 5. Re-export from SDK `lib.rs`
- [ ] 6. Add dummy function in SDK `types.rs`
- [ ] 7. Build SDK: `pnpm build`
- [ ] 8. Verify type appears in `pkg/bundler/queen_rbee_sdk.d.ts`
- [ ] 9. Import in TypeScript `index.ts`
- [ ] 10. **DELETE manual TypeScript definition**

## 🔍 Debugging

### Problem: Type doesn't appear in .d.ts

**Check:**
1. Did you add the dummy function in `types.rs`?
2. Did you re-export from `lib.rs`?
3. Did you rebuild the SDK?

### Problem: WASM compilation fails with "mio not supported"

**Fix:** Contract crate has non-WASM dependencies. Make them optional or remove them.

```toml
# Make non-WASM deps optional
heartbeat-registry = { path = "...", optional = true }

[features]
wasm = ["tsify", "wasm-bindgen"]  # Don't include non-WASM deps!
```

### Problem: TypeScript can't find the import

**Check:**
1. Is the import path correct? (`queen_rbee_sdk` not `rbee_sdk`)
2. Did you rebuild the SDK?
3. Is the type exported in `index.ts`?

## 📚 Examples

See these files for working examples:
- `bin/97_contracts/hive-contract/src/telemetry.rs` - Type definitions
- `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/src/types.rs` - Dummy functions
- `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/src/index.ts` - TypeScript imports

## 🎉 Success Criteria

You know it's working when:
1. ✅ SDK builds without errors
2. ✅ Types appear in `pkg/bundler/queen_rbee_sdk.d.ts`
3. ✅ TypeScript imports work without errors
4. ✅ NO manual TypeScript type definitions remain
5. ✅ Contract crate compiles to WASM

**SINGLE SOURCE OF TRUTH: RUST!** 🦀
