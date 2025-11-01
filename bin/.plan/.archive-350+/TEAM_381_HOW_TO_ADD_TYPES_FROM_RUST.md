# How to Add TypeScript Types from Rust (Using tsify)

**TEAM-381: Complete Guide**

## Overview

This guide shows you how to define types once in Rust and automatically generate TypeScript types using `tsify`.

## Prerequisites

1. Rust struct with `#[derive(Serialize, Deserialize)]`
2. `tsify` crate added to dependencies
3. `wasm` feature enabled

## Step-by-Step Guide

### Step 1: Add tsify to Your Contract Crate

```toml
# Example: bin/97_contracts/operations-contract/Cargo.toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Optional WASM support for TypeScript type generation
tsify = { version = "0.4", optional = true }
wasm-bindgen = { version = "0.2", optional = true }

[features]
wasm = ["tsify", "wasm-bindgen"]
```

### Step 2: Annotate Your Rust Struct

```rust
// Example: bin/97_contracts/operations-contract/src/responses.rs
use serde::{Deserialize, Serialize};

// Import tsify only when wasm feature is enabled
#[cfg(feature = "wasm")]
use tsify::Tsify;

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub struct ModelInfo {
    /// Model ID
    pub id: String,
    /// Model name
    pub name: String,
    /// Size in bytes
    pub size_bytes: u64,
    /// Download status
    pub status: String,
    /// Whether model is loaded in RAM
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loaded: Option<bool>,
    /// VRAM usage in MB (if loaded)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_mb: Option<u64>,
}
```

**Key points:**
- `#[cfg_attr(feature = "wasm", derive(Tsify))]` - Only derive Tsify when wasm feature is enabled
- `#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]` - Enable WASM bindings
- `#[serde(skip_serializing_if = "Option::is_none")]` - Optional fields won't be serialized if None

### Step 3: Enable wasm Feature in SDK

```toml
# Example: bin/20_rbee_hive/ui/packages/rbee-hive-sdk/Cargo.toml
[dependencies]
operations-contract = { path = "../../../../97_contracts/operations-contract", features = ["wasm"] }
```

### Step 4: Re-export in SDK lib.rs

```rust
// Example: bin/20_rbee_hive/ui/packages/rbee-hive-sdk/src/lib.rs
// Re-export types from operations-contract (with TypeScript generation)
pub use operations_contract::ModelInfo;
```

### Step 5: Re-export in TypeScript

```typescript
// Example: bin/20_rbee_hive/ui/packages/rbee-hive-sdk/src/index.ts
// Re-export WASM SDK types (including auto-generated TypeScript types from Rust)
export type { 
  HiveClient, 
  HeartbeatMonitor,
  OperationBuilder,
  ModelInfo, // Auto-generated from Rust via tsify
} from './pkg/bundler/rbee_hive_sdk'
```

### Step 6: Build

```bash
cd bin/20_rbee_hive/ui/packages/rbee-hive-sdk
pnpm build
```

### Step 7: Verify Generated Types

```bash
# Check the generated .d.ts file
cat bin/20_rbee_hive/ui/packages/rbee-hive-sdk/pkg/bundler/rbee_hive_sdk.d.ts

# Should see:
# export interface ModelInfo {
#     id: string;
#     name: string;
#     size_bytes: number;
#     status: string;
#     loaded?: boolean;
#     vram_mb?: number;
# }
```

## Type Mappings

| Rust Type | TypeScript Type | Example |
|-----------|----------------|---------|
| `String` | `string` | `pub id: String` → `id: string` |
| `u32`, `u64`, `i32`, `i64` | `number` | `pub size_bytes: u64` → `size_bytes: number` |
| `bool` | `boolean` | `pub active: bool` → `active: boolean` |
| `Option<T>` | `T \| undefined` | `pub loaded: Option<bool>` → `loaded?: boolean` |
| `Vec<T>` | `T[]` | `pub tags: Vec<String>` → `tags: string[]` |
| `HashMap<K, V>` | `Record<K, V>` | `pub meta: HashMap<String, String>` → `meta: Record<string, string>` |

## Common Patterns

### Optional Fields

```rust
#[derive(Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct User {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
}
```

**Generated TypeScript:**
```typescript
export interface User {
    id: string;
    email?: string;
}
```

### Nested Types

```rust
#[derive(Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct User {
    pub id: String,
    pub profile: UserProfile,
}

#[derive(Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct UserProfile {
    pub name: String,
    pub age: u32,
}
```

**Generated TypeScript:**
```typescript
export interface User {
    id: string;
    profile: UserProfile;
}

export interface UserProfile {
    name: string;
    age: number;
}
```

### Enums

```rust
#[derive(Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Active,
    Inactive,
    Pending,
}
```

**Generated TypeScript:**
```typescript
export type Status = "active" | "inactive" | "pending";
```

### Lists

```rust
#[derive(Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct ModelList {
    pub models: Vec<ModelInfo>,
}
```

**Generated TypeScript:**
```typescript
export interface ModelList {
    models: ModelInfo[];
}
```

## Best Practices

### ✅ DO

1. **Use `#[cfg_attr(feature = "wasm", derive(Tsify))]`** - Only enable for WASM builds
2. **Use `#[serde(skip_serializing_if = "Option::is_none")]`** - For optional fields
3. **Use descriptive field names** - They become TypeScript property names
4. **Add doc comments** - They appear in generated .d.ts file
5. **Use `#[serde(rename_all = "snake_case")]`** - For consistent naming

### ❌ DON'T

1. **Don't manually duplicate types in TypeScript** - Let tsify generate them
2. **Don't use Rust-specific types** - Stick to JSON-serializable types
3. **Don't forget to rebuild** - Types won't update until you run `pnpm build`
4. **Don't use lifetimes** - WASM types must be `'static`

## Troubleshooting

### Types not generating?

1. Check that `wasm` feature is enabled in SDK Cargo.toml
2. Verify `#[cfg_attr(feature = "wasm", derive(Tsify))]` is present
3. Rebuild: `cd bin/20_rbee_hive/ui/packages/rbee-hive-sdk && pnpm build`

### TypeScript errors about missing types?

1. Check that type is re-exported in `lib.rs`
2. Check that type is re-exported in `index.ts`
3. Rebuild SDK and restart TypeScript server

### Field names don't match?

Use `#[serde(rename = "newName")]`:
```rust
#[derive(Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct Model {
    #[serde(rename = "modelId")]
    pub model_id: String,
}
```

## Complete Example

**Rust (operations-contract/src/responses.rs):**
```rust
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use tsify::Tsify;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub size_bytes: u64,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loaded: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_mb: Option<u64>,
}
```

**Re-export (rbee-hive-sdk/src/lib.rs):**
```rust
pub use operations_contract::ModelInfo;
```

**TypeScript (rbee-hive-sdk/src/index.ts):**
```typescript
export type { ModelInfo } from './pkg/bundler/rbee_hive_sdk'
```

**Usage (React component):**
```typescript
import type { ModelInfo } from '@rbee/rbee-hive-react'

function MyComponent() {
  const model: ModelInfo = {
    id: "llama-3.2-1b",
    name: "LLaMA 3.2 1B",
    size_bytes: 1_000_000_000,
    status: "ready",
    loaded: true,
    vram_mb: 2048,
  }
  
  return <div>{model.name}: {(model.size_bytes / 1e9).toFixed(2)} GB</div>
}
```

## Summary

✅ **Define types once in Rust**  
✅ **Annotate with `#[cfg_attr(feature = "wasm", derive(Tsify))]`**  
✅ **Enable wasm feature in SDK**  
✅ **Re-export in lib.rs and index.ts**  
✅ **Build and use!**  

**No manual TypeScript type definitions needed!** 🎉
