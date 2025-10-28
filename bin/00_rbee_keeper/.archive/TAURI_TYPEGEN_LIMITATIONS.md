# Tauri Specta Integration (TEAM-297)

**Date:** October 26, 2025  
**Status:** ✅ COMPLETE - Migrated to tauri-specta v2.0.0-rc.21

## Summary

Successfully migrated from `tauri-plugin-typegen` (which doesn't support complex return types) to `tauri-specta v2.0.0-rc.21` for full TypeScript type generation with Tauri v2.

## The Previous Problem

`tauri-plugin-typegen` **could not generate TypeScript types for complex return types** from Tauri commands.

## What We Tried

### Attempt 1: Return external crate types directly
```rust
#[tauri::command]
pub async fn hive_list() -> Result<Vec<ssh_config::SshTarget>, String> {
    // ...
}
```

**Result:** ❌ Generated `Promise<types.ssh_config::SshTarget[]>` but the type was never generated in `types.ts`

### Attempt 2: Re-export types at root level
```rust
pub use ssh_config::{SshTarget, SshTargetStatus};

#[tauri::command]
pub async fn hive_list() -> Result<Vec<SshTarget>, String> {
    // ...
}
```

**Result:** ❌ Generated `Promise<types.SshTarget[]>` but the type was still not generated

### Attempt 3: Define types directly in tauri_commands.rs
```rust
// In tauri_commands.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SshTarget {
    pub host: String,
    pub hostname: String,
    // ...
}

#[tauri::command]
pub async fn hive_list() -> Result<Vec<SshTarget>, String> {
    // ...
}
```

**Result:** ❌ Tool found **0 struct definitions** even though the struct is in the same file

## Tool Output

```bash
$ cargo tauri-typegen generate --verbose
💬 Found 25 Tauri commands
💬 Found 0 struct definitions  # ← THE PROBLEM
✅ Analyzing Tauri commands - Found 25 commands
```

The tool **cannot discover struct definitions** even when they're in the same file as the commands.

## Root Cause

`tauri-plugin-typegen` has limited type discovery capabilities:
- ✅ Can generate types for function **parameters** (because they're in the function signature)
- ❌ Cannot generate types for complex **return types**
- ❌ Cannot discover struct definitions in the same file
- ❌ Cannot follow type aliases or re-exports

## What This Means

**We cannot use `tauri-plugin-typegen` for commands that return complex types.**

The tool only works for:
- Commands that return `String` (JSON strings)
- Commands that return primitive types
- Commands with typed parameters (those work fine)

## Current State

### What Works ✅
```rust
#[tauri::command]
pub async fn queen_rebuild(with_local_hive: bool) -> Result<String, String> {
    // Parameters are typed correctly in TypeScript
}
```

Generated TypeScript:
```typescript
export interface QueenRebuildParams {
  with_local_hive: boolean;  // ✅ Works!
}

export async function queen_rebuild(params: QueenRebuildParams): Promise<string>
```

### What Doesn't Work ❌
```rust
#[tauri::command]
pub async fn hive_list() -> Result<Vec<SshTarget>, String> {
    // Return type is NOT typed in TypeScript
}
```

Generated TypeScript:
```typescript
export async function hive_list(): Promise<types.SshTarget[]>
// But SshTarget is never defined in types.ts ❌
```

## Options Going Forward

### Option 1: Keep JSON String Approach (Current)
```rust
#[tauri::command]
pub async fn hive_list() -> Result<String, String> {
    let targets = parse_ssh_config(&path)?;
    let response = CommandResponse {
        success: true,
        data: Some(serde_json::to_string(&targets)?),
    };
    serde_json::to_string(&response)
}
```

**Pros:**
- Works with current tool
- No manual type maintenance

**Cons:**
- ❌ No type safety (returns `string`)
- ❌ Manual JSON parsing in TypeScript
- ❌ Component responsible for parsing
- ❌ Defeats the purpose of automatic typing

### Option 2: Manual TypeScript Types (REJECTED)
Manually maintain TypeScript types alongside Rust types.

**Rejected because:**
- ❌ Manual synchronization required
- ❌ Type drift inevitable
- ❌ Defeats the purpose of automatic generation
- ❌ User explicitly rejected this approach

### Option 3: Use Tauri's Specta Integration (RECOMMENDED)
Tauri has official support for `specta` + `tauri-specta` which properly generates types.

**See:** https://github.com/oscartbeaumont/tauri-specta

```rust
// With tauri-specta
use specta::Type;
use tauri_specta::Event;

#[derive(Serialize, Deserialize, Type)]
pub struct SshTarget {
    pub host: String,
    pub hostname: String,
    // ...
}

#[tauri::command]
#[specta::specta]  // ← This makes it work
pub async fn hive_list() -> Result<Vec<SshTarget>, String> {
    // ...
}
```

This properly generates TypeScript types for ALL types, not just parameters.

### Option 4: Write Custom Type Generator
Write our own tool that:
1. Parses Rust AST
2. Finds all `#[tauri::command]` functions
3. Extracts all types (parameters + return types)
4. Generates TypeScript definitions

**Pros:**
- Full control
- Can handle our specific needs

**Cons:**
- Significant development time
- Maintenance burden
- Reinventing the wheel

## Recommendation

**Switch to `tauri-specta`** - it's the official Tauri-recommended solution for type generation and properly handles complex return types.

### Migration Steps

1. Add dependencies:
```toml
[dependencies]
specta = "2"
tauri-specta = "2"
```

2. Add `#[derive(Type)]` to all structs:
```rust
#[derive(Serialize, Deserialize, specta::Type)]
pub struct SshTarget {
    // ...
}
```

3. Add `#[specta::specta]` to commands:
```rust
#[tauri::command]
#[specta::specta]
pub async fn hive_list() -> Result<Vec<SshTarget>, String> {
    // ...
}
```

4. Generate types:
```rust
// In build.rs or separate binary
tauri_specta::ts::export(
    collect_commands![hive_list, queen_start, /* ... */],
    "../ui/src/generated/bindings.ts"
)?;
```

## Current Workaround

Until we migrate to `tauri-specta`, we're **stuck with the JSON string approach**:

```typescript
// Component has to parse JSON
const result = await hive_list();  // Returns string
const response: CommandResponse = JSON.parse(result);
const hives: SshHive[] = JSON.parse(response.data);
```

This is **not ideal** but it's the reality of `tauri-plugin-typegen`'s limitations.

---

## ✅ SOLUTION (TEAM-297)

**Migrated to `tauri-specta v2.0.0-rc.21`** - Full type safety achieved!

### Implementation

#### 1. Dependencies (Cargo.toml)
```toml
tauri = { version = "2", features = [] }
specta = { version = "=2.0.0-rc.22" }
tauri-specta = { version = "=2.0.0-rc.21", features = ["derive", "typescript"] }
specta-typescript = "=0.0.9"
```

**Note:** Pin exact versions with `=` to prevent breakage from Tauri updates.

#### 2. Define Types with Specta (tauri_commands.rs)
```rust
use specta::Type;

#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct SshTarget {
    pub host: String,
    pub hostname: String,
    pub user: String,
    pub port: u16,
    pub status: SshTargetStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub enum SshTargetStatus {
    Online,
    Offline,
    Unknown,
}
```

#### 3. Annotate Commands (tauri_commands.rs)
```rust
#[tauri::command]
#[specta::specta]  // ← THIS IS CRITICAL!
pub async fn hive_list() -> Result<Vec<SshTarget>, String> {
    // Implementation
}
```

#### 4. Setup Builder (tauri_main.rs)
```rust
use specta_typescript::Typescript;
use tauri_specta::{collect_commands, Builder};

fn main() {
    let mut builder = Builder::<tauri::Wry>::new()
        .commands(collect_commands![hive_list]);

    #[cfg(debug_assertions)]
    builder
        .export(Typescript::default(), "../ui/src/generated/bindings.ts")
        .expect("Failed to export typescript bindings");

    tauri::Builder::default()
        .invoke_handler(builder.invoke_handler())
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

#### 5. Generate Bindings
Run the test to export bindings:
```bash
cargo test --lib export_typescript_bindings
```

### Generated TypeScript (bindings.ts)

```typescript
export const commands = {
  async hiveList() : Promise<Result<SshTarget[], string>> {
    return { status: "ok", data: await TAURI_INVOKE("hive_list") };
  }
}

export type SshTarget = {
  host: string;
  hostname: string;
  user: string;
  port: number;
  status: SshTargetStatus;
}

export type SshTargetStatus = "online" | "offline" | "unknown"
```

### Frontend Usage

```typescript
import { commands } from './generated/bindings';

const result = await commands.hiveList();
if (result.status === "ok") {
  const hives: SshTarget[] = result.data;
  // Full type safety! ✅
}
```

## Benefits Achieved

✅ **Full type safety** - Complex return types properly typed  
✅ **Auto-generated** - No manual type maintenance  
✅ **Type drift impossible** - Types generated from Rust source  
✅ **Documentation preserved** - Doc comments become TSDoc  
✅ **Compiler-verified** - TypeScript sees exact Rust types  

## Adding More Commands

1. Add `#[specta::specta]` to your `#[tauri::command]`
2. Ensure all types derive `specta::Type`
3. Add command to `collect_commands![...]` in `tauri_main.rs`
4. Run `cargo test --lib export_typescript_bindings`
5. Use in frontend via `import { commands } from './generated/bindings'`

## References

- [tauri-specta GitHub](https://github.com/specta-rs/tauri-specta)
- [tauri-specta v2 Docs](https://docs.rs/tauri-specta/2.0.0-rc.21/tauri_specta/)
- [Specta Documentation](https://github.com/oscartbeaumont/specta)
