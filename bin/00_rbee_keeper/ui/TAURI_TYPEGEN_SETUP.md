# Tauri TypeGen Setup Guide

**TEAM-296:** Automatic TypeScript bindings generation from Rust Tauri commands

## Overview

`tauri-plugin-typegen` automatically generates TypeScript types and bindings from your Rust `#[tauri::command]` functions, eliminating manual type synchronization.

## Installation

### 1. Install the CLI Tool

```bash
cargo install tauri-plugin-typegen
```

This installs the `cargo tauri-typegen` command globally.

### 2. Configuration

Already configured in `tauri.conf.json`:

```json
{
  "build": {
    "beforeDevCommand": "cargo tauri-typegen generate && cd ../ui && npm run dev",
    "beforeBuildCommand": "cargo tauri-typegen generate && cd ../ui && npm run build"
  },
  "plugins": {
    "tauri-typegen": {
      "project_path": "./src",
      "output_path": "../ui/src/generated",
      "validation_library": "none",
      "verbose": false,
      "visualize_deps": false
    }
  }
}
```

## Generated Files

When you run `cargo tauri-typegen generate`, it will create:

```
ui/src/generated/
├── commands.ts       # TypeScript command wrappers
├── types.ts          # TypeScript type definitions
└── index.ts          # Barrel export
```

## Usage

### Before (Manual)

```typescript
import { invoke } from "@tauri-apps/api/core";

// ❌ No type safety, prone to typos
const result = await invoke<string>("hive_list");
const response = JSON.parse(result);
```

### After (Generated)

```typescript
import { hiveList } from "@/generated/commands";

// ✅ Fully typed, autocomplete, compile-time checking
const hives = await hiveList();
// hives is already typed as SshHive[]
```

## Example: Rust Command

```rust
// src/tauri_commands.rs
#[derive(Serialize, Deserialize)]
pub struct SshHive {
    pub host: String,
    pub hostname: String,
    pub user: String,
    pub port: u16,
    pub status: String,
}

#[tauri::command]
pub async fn hive_list() -> Result<Vec<SshHive>, String> {
    // Implementation
}
```

## Generated TypeScript

```typescript
// ui/src/generated/types.ts
export interface SshHive {
  host: string;
  hostname: string;
  user: string;
  port: number;
  status: string;
}

// ui/src/generated/commands.ts
import { invoke } from "@tauri-apps/api/core";
import type { SshHive } from "./types";

export async function hiveList(): Promise<SshHive[]> {
  return await invoke("hive_list");
}
```

## Manual Generation

Generate bindings manually:

```bash
# From the Tauri project root (00_rbee_keeper)
cargo tauri-typegen generate

# With verbose output
cargo tauri-typegen generate --verbose

# Custom paths
cargo tauri-typegen generate \
  --project-path ./src \
  --output-path ../ui/src/generated
```

## Automatic Generation

Bindings are automatically generated:

1. **During Development:** When you run `npm run tauri dev`
2. **During Build:** When you run `npm run tauri build`

The `beforeDevCommand` and `beforeBuildCommand` in `tauri.conf.json` ensure types are always up-to-date.

## Migration Guide

### Step 1: Generate Bindings

```bash
cd /home/vince/Projects/llama-orch/bin/00_rbee_keeper
cargo tauri-typegen generate
```

### Step 2: Update Imports

**Old:**
```typescript
import { invoke } from "@tauri-apps/api/core";
import { COMMANDS } from "../api/commands.registry";

const result = await invoke<string>(COMMANDS.HIVE_LIST);
const response: CommandResponse = JSON.parse(result);
```

**New:**
```typescript
import { hiveList } from "@/generated/commands";

const hives = await hiveList();
// Fully typed, no parsing needed
```

### Step 3: Update Container

**Before:**
```typescript
// SshHivesContainer.tsx
async function fetchSshHives(): Promise<SshHive[]> {
  const result = await invoke<string>(COMMANDS.HIVE_LIST);
  const response: CommandResponse = JSON.parse(result);
  
  if (response.success && response.data) {
    return JSON.parse(response.data) as SshHive[];
  }
  
  throw new Error(response.message || "Failed to load SSH hives");
}
```

**After:**
```typescript
// SshHivesContainer.tsx
import { hiveList } from "@/generated/commands";

async function fetchSshHives(): Promise<SshHive[]> {
  return await hiveList();
}
```

## Benefits

### ✅ Type Safety
- Compile-time checking
- No manual type definitions
- Automatic type updates

### ✅ Developer Experience
- Full IDE autocomplete
- Inline documentation from Rust
- Jump to definition works

### ✅ Maintainability
- Single source of truth (Rust)
- No manual synchronization
- Refactoring is safe

### ✅ Error Prevention
- Typos caught at compile time
- Parameter mismatches prevented
- Return type mismatches prevented

## Comparison

| Feature | Manual Registry | Auto-Generated |
|---------|----------------|----------------|
| Type Safety | Partial | Full |
| Autocomplete | Command names only | Everything |
| Sync Effort | Manual | Automatic |
| Parameter Types | Manual | Generated |
| Return Types | Manual | Generated |
| Refactoring | Error-prone | Safe |

## Troubleshooting

### Bindings Not Generated

```bash
# Check if tool is installed
cargo tauri-typegen --version

# Generate with verbose output
cargo tauri-typegen generate --verbose
```

### Types Out of Sync

```bash
# Regenerate bindings
cargo tauri-typegen generate

# Or restart dev server (auto-generates)
npm run tauri dev
```

### Import Errors

Make sure TypeScript can resolve the `@/generated` path:

```json
// ui/tsconfig.json
{
  "compilerOptions": {
    "paths": {
      "@/*": ["./src/*"]
    }
  }
}
```

## Best Practices

### 1. Always Use Generated Bindings

```typescript
// ✅ Good
import { hiveList } from "@/generated/commands";
const hives = await hiveList();

// ❌ Bad - bypasses type safety
import { invoke } from "@tauri-apps/api/core";
const result = await invoke("hive_list");
```

### 2. Don't Edit Generated Files

Generated files are overwritten on each generation. Never edit them manually.

### 3. Commit Generated Files

Add generated files to git so team members don't need to generate them:

```bash
git add ui/src/generated/
git commit -m "Update generated Tauri bindings"
```

### 4. Regenerate After Rust Changes

After modifying Rust commands:

```bash
cargo tauri-typegen generate
```

Or just restart the dev server (auto-generates).

## Integration with Existing Code

You can gradually migrate from the manual registry pattern:

1. Generate bindings: `cargo tauri-typegen generate`
2. Keep `commands.registry.ts` for now
3. Migrate one component at a time
4. Remove registry once all components migrated

## Advanced: Custom Response Types

If your commands return custom response wrappers:

```rust
#[derive(Serialize)]
pub struct CommandResponse<T> {
    pub success: bool,
    pub message: String,
    pub data: Option<T>,
}

#[tauri::command]
pub async fn hive_list() -> Result<CommandResponse<Vec<SshHive>>, String> {
    // Implementation
}
```

Generated TypeScript will include the wrapper:

```typescript
export interface CommandResponse<T> {
  success: boolean;
  message: string;
  data?: T;
}

export async function hiveList(): Promise<CommandResponse<SshHive[]>> {
  return await invoke("hive_list");
}
```

## Resources

- [tauri-plugin-typegen GitHub](https://github.com/thwbh/tauri-typegen)
- [Tauri Commands Documentation](https://tauri.app/v2/develop/calling-rust/)
- **Generated Files:** `ui/src/generated/`

## Summary

1. **Install:** `cargo install tauri-plugin-typegen`
2. **Configure:** Already done in `tauri.conf.json`
3. **Generate:** `cargo tauri-typegen generate`
4. **Use:** Import from `@/generated/commands`
5. **Enjoy:** Full type safety with zero manual work!
