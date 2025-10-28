# Tauri Specta Migration Status

**Date:** October 26, 2025  
**Status:** ⚠️ IN PROGRESS - API incompatibility with latest versions

## Goal

Migrate from `tauri-plugin-typegen` (which cannot generate complex return types) to `tauri-specta` (which can).

## Current Status

### ✅ What's Working

1. **Dependencies Added:**
   - `tauri = "2"` (latest: 2.9.1)
   - `specta = "2.0.0-rc.22"` (latest RC)
   - `tauri-specta = "2.0.0-rc.21"` (latest RC)

2. **Rust Code Updated:**
   - Added `use specta::Type` to `tauri_commands.rs`
   - Created `SshTarget` and `SshTargetStatus` types with `#[derive(Type)]`
   - Added `#[specta::specta]` to `hive_list` command
   - Command returns `Result<Vec<SshTarget>, String>` (properly typed)

3. **Compilation:** ✅ Dependencies resolve correctly with latest versions

### ❌ What's Broken

**The tauri-specta v2 API has changed and the documentation examples don't match the actual API.**

#### Documentation Says:
```rust
use tauri_specta::{ts, collect_commands};

let builder = ts::builder()
    .commands(collect_commands![hive_list]);
```

#### Reality:
- `ts` module doesn't exist in `tauri_specta` root
- `Builder::new()` exists but doesn't have a `.path()` method
- The API structure is completely different from the docs

## The Problem

The official documentation at https://specta.dev/docs/tauri-specta/v2 shows API examples that **don't exist in the actual crate**.

This suggests:
1. The docs are for an older/newer version
2. The RC versions have breaking API changes
3. The docs haven't been updated for the latest RC

## What We Need

We need to find the **actual working API** for `tauri-specta = "2.0.0-rc.21"`.

Options:
1. Check the GitHub repo examples
2. Check the crate documentation (docs.rs)
3. Look at the source code
4. Try different RC versions until we find one that matches the docs

## Next Steps

1. **Check GitHub examples:** https://github.com/specta-rs/tauri-specta/tree/main/examples
2. **Check docs.rs:** https://docs.rs/tauri-specta/2.0.0-rc.21
3. **Try the API from actual source code**
4. **If all else fails:** Pin to the exact versions from the docs (beta.22, rc.12, rc.11)

## Current Code State

### tauri_commands.rs
```rust
use specta::Type;

#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct SshTarget {
    pub host: String,
    pub host_subtitle: Option<String>,
    pub hostname: String,
    pub user: String,
    pub port: u16,
    pub status: SshTargetStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Type)]
#[serde(rename_all = "lowercase")]
pub enum SshTargetStatus {
    Online,
    Offline,
    Unknown,
}

#[tauri::command]
#[specta::specta]
pub async fn hive_list() -> Result<Vec<SshTarget>, String> {
    // Implementation...
}
```

### tauri_main.rs (BROKEN)
```rust
use rbee_keeper::tauri_commands::*;
use tauri_specta::collect_commands;

fn main() {
    let invoke_handler = {
        let builder = tauri_specta::Builder::<tauri::Wry>::new()
            .commands(collect_commands![hive_list]);

        #[cfg(debug_assertions)]
        let builder = builder.path("../ui/src/generated/bindings.ts");  // ❌ No .path() method

        builder.build().unwrap()
    };

    tauri::Builder::default()
        .invoke_handler(invoke_handler)
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

## Errors

```
error[E0599]: no method named `path` found for struct `tauri_specta::Builder`
```

## Decision Point

**Do we:**
1. **Continue with latest versions** and figure out the correct API?
2. **Pin to beta versions** from the docs (which you rejected)?
3. **Wait for Specta v2 stable release**?
4. **Accept the JSON string limitation** for now?

## Recommendation

**Check the GitHub examples** to find the actual working API for the latest RC versions. If the examples also don't work, then Specta v2 is too unstable and we should either:
- Wait for stable release
- Accept JSON strings for now
- Use a different solution

## Files Modified

- `Cargo.toml` - Added specta dependencies
- `src/tauri_commands.rs` - Added Type derives and #[specta::specta]
- `src/tauri_main.rs` - Attempted to use tauri-specta (currently broken)
