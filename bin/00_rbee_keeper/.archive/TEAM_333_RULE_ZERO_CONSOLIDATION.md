# TEAM-333: RULE ZERO - Consolidate Duplicate Tauri Entry Points

**Status:** ✅ COMPLETE

**Mission:** Eliminate duplicate Tauri entry points that violated RULE ZERO.

---

## The Problem

**RULE ZERO VIOLATION:** Two different entry points doing the same thing = ENTROPY

```
src/main.rs        → Uses tauri::generate_handler![]  ← Actually running
src/tauri_main.rs  → Uses tauri_specta builder        ← Duplicate!
```

**Why this is bad:**
- ❌ Adding new commands requires updating BOTH files
- ❌ Easy to forget one (like we did with `ssh_list`)
- ❌ Maintenance burden doubles
- ❌ Confusion about which one is "correct"
- ❌ Permanent technical debt

**The bug that exposed this:**
- Added `ssh_list` to `tauri_main.rs` only
- Forgot to add it to `main.rs`
- Got "Command ssh_list not found" error
- Spent time debugging the wrong thing

---

## The Solution

**RULE ZERO:** Breaking changes > backwards compatibility

**Action taken:**
1. ✅ **DELETED** `src/tauri_main.rs` (duplicate entry point)
2. ✅ **UPDATED** `Cargo.toml` to remove duplicate binary entry
3. ✅ **KEPT** `src/main.rs` as single entry point for both CLI and GUI

**Result:**
- ✅ Single source of truth
- ✅ Add command once, works everywhere
- ✅ No confusion about which file to update
- ✅ Compiler enforces consistency

---

## How It Works Now

**Single entry point:** `src/main.rs`

```rust
#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // If no subcommand provided, launch Tauri GUI
    if cli.command.is_none() {
        launch_gui();  // ← Tauri GUI mode
        return Ok(());
    }
    
    handle_command(cli).await  // ← CLI mode
}

fn launch_gui() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            ssh_list,  // ← Add commands here ONCE
            // ... other commands
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

**Usage:**
```bash
# GUI mode (no args)
./rbee-keeper

# CLI mode (with args)
./rbee-keeper hive list
./rbee-keeper queen status
```

---

## Files Changed

- **DELETED:** `bin/00_rbee_keeper/src/tauri_main.rs` (entire file)
- **MODIFIED:** `bin/00_rbee_keeper/Cargo.toml` (-5 LOC)
  - Removed `[[bin]] rbee-keeper-gui` entry
  - Removed `default-run` setting
- **MODIFIED:** `bin/00_rbee_keeper/src/main.rs` (+1 LOC)
  - Added `ssh_list` to `tauri::generate_handler![]`

---

## Verification

✅ **Build successful:**
```bash
cargo build --bin rbee-keeper
```

✅ **Both modes work:**
- GUI: `./target/debug/rbee-keeper` (no args)
- CLI: `./target/debug/rbee-keeper --help`

---

## Lessons Learned

**RULE ZERO is not optional:**
- Duplicate code paths = entropy
- Entropy is permanent pain
- Breaking changes are temporary pain
- Always choose temporary over permanent

**How to spot violations:**
- "We have two ways to do X"
- "Update both files when you add Y"
- "Don't forget to also change Z"
- "Keep them in sync"

**Correct response:**
- Delete one
- Fix compilation errors
- Done

---

**TEAM-333 | Oct 28, 2025**
