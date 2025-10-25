# TEAM-295: Naked `./rbee` Launches Tauri GUI

**Status:** ✅ COMPLETE

**Mission:** When running `./rbee` with no arguments, launch the Tauri GUI instead of showing help text.

## Problem

Running `./rbee` with no arguments would show clap help text. User wanted:
- `./rbee queen start` → CLI mode (existing behavior)
- `./rbee` (naked, no args) → Launch Tauri GUI

## Solution

Made the `command` field optional in CLI parser. When no subcommand is provided, launch Tauri GUI directly in main.rs.

## Implementation Details

### 1. Made Command Optional

**File:** `bin/00_rbee_keeper/src/cli/commands.rs`

```rust
// TEAM-295: Made command optional - if no subcommand, launch Tauri GUI
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,  // ← Changed from Commands to Option<Commands>
}
```

### 2. Added GUI Launch Logic

**File:** `bin/00_rbee_keeper/src/main.rs`

```rust
#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // TEAM-295: If no subcommand provided, launch Tauri GUI instead
    if cli.command.is_none() {
        launch_gui();
        return Ok(());
    }
    
    handle_command(cli).await
}

// TEAM-295: Launch Tauri GUI (synchronous, blocks until window closes)
fn launch_gui() {
    use rbee_keeper::tauri_commands::*;
    
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            // Status
            get_status,
            // Queen commands
            queen_start,
            queen_stop,
            // ... (all tauri commands)
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### 3. Unwrap Command in Handler

**File:** `bin/00_rbee_keeper/src/main.rs`

```rust
async fn handle_command(cli: Cli) -> Result<()> {
    // ...
    
    // TEAM-295: Command is now Option, unwrap it here since we checked is_some earlier
    let command = cli.command.expect("Command should be Some if we reach here");

    match command {
        Commands::Status => handle_status(&queen_url).await,
        // ... rest of match arms
    }
}
```

## How It Works

1. User runs `./rbee` (shell script)
2. Shell script calls `xtask rbee` (with no args)
3. xtask forwards empty args to `rbee-keeper` binary
4. rbee-keeper parses CLI and sees `command: None`
5. rbee-keeper calls `launch_gui()`
6. Tauri window opens

## Codeflow Verification

```bash
# CLI mode (existing behavior)
./rbee queen start  # → xtask → rbee-keeper → handle_command → handle_queen

# GUI mode (new behavior)
./rbee             # → xtask → rbee-keeper → launch_gui (Tauri window opens)
```

## Files Modified

- ✅ `bin/00_rbee_keeper/src/cli/commands.rs` (1 line changed)
- ✅ `bin/00_rbee_keeper/src/main.rs` (48 lines added)
- ✅ `Cargo.toml` (removed broken rbee-sdk reference)

## Compilation

```bash
$ cargo check --bin rbee-keeper
✅ SUCCESS

$ cargo build --bin rbee-keeper
✅ SUCCESS (includes Tauri dependencies)
```

## Testing

```bash
# Test CLI mode (should work as before)
./rbee queen status

# Test GUI mode (should open Tauri window)
./rbee
```

## Architecture Notes

- **Separation:** CLI binary and GUI binary share the same codebase via `lib.rs`
- **Handlers:** Both CLI and GUI use the same handler functions
- **Consistency:** Single source of truth for business logic
- **Default:** `Cargo.toml` sets `default-run = "rbee-keeper-gui"` but `./rbee` script overrides this by calling `rbee-keeper` binary directly

## Engineering Rules Compliance

- ✅ **No TODO markers** - All functionality implemented
- ✅ **TEAM-295 signatures** - All changes marked with TEAM-295 comments
- ✅ **Complete implementation** - GUI launches on naked `./rbee`, CLI works with args
- ✅ **Compilation verified** - `cargo check` and `cargo build` both pass
- ✅ **No background testing** - All commands run in foreground

## Code Signatures

All modified code includes `// TEAM-295:` comments for historical tracking.

---

**TEAM-295 Complete**
**Date:** 2025-10-25
**Handoff:** Ready for testing
