# Smart `rbee` Wrapper

**Created by:** TEAM-162

## Overview

The `rbee` command is a smart wrapper around `rbee-keeper` that automatically:
1. ✅ Checks if the binary is up-to-date with source code
2. 🔨 Rebuilds only if needed (source files changed)
3. ⚡ Forwards all commands to `rbee-keeper`

## Usage

From the **workspace root**, run any `rbee-keeper` command using `./rbee`:

```bash
# Check queen status
./rbee queen status

# Start queen
./rbee queen start

# Run inference
./rbee infer --model llama3 --prompt "Hello"

# Any rbee-keeper command works!
./rbee --help
```

## How It Works

```
./rbee <args>
  ↓
cargo xtask rbee <args>
  ↓
xtask/src/tasks/rbee.rs
  ├─ Check: Is target/debug/rbee-keeper older than bin/00_rbee_keeper/**/*.rs?
  ├─ If YES: cargo build --bin rbee-keeper
  └─ If NO: Skip build
  ↓
Execute: target/debug/rbee-keeper <args>
```

## Implementation

### Files Created

1. **`/rbee`** - Root wrapper script
   - Forwards to `cargo xtask rbee`

2. **`xtask/src/tasks/rbee.rs`** - Smart build logic
   - `needs_rebuild()` - Checks file timestamps
   - `build_rbee_keeper()` - Runs `cargo build --bin rbee-keeper`
   - `run_rbee_keeper()` - Main entry point

3. **`xtask/src/cli.rs`** - Added `Cmd::Rbee` variant
   - Uses `trailing_var_arg = true` to capture all args

### Build Detection Logic

```rust
fn needs_rebuild(workspace_root: &Path) -> Result<bool> {
    // 1. Check if binary exists
    if !binary_path.exists() { return Ok(true); }
    
    // 2. Get binary modification time
    let binary_time = binary_meta.modified()?;
    
    // 3. Check if any .rs or Cargo.toml files are newer
    check_dir_newer(&keeper_dir, binary_time)
}
```

## Benefits

✅ **No manual builds** - Automatically rebuilds when needed  
✅ **Fast when up-to-date** - Skips build if nothing changed  
✅ **Simple interface** - Just use `./rbee` for everything  
✅ **Integration testing** - Perfect for testing the full stack  

## Examples

### Development Workflow

```bash
# Edit source code
vim bin/00_rbee_keeper/src/main.rs

# Run command - auto-rebuilds!
./rbee queen start
# Output: 🔨 Building rbee-keeper...
#         ✅ Build complete
#         [🧑‍🌾 rbee-keeper] Queen started...

# Run again - no rebuild needed
./rbee queen status
# Output: ✅ rbee-keeper is up-to-date
#         [🧑‍🌾 rbee-keeper] Queen is running...
```

### Integration Testing

```bash
# Test the entire integration without manual builds
./rbee queen start
./rbee hive register --pool gpu-0
./rbee worker start --hive gpu-0
./rbee infer --prompt "Test"
./rbee queen stop
```

## Comparison

### Before (Manual)
```bash
# Developer has to remember to rebuild
cd bin/00_rbee_keeper
cargo build
cd ../..
./target/debug/rbee-keeper queen status
```

### After (Automatic)
```bash
# Just run it!
./rbee queen status
```

## Technical Details

- **Timestamp-based detection** - Compares file modification times
- **Recursive scanning** - Checks all `.rs` and `Cargo.toml` files in `bin/00_rbee_keeper/`
- **Skips target dirs** - Ignores build artifacts
- **Exit code forwarding** - Preserves rbee-keeper's exit codes
- **Workspace-aware** - Always runs from workspace root

## Future Enhancements

Potential improvements:
- [ ] Cache build status in `.rbee-build-cache` for faster checks
- [ ] Support `RBEE_FORCE_REBUILD=1` env var
- [ ] Add `--no-rebuild` flag to skip build check
- [ ] Parallel build detection for other binaries (queen-rbee, rbee-hive)
