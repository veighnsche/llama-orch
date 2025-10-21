# 🐝 Quick Start: Using `rbee`

**Created by:** TEAM-162

## TL;DR

```bash
# From workspace root, just use ./rbee for everything:
./rbee queen start      # Auto-builds if needed, then starts queen
./rbee queen status     # Check queen status
./rbee --help          # See all commands
```

## What is `rbee`?

Smart wrapper that:
- ✅ Auto-detects if `rbee-keeper` needs rebuilding
- 🔨 Rebuilds only when source files changed
- ⚡ Forwards all commands to the binary

## Common Commands

```bash
# Queen management
./rbee queen start
./rbee queen status
./rbee queen stop

# Hive management
./rbee hive list
./rbee hive register --pool gpu-0

# Worker management
./rbee worker start --hive gpu-0
./rbee worker list

# Model management
./rbee model list
./rbee model download llama3

# Inference
./rbee infer --model llama3 --prompt "Hello, world!"
```

## How It Works

```
./rbee <command>
  ↓
Checks: Is source newer than binary?
  ├─ YES → cargo build --bin rbee-keeper
  └─ NO  → Skip build
  ↓
Runs: target/debug/rbee-keeper <command>
```

## Output Examples

### When rebuild is needed:
```
$ ./rbee queen status
🔨 Building rbee-keeper...
   Compiling rbee-keeper v0.1.0
✅ Build complete

[🧑‍🌾 rbee-keeper]
  ✅ Queen is running on http://localhost:8500
```

### When up-to-date:
```
$ ./rbee queen status
✅ rbee-keeper is up-to-date

[🧑‍🌾 rbee-keeper]
  ✅ Queen is running on http://localhost:8500
```

## Why Use This?

**Before:**
```bash
# Manual workflow - easy to forget!
cd bin/00_rbee_keeper
cargo build
cd ../..
./target/debug/rbee-keeper queen status
```

**After:**
```bash
# Just works!
./rbee queen status
```

## See Also

- Full documentation: `xtask/RBEE_WRAPPER.md`
- Implementation: `xtask/src/tasks/rbee.rs`
