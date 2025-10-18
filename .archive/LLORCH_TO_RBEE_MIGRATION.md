# llorch → rbee Migration Complete

**TEAM-035**  
**Date:** 2025-10-10

## Summary

Replaced all `llorch` references with `rbee` in active codebase files.

## Changes Made

### 1. Binary Names ✅

- CLI binary: `llorch` → `rbee`
- Output: `target/release/llorch` → `target/release/rbee`

### 2. Environment Variables ✅

- `LLORCH_WORKER_HOST` → `RBEE_WORKER_HOST`
- `LLORCH_MODEL_BASE_DIR` → `RBEE_MODEL_BASE_DIR`

### 3. Script Names ✅

- `scripts/llorch-models` → `scripts/rbee-models`

### 4. Function Names ✅

- `find_llorch_models_script()` → `find_rbee_models_script()`

### 5. Documentation ✅

**Files updated:**
- `bin/rbee-keeper/README.md` - All command examples
- `QUICKSTART_INFERENCE.md` - Architecture diagram and examples
- `test_inference.sh` - Environment variable

**Command examples changed:**
```bash
# Old
llorch pool models catalog --host mac.home.arpa
llorch pool git pull --host mac.home.arpa
llorch pool worker spawn metal --model tinyllama --host mac.home.arpa

# New
rbee pool models catalog --host mac.home.arpa
rbee pool git pull --host mac.home.arpa
rbee pool worker spawn metal --model tinyllama --host mac.home.arpa
```

### 6. Code Files ✅

**Modified:**
- `bin/rbee-hive/src/http/workers.rs` - Environment variable
- `bin/rbee-hive/src/provisioner/download.rs` - Script name, function names
- `bin/rbee-hive/src/commands/daemon.rs` - Environment variable

## Usage

### Environment Variables

```bash
# Set worker host (for localhost testing)
export RBEE_WORKER_HOST=127.0.0.1

# Set model base directory
export RBEE_MODEL_BASE_DIR=.test-models
```

### Running the CLI

```bash
# Build
cargo build --release -p rbee-keeper

# Binary location
target/release/rbee

# Example command
rbee infer --node localhost --model "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" \
    --prompt "Once upon a time" --max-tokens 20
```

### Model Download Script

```bash
# Old
./scripts/llorch-models download tinyllama

# New
./scripts/rbee-models download tinyllama
```

## What Was NOT Changed

Historical/archived documentation files were left as-is:
- `.business/naming/` - Historical naming discussions
- `bin/.specs/` - Historical specs
- `test-harness/.archive/` - Archived test reports

These are kept for historical reference and don't affect runtime behavior.

## Testing

All changes tested with:
```bash
# Start pool manager
RBEE_WORKER_HOST=127.0.0.1 cargo run -p rbee-hive -- daemon --addr 127.0.0.1:8080

# Run inference
cargo run --release -p rbee-keeper -- infer \
    --node localhost \
    --model "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" \
    --prompt "Once upon a time" \
    --max-tokens 20
```

✅ All environment variables work  
✅ Script renamed and functional  
✅ CLI commands work with new binary name  

## Backward Compatibility

⚠️ **Breaking changes:**
- Old `llorch` binary name no longer works
- Old `LLORCH_*` environment variables no longer recognized
- Old `scripts/llorch-models` script moved

**Migration path:**
1. Update any scripts/configs to use `rbee` instead of `llorch`
2. Update environment variables: `LLORCH_*` → `RBEE_*`
3. Update script references: `llorch-models` → `rbee-models`

---

**Status:** ✅ Migration complete  
**Impact:** Active codebase only (historical docs preserved)  
**Next:** Update any external scripts/configs that reference `llorch`
