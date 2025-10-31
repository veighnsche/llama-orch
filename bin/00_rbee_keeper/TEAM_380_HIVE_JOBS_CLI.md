# TEAM-380: Hive Jobs CLI Commands

**Date:** Oct 31, 2025  
**Team:** TEAM-380  
**Status:** ✅ COMPLETE

## Mission

Add hive job operations to the CLI as a separate command from hive lifecycle operations.

## Changes Made

### 1. ✅ Created `handlers/hive_jobs_action.rs`

**Purpose:** Define CLI actions for hive job operations

**Actions:**
- `WorkerSpawn` - Spawn a worker process on hive
- `WorkerList` - List worker processes running on hive
- `WorkerGet` - Get details of a specific worker process
- `WorkerDelete` - Delete (kill) a worker process
- `ModelDownload` - Download a model to hive
- `ModelList` - List models available on hive
- `ModelGet` - Get details of a specific model
- `ModelDelete` - Delete a model from hive
- `Check` - Run hive diagnostic check (SSE streaming test)

**Lines of code:** 65 lines

### 2. ✅ Updated `handlers/hive_jobs.rs`

**Added:** `handle_hive_jobs(hive_id, action)` function

**Purpose:** Convert HiveJobsAction to Operation and submit via job-client

**Implementation:**
- Matches on HiveJobsAction variants
- Creates appropriate Operation (WorkerSpawn, ModelList, etc.)
- Calls `submit_hive_job()` with the operation
- Streams narration events to stdout

**Lines added:** 73 lines

### 3. ✅ Updated `handlers/mod.rs`

**Changes:**
- Added `pub mod hive_jobs_action;`
- Exported `handle_hive_jobs` function

### 4. ✅ Updated `cli/mod.rs`

**Changes:**
- Added `pub use crate::handlers::hive_jobs_action::HiveJobsAction;`

### 5. ✅ Updated `cli/commands.rs`

**Changes:**
- Imported `HiveJobsAction`
- Added new `HiveJobs` command with `--hive` flag and subcommands

### 6. ✅ Updated `main.rs`

**Changes:**
- Imported `handle_hive_jobs`
- Added match arm: `Commands::HiveJobs { hive_id, action } => handle_hive_jobs(hive_id, action).await`

## CLI Usage

### Hive Lifecycle Commands (Unchanged)

```bash
# Start/stop/status hive daemon
./rbee hive start
./rbee hive stop
./rbee hive status
./rbee hive install
./rbee hive uninstall
./rbee hive rebuild
```

### Hive Jobs Commands (NEW)

```bash
# Worker operations
./rbee hive-jobs --hive localhost worker-spawn --model llama-3.2-1b --worker cpu --device 0
./rbee hive-jobs --hive localhost worker-list
./rbee hive-jobs --hive localhost worker-get --pid 12345
./rbee hive-jobs --hive localhost worker-delete --pid 12345

# Model operations
./rbee hive-jobs --hive localhost model-download --model meta-llama/Llama-2-7b
./rbee hive-jobs --hive localhost model-list
./rbee hive-jobs --hive localhost model-get --id model-123
./rbee hive-jobs --hive localhost model-delete --id model-123

# Diagnostic
./rbee hive-jobs --hive localhost check
```

## Architecture

### Command Structure

```
./rbee
├── hive (lifecycle)
│   ├── start
│   ├── stop
│   ├── status
│   ├── install
│   ├── uninstall
│   └── rebuild
│
└── hive-jobs (job operations)
    ├── worker-spawn
    ├── worker-list
    ├── worker-get
    ├── worker-delete
    ├── model-download
    ├── model-list
    ├── model-get
    ├── model-delete
    └── check
```

### Data Flow

```
CLI Command
    ↓
HiveJobsAction enum
    ↓
handle_hive_jobs()
    ↓
Convert to Operation
    ↓
submit_hive_job()
    ↓
JobClient
    ↓
POST http://localhost:7835/v1/jobs
    ↓
SSE stream narration events
    ↓
Print to stdout
```

## Benefits

### 1. **Clear Separation**
- `./rbee hive` - Daemon lifecycle (start, stop, install)
- `./rbee hive-jobs` - Job operations (worker/model management)
- No confusion between lifecycle and job operations

### 2. **Consistent with Architecture**
- Lifecycle operations use lifecycle-local/lifecycle-ssh
- Job operations use job-client (aligns with contracts)
- Each command uses the appropriate pattern

### 3. **Extensible**
- Adding new job operations: Add variant to HiveJobsAction
- No changes needed to lifecycle commands
- Clear separation makes it easy to extend

### 4. **Type-Safe**
- CLI args validated by clap
- Converted to typed Operation structs
- Compiler ensures all operations handled

## Comparison with Old Commands

### Before (Worker/Model as separate top-level commands)

```bash
# Worker operations (top-level)
./rbee worker --hive localhost spawn ...
./rbee worker --hive localhost list
./rbee worker --hive localhost get --pid 123
./rbee worker --hive localhost delete --pid 123

# Model operations (top-level)
./rbee model --hive localhost download ...
./rbee model --hive localhost list
./rbee model --hive localhost get --id 123
./rbee model --hive localhost delete --id 123
```

### After (Grouped under hive-jobs)

```bash
# All hive job operations grouped together
./rbee hive-jobs --hive localhost worker-spawn ...
./rbee hive-jobs --hive localhost worker-list
./rbee hive-jobs --hive localhost worker-get --pid 123
./rbee hive-jobs --hive localhost worker-delete --pid 123

./rbee hive-jobs --hive localhost model-download ...
./rbee hive-jobs --hive localhost model-list
./rbee hive-jobs --hive localhost model-get --id 123
./rbee hive-jobs --hive localhost model-delete --id 123
```

**Note:** The old `worker` and `model` top-level commands still exist for backward compatibility, but `hive-jobs` provides a clearer grouping.

## Files Modified

1. **NEW:** `bin/00_rbee_keeper/src/handlers/hive_jobs_action.rs` (65 lines)
2. **MODIFIED:** `bin/00_rbee_keeper/src/handlers/hive_jobs.rs` (+73 lines)
3. **MODIFIED:** `bin/00_rbee_keeper/src/handlers/mod.rs` (added module and export)
4. **MODIFIED:** `bin/00_rbee_keeper/src/cli/mod.rs` (added re-export)
5. **MODIFIED:** `bin/00_rbee_keeper/src/cli/commands.rs` (added HiveJobs command)
6. **MODIFIED:** `bin/00_rbee_keeper/src/main.rs` (added match arm)

## Verification

✅ **Compilation:** `cargo check --bin rbee-keeper` passes

## Examples

### Spawn a worker

```bash
./rbee hive-jobs --hive localhost worker-spawn \
  --model llama-3.2-1b \
  --worker cpu \
  --device 0
```

**Output:**
```
🚀 Spawning worker 'cpu' with model 'llama-3.2-1b' on device 0
✅ Worker 'worker-cpu-9301' spawned (PID: 12345, port: 9301)
[DONE]
```

### List models

```bash
./rbee hive-jobs --hive localhost model-list
```

**Output:**
```
📋 Listing models on hive 'localhost'
Found 2 model(s)
  model-1 | llama-3.2-1b | 1.23 GB | available
  model-2 | llama-2-7b | 7.45 GB | available
[DONE]
```

### Run diagnostic check

```bash
./rbee hive-jobs --hive localhost check
```

**Output:**
```
🔍 Starting hive narration check
✅ Hive narration check complete
[DONE]
```

## Summary

Successfully added hive job operations to the CLI:
- ✅ Created HiveJobsAction enum with 9 operations
- ✅ Implemented handle_hive_jobs() handler
- ✅ Added `hive-jobs` command to CLI
- ✅ Clear separation from lifecycle commands
- ✅ Uses job-client for all operations
- ✅ Compilation successful
