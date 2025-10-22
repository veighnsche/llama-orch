# rbee-keeper CLI Migration Status

**TEAM-151 Progress Report**  
**Date:** 2025-10-20  
**Status:** ✅ CLI Entry Point Complete

---

## ✅ Completed: CLI Binary Migration

### What Was Migrated

**Source:** `bin/old.rbee-keeper/src/cli.rs` + `main.rs`  
**Destination:** `bin/00_rbee_keeper/src/main.rs`

**Key Changes:**
1. ✅ Complete CLI structure with all commands
2. ✅ Clap argument parsing (Parser + Subcommand derives)
3. ✅ All command variants from old structure preserved
4. ✅ Clean dispatch pattern with TODO stubs for commands crate
5. ✅ Compiles without errors or warnings
6. ✅ Help system works correctly

### Command Structure

```
rbee-keeper
├── infer                    (happy flow entry point)
├── setup
│   ├── add-node
│   ├── list-nodes
│   ├── remove-node
│   └── install
├── hive
│   ├── models (download/list/catalog/register)
│   ├── worker (spawn/list/stop)
│   ├── git (pull/status/build)
│   └── status
├── workers
│   ├── list
│   ├── health
│   └── shutdown
├── logs
└── install
```

### Test Results

**Compilation:**
```bash
cargo build --bin rbee-keeper
# ✅ Success - 0 errors, 0 warnings
```

**CLI Help:**
```bash
./target/debug/rbee-keeper --help
# ✅ Displays all commands correctly
```

**Infer Command (Happy Flow Entry):**
```bash
./target/debug/rbee-keeper infer "hello world" --model HF:author/minillama
# ✅ Parses arguments correctly
# Output:
#   Model: HF:author/minillama
#   Prompt: hello world
#   Max tokens: 20
#   Temperature: 0.7
```

---

## 🔄 Next Steps (In Order)

### Critical Path: Shared Crates First

These **MUST** be completed before command implementations:

#### 1. `daemon-lifecycle` (bin/99_shared_crates/)
**Priority:** 🔴 CRITICAL  
**Needed by:** rbee-keeper-queen-lifecycle crate  
**Extract from:** `old.rbee-keeper/src/queen_lifecycle.rs` lines 64-132

**Required API:**
```rust
pub struct DaemonManager;
impl DaemonManager {
    pub async fn start(&self, binary: &str, args: Vec<String>) -> Result<Child>;
    pub async fn health_check(&self, url: &str) -> Result<bool>;
    pub async fn stop(&self) -> Result<()>;
}
```

#### 2. `rbee-http-client` (bin/99_shared_crates/)
**Priority:** 🔴 CRITICAL  
**Needed by:** All commands that make HTTP requests  
**Extract from:** `old.rbee-keeper/src/pool_client.rs` + `commands/infer.rs`

**Required Features:**
- GET/POST requests with timeout
- Retry logic with exponential backoff
- SSE streaming support
- Error handling

#### 3. `rbee-types` (bin/99_shared_crates/)
**Priority:** 🔴 CRITICAL  
**Needed by:** All commands for request/response types

---

### Then: rbee-keeper Crates

#### 4. `rbee-keeper-queen-lifecycle` (bin/05_rbee_keeper_crates/queen-lifecycle/)
**Source:** `old.rbee-keeper/src/queen_lifecycle.rs`  
**Depends on:** daemon-lifecycle, rbee-http-client

**Key Function:**
```rust
pub async fn ensure_queen_running(base_url: &str) -> Result<()>
```

#### 5. `rbee-keeper-polling` (bin/05_rbee_keeper_crates/polling/)
**Source:** Extract from queen_lifecycle.rs (polling logic)  
**Depends on:** rbee-http-client

#### 6. `rbee-keeper-config` (bin/05_rbee_keeper_crates/config/)
**Source:** `old.rbee-keeper/src/config.rs`  
**Simple copy** with path updates

#### 7. `rbee-keeper-commands` (bin/05_rbee_keeper_crates/commands/)
**Source:** `old.rbee-keeper/src/commands/` (all 7 files)  
**Depends on:** All above crates

**Files to migrate:**
- `infer.rs` (272 lines) - **HIGHEST PRIORITY** (happy flow)
- `setup.rs` (8545 lines)
- `hive.rs` (4013 lines)
- `workers.rs` (6835 lines)
- `logs.rs` (1288 lines)
- `install.rs` (4240 lines)

---

### Finally: Wire Everything Together

#### 8. Update main.rs to use commands crate
**Replace TODO stubs with:**
```rust
Commands::Infer { ... } => {
    rbee_keeper_commands::infer::handle(...).await
}
```

**Update Cargo.toml:**
```toml
rbee-keeper-commands = { path = "../05_rbee_keeper_crates/commands" }
```

---

## 📊 Architecture Compliance

### ✅ Follows Minimal Binary Pattern
- Binary contains **only** CLI parsing
- All logic will be in crates (once implemented)
- Clean separation of concerns

### ✅ Matches Happy Flow
From `a_human_wrote_this.md`:
```bash
rbee-keeper infer "hello" HF:author/minillama
```

Maps to:
```
Commands::Infer { model, prompt, ... }
  → rbee-keeper-commands::infer::handle()
    → rbee-keeper-queen-lifecycle::ensure_queen_running()
      → daemon-lifecycle (spawn queen-rbee)
      → rbee-http-client (health check + SSE)
```

### ✅ Ports Aligned
- Queen: `:8500` (not :8080 from old code!)
- Hive: `:8600`
- Worker: `:8601`

---

## 📋 Migration Checklist Update

**From WORK_UNITS_CHECKLIST.md:**

### UNIT 2-A: rbee-keeper Commands (3-4h)
- [x] CLI structure migrated
- [ ] Commands crate created
- [ ] infer.rs migrated
- [ ] Other commands migrated

### Prerequisites
- [ ] UNIT 2-J: daemon-lifecycle
- [ ] UNIT 2-K: rbee-http-client  
- [ ] UNIT 2-L: rbee-types

---

## 🎯 Ready for Next Step

**Team 151 should now:**

1. **Choose path:**
   - **Option A:** Continue with commands crate (requires shared crates first)
   - **Option B:** Help with shared crates (daemon-lifecycle, rbee-http-client, rbee-types)

2. **Recommended:** Start with shared crates since they're blocking

---

**Files Modified:**
- ✅ `bin/00_rbee_keeper/src/main.rs` (340 lines)
- ✅ `bin/00_rbee_keeper/Cargo.toml` (dependencies added)

**Files Ready for Deletion (after full migration):**
- `bin/old.rbee-keeper/src/cli.rs`
- `bin/old.rbee-keeper/src/main.rs`

**Compilation Status:** ✅ Clean build, 0 warnings
