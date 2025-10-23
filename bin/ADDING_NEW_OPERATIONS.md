# Adding New Operations - Quick Reference

**Last Updated:** Oct 24, 2025 (TEAM-283)

## The 3-File Pattern

Every new operation requires updates to exactly **3 files** in this order:

```
1. rbee-operations/src/lib.rs  (Contract)
2. queen-rbee/src/job_router.rs (Router)
3. rbee-keeper/src/handlers/    (CLI Handler)
```

## TEAM-277 Update: Package Operations

**New declarative operations added:**
- `PackageSync` - Sync all hives to match config (uses daemon-sync crate)
- `PackageStatus` - Check package status and detect drift
- `PackageInstall` - Install all components
- `PackageUninstall` - Uninstall components
- `PackageValidate` - Validate declarative config
- `PackageMigrate` - Generate config from current state

**Removed imperative operations:**
- ~~`HiveInstall`~~ - Replaced by PackageSync
- ~~`HiveUninstall`~~ - Replaced by PackageUninstall
- ~~`WorkerDownload`~~ - Replaced by PackageSync
- ~~`WorkerBuild`~~ - Replaced by PackageSync
- ~~`WorkerBinaryList`~~ - Replaced by PackageStatus
- ~~`WorkerBinaryGet`~~ - Replaced by PackageStatus
- ~~`WorkerBinaryDelete`~~ - Replaced by PackageUninstall

---

## Step 1: Define Operation (Contract)

**File:** `bin/99_shared_crates/rbee-operations/src/lib.rs`

### 1a. Add enum variant (~line 54)

```rust
pub enum Operation {
    // ... existing operations ...
    
    HiveYourNewOp {
        alias: String,
        // Add your fields here
    },
}
```

### 1b. Add to `Operation::name()` (~line 148)

```rust
impl Operation {
    pub fn name(&self) -> &'static str {
        match self {
            // ... existing cases ...
            Operation::HiveYourNewOp { .. } => "hive_your_new_op",
        }
    }
}
```

### 1c. Add to `Operation::hive_id()` if needed (~line 173)

```rust
pub fn hive_id(&self) -> Option<&str> {
    match self {
        // ... existing cases ...
        Operation::HiveYourNewOp { alias } => Some(alias),
    }
}
```

### 1d. Add constant if needed (~line 204)

```rust
pub mod constants {
    pub const OP_HIVE_YOUR_NEW_OP: &str = "hive_your_new_op";
}
```

---

## Step 2: Route Operation (Server)

**File:** `bin/10_queen_rbee/src/job_router.rs`

### 2a. Import request types if using lifecycle crate (~line 30)

```rust
use queen_rbee_hive_lifecycle::{
    execute_your_new_op,
    YourNewOpRequest,
    // ... other imports
};
```

### 2b. Add match arm (~line 149)

```rust
match operation {
    // ... existing operations ...
    
    Operation::HiveYourNewOp { alias } => {
        let request = YourNewOpRequest { alias };
        execute_your_new_op(request, state.config.clone(), &job_id).await?;
    }
}
```

---

## Step 3: Add CLI Command (Client)

**File:** `bin/00_rbee_keeper/src/main.rs`

### 3a. Add to appropriate action enum (~line 188 for HiveAction)

```rust
pub enum HiveAction {
    // ... existing actions ...
    
    YourNewOp {
        /// Hive alias from ~/.config/rbee/hives.conf
        #[arg(short = 'a', long = "host")]
        alias: String,
    },
}
```

### 3b. Add match arm in `handle_command()` (~line 388 for Hive)

```rust
Commands::Hive { action } => {
    let operation = match action {
        // ... existing actions ...
        HiveAction::YourNewOp { alias } => Operation::HiveYourNewOp { alias },
    };
    submit_and_stream_job(&client, &queen_url, operation).await
}
```

---

## Examples by Category

### System-Wide Operation (no hive)
- **Example:** `Operation::Status`
- **Pattern:** No hive_id, direct implementation in job_router.rs
- **CLI:** Top-level command in `Commands` enum

### Hive Operation (managed by queen)
- **Example:** `Operation::HiveStart`, `Operation::HiveRefreshCapabilities`
- **Pattern:** Uses `queen-rbee-hive-lifecycle` crate
- **CLI:** Subcommand under `Commands::Hive`

### Worker/Model Operation (forwarded to hive)
- **Example:** `Operation::WorkerSpawn`, `Operation::ModelDownload`
- **Pattern:** Forward to hive via HTTP (TODO: not yet implemented)
- **CLI:** Subcommand under `Commands::Worker` or `Commands::Model`

### Inference Operation
- **Example:** `Operation::Infer`
- **Pattern:** Forward to hive, hive handles worker selection
- **CLI:** Top-level command with many parameters

---

## Testing Checklist

After adding a new operation:

- [ ] `cargo check -p rbee-operations` passes
- [ ] `cargo check -p queen-rbee` passes
- [ ] `cargo check -p rbee-keeper` passes
- [ ] `cargo build --bin rbee-keeper` succeeds
- [ ] CLI help shows new command: `./rbee hive --help`
- [ ] Command executes without panic: `./rbee hive your-new-op -a localhost`
- [ ] SSE streaming works (narration appears in real-time)

---

## Common Pitfalls

1. **Forgot to add to `Operation::name()`** → Compilation succeeds but runtime panic
2. **Forgot to pass `job_id` in lifecycle crate** → SSE events go to stdout, not stream
3. **Used `-h` for short option** → Conflicts with `--help`, use `-a` instead
4. **Cloned values unnecessarily** → Use `&action` in match or move owned values
5. **Added operation but no CLI command** → Users can't access it

---

## Architecture Principles

- **rbee-keeper**: Thin HTTP client, no business logic
- **rbee-operations**: Type-safe contract between client and server
- **job_router**: Dispatcher that routes to appropriate handlers
- **hive-lifecycle**: Shared crate for hive management operations

**Why 3 files?**
- Clean separation of concerns
- Type safety via shared contract
- Single responsibility per file
- Easy to test each layer independently

**Why not consolidate?**
- Would blur CLI/server boundaries
- Would lose type safety benefits
- Would make testing harder
- Current pattern is already well-established

---

## Need Help?

See existing operations for examples:
- Simple: `Operation::HiveList` (no parameters)
- With parameters: `Operation::HiveStart` (alias)
- Complex: `Operation::Infer` (many parameters, streaming)
- System-wide: `Operation::Status` (no hive, uses registry)
