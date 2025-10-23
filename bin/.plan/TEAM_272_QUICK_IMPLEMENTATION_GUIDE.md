# TEAM-272: Quick Implementation Guide for New Operations

**Date:** Oct 23, 2025  
**Reference:** See TEAM_272_NEW_OPERATIONS_CHECKLIST.md for full checklist

---

## 🚀 Quick Start: Implementing a New Operation

### Step 1: Operation Already Defined ✅

All 28 operations are already defined in `rbee-operations/src/lib.rs`:
- ✅ Enum variants added
- ✅ `name()` method updated
- ✅ `hive_id()` method updated
- ✅ `should_forward_to_hive()` updated

**You can skip Step 1 for all new operations!**

---

### Step 2: Add Router Handler

**For HIVE operations** (forwarded to hive):

**File:** `bin/20_rbee_hive/src/job_router.rs`

```rust
// Add to match statement in route_operation()
Operation::WorkerBinaryList { hive_id } => {
    NARRATE
        .action("worker_binary_list_start")
        .job_id(&job_id)
        .context(&hive_id)
        .human("📋 Listing worker binaries on hive '{}'")
        .emit();

    let binaries = state.worker_catalog.list();
    
    for binary in &binaries {
        NARRATE
            .action("worker_binary_entry")
            .job_id(&job_id)
            .context(&binary.name)
            .context(&binary.size.to_string())
            .human("  {} | {} bytes")
            .emit();
    }
}
```

**For QUEEN operations** (handled by queen):

**File:** `bin/10_queen_rbee/src/job_router.rs`

```rust
// Add to match statement in route_operation()
Operation::ActiveWorkerList => {
    NARRATE
        .action("active_worker_list_start")
        .job_id(&job_id)
        .human("📋 Listing active workers")
        .emit();

    let workers = state.worker_registry.list_active();
    
    for worker in &workers {
        NARRATE
            .action("active_worker_entry")
            .job_id(&job_id)
            .context(&worker.id)
            .context(&worker.hive_id)
            .context(&worker.model_id)
            .human("  {} | {} | {}")
            .emit();
    }
}
```

---

### Step 3: Add CLI Command

**File:** `bin/00_rbee_keeper/src/main.rs`

#### 3a. Add to Action Enum

```rust
// For worker operations
#[derive(Debug, Subcommand)]
pub enum WorkerAction {
    // ... existing actions ...
    
    /// List worker binaries on hive
    BinaryList {
        /// Hive to list binaries from
        #[arg(short = 'H', long = "hive", default_value = "localhost")]
        hive_id: String,
    },
}
```

#### 3b. Add to Command Handler

```rust
Commands::Worker { action } => {
    let operation = match action {
        // ... existing actions ...
        
        WorkerAction::BinaryList { hive_id } => {
            Operation::WorkerBinaryList { hive_id }
        }
    };
    submit_and_stream_job(&client, &queen_url, operation).await
}
```

---

## 📋 Implementation Templates

### Template 1: Simple List Operation

**Use for:** WorkerBinaryList, WorkerProcessList, ActiveWorkerList

```rust
Operation::YourList { hive_id } => {
    NARRATE
        .action("your_list_start")
        .job_id(&job_id)
        .context(&hive_id)
        .human("📋 Listing items on hive '{}'")
        .emit();

    let items = state.your_catalog.list();
    
    if items.is_empty() {
        NARRATE
            .action("your_list_empty")
            .job_id(&job_id)
            .human("No items found")
            .emit();
    } else {
        for item in &items {
            NARRATE
                .action("your_list_entry")
                .job_id(&job_id)
                .context(&item.name)
                .human("  {}")
                .emit();
        }
    }
}
```

### Template 2: Get Operation

**Use for:** WorkerBinaryGet, WorkerProcessGet, ActiveWorkerGet

```rust
Operation::YourGet { hive_id, id } => {
    NARRATE
        .action("your_get_start")
        .job_id(&job_id)
        .context(&hive_id)
        .context(&id)
        .human("🔍 Getting item '{}' on hive '{}'")
        .emit();

    let item = state.your_catalog.get(&id)?;
    let json = serde_json::to_string_pretty(&item)?;
    
    NARRATE
        .action("your_get_found")
        .job_id(&job_id)
        .human(&json)
        .emit();
}
```

### Template 3: Delete Operation

**Use for:** WorkerBinaryDelete, WorkerProcessDelete

```rust
Operation::YourDelete { hive_id, id } => {
    NARRATE
        .action("your_delete_start")
        .job_id(&job_id)
        .context(&hive_id)
        .context(&id)
        .human("🗑️  Deleting item '{}' on hive '{}'")
        .emit();

    state.your_catalog.remove(&id)?;
    
    NARRATE
        .action("your_delete_complete")
        .job_id(&job_id)
        .context(&id)
        .human("✅ Item '{}' deleted")
        .emit();
}
```

### Template 4: Download/Build Operation

**Use for:** WorkerDownload, WorkerBuild, ModelDownload

```rust
Operation::YourDownload { hive_id, name } => {
    NARRATE
        .action("your_download_start")
        .job_id(&job_id)
        .context(&hive_id)
        .context(&name)
        .human("📥 Downloading '{}' on hive '{}'")
        .emit();

    // Stream progress
    state.your_provisioner
        .download(&name, |progress| {
            NARRATE
                .action("your_download_progress")
                .job_id(&job_id)
                .context(&progress.to_string())
                .human("Progress: {}%")
                .emit();
        })
        .await?;
    
    NARRATE
        .action("your_download_complete")
        .job_id(&job_id)
        .context(&name)
        .human("✅ Download complete: {}")
        .emit();
}
```

---

## 🎯 Priority Order for Implementation

### Week 1: Worker Binary Operations
1. **WorkerBinaryList** (easiest, 4-6h)
2. **WorkerBinaryGet** (easy, 4-6h)
3. **WorkerDownload** (medium, 8-12h)
4. **WorkerBinaryDelete** (easy, 4-6h)
5. **WorkerBuild** (hard, 12-16h)

### Week 2: Worker Process Operations
1. **WorkerProcessList** (medium, 6-8h)
2. **WorkerProcessGet** (easy, 4-6h)
3. **WorkerProcessDelete** (easy, 4-6h) - uses existing delete_worker()

### Week 3: Active Worker Operations
**BLOCKER:** Need worker registry first!

1. Implement worker registry (20-30h)
2. **ActiveWorkerList** (medium, 12-16h)
3. **ActiveWorkerGet** (easy, 6-8h)
4. **ActiveWorkerRetire** (medium, 8-12h)

### Week 4-5: Model & Inference
**BLOCKER:** Need model provisioner first!

1. Implement model provisioner (16-24h)
2. **ModelDownload** (medium, 16-24h)
3. Implement inference scheduler (40-60h)
4. **Infer** (hard, 40-60h)

---

## ✅ Testing Checklist (Per Operation)

After implementing each operation:

```bash
# 1. Compilation
cargo check --bin rbee-hive        # For hive operations
cargo check --bin queen-rbee       # For queen operations
cargo check --bin rbee-keeper      # For CLI

# 2. Build
cargo build --bin rbee-keeper

# 3. CLI Help
./target/debug/rbee worker --help  # Should show new command

# 4. Manual Test
./target/debug/rbee worker binary list --hive localhost

# 5. Verify SSE streaming
# Should see narration events in real-time, not all at once
```

---

## 🔑 Key Points to Remember

### 1. Always Include job_id
```rust
NARRATE
    .action("your_action")
    .job_id(&job_id)  // ← CRITICAL for SSE routing
    .human("Your message")
    .emit();
```

### 2. Use Existing Catalogs/Registries
- `state.worker_catalog` - Worker binaries
- `state.model_catalog` - Models
- `state.worker_registry` - Active workers (TODO)

### 3. Follow Naming Conventions
- Actions: `your_operation_start`, `your_operation_complete`, `your_operation_error`
- Emojis: 📋 (list), 🔍 (get), 🗑️ (delete), 📥 (download), 🔨 (build)

### 4. Error Handling
```rust
// Return errors, don't panic
let item = state.catalog.get(&id)?;

// Provide helpful error messages
return Err(anyhow::anyhow!(
    "Item '{}' not found on hive '{}'",
    id, hive_id
));
```

### 5. State Management
```rust
// Hive operations use JobState
pub struct JobState {
    pub registry: Arc<JobRegistry<String>>,
    pub model_catalog: Arc<ModelCatalog>,
    pub worker_catalog: Arc<WorkerCatalog>,  // TODO: Add this
}

// Queen operations use JobState
pub struct JobState {
    pub registry: Arc<JobRegistry<String>>,
    pub config: Arc<RbeeConfig>,
    pub hive_registry: Arc<HiveRegistry>,
    pub worker_registry: Arc<WorkerRegistry>,  // TODO: Add this
}
```

---

## 📝 Documentation Requirements

For each operation, update:

1. **This checklist** - Mark operation as complete
2. **Handoff document** - Add implementation notes
3. **ADDING_NEW_OPERATIONS.md** - Add example if pattern is new
4. **CLI help text** - Add clear description

---

## 🚨 Common Mistakes to Avoid

1. ❌ Forgetting `job_id` in narration → SSE doesn't work
2. ❌ Not checking if catalog/registry exists → Panics
3. ❌ Using blocking operations → Hangs the server
4. ❌ Not handling errors → Crashes on failure
5. ❌ Hardcoding values → Not configurable
6. ❌ Not streaming progress → User sees nothing until complete
7. ❌ Not adding CLI command → Operation not accessible

---

## 🎉 Success Criteria

**Operation is complete when:**
- ✅ Compiles without errors
- ✅ CLI command works
- ✅ SSE streaming works
- ✅ Error handling works
- ✅ Documentation updated
- ✅ Tested manually
- ✅ Handoff document created

---

**Ready to implement! Start with WorkerBinaryList - it's the easiest! 🚀**
