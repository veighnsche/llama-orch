# TEAM-190 Implementation Guide - Part 2: Step-by-Step

## Step-by-Step Implementation Checklist

Use this checklist when implementing ANY new operation.

---

### Step 1: Define CLI Command

**File:** `bin/00_rbee_keeper/src/main.rs`

- [ ] Add variant to appropriate `Action` enum (HiveAction, WorkerAction, etc.)
- [ ] Add doc comments for help text
- [ ] Add TEAM-XXX annotation
- [ ] Use `default_value` for common parameters
- [ ] Keep parameter names short and consistent

**Example:**
```rust
/// Start the hive daemon
/// TEAM-190: Spawns hive process and waits for health check
Start {
    /// Hive ID (defaults to localhost)
    #[arg(default_value = "localhost")]
    id: String,
},
```

---

### Step 2: Define Operation Type

**File:** `bin/99_shared_crates/rbee-operations/src/lib.rs`

- [ ] Add variant to `Operation` enum
- [ ] Add doc comment with TEAM-XXX
- [ ] Use `#[serde(default = "...")]` for defaults
- [ ] Add to `name()` method
- [ ] Add to `hive_id()` extraction if applicable
- [ ] Add constant in `constants` module

**Example:**
```rust
// In enum
/// TEAM-190: Start hive daemon
HiveStart {
    #[serde(default = "default_hive_id")]
    hive_id: String,
},

// In name() method
Operation::HiveStart { .. } => constants::OP_HIVE_START,

// In hive_id() method
Operation::HiveStart { hive_id } => Some(hive_id),

// In constants module
pub const OP_HIVE_START: &str = "hive_start"; // TEAM-190
```

---

### Step 3: Add CLI Routing

**File:** `bin/00_rbee_keeper/src/main.rs` (in `handle_command()`)

- [ ] Match on your action variant
- [ ] Create corresponding `Operation` variant
- [ ] Map parameter names correctly

**Example:**
```rust
Commands::Hive { action } => {
    let operation = match action {
        HiveAction::Start { id } => Operation::HiveStart { hive_id: id },
        // ... other actions
    };
    submit_and_stream_job(&client, &queen_url, operation).await
}
```

---

### Step 4: Implement Server Handler

**File:** `bin/10_queen_rbee/src/job_router.rs`

- [ ] Add match arm in `route_operation()`
- [ ] Add TEAM-XXX comment at top
- [ ] Follow the narration pattern (below)
- [ ] Handle all error cases
- [ ] Provide actionable error messages

**Template:**
```rust
Operation::YourOperation { param } => {
    // TEAM-XXX: Brief description of what this does
    
    // Step 1: Initial narration
    Narration::new(ACTOR_QUEEN_ROUTER, "your_operation", &param)
        .human(format!("🔧 Starting operation for '{}'", param))
        .emit();
    
    // Step 2: Pre-flight checks
    Narration::new(ACTOR_QUEEN_ROUTER, "your_operation_preflight", &param)
        .human("📋 Running pre-flight checks...")
        .emit();
    
    let resource = state.catalog.get(&param).await?;
    if resource.is_none() {
        Narration::new(ACTOR_QUEEN_ROUTER, "your_operation_not_found", &param)
            .human(format!(
                "❌ Resource '{}' not found.\n\
                 \n\
                 To create it:\n\
                 \n\
                   ./rbee resource create",
                param
            ))
            .emit();
        return Err(anyhow::anyhow!("Resource not found"));
    }
    
    // Step 3: Do the actual work
    Narration::new(ACTOR_QUEEN_ROUTER, "your_operation_work", &param)
        .human("⚙️  Executing operation...")
        .emit();
    
    // ... your implementation ...
    
    // Step 4: Success narration
    Narration::new(ACTOR_QUEEN_ROUTER, "your_operation_success", &param)
        .human(format!("✅ Operation completed successfully"))
        .emit();
}
```

---

### Step 5: Add Dependencies (if needed)

**File:** `bin/10_queen_rbee/Cargo.toml`

- [ ] Add required crates with TEAM-XXX comment
- [ ] Use workspace dependencies when available

**Example:**
```toml
daemon-lifecycle = { path = "../99_shared_crates/daemon-lifecycle" }  # TEAM-190: For spawning processes
```

---

### Step 6: Test Your Implementation

- [ ] Build: `cargo build --bin rbee-keeper --bin queen-rbee`
- [ ] Test happy path: `./rbee your command`
- [ ] Test error cases (not found, already running, etc.)
- [ ] Test with different parameters
- [ ] Verify narration is clear and helpful
- [ ] Verify error messages provide actionable guidance

---

### Step 7: Add TEAM Annotations

- [ ] CLI command definition
- [ ] Operation variant
- [ ] Operation constant
- [ ] Server handler
- [ ] Any new dependencies
- [ ] Update or create TEAM-XXX-SUMMARY.md

---

## Narration Best Practices

### Use Emojis Consistently

- 🔧 Starting/Installing
- 🗑️ Uninstalling/Deleting
- 🚀 Starting/Launching
- 🛑 Stopping
- 📋 Checking/Verifying
- ✅ Success
- ❌ Error
- ⚠️ Warning
- 📝 Writing/Saving
- 🔍 Searching/Looking
- ⏳ Waiting
- 📤 Sending
- 📊 Listing/Showing

### Narration Stages

1. **Initial**: What you're about to do
   ```rust
   "🔧 Installing hive 'localhost'"
   ```

2. **Progress**: What you're doing now
   ```rust
   "📋 Checking if hive is already installed..."
   ```

3. **Decision**: What you found/decided
   ```rust
   "✅ Hive not found in catalog - proceeding with installation"
   ```

4. **Action**: What specific action you're taking
   ```rust
   "🔍 Looking for rbee-hive binary in target/debug..."
   ```

5. **Result**: What happened
   ```rust
   "✅ Found binary at: target/debug/rbee-hive"
   ```

6. **Final**: Overall result
   ```rust
   "✅ Hive 'localhost' installed successfully!"
   ```

---

## Common Gotchas

### 1. Async/Await

❌ **Wrong:**
```rust
let result = state.catalog.get(&id);  // Missing .await
```

✅ **Right:**
```rust
let result = state.catalog.get(&id).await?;
```

### 2. Error Handling

❌ **Wrong:**
```rust
return Err("not found".into());
```

✅ **Right:**
```rust
return Err(anyhow::anyhow!("Hive '{}' not found", id));
```

### 3. Parameter Names

❌ **Wrong:** Inconsistent names
```rust
// CLI uses 'id', Operation uses 'hive_name'
```

✅ **Right:** Consistent mapping
```rust
// CLI: id → Operation: hive_id (always)
```

### 4. Default Values

❌ **Wrong:** No default for common case
```rust
#[arg(long)]
id: String,  // User must always type --id localhost
```

✅ **Right:** Default for common case
```rust
#[arg(long, default_value = "localhost")]
id: String,  // ./rbee hive start (defaults to localhost)
```

### 5. Narration Timing

❌ **Wrong:** Narrate after the work
```rust
do_expensive_work().await?;
Narration::new(...).human("🔧 Doing work...").emit();
```

✅ **Right:** Narrate before the work
```rust
Narration::new(...).human("🔧 Doing work...").emit();
do_expensive_work().await?;
```

---

## Testing Commands

```bash
# Build everything
cargo build --bin rbee-keeper --bin queen-rbee --quiet

# Test your operation
./rbee your command --param value

# Test with default params
./rbee your command

# Test error cases
./rbee your command --id nonexistent

# Check narration flow
./rbee your command 2>&1 | grep "queen-router"

# Test end-to-end
./rbee your command && echo "✅ Success" || echo "❌ Failed"
```

---

## Example: Implementing `hive restart`

Following the complete pattern:

### 1. CLI (`bin/00_rbee_keeper/src/main.rs`)
```rust
/// Restart a hive daemon
/// TEAM-190: Stops and starts the hive
Restart {
    #[arg(default_value = "localhost")]
    id: String,
},
```

### 2. Operation (`bin/99_shared_crates/rbee-operations/src/lib.rs`)
```rust
/// TEAM-190: Restart hive daemon
HiveRestart {
    #[serde(default = "default_hive_id")]
    hive_id: String,
},

// In name()
Operation::HiveRestart { .. } => constants::OP_HIVE_RESTART,

// In hive_id()
Operation::HiveRestart { hive_id } => Some(hive_id),

// In constants
pub const OP_HIVE_RESTART: &str = "hive_restart"; // TEAM-190
```

### 3. Routing (`bin/00_rbee_keeper/src/main.rs`)
```rust
HiveAction::Restart { id } => Operation::HiveRestart { hive_id: id },
```

### 4. Implementation (`bin/10_queen_rbee/src/job_router.rs`)
```rust
Operation::HiveRestart { hive_id } => {
    // TEAM-190: Restart hive by stopping then starting
    
    Narration::new(ACTOR_QUEEN_ROUTER, "hive_restart", &hive_id)
        .human(format!("🔄 Restarting hive '{}'", hive_id))
        .emit();
    
    // Stop first
    Narration::new(ACTOR_QUEEN_ROUTER, "hive_restart_stop", &hive_id)
        .human("🛑 Stopping hive...")
        .emit();
    
    // Call existing stop logic
    route_operation(Operation::HiveStop { hive_id: hive_id.clone() }, state.clone()).await?;
    
    // Then start
    Narration::new(ACTOR_QUEEN_ROUTER, "hive_restart_start", &hive_id)
        .human("🚀 Starting hive...")
        .emit();
    
    // Call existing start logic
    route_operation(Operation::HiveStart { hive_id: hive_id.clone() }, state).await?;
    
    Narration::new(ACTOR_QUEEN_ROUTER, "hive_restart_success", &hive_id)
        .human(format!("✅ Hive '{}' restarted successfully", hive_id))
        .emit();
}
```

Done! 🎉
