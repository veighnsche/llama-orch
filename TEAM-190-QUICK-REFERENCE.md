# TEAM-190 Quick Reference Card

## The 4 Files You'll Always Edit

```
1. bin/00_rbee_keeper/src/main.rs        ← Add CLI command
2. bin/99_shared_crates/rbee-operations/src/lib.rs  ← Add Operation variant
3. bin/00_rbee_keeper/src/main.rs        ← Add routing (again)
4. bin/10_queen_rbee/src/job_router.rs   ← Add implementation
```

---

## Copy-Paste Templates

### Template 1: CLI Command
```rust
/// Brief description
/// TEAM-XXX: What this does
CommandName {
    /// Parameter description (defaults to ...)
    #[arg(default_value = "...")]
    param: String,
},
```

### Template 2: Operation Variant
```rust
/// TEAM-XXX: Brief description
OperationName {
    #[serde(default = "default_hive_id")]  // or other default
    param: String,
},
```

### Template 3: Add to name()
```rust
Operation::OperationName { .. } => constants::OP_OPERATION_NAME,
```

### Template 4: Add to hive_id() (if hive operation)
```rust
Operation::OperationName { hive_id } => Some(hive_id),
```

### Template 5: Add constant
```rust
pub const OP_OPERATION_NAME: &str = "operation_name"; // TEAM-XXX
```

### Template 6: Routing
```rust
Action::CommandName { param } => Operation::OperationName { param },
```

### Template 7: Implementation
```rust
Operation::OperationName { param } => {
    // TEAM-XXX: Brief description
    
    Narration::new(ACTOR_QUEEN_ROUTER, "operation_name", &param)
        .human(format!("🔧 Starting operation"))
        .emit();
    
    // Do work here
    
    Narration::new(ACTOR_QUEEN_ROUTER, "operation_name_success", &param)
        .human("✅ Success")
        .emit();
}
```

---

## Error Message Template

```rust
Narration::new(ACTOR_QUEEN_ROUTER, "operation_error", &param)
    .human(format!(
        "❌ [Problem statement]\n\
         \n\
         [Explanation if needed]\n\
         \n\
           [Exact command to fix]",
        param
    ))
    .emit();
return Err(anyhow::anyhow!("Short error message for logs"));
```

---

## Build & Test Commands

```bash
# Build both binaries
cargo build --bin rbee-keeper --bin queen-rbee --quiet

# Kill and restart queen
pkill -9 queen-rbee; pkill -9 rbee-keeper; sleep 1

# Test your command
./rbee your command

# Test with verbose output
./rbee your command 2>&1 | grep "queen-router"

# Test error case
./rbee your command --id nonexistent
```

---

## Common Operations Reference

### Check if resource exists
```rust
let resource = state.catalog.get(&id).await?;
if resource.is_none() {
    // Show error, return
}
let resource = resource.unwrap();
```

### Health check
```rust
let url = format!("http://{}:{}/health", host, port);
let client = reqwest::Client::builder()
    .timeout(tokio::time::Duration::from_secs(2))
    .build()?;
    
if let Ok(resp) = client.get(&url).send().await {
    if resp.status().is_success() {
        // Running
    }
}
```

### Spawn process
```rust
use daemon_lifecycle::DaemonManager;
let manager = DaemonManager::new(
    PathBuf::from(binary),
    vec!["--port".to_string(), port.to_string()],
);
let _child = manager.spawn().await?;
```

### Stop process (graceful)
```rust
// SIGTERM
tokio::process::Command::new("pkill")
    .args(&["-TERM", "process-name"])
    .output()
    .await?;

// Wait with timeout
for i in 1..=5 {
    tokio::time::sleep(Duration::from_secs(1)).await;
    // Check if stopped
    if stopped { break; }
    if i == 5 {
        // SIGKILL
        tokio::process::Command::new("pkill")
            .args(&["-KILL", "process-name"])
            .output()
            .await?;
    }
}
```

---

## Emoji Guide

| Emoji | Use Case | Example |
|-------|----------|---------|
| 🔧 | Install, Configure | "🔧 Installing hive" |
| 🗑️ | Uninstall, Delete | "🗑️  Uninstalling hive" |
| 🚀 | Start, Launch | "🚀 Starting hive" |
| 🛑 | Stop, Halt | "🛑 Stopping hive" |
| 📋 | Check, Verify | "📋 Checking if hive exists" |
| ✅ | Success, Found | "✅ Hive installed successfully" |
| ❌ | Error, Not Found | "❌ Hive not found" |
| ⚠️  | Warning | "⚠️  Hive already running" |
| 📝 | Write, Save | "📝 Saving to database" |
| 🔍 | Search, Look | "🔍 Looking for binary" |
| ⏳ | Wait | "⏳ Waiting for startup" |
| 📤 | Send | "📤 Sending SIGTERM" |
| 🔄 | Restart, Reload | "🔄 Restarting hive" |
| 📊 | List, Show | "📊 Listing hives" |
| 🏠 | Local/Localhost | "🏠 Localhost installation" |
| 🌐 | Remote/SSH | "🌐 Remote installation" |

---

## Remember

1. ✅ **Always narrate before doing work**
2. ✅ **Always provide actionable error messages**
3. ✅ **Always add TEAM-XXX comments**
4. ✅ **Always test happy path AND error cases**
5. ✅ **Always use default_value for common params**
6. ✅ **Always use consistent parameter names**
7. ✅ **Always emit narration, don't println!()**
8. ✅ **Always return Result, never panic**

---

## Study These Examples

Best implementations to learn from:
- `Operation::HiveInstall` - Pre-flight checks, binary discovery
- `Operation::HiveUninstall` - Pre-flight checks, friendly errors
- `Operation::HiveStart` - Process spawning, health polling
- `Operation::HiveStop` - Graceful shutdown, SIGKILL fallback
- `Operation::HiveStatus` - Health checks, error messages

**Location:** `bin/10_queen_rbee/src/job_router.rs` lines 130-700

---

## Get Help

1. Read TEAM-189-SUMMARY.md for overview
2. Read TEAM-190-GUIDE-PART-1.md for architecture
3. Read TEAM-190-GUIDE-PART-2.md for step-by-step
4. Search existing operations for patterns:
   ```bash
   grep -n "Operation::" bin/10_queen_rbee/src/job_router.rs
   ```
5. Check narration patterns:
   ```bash
   grep "Narration::new" bin/10_queen_rbee/src/job_router.rs | head -20
   ```

Good luck! 🚀
