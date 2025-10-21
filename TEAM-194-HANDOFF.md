# TEAM-194 HANDOFF

**Mission:** Replace SQLite-based hive catalog with file-based config  
**Status:** ✅ 60% Complete - Infrastructure done, handlers need refactoring  
**Time Invested:** 4 hours  
**Remaining:** 2-3 hours

---

## ✅ WHAT'S DONE

### Infrastructure (100% Complete)
1. **Dependencies:** `queen-rbee-hive-catalog` → `rbee-config` ✅
2. **AppState:** Uses `RbeeConfig` instead of `HiveCatalog` ✅
3. **Operation Enum:** Simplified to alias-based (removed 50+ LOC) ✅
4. **CLI:** All commands use `-h <alias>` instead of detailed args ✅

### Verification
```bash
cargo check --package rbee-operations  # ✅ PASSES
cargo check --bin rbee-keeper          # ✅ PASSES
cargo check --bin queen-rbee           # ❌ 26 errors (job_router.rs only)
```

---

## 🚧 WHAT REMAINS

### 7 Handlers in `job_router.rs` Need Updates

**All errors are in:** `bin/10_queen_rbee/src/job_router.rs`

**Pattern:** Replace `state.hive_catalog.*` with `state.config.hives.*`

---

## 📝 CODE EXAMPLES

### Example 1: SshTest Handler (EASIEST - Start Here)

**Current (Line 181):**
```rust
Operation::SshTest { ssh_host, ssh_port, ssh_user } => {
    let request = SshTestRequest { ssh_host, ssh_port, ssh_user };
    let response = execute_ssh_test(request).await?;
    // ...
}
```

**Target:**
```rust
Operation::SshTest { alias } => {
    // Get SSH details from config
    let hive = state.config.hives.get(&alias)
        .ok_or_else(|| anyhow::anyhow!(
            "Hive '{}' not found in hives.conf", alias
        ))?;
    
    let request = SshTestRequest {
        ssh_host: hive.hostname.clone(),
        ssh_port: hive.ssh_port,
        ssh_user: hive.ssh_user.clone(),
    };
    
    let response = execute_ssh_test(request).await?;
    // ... rest unchanged
}
```

### Example 2: HiveList Handler

**Current (Line 697):**
```rust
Operation::HiveList => {
    let hives = state.hive_catalog.list_hives().await?;
    
    if hives.is_empty() {
        // ... empty message
        return Ok(());
    }
    
    let hives_json: Vec<serde_json::Value> = hives.iter()
        .map(|h| serde_json::json!({
            "id": h.id,
            "host": h.host,
            "port": h.port,
            "binary_path": h.binary_path.as_ref().unwrap_or(&"-".to_string()),
        }))
        .collect();
}
```

**Target:**
```rust
Operation::HiveList => {
    let hives = state.config.hives.all();
    
    if hives.is_empty() {
        NARRATE_ROUTER.action("hive_list_empty")
            .human("No hives configured.\n\nAdd hives to ~/.config/rbee/hives.conf")
            .emit();
        return Ok(());
    }
    
    let hives_json: Vec<serde_json::Value> = hives.iter()
        .map(|h| {
            let status = if state.config.capabilities.get(&h.alias).is_some() {
                "🟢 RUNNING"
            } else {
                "⚫ STOPPED"
            };
            
            serde_json::json!({
                "alias": h.alias,
                "status": status,
                "host": h.hostname,
                "port": h.hive_port,
                "binary_path": h.binary_path.as_ref().unwrap_or(&"-".to_string()),
            })
        })
        .collect();
}
```

### Example 3: HiveStart Handler

**Current (Line 514):**
```rust
Operation::HiveStart { hive_id } => {
    let hive = state.hive_catalog.get_hive(&hive_id).await?
        .ok_or_else(|| anyhow::anyhow!("Hive '{}' not found", hive_id))?;
    
    // Check if already running
    let health_url = format!("http://{}:{}/health", hive.host, hive.port);
    // ...
}
```

**Target:**
```rust
Operation::HiveStart { alias } => {
    let hive = state.config.hives.get(&alias)
        .ok_or_else(|| anyhow::anyhow!(
            "Hive '{}' not found in hives.conf", alias
        ))?;
    
    // Check if already running (via capabilities)
    if let Some(caps) = state.config.capabilities.get(&alias) {
        let age_ms = chrono::Utc::now().timestamp_millis() - caps.last_updated_ms;
        if age_ms < 30_000 {  // 30 seconds
            Narration::new(ACTOR_QUEEN_ROUTER, "hive_start_already_running", &alias)
                .human(format!("✅ Hive '{}' is already running", alias))
                .emit();
            return Ok(());
        }
    }
    
    let health_url = format!("http://{}:{}/health", hive.hostname, hive.hive_port);
    // ... rest of logic unchanged
}
```

---

## 🎯 EXECUTION CHECKLIST

### Quick Fixes (10 minutes)
- [ ] Line 207: Delete `use queen_rbee_hive_catalog::HiveRecord;`
- [ ] Lines 483-512: Delete entire `HiveUpdate` handler
- [ ] Search/replace pattern matches: `hive_id` → `alias` (7 occurrences)
- [ ] Line 372: Remove `catalog_only: _` from HiveUninstall pattern

### Handler Updates (2-3 hours)
- [ ] SshTest (181-204) - 10 min
- [ ] HiveInstall (205-371) - 45 min  
- [ ] HiveUninstall (372-478) - 30 min
- [ ] HiveStart (514-593) - 20 min
- [ ] HiveStop (598-696) - 15 min
- [ ] HiveList (697-737) - 20 min
- [ ] HiveStatus (751-840) - 15 min

### Final Verification
```bash
cargo check --bin queen-rbee
cargo clippy --bin queen-rbee
```

---

## 📚 API REFERENCE

```rust
// Get hive config
let hive = state.config.hives.get(&alias)?;
// Fields: alias, hostname, ssh_port, ssh_user, hive_port, binary_path

// Check if running
if let Some(caps) = state.config.capabilities.get(&alias) { ... }

// List all hives
let hives = state.config.hives.all();  // Vec<&HiveEntry>
```

---

**Created by:** TEAM-194  
**For:** Next team (continue as TEAM-194 or TEAM-195)  
**Files:** See `TEAM-194-FINAL-SUMMARY.md` for complete details
