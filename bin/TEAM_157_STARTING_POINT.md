# TEAM-157 STARTING POINT

**Previous Team:** TEAM-156 (Hive Catalog & "No Hives Found")  
**Your Mission:** Add Local PC to Hive Catalog (Happy Flow Lines 29-48)

---

## 🎯 What TEAM-156 Built for You

### 1. Working Hive Catalog ✅
**Location:** `bin/15_queen_rbee_crates/hive-catalog/`

**Available APIs:**
```rust
// Already implemented - use these!
let catalog = HiveCatalog::new(path).await?;
let hives = catalog.list_hives().await?;
let hive = catalog.get_hive("localhost").await?;

// You'll need these:
catalog.add_hive(hive_record).await?;
catalog.update_hive_status("localhost", HiveStatus::Online).await?;
catalog.update_heartbeat("localhost", timestamp_ms).await?;
```

### 2. "No Hives Found" Detection ✅
**Location:** `bin/10_queen_rbee/src/http/jobs.rs` line 87-104

**Current Flow:**
```rust
if hives.is_empty() {
    // Stream "No hives found."
    // TODO TEAM-157: Add local PC to hive catalog (lines 29-48)
    // ← START HERE
}
```

---

## 📋 Your TODO List (Lines 29-48)

From `a_human_wrote_this.md`:

1. **Add local PC to hive catalog**
   - Create HiveRecord for localhost:8600
   - Call `catalog.add_hive(record).await?`
   - Narration: "Adding local pc to hive catalog."

2. **Start rbee-hive locally**
   - Build rbee-hive binary
   - Spawn subprocess on port 8600
   - Narration: "Waking up the bee hive at localhost"

3. **Wait for heartbeat**
   - Don't poll - wait for HTTP POST from hive
   - Hive will send heartbeat to queen
   - Narration: "First heartbeat from a bee hive is received from localhost"

4. **Check hive capabilities**
   - When heartbeat received, check catalog for devices
   - Will be undefined (first time)
   - Narration: "Unknown capabilities of beehive localhost. Asking the beehive to detect devices"

5. **Request device detection**
   - Call rbee-hive API to detect devices
   - Hive responds with CPU, GPUs, model catalog, worker catalog
   - Update catalog with devices
   - Narration: "The beehive localhost has cpu, gpu0, gpu1, ..."

---

## 🔧 Implementation Hints

### Where to Add Code

**File:** `bin/10_queen_rbee/src/http/jobs.rs`  
**Function:** `handle_create_job()`  
**Line:** 101 (after "No hives found" check)

### Suggested Structure

```rust
if hives.is_empty() {
    // TEAM-156: Stream "No hives found."
    tx.send("No hives found.".to_string())?;
    
    // TEAM-157: Add local PC to hive catalog
    let localhost_record = HiveRecord {
        id: "localhost".to_string(),
        host: "127.0.0.1".to_string(),
        port: 8600,
        ssh_host: None,
        ssh_port: None,
        ssh_user: None,
        status: HiveStatus::Unknown,
        last_heartbeat_ms: None,
        created_at_ms: chrono::Utc::now().timestamp_millis(),
        updated_at_ms: chrono::Utc::now().timestamp_millis(),
    };
    
    hive_catalog.add_hive(localhost_record).await?;
    tx.send("Adding local pc to hive catalog.".to_string())?;
    
    // TEAM-157: Start rbee-hive subprocess
    // TODO: Implement hive spawning
    
    // TEAM-157: Wait for heartbeat
    // TODO: Implement heartbeat listener
    
    // TEAM-157: Device detection
    // TODO: Implement device detection flow
}
```

---

## 🚀 Next Steps

### Step 1: Add Hive to Catalog
- Create HiveRecord for localhost
- Call `add_hive()`
- Stream narration

### Step 2: Start Rbee-Hive
- Find rbee-hive binary in target/
- Use `tokio::process::Command`
- Pass port 8600

### Step 3: Heartbeat Endpoint
- Add POST /heartbeat endpoint to queen
- Hive will call this automatically
- Update catalog when received

### Step 4: Device Detection
- Call rbee-hive GET /devices
- Parse response (CPU, GPUs)
- Update catalog with capabilities

### Step 5: Update Registry
- Store devices in RAM registry
- Update hive status to Online
- Stream final narration

---

## 📚 Reference Files

**Read these first:**
- `bin/a_human_wrote_this.md` (lines 29-48) - Your mission
- `bin/TEAM_156_SUMMARY.md` - What we built
- `bin/15_queen_rbee_crates/hive-catalog/src/lib.rs` - Catalog API
- `bin/10_queen_rbee/src/http/jobs.rs` - Where to add code

**Hive binary location:**
- `target/debug/rbee-hive` (after building)

---

## ⚠️ Important Notes

### Don't Implement These Yet
- Model provisioning (later)
- Worker provisioning (later)
- Actual inference forwarding (later)

### Your Job
- Just add localhost to catalog
- Start hive
- Wait for heartbeat
- Detect devices
- Update catalog

### After Your Work
The flow should be:
```
keeper → queen → check catalog → empty → add localhost → start hive → 
wait for heartbeat → detect devices → update catalog → ready!
```

---

## 🎓 Tips

1. **Use existing patterns** - Follow TEAM-156's style
2. **Narration everywhere** - Every step gets narration
3. **Test incrementally** - Build → test → integrate → test
4. **Keep it simple** - Hardcode localhost for now
5. **Stream via SSE** - Keeper needs to see all messages

---

## 📞 Questions?

Check these files:
- `bin/TEAM_156_SUMMARY.md` - Full implementation details
- `bin/15_queen_rbee_crates/hive-catalog/src/lib.rs` - API reference
- `bin/a_human_wrote_this.md` - Original requirements

---

**Good luck, TEAM-157! 🚀**

**You got this! 💪**
