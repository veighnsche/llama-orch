# TEAM-156 Instructions: Hive Catalog & "No Hives Found" Flow

**Date:** 2025-10-20  
**Previous Team:** TEAM-155 (SSE streaming foundation)  
**Mission:** Implement hive catalog checking and "no hives found" narration

---

## 📋 Your Mission

Implement lines 25-27 of the happy flow from `a_human_wrote_this.md`:

```
The queen bee looks at the hive catalog (missing crate hive catalog is sqlite) for valid hives.
No hives are found in the hive catalog. (because clean install)
narration (queen bee -> sse -> bee keeper -> stdout): No hives found.
```

---

## 🎯 What TEAM-155 Built

**Working foundation:**
- ✅ POST /jobs endpoint - Creates job, returns job_id + sse_url
- ✅ GET /jobs/{job_id}/stream - SSE streaming endpoint
- ✅ rbee-keeper submits jobs and streams results
- ✅ Narration system with emojis and multi-line format

**Current flow:**
```
[🧑‍🌾 rbee-keeper]
  Submitting job to queen

[👑 queen-rbee]
  Job created: job-xxx

[🧑‍🌾 rbee-keeper]
  Connecting to SSE stream

[👑 queen-rbee]
  Job has no token receiver  ← THIS IS WHERE YOU START
```

---

## 📦 What You Need to Build

### 1. Hive Catalog Crate (SQLite)

**Location:** `bin/99_shared_crates/hive-catalog/` NO - bin/15_queen_rbee_crates/hive-catalog - It's not shared.

**Purpose:** Persistent storage for registered hives

**Schema:**
```sql
CREATE TABLE hives (
    id TEXT PRIMARY KEY,           -- e.g., "localhost", "workstation-1"
    host TEXT NOT NULL,            -- e.g., "localhost", "192.168.1.100"
    port INTEGER NOT NULL,         -- e.g., 8600
    ssh_host TEXT,                 -- For remote hives
    ssh_port INTEGER,              -- For remote hives
    ssh_user TEXT,                 -- For remote hives
    status TEXT NOT NULL,          -- "unknown", "online", "offline"
    last_heartbeat_ms INTEGER,     -- Timestamp of last heartbeat
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL
);
```

**API:**
```rust
pub struct HiveCatalog {
    db: rusqlite::Connection,
}

impl HiveCatalog {
    pub fn new(db_path: &Path) -> Result<Self>;
    pub fn list_hives(&self) -> Result<Vec<HiveRecord>>;
    pub fn get_hive(&self, id: &str) -> Result<Option<HiveRecord>>;
    pub fn add_hive(&self, hive: HiveRecord) -> Result<()>;
    pub fn update_hive_status(&self, id: &str, status: HiveStatus) -> Result<()>;
    pub fn update_heartbeat(&self, id: &str, timestamp_ms: u64) -> Result<()>;
}

pub struct HiveRecord {
    pub id: String,
    pub host: String,
    pub port: u16,
    pub status: HiveStatus,
    // ... other fields
}

pub enum HiveStatus {
    Unknown,
    Online,
    Offline,
}
```

**Actor constant:**
```rust
const ACTOR_HIVE_CATALOG: &str = "(crown emoji here) queen bee / ⚙️ hive-catalog";
```

---

### 2. Update Queen-Rbee to Check Catalog

**File:** `bin/10_queen_rbee/src/http/jobs.rs`

**In `handle_create_job()` after creating the job:**

```rust
// TEAM-156: Check hive catalog for available hives
let hive_catalog = state.hive_catalog.clone();
let hives = hive_catalog.list_hives()?;

if hives.is_empty() {
    // TEAM-156: No hives found - send narration via SSE
    let (tx, rx) = tokio::sync::mpsc::channel(100);
    state.registry.set_token_receiver(&job_id, rx);
    
    // Send "no hives found" message
    tx.send("No hives found.".to_string()).await?;
    
    Narration::new(ACTOR_QUEEN_HTTP, ACTION_JOB_CREATE, &job_id)
        .human("No hives found in catalog")
        .emit();
    
    // TODO TEAM-157: Add local PC to hive catalog (lines 29-48)
    // For now, just close the stream
    drop(tx);
}
```

---

### 3. Update Queen-Rbee State

**File:** `bin/10_queen_rbee/src/main.rs`

**Add hive catalog to state:**

```rust
use hive_catalog::HiveCatalog;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    // TEAM-156: Initialize hive catalog
    let catalog_path = args.database
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("queen-hive-catalog.db"));
    
    let hive_catalog = Arc::new(HiveCatalog::new(&catalog_path)?);
    
    Narration::new(ACTOR_QUEEN_RBEE, ACTION_START, &catalog_path.display().to_string())
        .human(format!("Initialized hive catalog at {}", catalog_path.display()))
        .emit();
    
    // ... rest of initialization
}
```

**Update router state:**

```rust
pub struct QueenJobState {
    pub registry: Arc<JobRegistry<String>>,
    pub hive_catalog: Arc<HiveCatalog>,  // TEAM-156: Added
}
```

---

### 4. Update Cargo.toml

**File:** `bin/10_queen_rbee/Cargo.toml`

```toml
[dependencies]
# ... existing deps
hive-catalog = { path = "../99_shared_crates/hive-catalog" }
```

---

## 🧪 Testing

### Manual Test

```bash
# Clean install (no hives in catalog)
rm -f queen-hive-catalog.db

# Run the test
./target/debug/rbee-keeper infer "hello" --model HF:author/minillama
```

**Expected output:**
```
[🧑‍🌾 rbee-keeper]
  Submitting job to queen

[👑 queen-rbee]
  Job created: job-xxx

[🧑‍🌾 rbee-keeper]
  Connecting to SSE stream

[👑 queen-rbee]
  No hives found in catalog

[🧑‍🌾 rbee-keeper]
  Event: No hives found.

[🧑‍🌾 rbee-keeper]
  Event: [DONE]
```

### BDD Test

**File:** `bin/10_queen_rbee/bdd/tests/features/hive_catalog.feature`

```gherkin
Feature: Hive Catalog Management
  As queen-rbee
  I want to check the hive catalog for available hives
  So that I can route jobs appropriately

  Scenario: No hives found on clean install
    Given the hive catalog is empty
    When I submit a job to queen-rbee
    Then the SSE stream should contain "No hives found."
    And the job should complete with [DONE]

  Scenario: Hive catalog is initialized
    Given queen-rbee starts with database path "test-catalog.db"
    Then the hive catalog should be created
    And the hive catalog should be empty
```

---

## 📊 Deliverables

### Must Complete

1. ✅ **hive-catalog crate** - SQLite-based hive registry
2. ✅ **Catalog checking in queen-rbee** - Check for hives before processing job
3. ✅ **"No hives found" narration** - Stream message via SSE to keeper
4. ✅ **BDD tests** - Test empty catalog scenario
5. ✅ **Documentation** - Update handoff with what you built

### Success Criteria

- [ ] `cargo build --bin queen-rbee` compiles
- [ ] `cargo test --bin queen-rbee` passes
- [ ] Manual test shows "No hives found." in SSE stream
- [ ] Narration uses correct format: `[👑 queen-rbee]\n  Message`
- [ ] Database file is created on first run
- [ ] Handoff document ≤ 2 pages with code examples

---

## 🚫 Out of Scope (For TEAM-157)

**DO NOT implement these (lines 29-48):**
- Adding local PC to hive catalog
- Starting rbee-hive locally
- Heartbeat detection
- Device detection
- Worker catalog

**Your job:** Just check the catalog and report "no hives found"

---

## 📚 Reference Files

**Read these first:**
- `bin/a_human_wrote_this.md` (lines 25-27) - Your mission
- `bin/TEAM_155_FINAL_SUMMARY.md` - What TEAM-155 built
- `bin/10_queen_rbee/src/http/jobs.rs` - Where to add catalog check
- `bin/99_shared_crates/job-registry/` - Example of how to structure your crate

**Narration format:**
- Actor: `[👑 queen-rbee]` or `[⚙️ hive-catalog]`
- Multi-line: `[actor]\n  message`
- Use `.emit()` not `narrate!()` macro

---

## 🎓 Tips

1. **Start with the crate** - Build hive-catalog first, test it standalone -- alread exists at the 15_queen_rbee_crates/hive-catalog
2. **Keep it simple** - Just list_hives() for now, don't over-engineer
3. **Use rusqlite** - It's already in the workspace
4. **Follow job-registry pattern** - Similar structure, different domain
5. **Test incrementally** - Build → test → integrate → test
6. **Narration everywhere** - Every significant action gets narration

---

## ⚠️ Common Pitfalls

1. **Don't implement auto-add** - That's TEAM-157's job (lines 29-48)
2. **Don't start rbee-hive** - That's also TEAM-157
3. **Don't add heartbeat logic** - TEAM-157 again
4. **Don't over-complicate the schema** - Keep it simple for now
5. **Don't forget to stream via SSE** - The keeper needs to see the message!

---

## 📝 Handoff Template

```markdown
# TEAM-156 SUMMARY

**Mission:** Hive catalog checking and "no hives found" narration

## Deliverables

1. **hive-catalog crate** - [Lines of code, key functions]
2. **Queen integration** - [What you changed in queen-rbee]
3. **Testing** - [What tests you wrote]

## Code Examples

[Show actual code you wrote, not TODOs]

## Verification

- [ ] Compilation: SUCCESS
- [ ] Tests: X/Y passing
- [ ] Manual test: "No hives found" appears in SSE stream

## Next Steps for TEAM-157

[What they need to do - lines 29-48]
```

---

**Good luck, TEAM-156! 🚀**

**Remember:** 
- Read the engineering rules in `engineering-rules.md`
- No TODO markers - implement everything
- Handoff ≤ 2 pages
- Show actual progress with code examples

**You got this! 💪**
