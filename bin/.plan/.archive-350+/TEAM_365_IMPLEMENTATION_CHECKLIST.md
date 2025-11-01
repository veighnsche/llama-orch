# TEAM-365: Bidirectional Handshake Implementation Checklist

**Date:** Oct 30, 2025  
**Status:** 🚀 READY TO IMPLEMENT  
**Canonical Spec:** `bin/.specs/HEARTBEAT_ARCHITECTURE.md`  
**Previous Analysis:** `bin/.plan/TEAM_364_MISSING_HANDSHAKE_LOGIC.md`

---

## 🎯 MISSION

Implement the **bidirectional discovery handshake** protocol specified in HEARTBEAT_ARCHITECTURE.md to enable Queen and Hive to discover each other regardless of startup order.

---

## 📋 IMPLEMENTATION PHASES

### **Phase 1: Create Shared SSH Config Parser Crate** (1 hour)

**Goal:** Extract SSH config parsing logic into reusable shared crate

**Location:** `bin/99_shared_crates/ssh-config-parser/`

**Tasks:**
- [ ] Create crate structure (Cargo.toml, src/lib.rs, README.md)
- [ ] Copy `parse_ssh_config()` from `bin/00_rbee_keeper/src/ssh_resolver.rs:93-170`
- [ ] Add `SshTarget` struct with fields: `host`, `hostname`, `user`, `port`
- [ ] Add `get_default_ssh_config_path()` helper (returns `~/.ssh/config`)
- [ ] Add `parse_ssh_config(path: &Path) -> Result<Vec<SshTarget>>`
- [ ] Copy unit tests from ssh_resolver.rs
- [ ] Add TEAM-365 signatures to all code
- [ ] Update `bin/00_rbee_keeper/Cargo.toml` to depend on new crate
- [ ] Update `bin/00_rbee_keeper/src/tauri_commands.rs:431` to use new crate
- [ ] Verify compilation: `cargo check -p rbee-keeper`

**API Design:**
```rust
// TEAM-365: Created by TEAM-365
pub struct SshTarget {
    pub host: String,      // Alias from SSH config
    pub hostname: String,  // Actual hostname/IP
    pub user: String,
    pub port: u16,
}

pub fn parse_ssh_config(path: &Path) -> Result<Vec<SshTarget>>;
pub fn get_default_ssh_config_path() -> PathBuf;  // ~/.ssh/config
```

**Files to Create:**
- `bin/99_shared_crates/ssh-config-parser/Cargo.toml`
- `bin/99_shared_crates/ssh-config-parser/src/lib.rs`
- `bin/99_shared_crates/ssh-config-parser/README.md`

**Files to Modify:**
- `bin/00_rbee_keeper/Cargo.toml` (add dependency)
- `bin/00_rbee_keeper/src/tauri_commands.rs` (use new crate)

**Verification:**
```bash
cargo check -p ssh-config-parser
cargo test -p ssh-config-parser
cargo check -p rbee-keeper
```

---

### **Phase 2: Enhance Hive Capabilities Endpoint** (1 hour)

**Goal:** Add `queen_url` query parameter handling to `/capabilities` endpoint

**Location:** `bin/20_rbee_hive/src/main.rs:184-242`

**Tasks:**
- [ ] Add `CapabilitiesQuery` struct with `queen_url: Option<String>` field
- [ ] Change `get_capabilities()` signature to accept `Query<CapabilitiesQuery>` and `State<Arc<HiveState>>`
- [ ] Add logic to extract `queen_url` from query parameter
- [ ] Store `queen_url` in HiveState (Phase 3 dependency)
- [ ] Trigger heartbeat task if `queen_url` is provided (Phase 3 dependency)
- [ ] Add narration for queen_url reception
- [ ] Add TEAM-365 signatures

**Code Pattern:**
```rust
// TEAM-365: Query parameter for Queen discovery
#[derive(Debug, Deserialize)]
struct CapabilitiesQuery {
    queen_url: Option<String>,
}

// TEAM-365: Enhanced capabilities endpoint with Queen discovery
async fn get_capabilities(
    Query(params): Query<CapabilitiesQuery>,
    State(state): State<Arc<HiveState>>,
) -> Json<CapabilitiesResponse> {
    n!("caps_request", "📡 Received capabilities request from queen");
    
    // TEAM-365: Handle queen_url parameter for discovery
    if let Some(queen_url) = params.queen_url {
        n!("caps_queen_url", "🔗 Queen URL received: {}", queen_url);
        state.set_queen_url(queen_url.clone()).await;
        state.start_heartbeat_task(queen_url).await;
    }
    
    // ... existing device detection code ...
}
```

**Files to Modify:**
- `bin/20_rbee_hive/src/main.rs` (lines 184-242)

**Verification:**
```bash
# Test without queen_url (existing behavior)
curl http://localhost:7835/capabilities

# Test with queen_url (new behavior)
curl "http://localhost:7835/capabilities?queen_url=http://localhost:7833"
```

---

### **Phase 3: Add HiveState with Dynamic Queen URL** (1 hour)

**Goal:** Create shared state to store queen_url dynamically and control heartbeat task

**Location:** `bin/20_rbee_hive/src/main.rs`

**Tasks:**
- [ ] Define `HiveState` struct with fields:
  - `registry: Arc<JobRegistry<String>>`
  - `model_catalog: Arc<ModelCatalog>`
  - `worker_catalog: Arc<WorkerCatalog>`
  - `queen_url: Arc<RwLock<Option<String>>>` (TEAM-365: Dynamic queen URL)
  - `heartbeat_running: Arc<AtomicBool>` (TEAM-365: Prevent duplicate tasks)
  - `hive_info: HiveInfo` (TEAM-365: For heartbeat)
- [ ] Implement `set_queen_url(&self, url: String)` method
- [ ] Implement `start_heartbeat_task(&self, queen_url: String)` method
- [ ] Initialize HiveState in main()
- [ ] Pass HiveState to router via `.with_state(Arc::new(state))`
- [ ] Update capabilities endpoint to use State extractor
- [ ] Add TEAM-365 signatures

**Code Pattern:**
```rust
// TEAM-365: Shared state for dynamic Queen URL and heartbeat control
#[derive(Clone)]
pub struct HiveState {
    pub registry: Arc<JobRegistry<String>>,
    pub model_catalog: Arc<ModelCatalog>,
    pub worker_catalog: Arc<WorkerCatalog>,
    pub queen_url: Arc<RwLock<Option<String>>>,  // TEAM-365: Dynamic queen URL
    pub heartbeat_running: Arc<AtomicBool>,      // TEAM-365: Prevent duplicate tasks
    pub hive_info: HiveInfo,                     // TEAM-365: For heartbeat
}

impl HiveState {
    // TEAM-365: Store queen URL dynamically
    pub async fn set_queen_url(&self, url: String) {
        *self.queen_url.write().await = Some(url);
    }
    
    // TEAM-365: Start heartbeat task (idempotent - only starts once)
    pub async fn start_heartbeat_task(&self, queen_url: String) {
        // Only start if not already running
        if self.heartbeat_running.swap(true, Ordering::SeqCst) {
            n!("heartbeat_skip", "💓 Heartbeat already running, skipping");
            return;
        }
        
        n!("heartbeat_start", "💓 Starting heartbeat task to {}", queen_url);
        let hive_info = self.hive_info.clone();
        heartbeat::start_heartbeat_task(hive_info, queen_url);
    }
}
```

**Files to Modify:**
- `bin/20_rbee_hive/src/main.rs` (add HiveState struct and impl)

**Verification:**
```bash
cargo check -p rbee-hive
```

---

### **Phase 4: Implement Exponential Backoff Discovery** (2 hours)

**Goal:** Add discovery phase with exponential backoff to Hive heartbeat

**Location:** `bin/20_rbee_hive/src/heartbeat.rs`

**Tasks:**
- [ ] Rename `start_heartbeat_task()` to `start_normal_telemetry_task()`
- [ ] Create new `start_heartbeat_task()` that calls `start_discovery_with_backoff()`
- [ ] Implement `start_discovery_with_backoff()` function:
  - 5 attempts with delays: [0s, 2s, 4s, 8s, 16s]
  - Send full heartbeat on each attempt
  - On first 200 OK: transition to normal telemetry
  - After 5 failures: stop and wait for Queen discovery
- [ ] Add narration for each discovery attempt
- [ ] Add narration for success/failure outcomes
- [ ] Add TEAM-365 signatures

**Code Pattern:**
```rust
// TEAM-365: Start heartbeat with discovery phase
pub fn start_heartbeat_task(hive_info: HiveInfo, queen_url: String) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        start_discovery_with_backoff(hive_info, queen_url).await;
    })
}

// TEAM-365: Discovery phase with exponential backoff
async fn start_discovery_with_backoff(hive_info: HiveInfo, queen_url: String) {
    let delays = [0, 2, 4, 8, 16];  // Exponential backoff in seconds
    
    for (attempt, delay) in delays.iter().enumerate() {
        if *delay > 0 {
            tokio::time::sleep(tokio::time::Duration::from_secs(*delay)).await;
        }
        
        tracing::info!("🔍 Discovery attempt {} (delay: {}s)", attempt + 1, delay);
        
        // Send discovery heartbeat (same format as normal)
        match send_heartbeat_to_queen(&hive_info, &queen_url).await {
            Ok(_) => {
                tracing::info!("✅ Discovery successful! Starting normal telemetry");
                // Start normal telemetry task
                start_normal_telemetry_task(hive_info, queen_url).await;
                return;
            }
            Err(e) => {
                tracing::warn!("❌ Discovery attempt {} failed: {}", attempt + 1, e);
            }
        }
    }
    
    // All 5 attempts failed
    tracing::warn!("⏸️  All discovery attempts failed. Waiting for Queen to discover us via /capabilities");
}

// TEAM-365: Normal telemetry task (runs after discovery)
async fn start_normal_telemetry_task(hive_info: HiveInfo, queen_url: String) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
        
        loop {
            interval.tick().await;
            
            if let Err(e) = send_heartbeat_to_queen(&hive_info, &queen_url).await {
                tracing::warn!("Failed to send hive telemetry: {}", e);
            }
        }
    });
}
```

**Files to Modify:**
- `bin/20_rbee_hive/src/heartbeat.rs`

**Verification:**
```bash
# Test: Start Hive before Queen (should see 5 discovery attempts)
cargo run --bin rbee-hive

# Test: Start Queen after 3rd attempt (should see success on 4th attempt)
cargo run --bin queen-rbee
```

---

### **Phase 5: Implement Queen Discovery Module** (2 hours)

**Goal:** Create discovery module for Queen to discover hives via SSH config

**Location:** `bin/10_queen_rbee/src/discovery.rs` (NEW FILE)

**Tasks:**
- [ ] Create `bin/10_queen_rbee/src/discovery.rs`
- [ ] Add `mod discovery;` to `bin/10_queen_rbee/src/main.rs`
- [ ] Implement `discover_hives_on_startup()` function:
  - Wait 5 seconds for services to stabilize
  - Read SSH config using shared crate
  - Send parallel `GET /capabilities?queen_url=X` to all hives
  - Store capabilities in HiveRegistry (TODO: needs HiveRegistry implementation)
- [ ] Implement `discover_single_hive()` helper function
- [ ] Add comprehensive narration
- [ ] Add TEAM-365 signatures

**Code Pattern:**
```rust
// TEAM-365: Created by TEAM-365
//! Queen hive discovery module
//!
//! Implements pull-based discovery: Queen reads SSH config and sends
//! GET /capabilities?queen_url=X to all configured hives.

use anyhow::Result;
use observability_narration_core::n;
use ssh_config_parser::{parse_ssh_config, get_default_ssh_config_path, SshTarget};
use std::time::Duration;

/// Discover all hives on Queen startup
///
/// TEAM-365: Pull-based discovery (Scenario 1 from HEARTBEAT_ARCHITECTURE.md)
///
/// # Flow
/// 1. Wait 5 seconds for services to stabilize
/// 2. Read SSH config
/// 3. Send parallel GET /capabilities?queen_url=X to all hives
/// 4. Store capabilities in HiveRegistry
pub async fn discover_hives_on_startup(queen_url: &str) -> Result<()> {
    n!("discovery_start", "🔍 Starting hive discovery (waiting 5s for services to stabilize)");
    
    // Wait for services to stabilize
    tokio::time::sleep(Duration::from_secs(5)).await;
    
    // Read SSH config
    let ssh_config_path = get_default_ssh_config_path();
    let targets = match parse_ssh_config(&ssh_config_path) {
        Ok(targets) => targets,
        Err(e) => {
            n!("discovery_no_config", "⚠️  No SSH config found: {}. Only localhost will be discovered.", e);
            vec![]
        }
    };
    
    n!("discovery_targets", "📋 Found {} SSH targets to discover", targets.len());
    
    // Discover all hives in parallel
    let mut tasks = vec![];
    for target in targets {
        let queen_url = queen_url.to_string();
        
        tasks.push(tokio::spawn(async move {
            discover_single_hive(&target, &queen_url).await
        }));
    }
    
    // Wait for all discoveries
    let mut success_count = 0;
    let mut failure_count = 0;
    
    for task in tasks {
        match task.await {
            Ok(Ok(_)) => success_count += 1,
            Ok(Err(_)) => failure_count += 1,
            Err(_) => failure_count += 1,
        }
    }
    
    n!("discovery_complete", "✅ Discovery complete: {} successful, {} failed", success_count, failure_count);
    
    Ok(())
}

/// Discover a single hive
///
/// TEAM-365: Send GET /capabilities?queen_url=X to hive
async fn discover_single_hive(target: &SshTarget, queen_url: &str) -> Result<()> {
    let url = format!(
        "http://{}:7835/capabilities?queen_url={}",
        target.hostname,
        urlencoding::encode(queen_url)
    );
    
    n!("discovery_hive", "🔍 Discovering hive: {} ({})", target.host, target.hostname);
    
    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .timeout(Duration::from_secs(10))
        .send()
        .await?;
    
    if response.status().is_success() {
        n!("discovery_success", "✅ Discovered hive: {}", target.host);
        
        // TODO: Store capabilities in HiveRegistry when implemented
        // let capabilities: CapabilitiesResponse = response.json().await?;
        // hive_registry.register_hive(target.host.clone(), capabilities);
    } else {
        n!("discovery_failed", "❌ Failed to discover hive {}: {}", target.host, response.status());
    }
    
    Ok(())
}
```

**Files to Create:**
- `bin/10_queen_rbee/src/discovery.rs`

**Files to Modify:**
- `bin/10_queen_rbee/src/main.rs` (add `mod discovery;`)
- `bin/10_queen_rbee/Cargo.toml` (add dependencies: `ssh-config-parser`, `urlencoding`)

**Verification:**
```bash
cargo check -p queen-rbee
```

---

### **Phase 6: Wire Up Queen Discovery in Startup** (30 minutes)

**Goal:** Call discovery function from Queen main.rs after server starts

**Location:** `bin/10_queen_rbee/src/main.rs`

**Tasks:**
- [ ] Import discovery module: `use crate::discovery;`
- [ ] Add queen_url configuration (default: `http://localhost:7833`)
- [ ] Spawn discovery task after server starts
- [ ] Add narration for discovery startup
- [ ] Add TEAM-365 signatures

**Code Pattern:**
```rust
// In main() function, after server starts:

// TEAM-365: Start hive discovery task
let queen_url = format!("http://127.0.0.1:{}", args.port);
tokio::spawn(async move {
    if let Err(e) = discovery::discover_hives_on_startup(&queen_url).await {
        n!("discovery_error", "❌ Hive discovery failed: {}", e);
    }
});

n!("ready", "Ready to accept connections");
```

**Files to Modify:**
- `bin/10_queen_rbee/src/main.rs`

**Verification:**
```bash
# Start Queen, should see discovery logs after 5s
cargo run --bin queen-rbee
```

---

### **Phase 7: Integration Testing** (1 hour)

**Goal:** Test all 3 startup scenarios from HEARTBEAT_ARCHITECTURE.md

**Tasks:**
- [ ] **Scenario 1: Queen starts first**
  - Start Queen
  - Wait 10 seconds (discovery should complete)
  - Start Hive
  - Verify: Hive sends heartbeats immediately (no exponential backoff)
  - Verify: Queen logs show hive discovered
- [ ] **Scenario 2: Hive starts first**
  - Start Hive
  - Verify: 5 discovery attempts with exponential backoff
  - Verify: Hive stops after 5 failures
  - Start Queen
  - Verify: Queen discovers hive via /capabilities
  - Verify: Hive starts normal heartbeats
- [ ] **Scenario 3: Both start simultaneously**
  - Start both at same time
  - Verify: Both mechanisms work in parallel
  - Verify: First success wins (no duplicate heartbeats)
  - Verify: Normal operation begins smoothly
- [ ] Check logs for clean output (no error spam)
- [ ] Verify no duplicate heartbeat tasks

**Test Commands:**
```bash
# Scenario 1: Queen first
cargo run --bin queen-rbee &
sleep 10
cargo run --bin rbee-hive

# Scenario 2: Hive first
cargo run --bin rbee-hive &
sleep 20
cargo run --bin queen-rbee

# Scenario 3: Both simultaneously
cargo run --bin queen-rbee & cargo run --bin rbee-hive
```

**Acceptance Criteria:**
- [ ] No error spam in logs
- [ ] Discovery completes successfully in all scenarios
- [ ] Heartbeats flow normally after discovery
- [ ] No duplicate heartbeat tasks
- [ ] Clean shutdown

---

### **Phase 8: Create Handoff Document** (30 minutes)

**Goal:** Document implementation for next team (max 2 pages)

**Location:** `bin/.plan/TEAM_365_HANDOFF.md`

**Required Content:**
- [ ] Status: ✅ COMPLETE
- [ ] Mission summary
- [ ] Code examples of what was implemented
- [ ] Actual progress (LOC added, files created)
- [ ] Verification checklist (all boxes checked)
- [ ] NO TODO lists for next team
- [ ] NO "next team should implement X"

**Template:**
```markdown
# TEAM-365: Bidirectional Handshake Implementation

**Status:** ✅ COMPLETE  
**Date:** Oct 30, 2025

## Mission

Implemented bidirectional discovery handshake per HEARTBEAT_ARCHITECTURE.md spec.

## Deliverables

1. **ssh-config-parser crate** (XXX LOC)
2. **Enhanced capabilities endpoint** (XX LOC)
3. **HiveState with dynamic queen_url** (XX LOC)
4. **Exponential backoff discovery** (XX LOC)
5. **Queen discovery module** (XXX LOC)

## Code Examples

[Show actual code snippets]

## Verification

- [x] All 3 scenarios tested
- [x] No error spam
- [x] Clean logs
- [x] Compilation passes
- [x] Integration tests pass

## Files Created

- bin/99_shared_crates/ssh-config-parser/
- bin/10_queen_rbee/src/discovery.rs

## Files Modified

- bin/20_rbee_hive/src/main.rs
- bin/20_rbee_hive/src/heartbeat.rs
- bin/10_queen_rbee/src/main.rs
- bin/00_rbee_keeper/src/tauri_commands.rs
```

---

## 🎯 ACCEPTANCE CRITERIA

### **Scenario 1: Hive Starts First**
- [ ] Hive sends 5 discovery heartbeats with exponential backoff (0s, 2s, 4s, 8s, 16s)
- [ ] Hive stops after 5 failures
- [ ] Hive waits silently for Queen discovery
- [ ] When Queen starts and sends `/capabilities?queen_url=X`, hive starts normal heartbeats
- [ ] No error spam in logs

### **Scenario 2: Queen Starts First**
- [ ] Queen waits 5 seconds
- [ ] Queen reads SSH config
- [ ] Queen sends `GET /capabilities?queen_url=X` to all hives
- [ ] Hives respond and start heartbeats immediately
- [ ] Queen knows about all hives immediately

### **Scenario 3: Both Start Simultaneously**
- [ ] Both mechanisms work in parallel
- [ ] First success wins (no duplicate heartbeats)
- [ ] Normal operation begins smoothly

---

## 📊 ESTIMATED EFFORT

| Phase | Estimated Time | Actual Time |
|-------|---------------|-------------|
| Phase 1: SSH config parser | 1 hour | ___ |
| Phase 2: Capabilities endpoint | 1 hour | ___ |
| Phase 3: HiveState | 1 hour | ___ |
| Phase 4: Exponential backoff | 2 hours | ___ |
| Phase 5: Queen discovery | 2 hours | ___ |
| Phase 6: Wire up startup | 30 minutes | ___ |
| Phase 7: Integration testing | 1 hour | ___ |
| Phase 8: Handoff document | 30 minutes | ___ |
| **TOTAL** | **9 hours** | ___ |

---

## 🚨 CRITICAL REMINDERS

1. **RULE ZERO:** Update existing functions, don't create new ones
2. **Add TEAM-365 signatures** to ALL code
3. **No TODO markers** - implement or delete
4. **Handoff max 2 pages** - code examples, not analysis
5. **Complete previous team's TODO** - TEAM-364 identified the gap, TEAM-365 implements it
6. **No background testing** - foreground only, see full output
7. **Consult existing docs** - HEARTBEAT_ARCHITECTURE.md is canonical

---

## 📚 REFERENCE DOCUMENTS

- **Canonical Spec:** `bin/.specs/HEARTBEAT_ARCHITECTURE.md`
- **Gap Analysis:** `bin/.plan/TEAM_364_MISSING_HANDSHAKE_LOGIC.md`
- **Engineering Rules:** `.windsurf/rules/engineering-rules.md`
- **SSH Config Parser:** `bin/00_rbee_keeper/src/ssh_resolver.rs` (source to extract)
- **Current Hive Heartbeat:** `bin/20_rbee_hive/src/heartbeat.rs`
- **Current Capabilities:** `bin/20_rbee_hive/src/main.rs:184-242`

---

**TEAM-365: Let's implement this! 🚀**
