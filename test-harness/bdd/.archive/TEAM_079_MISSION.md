# TEAM-079 MISSION: BDD Product Integration
# Date: 2025-10-11
# Status: 🚀 READY TO START

## Your Mission

**Transform 84 stub functions into a living, breathing test suite by wiring them to real product code.**

TEAM-078 built the scaffolding. Now you bring it to life. This is where BDD proves its value - when tests drive real implementation and catch real bugs.

## Why This Matters

Right now, we have **15 beautifully organized M1 feature files** with **84 step definitions** that do nothing but log. Your job is to make them **actually test the product**.

**This is not busywork.** Every function you wire up:
- ✅ Validates a real user scenario
- ✅ Catches integration bugs early
- ✅ Documents how the system actually works
- ✅ Enables confident refactoring

## The Challenge

**Minimum requirement:** 10+ functions with real API calls  
**Your goal:** Wire up as many as you can. Every function counts.

### Why Go Beyond Minimum?

1. **Momentum** - The more you wire up now, the less technical debt later
2. **Context** - You have the full picture fresh in your mind
3. **Impact** - A 50% wired test suite is infinitely more valuable than 12% (10/84)
4. **Pride** - Be the team that made BDD actually work

## What TEAM-078 Left You

### ✅ Infrastructure Ready
- 15 M1 feature files (test-001.feature DELETED!)
- 84 step function stubs with clear comments
- World struct with `last_action` tracking
- Compilation green, ready to extend

### 📋 Your Targets (Prioritized)

#### Priority 1: Model Catalog (18 functions) - EASIEST START
**File:** `src/steps/model_catalog.rs`  
**Product code:** Create `bin/rbee-hive/src/model_catalog.rs`

This is your **warm-up**. SQLite queries are straightforward:

```rust
// BEFORE (stub):
#[when(expr = "rbee-hive checks the model catalog")]
pub async fn when_rbee_hive_checks_catalog(world: &mut World) {
    tracing::info!("TEAM-078: Checking model catalog");
    world.last_action = Some("catalog_checked".to_string());
}

// AFTER (wired):
#[when(expr = "rbee-hive checks the model catalog")]
pub async fn when_rbee_hive_checks_catalog(world: &mut World) {
    // TEAM-079: Wire to real SQLite catalog
    use rbee_hive::model_catalog::ModelCatalog;
    
    let catalog = ModelCatalog::new(
        world.model_catalog_path.as_ref().unwrap()
    );
    
    let result = catalog.find_model("tinyllama-q4").await;
    world.last_action = Some(format!("catalog_result_{:?}", result.is_some()));
    
    tracing::info!("TEAM-079: Model catalog query returned: {:?}", result.is_some());
}
```

**Product code to create:**
```rust
// bin/rbee-hive/src/model_catalog.rs
pub struct ModelCatalog {
    db_path: PathBuf,
}

impl ModelCatalog {
    pub fn new(path: &Path) -> Self { /* ... */ }
    
    pub async fn find_model(&self, model_id: &str) -> Option<ModelEntry> {
        // Real SQLite query here
    }
    
    pub async fn insert(&self, entry: ModelEntry) -> Result<()> {
        // Real SQLite insert here
    }
    
    pub async fn query_by_provider(&self, provider: &str) -> Vec<ModelEntry> {
        // Real SQLite query here
    }
}
```

**Why start here:** 
- Simple CRUD operations
- No HTTP complexity
- Builds confidence
- **Target: Wire all 18 functions** (achievable in 2-3 hours)

#### Priority 2: queen-rbee Registry (22 functions) - HIGH IMPACT
**File:** `src/steps/queen_rbee_registry.rs`  
**Product code:** Create `bin/queen-rbee/src/worker_registry.rs`

This is the **heart of the orchestration system**. In-memory HashMap with HTTP API:

```rust
// bin/queen-rbee/src/worker_registry.rs
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[derive(Clone)]
pub struct WorkerRegistry {
    workers: Arc<RwLock<HashMap<String, WorkerEntry>>>,
}

impl WorkerRegistry {
    pub fn new() -> Self {
        Self {
            workers: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub fn register(&self, worker: WorkerEntry) {
        let mut workers = self.workers.write().unwrap();
        workers.insert(worker.id.clone(), worker);
    }
    
    pub fn list(&self) -> Vec<WorkerEntry> {
        let workers = self.workers.read().unwrap();
        workers.values().cloned().collect()
    }
    
    pub fn filter_by_capability(&self, capability: &str) -> Vec<WorkerEntry> {
        let workers = self.workers.read().unwrap();
        workers.values()
            .filter(|w| w.capabilities.contains(&capability.to_string()))
            .cloned()
            .collect()
    }
    
    pub fn remove(&self, worker_id: &str) -> Option<WorkerEntry> {
        let mut workers = self.workers.write().unwrap();
        workers.remove(worker_id)
    }
    
    pub fn cleanup_stale(&self, max_age_secs: u64) {
        let mut workers = self.workers.write().unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        workers.retain(|_, w| {
            now - w.last_heartbeat_unix < max_age_secs
        });
    }
}

#[derive(Clone, Debug)]
pub struct WorkerEntry {
    pub id: String,
    pub rbee_hive_url: String,
    pub capabilities: Vec<String>,
    pub models_loaded: Vec<String>,
    pub state: String,
    pub last_heartbeat_unix: u64,
}
```

**Why this matters:**
- Tests the core orchestration logic
- Validates worker lifecycle
- Catches race conditions
- **Target: Wire 15+ functions** (achievable in 3-4 hours)

#### Priority 3: Worker Provisioning (18 functions) - MOST INTERESTING
**File:** `src/steps/worker_provisioning.rs`  
**Product code:** Create `bin/rbee-hive/src/worker_provisioner.rs`

This is where it gets **fun** - you're testing actual `cargo build` commands:

```rust
// bin/rbee-hive/src/worker_provisioner.rs
use std::process::Command;

pub struct WorkerProvisioner {
    workspace_root: PathBuf,
}

impl WorkerProvisioner {
    pub async fn build_worker(
        &self,
        worker_type: &str,
        features: &[String],
    ) -> Result<PathBuf> {
        let mut cmd = Command::new("cargo");
        cmd.arg("build")
           .arg("--release")
           .arg("--bin")
           .arg(worker_type)
           .current_dir(&self.workspace_root);
        
        if !features.is_empty() {
            cmd.arg("--features").arg(features.join(","));
        }
        
        let output = cmd.output()?;
        
        if !output.status.success() {
            return Err(anyhow::anyhow!(
                "Build failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        
        let binary_path = self.workspace_root
            .join("target/release")
            .join(worker_type);
        
        Ok(binary_path)
    }
    
    pub fn verify_binary(&self, path: &Path) -> Result<()> {
        if !path.exists() {
            return Err(anyhow::anyhow!("Binary not found: {:?}", path));
        }
        
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let metadata = std::fs::metadata(path)?;
            let permissions = metadata.permissions();
            if permissions.mode() & 0o111 == 0 {
                return Err(anyhow::anyhow!("Binary not executable"));
            }
        }
        
        Ok(())
    }
}
```

**Why this is cool:**
- Tests actual build process
- Validates feature flags
- Catches missing dependencies
- **Target: Wire 10+ functions** (achievable in 3-4 hours)

#### Priority 4: SSH Preflight (14 functions) - DEVOPS CRITICAL
**File:** `src/steps/ssh_preflight.rs`  
**Product code:** Create `bin/queen-rbee/src/preflight/ssh.rs`

**DevOps will love you** for making SSH validation bulletproof:

```rust
// bin/queen-rbee/src/preflight/ssh.rs
use ssh2::Session;
use std::net::TcpStream;
use std::time::Duration;

pub struct SshPreflight {
    host: String,
    port: u16,
    user: String,
    key_path: Option<PathBuf>,
}

impl SshPreflight {
    pub async fn validate_connection(&self) -> Result<()> {
        let addr = format!("{}:{}", self.host, self.port);
        
        // Test TCP connection with timeout
        let tcp = TcpStream::connect_timeout(
            &addr.parse()?,
            Duration::from_secs(10)
        )?;
        
        // Test SSH handshake
        let mut sess = Session::new()?;
        sess.set_tcp_stream(tcp);
        sess.handshake()?;
        
        // Test authentication
        if let Some(key_path) = &self.key_path {
            sess.userauth_pubkey_file(&self.user, None, key_path, None)?;
        }
        
        if !sess.authenticated() {
            return Err(anyhow::anyhow!("SSH authentication failed"));
        }
        
        Ok(())
    }
    
    pub async fn execute_command(&self, cmd: &str) -> Result<String> {
        // Execute command and return stdout
    }
    
    pub async fn measure_latency(&self) -> Result<Duration> {
        let start = std::time::Instant::now();
        self.execute_command("echo test").await?;
        Ok(start.elapsed())
    }
    
    pub async fn check_binary_exists(&self, binary: &str) -> Result<bool> {
        let output = self.execute_command(&format!("which {}", binary)).await?;
        Ok(!output.trim().is_empty())
    }
}
```

**Why this matters:**
- Prevents deployment failures
- Validates SSH keys early
- Measures network latency
- **Target: Wire 8+ functions** (achievable in 2-3 hours)

#### Priority 5: rbee-hive Preflight (12 functions) - PLATFORM HEALTH
**File:** `src/steps/rbee_hive_preflight.rs`  
**Product code:** Create `bin/queen-rbee/src/preflight/rbee_hive.rs`

Simple HTTP health checks that **prevent production incidents**:

```rust
// bin/queen-rbee/src/preflight/rbee_hive.rs
use reqwest::Client;
use semver::Version;

pub struct RbeeHivePreflight {
    base_url: String,
    client: Client,
}

impl RbeeHivePreflight {
    pub async fn check_health(&self) -> Result<HealthResponse> {
        let resp = self.client
            .get(&format!("{}/health", self.base_url))
            .timeout(Duration::from_secs(5))
            .send()
            .await?;
        
        if !resp.status().is_success() {
            return Err(anyhow::anyhow!("Health check failed: {}", resp.status()));
        }
        
        let health: HealthResponse = resp.json().await?;
        Ok(health)
    }
    
    pub async fn check_version_compatibility(
        &self,
        required: &str,
    ) -> Result<bool> {
        let health = self.check_health().await?;
        let version = Version::parse(&health.version)?;
        let required_version = Version::parse(required)?;
        
        Ok(version >= required_version)
    }
    
    pub async fn query_backends(&self) -> Result<Vec<Backend>> {
        let resp = self.client
            .get(&format!("{}/v1/backends", self.base_url))
            .send()
            .await?;
        
        Ok(resp.json().await?)
    }
    
    pub async fn query_resources(&self) -> Result<ResourceInfo> {
        let resp = self.client
            .get(&format!("{}/v1/resources", self.base_url))
            .send()
            .await?;
        
        Ok(resp.json().await?)
    }
}
```

**Why this matters:**
- Validates platform readiness
- Checks version compatibility
- Verifies resource availability
- **Target: Wire all 12 functions** (achievable in 2 hours)

## Your Roadmap

### Hour 1-3: Model Catalog (Priority 1)
- Create `bin/rbee-hive/src/model_catalog.rs`
- Wire up all 18 functions in `src/steps/model_catalog.rs`
- Run: `LLORCH_BDD_FEATURE_PATH=tests/features/020-model-catalog.feature cargo test --package test-harness-bdd`
- **Goal: 100% pass rate for model catalog scenarios**

### Hour 4-7: queen-rbee Registry (Priority 2)
- Create `bin/queen-rbee/src/worker_registry.rs`
- Wire up 15+ functions in `src/steps/queen_rbee_registry.rs`
- Run: `LLORCH_BDD_FEATURE_PATH=tests/features/050-queen-rbee-worker-registry.feature cargo test --package test-harness-bdd`
- **Goal: 80%+ pass rate for registry scenarios**

### Hour 8-11: Worker Provisioning (Priority 3)
- Create `bin/rbee-hive/src/worker_provisioner.rs`
- Wire up 10+ functions in `src/steps/worker_provisioning.rs`
- Run: `LLORCH_BDD_FEATURE_PATH=tests/features/040-worker-provisioning.feature cargo test --package test-harness-bdd`
- **Goal: 70%+ pass rate for provisioning scenarios**

### Hour 12-14: SSH Preflight (Priority 4)
- Create `bin/queen-rbee/src/preflight/ssh.rs`
- Wire up 8+ functions in `src/steps/ssh_preflight.rs`
- Run: `LLORCH_BDD_FEATURE_PATH=tests/features/070-ssh-preflight-validation.feature cargo test --package test-harness-bdd`
- **Goal: 60%+ pass rate for SSH scenarios**

### Hour 15-16: rbee-hive Preflight (Priority 5)
- Create `bin/queen-rbee/src/preflight/rbee_hive.rs`
- Wire up all 12 functions in `src/steps/rbee_hive_preflight.rs`
- Run: `LLORCH_BDD_FEATURE_PATH=tests/features/080-rbee-hive-preflight-validation.feature cargo test --package test-harness-bdd`
- **Goal: 100% pass rate for preflight scenarios**

## Success Metrics

### Minimum (Don't Stop Here!)
- ✅ 10+ functions wired with real API calls
- ✅ Compilation green
- ✅ At least 1 feature file passing

### Good Progress
- ✅ 30+ functions wired (35% coverage)
- ✅ 2-3 feature files passing
- ✅ Model catalog + Registry complete

### Excellent Work
- ✅ 50+ functions wired (60% coverage)
- ✅ 4-5 feature files passing
- ✅ All Priority 1-3 complete

### Outstanding Achievement
- ✅ 70+ functions wired (83% coverage)
- ✅ All 5 priorities complete
- ✅ Pass rate 70%+ across all wired features

## Critical Reminders

### ⚠️ GPU FAIL FAST Policy
When implementing worker resource preflight (EH-005a, EH-009a/b):
```rust
// ❌ WRONG - NO CPU FALLBACK
if !gpu_available {
    tracing::warn!("GPU not available, falling back to CPU");
    use_cpu_backend();
}

// ✅ CORRECT - FAIL FAST
if !gpu_available {
    tracing::error!("GPU FAIL FAST! NO CPU FALLBACK!");
    return Err(anyhow::anyhow!("GPU required but not available"));
}
```

### 🏷️ Add Your Signature
```rust
// TEAM-079: Implemented real SQLite queries for model catalog
// TEAM-079: Wired queen-rbee registry with in-memory HashMap
```

### 🚫 NO TODO Markers
```rust
// ❌ BANNED
pub async fn some_function(world: &mut World) {
    // TODO: implement this later
}

// ✅ REQUIRED - Implement it now or ask for help
pub async fn some_function(world: &mut World) {
    // TEAM-079: Real implementation
    let catalog = ModelCatalog::new(...);
    let result = catalog.find_model(...).await;
    // ...
}
```

### 📏 Handoff ≤2 Pages
When you hand off to TEAM-080:
- Show **actual code** you implemented
- Show **pass rates** you achieved
- Show **function count** you wired
- NO TODO lists for next team

## Testing Commands

```bash
# Test specific feature
LLORCH_BDD_FEATURE_PATH=tests/features/020-model-catalog.feature \
  cargo test --package test-harness-bdd -- --nocapture

# Test all features (after wiring multiple)
cargo test --package test-harness-bdd -- --nocapture

# Check compilation
cargo check --package test-harness-bdd

# Run with debug logging
RUST_LOG=debug cargo test --package test-harness-bdd -- --nocapture
```

## Why You Should Go Big

**Scenario 1: Minimum Effort (10 functions)**
- You wire 10 functions
- Next team wires 10 functions
- Team after that wires 10 functions
- **Result:** 8 teams needed to finish, context lost, bugs multiply

**Scenario 2: Your Excellence (50+ functions)**
- You wire 50 functions in one focused session
- Next team wires remaining 34 functions
- **Result:** 2 teams total, momentum maintained, quality high

**Which team do you want to be?**

## Resources

### Existing Code to Learn From
- `src/steps/model_provisioning.rs` - Has some real API calls already
- `src/steps/beehive_registry.rs` - Shows HTTP client patterns
- `src/steps/worker_startup.rs` - Shows process spawning

### Dependencies Available
```toml
# Already in Cargo.toml:
reqwest = { workspace = true }  # HTTP client
tokio = { workspace = true }    # Async runtime
serde = { workspace = true }    # JSON serialization
anyhow = { workspace = true }   # Error handling
tracing = { workspace = true }  # Logging

# You may need to add:
rusqlite = "0.30"              # SQLite
ssh2 = "0.9"                   # SSH client
semver = "1.0"                 # Version parsing
```

## The Bottom Line

TEAM-078 gave you a **clean slate** - 15 organized feature files, 84 well-documented stubs, and a clear path forward.

**Don't just meet the minimum. Seize the opportunity.**

Wire up as many functions as you can. Make the tests actually test. Build something you're proud of.

The codebase is waiting. The scenarios are ready. The product code is yours to create.

**Go make BDD work.** 🚀

---

**TEAM-078 says:** The scaffolding is solid. Now build the house. Every function you wire is a bug you prevent. Don't stop at 10 - aim for 50+. You've got this! 🐝
