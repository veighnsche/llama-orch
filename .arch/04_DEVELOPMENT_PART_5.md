# rbee Architecture Overview - Part 5: Development Patterns

**Version:** 1.0.0  
**Date:** October 23, 2025  
**Status:** Living Document

---

## Crate Structure

### Workspace Organization

```
llama-orch/ (workspace root)
├── bin/                           # All binaries and their crates
│   ├── 00_rbee_keeper/           # CLI (user interface)
│   ├── 05_rbee_keeper_crates/    # Keeper-specific crates
│   ├── 10_queen_rbee/            # Brain daemon
│   ├── 15_queen_rbee_crates/     # Queen-specific crates
│   ├── 20_rbee_hive/             # Pool daemon
│   ├── 25_rbee_hive_crates/      # Hive-specific crates
│   ├── 30_llm_worker_rbee/       # Worker daemon
│   └── 99_shared_crates/         # Shared infrastructure
├── consumers/                     # External SDKs
│   ├── rbee-sdk/                 # Rust SDK
│   └── rbee-utils/               # TypeScript utilities
├── contracts/                     # Shared types and specs
│   ├── api-types/                # API types
│   ├── openapi/                  # OpenAPI specs
│   └── config-schema/            # Config schemas
├── tools/                         # Development tools
│   ├── openapi-client/           # Generated client
│   ├── readme-index/             # README generator
│   └── spec-extract/             # Spec extraction
├── xtask/                         # Workspace automation
├── docs/                          # Documentation
├── requirements/                  # Requirements (YAML)
└── Cargo.toml                     # Workspace manifest
```

### Naming Convention

**Binary Crates:**
- `00_` - User interface (keeper)
- `10_` - Control plane (queen)
- `20_` - Data plane (hive)
- `30_` - Execution plane (worker)

**Supporting Crates:**
- `05_` - Keeper-specific
- `15_` - Queen-specific
- `25_` - Hive-specific
- `99_` - Shared across all

**Why Numbers?**
- Clear hierarchy
- Sorted by importance
- Easy to navigate
- Prevents name conflicts

### Crate Dependencies

**Binary → Crate:**
```
rbee-keeper
    ├── job-client (99_shared_crates)
    ├── rbee-operations (99_shared_crates)
    └── narration-core (99_shared_crates)

queen-rbee
    ├── job-server (99_shared_crates)
    ├── job-client (99_shared_crates)
    ├── rbee-operations (99_shared_crates)
    ├── hive-lifecycle (15_queen_rbee_crates)
    ├── hive-registry (15_queen_rbee_crates)
    └── narration-core (99_shared_crates)

rbee-hive
    ├── job-server (99_shared_crates)
    ├── rbee-operations (99_shared_crates)
    ├── device-detection (25_rbee_hive_crates)
    └── narration-core (99_shared_crates)
```

**Dependency Rules:**
1. Binaries can depend on any crate
2. Shared crates cannot depend on binary-specific crates
3. Binary-specific crates can depend on shared crates
4. No circular dependencies

---

## BDD Testing Strategy

### Test Organization

```
bin/00_rbee_keeper/
├── src/                          # Production code
└── bdd/                          # BDD tests
    ├── features/                 # Gherkin scenarios
    │   ├── hive_lifecycle.feature
    │   └── worker_management.feature
    ├── step_definitions/         # Rust step implementations
    │   ├── hive_steps.rs
    │   └── worker_steps.rs
    └── Cargo.toml                # Test dependencies
```

### Gherkin Example

```gherkin
# bin/00_rbee_keeper/bdd/features/hive_lifecycle.feature

Feature: Hive Lifecycle Management
  As an operator
  I want to manage hive daemons
  So that I can orchestrate LLM workers

  Scenario: Start a stopped hive
    Given a hive "localhost" is installed
    And the hive "localhost" is stopped
    When I run "rbee-keeper hive start localhost"
    Then I should see "🚀 Starting hive"
    And I should see "✅ Hive started"
    And the hive "localhost" should be running

  Scenario: Start an already running hive
    Given a hive "localhost" is installed
    And the hive "localhost" is running
    When I run "rbee-keeper hive start localhost"
    Then I should see "ℹ️  Hive is already running"
    And the exit code should be 0
```

### Step Definitions

```rust
// bin/00_rbee_keeper/bdd/step_definitions/hive_steps.rs

use cucumber::{given, when, then};

#[given(expr = "a hive {string} is installed")]
async fn hive_is_installed(world: &mut World, alias: String) {
    // Setup: Install hive if not exists
    world.ensure_hive_installed(&alias).await;
}

#[given(expr = "the hive {string} is stopped")]
async fn hive_is_stopped(world: &mut World, alias: String) {
    // Setup: Stop hive if running
    world.ensure_hive_stopped(&alias).await;
}

#[when(expr = "I run {string}")]
async fn run_command(world: &mut World, command: String) {
    // Execute: Run command, capture output
    let output = world.run_keeper_command(&command).await;
    world.last_output = output;
}

#[then(expr = "I should see {string}")]
async fn should_see(world: &mut World, expected: String) {
    // Assert: Check output contains expected string
    assert!(
        world.last_output.contains(&expected),
        "Expected to see '{}' in output:\n{}",
        expected,
        world.last_output
    );
}
```

### World Context

```rust
// bin/00_rbee_keeper/bdd/src/world.rs

pub struct World {
    pub last_output: String,
    pub last_exit_code: i32,
    pub hives: HashMap<String, HiveState>,
    pub workers: HashMap<String, WorkerState>,
}

impl World {
    pub async fn run_keeper_command(&mut self, command: &str) -> String {
        let args: Vec<&str> = command.split_whitespace().collect();
        let output = Command::new(&args[0])
            .args(&args[1..])
            .output()
            .await
            .expect("Failed to execute command");
        
        self.last_exit_code = output.status.code().unwrap_or(-1);
        String::from_utf8_lossy(&output.stdout).to_string()
    }
    
    pub async fn ensure_hive_installed(&mut self, alias: &str) {
        // Check if hive is in catalog
        // If not, install it
    }
}
```

### Running BDD Tests

```bash
# Run all BDD tests
cargo test --package rbee-keeper-bdd

# Run specific feature
cargo test --package rbee-keeper-bdd --test hive_lifecycle

# Run with verbose output
cargo test --package rbee-keeper-bdd -- --nocapture
```

### BDD Benefits

1. **Living Documentation:** Features describe behavior
2. **Stakeholder Communication:** Non-technical readable
3. **Regression Protection:** Scenarios must pass
4. **Refactoring Safety:** Behavior preserved
5. **Examples:** Concrete usage examples

---

## Character-Driven Development

### The Six Teams

rbee is built by **6 specialized AI teams** with distinct personalities:

#### 1. Testing Team 🔍

**Personality:** Obsessively paranoid, zero tolerance for false positives

**Focus:**
- BDD scenarios
- Test coverage
- Edge cases
- False positive prevention

**Example Contribution:**
```gherkin
Scenario: Handle network timeout gracefully
  Given the queen is unreachable
  When I run "rbee-keeper hive start localhost"
  Then I should see "❌ Error: Connection timeout"
  And the exit code should be 1
```

#### 2. Security Team (auth-min) 🎭

**Personality:** Trickster guardians, timing-safe everything

**Focus:**
- Authentication primitives
- Timing-safe comparison
- Token fingerprinting
- Constant-time operations

**Example Contribution:**
```rust
// auth-min/src/lib.rs
pub fn compare_tokens(a: &str, b: &str) -> bool {
    use subtle::ConstantTimeEq;
    a.as_bytes().ct_eq(b.as_bytes()).into()
}
```

#### 3. Performance Team ⏱️

**Personality:** Obsessive timekeepers, every millisecond counts

**Focus:**
- Latency optimization
- Resource efficiency
- Benchmark-driven
- Zero-copy where possible

**Example Contribution:**
```rust
// Avoid clone when Arc<T> is sufficient
let config = Arc::clone(&state.config);  // ✅ Fast
// let config = state.config.clone();    // ❌ Slow
```

#### 4. Audit Logging Team 🔒

**Personality:** Compliance engine, immutable trails

**Focus:**
- GDPR compliance
- Tamper detection
- 7-year retention
- Structured logging

**Example Contribution:**
```rust
audit_logger.log(AuditEvent::WorkerSpawned {
    timestamp: Utc::now(),
    worker_id,
    model,
    device,
    operator_id,
});
```

#### 5. Narration Core Team 🎀

**Personality:** Observability artists, secret redaction experts

**Focus:**
- Human-readable events
- SSE routing
- Secret redaction
- Consistent formatting

**Example Contribution:**
```rust
NARRATE
    .action("worker_spawn")
    .job_id(&job_id)
    .context(&worker_id)
    .human("🚀 Spawning worker {}")
    .emit();
```

#### 6. Developer Experience Team 🎨

**Personality:** Readability minimalists, policy hunters

**Focus:**
- API ergonomics
- Error messages
- Documentation
- Code organization

**Example Contribution:**
```rust
// ❌ Before: Cryptic error
Err(anyhow!("Failed"))

// ✅ After: Helpful error
Err(anyhow!(
    "Failed to spawn worker: GPU-0 not found\n\
     Available devices:\n  \
     - GPU-0 (NVIDIA RTX 3090, 24GB)\n  \
     - CPU-0 (16 cores, 64GB RAM)"
))
```

### TEAM-XXX Pattern

Every code change is attributed to a team:

```rust
// TEAM-261: Simplified heartbeat architecture
// Workers send heartbeats directly to queen (not through hive)
pub async fn send_heartbeat_to_queen(...) { ... }
```

**Benefits:**
1. **Attribution:** Know who made each decision
2. **Context:** Understand why changes were made
3. **Continuity:** Teams pick up where others left off
4. **Accountability:** Clear ownership

### Handoff Documents

Teams create handoff documents:

```markdown
# TEAM-261: Hive Simplification - COMPLETE

**Status:** ✅ COMPLETE  
**Mission:** Remove hive heartbeat, simplify architecture

**Deliverables:**
1. Removed hive heartbeat task (~30 LOC)
2. Deleted heartbeat.rs (80 LOC)
3. Added worker heartbeat endpoint to queen (+40 LOC)
4. Updated worker to send to queen (~20 LOC modified)

**Impact:** ~110 LOC removed, simpler architecture

**Next Team:** TEAM-262 can implement worker registry in queen
```

---

## Code Organization Principles

### 1. Binary-Thin, Crate-Fat

**Principle:** Keep binary code minimal, put logic in crates.

**Binary (bin/XX/src/main.rs):**
```rust
// ONLY HTTP server setup and routing
#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let router = create_router();
    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, router).await?;
    Ok(())
}
```

**Crate (bin/XX_crates/lifecycle/):**
```rust
// ALL business logic
pub async fn execute_hive_start(...) -> Result<()> {
    // Actual implementation
}
```

**Benefits:**
- Testable (crates can be unit tested)
- Reusable (crates can be used by other binaries)
- Maintainable (logic separate from HTTP)

### 2. Operation-Based Routing

**Principle:** Use enum for type-safe operation routing.

```rust
#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Operation {
    HiveStart { alias: String },
    WorkerSpawn { model: String },
    // ... more operations
}

// Routing
match operation {
    Operation::HiveStart { alias } => execute_hive_start(alias).await,
    Operation::WorkerSpawn { model } => execute_worker_spawn(model).await,
}
```

**Benefits:**
- Type-safe (compile-time checking)
- Exhaustive (compiler ensures all cases handled)
- Serializable (same enum for HTTP and internal use)

### 3. SSE-First Design

**Principle:** All long-running operations provide SSE feedback.

```rust
pub async fn execute_long_operation(
    request: Request,
) -> Result<()> {
    NARRATE.action("start")
        .job_id(&request.job_id)
        .human("Starting...")
        .emit();
    
    // Do work...
    
    NARRATE.action("progress")
        .job_id(&request.job_id)
        .human("50% complete")
        .emit();
    
    // More work...
    
    NARRATE.action("complete")
        .job_id(&request.job_id)
        .human("✅ Complete")
        .emit();
    
    Ok(())
}
```

**Benefits:**
- User visibility (real-time feedback)
- Debuggable (see exactly what's happening)
- Cancellable (can interrupt mid-operation)

### 4. Config-File Based (Cross-Platform)

**Principle:** Use files for config, not environment variables. Support all platforms.

**Why Files?**
- Version controllable (git-trackable)
- Easier to edit (no shell syntax)
- Hierarchical (TOML/YAML structure)
- Validatable (schema checking)
- Cross-platform (same format everywhere)

**Platform-Specific Locations:**

| Platform | Config | Cache |
|----------|--------|-------|
| Linux | `~/.config/rbee/` | `~/.cache/rbee/` |
| macOS | `~/Library/Application Support/rbee/` | `~/Library/Caches/rbee/` |
| Windows | `%APPDATA%\rbee\` | `%LOCALAPPDATA%\rbee\` |

**Implementation:**
```toml
# config.toml (same format on all platforms)
[queen]
port = 8500

[hives.localhost]
host = "localhost"
port = 9000
```

```rust
pub struct RbeeConfig {
    pub queen: QueenConfig,
    pub hives: HashMap<String, HiveConfig>,
}

impl RbeeConfig {
    /// Load config from platform-specific directory
    pub fn load() -> Result<Self> {
        // Uses dirs crate for cross-platform support
        let path = dirs::config_dir()
            .ok_or_else(|| anyhow!("Cannot determine config directory"))?
            .join("rbee/config.toml");
        
        let content = fs::read_to_string(path)?;
        let config: RbeeConfig = toml::from_str(&content)?;
        Ok(config)
    }
}
```

**See:** `bin/.plan/CROSS_PLATFORM_CONFIG_PLAN.md` for full implementation details.

### 5. Error Context

**Principle:** Provide helpful error messages with context.

```rust
// ❌ Bad
Err(anyhow!("Failed"))

// ✅ Good
Err(anyhow!(
    "Failed to start hive '{}': process exited with code {}\n\
     Hint: Check that port {} is not already in use",
    alias, exit_code, port
))
```

**Benefits:**
- Debuggable (clear what went wrong)
- Actionable (hints for resolution)
- User-friendly (no cryptic errors)

---

## Development Workflow

### 1. Write Gherkin Feature

```gherkin
Feature: Worker Spawning
  Scenario: Spawn worker on available GPU
    Given a hive "localhost" is running
    When I run "rbee-keeper worker spawn --model llama-3-8b --device GPU-0"
    Then I should see "✅ Worker spawned"
```

### 2. Implement Step Definitions

```rust
#[when(expr = "I run {string}")]
async fn run_command(world: &mut World, command: String) {
    world.last_output = world.run_keeper_command(&command).await;
}
```

### 3. Run BDD Test (Fails)

```bash
$ cargo test --package rbee-keeper-bdd
# ❌ Fails: Feature not implemented
```

### 4. Implement Feature

```rust
// bin/15_queen_rbee_crates/hive-lifecycle/src/worker.rs
pub async fn execute_worker_spawn(...) -> Result<()> {
    // Implementation
}
```

### 5. Wire Up Binary

```rust
// bin/10_queen_rbee/src/job_router.rs
Operation::WorkerSpawn { .. } => {
    hive_forwarder::forward_to_hive(&job_id, op, config).await
}
```

### 6. Run BDD Test (Passes)

```bash
$ cargo test --package rbee-keeper-bdd
# ✅ Passes: Feature implemented
```

### 7. Handoff Document

```markdown
# TEAM-XXX: Worker Spawning - COMPLETE

**Deliverables:**
- BDD scenario written
- Step definitions implemented
- Worker spawn logic in hive-lifecycle crate
- Operation routing in queen-rbee

**Status:** ✅ COMPLETE
```

---

## Next: Part 6 - Security & Compliance

The final document covers security architecture and GDPR compliance:
- Defense in Depth
- GDPR Compliance
- Audit Logging
- Threat Model

**See:** `.arch/05_SECURITY_PART_6.md`
