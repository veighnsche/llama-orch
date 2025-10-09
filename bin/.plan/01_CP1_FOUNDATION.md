# Checkpoint 1: Foundation - Shared Crates + CLI Skeletons

**Created by:** TEAM-022  
**Checkpoint:** CP1  
**Duration:** Week 1 (5 days)  
**Status:** Pending

---

## Objective

Build the foundational infrastructure:
1. Create `pool-core` shared crate
2. Create `rbee-hive` CLI binary
3. Create `rbee-keeper` CLI binary (minimal SSH wrapper)

**Why This First:** Need basic CLI structure before implementing catalog and automation.

---

## Work Units

### WU1.1: Create pool-core Shared Crate (Day 1-2)

**Location:** `bin/shared-crates/pool-core/`

**Tasks:**
1. Create crate structure
2. Define core types
3. Implement basic validation
4. Write unit tests

**Files to Create:**
```
bin/shared-crates/pool-core/
├── Cargo.toml
├── README.md
├── .specs/
│   └── 00_pool-core.md
└── src/
    ├── lib.rs
    ├── worker.rs          # Worker types
    ├── catalog.rs         # Model catalog types
    ├── config.rs          # Configuration types
    └── error.rs           # Error types
```

**Core Types:**
```rust
// worker.rs
pub struct WorkerInfo {
    pub id: String,
    pub backend: Backend,
    pub model_ref: String,
    pub gpu_id: Option<u32>,
    pub port: u16,
    pub pid: u32,
}

pub enum Backend {
    Cpu,
    Metal,
    Cuda,
}

// catalog.rs
pub struct ModelCatalog {
    pub version: String,
    pub pool_id: String,
    pub models: Vec<ModelEntry>,
}

pub struct ModelEntry {
    pub id: String,
    pub name: String,
    pub path: PathBuf,
    pub architecture: String,
    pub downloaded: bool,
    pub backends: Vec<Backend>,
}
```

**Success Criteria:**
- [ ] Crate compiles
- [ ] All types have serde support
- [ ] Unit tests pass
- [ ] README documents public API

---

### WU1.2: Create rbee-hive CLI Binary (Day 2-3)

**Location:** `bin/rbee-hive/`

**Tasks:**
1. Create binary structure with clap
2. Implement command skeleton
3. Wire up pool-core types
4. Add basic commands (help, version, status)

**Files to Create:**
```
bin/rbee-hive/
├── Cargo.toml
├── README.md
├── .specs/
│   └── 00_rbee-hive.md
└── src/
    ├── main.rs
    ├── cli.rs             # Clap definitions
    └── commands/
        ├── mod.rs
        ├── models.rs      # rbee-hive models ...
        ├── worker.rs      # rbee-hive worker ...
        └── status.rs      # rbee-hive status
```

**CLI Structure:**
```rust
// cli.rs
#[derive(Parser)]
#[command(name = "rbee-hive")]
#[command(about = "Pool manager control CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Models {
        #[command(subcommand)]
        action: ModelsAction,
    },
    Worker {
        #[command(subcommand)]
        action: WorkerAction,
    },
    Status,
}

#[derive(Subcommand)]
enum ModelsAction {
    Download { model: String },
    List,
    Catalog,
}

#[derive(Subcommand)]
enum WorkerAction {
    Spawn {
        backend: String,
        #[arg(long)]
        model: String,
        #[arg(long, default_value = "0")]
        gpu: u32,
    },
    List,
    Stop { worker_id: String },
}
```

**Success Criteria:**
- [ ] Binary compiles
- [ ] `rbee-hive --help` works
- [ ] `rbee-hive --version` works
- [ ] `rbee-hive status` shows placeholder message
- [ ] All subcommands show help

---

### WU1.3: Create rbee-keeper CLI Binary (Day 3-4)

**Location:** `bin/rbee-keeper/`

**Tasks:**
1. Create binary structure with clap
2. Implement SSH wrapper
3. Wire up remote pool commands
4. Add basic commands (help, version)

**Files to Create:**
```
bin/rbee-keeper/
├── Cargo.toml
├── README.md
├── .specs/
│   └── 00_rbee-keeper.md
└── src/
    ├── main.rs
    ├── cli.rs             # Clap definitions
    ├── ssh.rs             # SSH client wrapper
    └── commands/
        ├── mod.rs
        └── pool.rs        # rbee pool ...
```

**CLI Structure:**
```rust
// cli.rs
#[derive(Parser)]
#[command(name = "llorch")]
#[command(about = "Orchestrator control CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Pool {
        #[command(subcommand)]
        action: PoolAction,
    },
}

#[derive(Subcommand)]
enum PoolAction {
    Models {
        #[command(subcommand)]
        action: ModelsAction,
        #[arg(long)]
        host: String,
    },
    Worker {
        #[command(subcommand)]
        action: WorkerAction,
        #[arg(long)]
        host: String,
    },
    Status {
        #[arg(long)]
        host: String,
    },
}
```

**SSH Wrapper:**
```rust
// ssh.rs
pub fn execute_remote_command(host: &str, command: &str) -> Result<String> {
    let output = Command::new("ssh")
        .arg(host)
        .arg(command)
        .output()?;
    
    if !output.status.success() {
        anyhow::bail!("SSH command failed: {}", 
            String::from_utf8_lossy(&output.stderr));
    }
    
    Ok(String::from_utf8(output.stdout)?)
}
```

**Success Criteria:**
- [ ] Binary compiles
- [ ] `llorch --help` works
- [ ] `llorch --version` works
- [ ] `llorch pool status --host mac` executes SSH command
- [ ] SSH wrapper handles errors gracefully

---

### WU1.4: Integration Testing (Day 4-5)

**Tasks:**
1. Test rbee-hive locally
2. Test rbee-keeper remote execution
3. Verify SSH connectivity to all pools
4. Document usage

**Test Cases:**
```bash
# Local rbee-hive
rbee-hive --help
rbee-hive --version
rbee-hive status
rbee-hive models --help
rbee-hive worker --help

# Remote rbee-keeper
llorch --help
llorch --version
llorch pool status --host mac.home.arpa
llorch pool status --host workstation.home.arpa
```

**Success Criteria:**
- [ ] All commands execute without errors
- [ ] SSH connectivity verified to all pools
- [ ] Help text is clear and accurate
- [ ] Error messages are helpful

---

### WU1.5: Documentation (Day 5)

**Tasks:**
1. Write pool-core README
2. Write rbee-hive README
3. Write rbee-keeper README
4. Update bin/.specs with implementation notes

**Documentation Requirements:**
- Installation instructions
- Command reference
- Examples
- Architecture diagrams
- Error handling guide

**Success Criteria:**
- [ ] All READMEs complete
- [ ] Examples are copy-pastable
- [ ] Architecture is clear
- [ ] Specs updated with actual implementation

---

## Checkpoint Gate: CP1 Verification

**Before proceeding to CP2, verify:**

### Compilation
- [ ] `cargo build --release -p pool-core` succeeds
- [ ] `cargo build --release -p rbee-hive` succeeds
- [ ] `cargo build --release -p rbee-keeper` succeeds

### Functionality
- [ ] `rbee-hive --help` shows all commands
- [ ] `rbee-hive status` runs without error
- [ ] `llorch --help` shows all commands
- [ ] `llorch pool status --host mac` executes via SSH

### Testing
- [ ] `cargo test -p pool-core` passes
- [ ] All unit tests pass
- [ ] SSH connectivity verified

### Documentation
- [ ] All READMEs exist and are complete
- [ ] Specs updated
- [ ] Examples tested

### Code Quality
- [ ] `cargo fmt --all` clean
- [ ] `cargo clippy --all` clean
- [ ] Team signatures added to all new files

---

## Deliverables

**Binaries:**
- `target/release/rbee-hive` (rbee-hive)
- `target/release/llorch` (rbee-keeper)

**Crates:**
- `bin/shared-crates/pool-core/`

**Documentation:**
- `bin/rbee-hive/README.md`
- `bin/rbee-keeper/README.md`
- `bin/shared-crates/pool-core/README.md`

---

## Dependencies

**Cargo.toml for pool-core:**
```toml
[package]
name = "pool-core"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
thiserror = "1.0"
```

**Cargo.toml for rbee-hive:**
```toml
[package]
name = "rbee-hive"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "rbee-hive"
path = "src/main.rs"

[dependencies]
pool-core = { path = "../shared-crates/pool-core" }
clap = { version = "4.5", features = ["derive"] }
anyhow = "1.0"
colored = "2.0"
```

**Cargo.toml for rbee-keeper:**
```toml
[package]
name = "rbee-keeper"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "llorch"
path = "src/main.rs"

[dependencies]
clap = { version = "4.5", features = ["derive"] }
anyhow = "1.0"
colored = "2.0"
```

---

## Risk Mitigation

**Risk 1:** SSH connectivity issues  
**Mitigation:** Test SSH manually before implementing wrapper

**Risk 2:** Clap API complexity  
**Mitigation:** Start with simple commands, add complexity incrementally

**Risk 3:** Type design mistakes  
**Mitigation:** Review specs carefully, align with existing contracts

---

## Next Checkpoint

After CP1 gate passes, proceed to `02_CP2_MODEL_CATALOG.md`.

---

**Status:** Ready to start  
**Estimated Duration:** 5 days  
**Blocking:** None
