# Auto-Build System Plan

**Created by:** TEAM-193  
**Date:** 2025-10-21  
**Status:** PLANNING

## Executive Summary

Implement auto-rebuild detection for daemon binaries across the rbee ecosystem:
- **Keeper → Queen:** Already exists in xtask (TEAM-162)
- **Queen → Hive:** NEW - needs implementation
- **Hive → Worker:** NEW - needs implementation (future)

## Current State Analysis

### ✅ EXISTING: Keeper Auto-Build (xtask)

**Location:** `/home/vince/Projects/llama-orch/xtask/src/tasks/rbee.rs`

**How it works:**
1. `./rbee` wrapper calls `xtask::run_rbee_keeper(args)`
2. xtask checks if `target/debug/rbee-keeper` needs rebuild
3. Compares binary mtime vs source files in `bin/00_rbee_keeper/`
4. Rebuilds if needed: `cargo build --bin rbee-keeper`
5. Forwards command to keeper binary

**Key functions:**
- `needs_rebuild(workspace_root: &Path) -> Result<bool>`
- `check_dir_newer(dir: &Path, reference_time: SystemTime) -> Result<bool>`
- `build_rbee_keeper(workspace_root: &Path) -> Result<()>`

### 🆕 NEEDED: Queen Auto-Build

**Trigger:** `./rbee queen update` (NEW COMMAND)

**Location:** Keeper spawns queen via `queen_lifecycle::ensure_queen_running()`

**Current behavior:**
- `DaemonManager::find_in_target("queen-rbee")` finds existing binary
- Spawns queen with `--port 8500`
- NO rebuild check

**Desired behavior:**
1. Check if queen binary needs rebuild (compare mtime)
2. If stale → rebuild: `cargo build --bin queen-rbee`
3. Spawn updated binary

### 🆕 NEEDED: Hive Auto-Build

**Trigger:** Queen spawns hive via `hive_lifecycle::execute_hive_start()`

**Location:** `/home/vince/Projects/llama-orch/bin/15_queen_rbee_crates/hive-lifecycle/src/lib.rs`

**Current behavior:**
- Hardcoded path: `target/debug/rbee-hive`
- Spawns hive with `--port {port} --queen-url {queen_url}`
- NO rebuild check

**Desired behavior:**
1. Check if hive binary needs rebuild (compare mtime)
2. If stale → rebuild: `cargo build --bin rbee-hive`
3. Spawn updated binary

### 🔮 FUTURE: Worker Auto-Build

**Trigger:** Hive spawns worker (not in scope for this plan)

**Note:** Same pattern will apply when hive spawns llm-worker-rbee

## Architecture Pattern

### Shared Pattern (All 3 Levels)

```rust
// 1. Check if rebuild needed
if needs_rebuild(binary_name, source_dir)? {
    // 2. Rebuild binary
    rebuild_binary(binary_name)?;
}

// 3. Spawn daemon
let binary_path = find_binary(binary_name)?;
spawn_daemon(binary_path, args).await?;
```

### Configuration Control

**NEW Config Field:** `auto_build: bool` (default: `true`)

**Locations:**
1. **Keeper config:** `~/.config/rbee/config.toml`
   ```toml
   queen_port = 8500
   auto_build_queen = true  # NEW
   ```

2. **Queen config:** (needs investigation - where does queen store config?)
   ```toml
   auto_build_hive = true  # NEW
   ```

3. **Hive config:** (needs investigation - where does hive store config?)
   ```toml
   auto_build_worker = true  # NEW
   ```

## Implementation Plan

### Phase 1: Shared Auto-Build Crate

**Create:** `bin/99_shared_crates/auto-build/`

**Purpose:** Reusable auto-build logic for all binaries

**API:**
```rust
pub struct AutoBuilder {
    binary_name: String,
    source_dir: PathBuf,
    workspace_root: PathBuf,
}

impl AutoBuilder {
    pub fn new(binary_name: &str, source_dir: PathBuf) -> Result<Self>;
    
    /// Check if binary needs rebuild
    pub fn needs_rebuild(&self) -> Result<bool>;
    
    /// Rebuild the binary
    pub fn rebuild(&self) -> Result<()>;
    
    /// Find binary in target directory
    pub fn find_binary(&self) -> Result<PathBuf>;
    
    /// Full workflow: check, rebuild if needed, return path
    pub async fn ensure_built(&self) -> Result<PathBuf>;
}
```

**Narration:**
- Actor: `"🔨 auto-build"`
- Actions: `"check_rebuild"`, `"rebuild"`, `"find_binary"`

### Phase 2: Update Keeper Config

**File:** `bin/00_rbee_keeper/src/config.rs`

**Changes:**
```rust
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    #[serde(default = "default_queen_port")]
    pub queen_port: u16,
    
    // NEW: Auto-build control
    #[serde(default = "default_auto_build")]
    pub auto_build_queen: bool,
}

fn default_auto_build() -> bool {
    true
}
```

### Phase 3: Update Queen Lifecycle

**File:** `bin/00_rbee_keeper/src/queen_lifecycle.rs`

**Changes:**
```rust
use auto_build::AutoBuilder;

pub async fn ensure_queen_running(base_url: &str, config: &Config) -> Result<QueenHandle> {
    // ... existing health check ...
    
    // NEW: Auto-build check
    if config.auto_build_queen {
        let builder = AutoBuilder::new(
            "queen-rbee",
            PathBuf::from("bin/10_queen_rbee"),
        )?;
        
        if builder.needs_rebuild()? {
            NARRATE.action("queen_rebuild")
                .human("🔨 Queen binary is stale, rebuilding...")
                .emit();
            builder.rebuild()?;
        }
    }
    
    // Find binary (existing logic)
    let queen_binary = if config.auto_build_queen {
        AutoBuilder::new("queen-rbee", PathBuf::from("bin/10_queen_rbee"))?
            .find_binary()?
    } else {
        DaemonManager::find_in_target("queen-rbee")?
    };
    
    // ... rest of existing logic ...
}
```

### Phase 4: Add `./rbee queen update` Command

**File:** `bin/00_rbee_keeper/src/main.rs`

**Changes:**
```rust
#[derive(Subcommand)]
pub enum QueenAction {
    /// Start queen-rbee daemon
    Start,
    /// Stop queen-rbee daemon
    Stop,
    /// Check queen-rbee daemon status
    Status,
    /// Update queen-rbee binary (rebuild from source)
    Update,  // NEW
}

// In handle_command:
Commands::Queen { action } => match action {
    // ... existing actions ...
    
    QueenAction::Update => {
        NARRATE.action("queen_update")
            .human("🔨 Rebuilding queen-rbee binary...")
            .emit();
        
        let builder = AutoBuilder::new(
            "queen-rbee",
            PathBuf::from("bin/10_queen_rbee"),
        )?;
        
        builder.rebuild()?;
        
        NARRATE.action("queen_update")
            .human("✅ Queen binary updated successfully")
            .emit();
        
        Ok(())
    }
}
```

### Phase 5: Investigate Queen/Hive Config

**TODO:** Find where queen-rbee and rbee-hive store their configuration

**Questions:**
1. Does queen have a config file? Where?
2. Does hive have a config file? Where?
3. Or are they purely CLI-driven?

**If no config exists:**
- Add config support to queen-rbee
- Add config support to rbee-hive
- Follow same pattern as keeper: `~/.config/rbee/queen.toml`, `~/.config/rbee/hive.toml`

### Phase 6: Update Hive Lifecycle

**File:** `bin/15_queen_rbee_crates/hive-lifecycle/src/lib.rs`

**Changes:**
```rust
use auto_build::AutoBuilder;

async fn spawn_hive(port: u16, queen_url: &str, auto_build: bool) -> Result<()> {
    // NEW: Auto-build check
    if auto_build {
        let builder = AutoBuilder::new(
            "rbee-hive",
            PathBuf::from("bin/20_rbee_hive"),
        )?;
        
        if builder.needs_rebuild()? {
            Narration::new(ACTOR_HIVE_LIFECYCLE, "hive_rebuild", &format!("port:{}", port))
                .human("🔨 Hive binary is stale, rebuilding...")
                .emit();
            builder.rebuild()?;
        }
    }
    
    // Find binary
    let hive_binary = if auto_build {
        AutoBuilder::new("rbee-hive", PathBuf::from("bin/20_rbee_hive"))?
            .find_binary()?
    } else {
        PathBuf::from("target/debug/rbee-hive")
    };
    
    // Spawn with found binary
    let _child = Command::new(&hive_binary)
        .arg("--port")
        .arg(port.to_string())
        .arg("--queen-url")
        .arg(queen_url)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("Failed to spawn rbee-hive")?;
    
    // ... rest of existing logic ...
}
```

## Configuration UX

### Enable/Disable Auto-Build

**Keeper → Queen:**
```bash
# Enable (default)
echo 'auto_build_queen = true' >> ~/.config/rbee/config.toml

# Disable
echo 'auto_build_queen = false' >> ~/.config/rbee/config.toml
```

**Queen → Hive:**
```bash
# Enable (default)
echo 'auto_build_hive = true' >> ~/.config/rbee/queen.toml

# Disable
echo 'auto_build_hive = false' >> ~/.config/rbee/queen.toml
```

### Manual Update Commands

**Force rebuild queen:**
```bash
./rbee queen update
```

**Force rebuild hive:**
```bash
./rbee hive update --id localhost  # Future command
```

## Testing Strategy

### Unit Tests

**auto-build crate:**
- `test_needs_rebuild_binary_missing()` → true
- `test_needs_rebuild_source_newer()` → true
- `test_needs_rebuild_binary_newer()` → false
- `test_rebuild_success()`
- `test_find_binary_debug()`
- `test_find_binary_release()`

### Integration Tests

**Keeper → Queen:**
1. Delete queen binary
2. Run `./rbee queen start`
3. Verify auto-rebuild triggered
4. Verify queen started successfully

**Queen → Hive:**
1. Delete hive binary
2. Run `./rbee hive start --id localhost`
3. Verify auto-rebuild triggered
4. Verify hive started successfully

### E2E Tests

**Full stack:**
1. Delete all binaries (keeper, queen, hive)
2. Run `./rbee infer --model test --prompt "hello"`
3. Verify cascade rebuild: keeper → queen → hive
4. Verify inference succeeds

## Edge Cases

### 1. Concurrent Rebuilds

**Problem:** Multiple processes try to rebuild same binary simultaneously

**Solution:** Use file-based lock (e.g., `target/.rebuild-lock-queen-rbee`)

```rust
impl AutoBuilder {
    pub fn rebuild(&self) -> Result<()> {
        let lock_file = format!("target/.rebuild-lock-{}", self.binary_name);
        let _lock = FileLock::acquire(&lock_file)?;
        
        // Rebuild with lock held
        self.rebuild_inner()?;
        
        Ok(())
    }
}
```

### 2. Build Failures

**Problem:** Source code has errors, rebuild fails

**Solution:** Fail fast with clear error message

```rust
if !status.success() {
    NARRATE.action("rebuild")
        .context(self.binary_name.clone())
        .human("❌ Failed to rebuild {}")
        .error_kind("build_failed")
        .emit();
    anyhow::bail!("Build failed for {}", self.binary_name);
}
```

### 3. Workspace Root Detection

**Problem:** Auto-build needs workspace root, but binaries run from anywhere

**Solution:** Walk up directory tree to find `Cargo.toml` with `[workspace]`

```rust
pub fn find_workspace_root() -> Result<PathBuf> {
    let mut current = std::env::current_dir()?;
    
    loop {
        let cargo_toml = current.join("Cargo.toml");
        if cargo_toml.exists() {
            let contents = std::fs::read_to_string(&cargo_toml)?;
            if contents.contains("[workspace]") {
                return Ok(current);
            }
        }
        
        current = current.parent()
            .ok_or_else(|| anyhow!("Workspace root not found"))?
            .to_path_buf();
    }
}
```

## Migration Path

### Step 1: Create auto-build crate (no breaking changes)
- New crate, no existing code affected
- Write unit tests

### Step 2: Update keeper (opt-in via config)
- Add `auto_build_queen` config field (default: true)
- Update `queen_lifecycle.rs` to use auto-build
- Add `./rbee queen update` command
- Existing behavior unchanged if config = false

### Step 3: Update queen (opt-in via config)
- Add queen config file support
- Add `auto_build_hive` config field (default: true)
- Update `hive-lifecycle` crate to use auto-build
- Existing behavior unchanged if config = false

### Step 4: Documentation
- Update README with auto-build feature
- Add troubleshooting guide
- Document config options

## Success Criteria

### ✅ Phase 1 Complete When:
- [ ] `auto-build` crate exists with full API
- [ ] Unit tests pass (100% coverage)
- [ ] Documentation complete

### ✅ Phase 2-4 Complete When:
- [ ] `./rbee queen update` command works
- [ ] Keeper auto-rebuilds queen when source changes
- [ ] Config option works (enable/disable)
- [ ] Integration tests pass

### ✅ Phase 5-6 Complete When:
- [ ] Queen auto-rebuilds hive when source changes
- [ ] Config option works (enable/disable)
- [ ] E2E test passes (full cascade rebuild)

## Open Questions

1. **Queen config location:** Where should queen store its config?
   - Option A: `~/.config/rbee/queen.toml`
   - Option B: Embedded in keeper config
   - Option C: CLI flags only (no config file)

2. **Hive config location:** Where should hive store its config?
   - Option A: `~/.config/rbee/hive.toml`
   - Option B: Embedded in queen config
   - Option C: CLI flags only (no config file)

3. **Build parallelism:** Should we allow parallel builds?
   - Pro: Faster when rebuilding multiple binaries
   - Con: More complex, potential race conditions
   - Recommendation: Start with sequential, add parallel later

4. **Release builds:** Should auto-build support release mode?
   - Current: Only debug builds
   - Future: Add `--release` flag support?

5. **Cross-compilation:** What about remote hives with different architectures?
   - Current: Assumes same architecture
   - Future: Add cross-compilation support?

## Related Files

### Existing Code
- `xtask/src/tasks/rbee.rs` - Keeper auto-build (reference implementation)
- `bin/00_rbee_keeper/src/queen_lifecycle.rs` - Queen spawning
- `bin/15_queen_rbee_crates/hive-lifecycle/src/lib.rs` - Hive spawning
- `bin/99_shared_crates/daemon-lifecycle/src/lib.rs` - Daemon spawning utilities

### New Files (To Create)
- `bin/99_shared_crates/auto-build/src/lib.rs` - Shared auto-build logic
- `bin/99_shared_crates/auto-build/Cargo.toml` - Crate manifest
- `bin/99_shared_crates/auto-build/README.md` - Documentation

### Modified Files
- `bin/00_rbee_keeper/src/config.rs` - Add auto_build_queen field
- `bin/00_rbee_keeper/src/main.rs` - Add queen update command
- `bin/00_rbee_keeper/src/queen_lifecycle.rs` - Use auto-build
- `bin/15_queen_rbee_crates/hive-lifecycle/src/lib.rs` - Use auto-build

## Timeline Estimate

- **Phase 1 (auto-build crate):** 4-6 hours
- **Phase 2-4 (keeper → queen):** 6-8 hours
- **Phase 5 (investigation):** 2-3 hours
- **Phase 6 (queen → hive):** 4-6 hours
- **Testing & Documentation:** 4-6 hours

**Total:** 20-29 hours (2.5-4 days)

## Notes

- This pattern emerges naturally from the hierarchical daemon architecture
- Same pattern will apply to worker spawning in the future
- Auto-build is development-focused (production uses pre-built binaries)
- Config-driven approach allows users to disable if needed
- Reuses existing patterns from xtask implementation
