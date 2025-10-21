# CRITICAL BUG: Dependency Tracking in Auto-Build

**Discovered by:** User  
**Date:** 2025-10-21  
**Severity:** HIGH - Silent failures, stale binaries

## The Problem

### Current xtask Implementation (BROKEN)

```rust
// xtask/src/tasks/rbee.rs
fn needs_rebuild(workspace_root: &Path) -> Result<bool> {
    let binary_path = workspace_root.join("target/debug/rbee-keeper");
    let binary_time = binary_meta.modified()?;
    
    // ❌ ONLY checks bin/00_rbee_keeper/ directory
    let keeper_dir = workspace_root.join("bin/00_rbee_keeper");
    let needs_rebuild = check_dir_newer(&keeper_dir, binary_time)?;
    
    Ok(needs_rebuild)
}
```

**What it misses:**
- Changes in `daemon-lifecycle` crate
- Changes in `observability-narration-core` crate
- Changes in `timeout-enforcer` crate
- Changes in `rbee-operations` crate
- Changes in `rbee-config` crate

### Real-World Scenario (BUG REPRODUCTION)

```bash
# 1. Build rbee-keeper
cargo build --bin rbee-keeper
# Binary timestamp: 2025-10-21 10:00:00

# 2. Edit shared crate
vim bin/99_shared_crates/daemon-lifecycle/src/lib.rs
# Add new function, save
# File timestamp: 2025-10-21 10:05:00

# 3. Run via xtask
./rbee queen start

# ❌ BUG: xtask sees no changes in bin/00_rbee_keeper/
# ❌ Skips rebuild
# ❌ Runs OLD binary with OLD daemon-lifecycle code
# ❌ New function doesn't exist → runtime error or silent bug
```

## Dependency Graph Analysis

### rbee-keeper Dependencies

```
rbee-keeper (bin/00_rbee_keeper/)
├── daemon-lifecycle (bin/99_shared_crates/daemon-lifecycle/)
├── observability-narration-core (bin/99_shared_crates/narration-core/)
├── timeout-enforcer (bin/99_shared_crates/timeout-enforcer/)
├── rbee-operations (bin/99_shared_crates/rbee-operations/)
└── rbee-config (bin/15_queen_rbee_crates/rbee-config/)
```

### queen-rbee Dependencies

```
queen-rbee (bin/10_queen_rbee/)
├── daemon-lifecycle (bin/99_shared_crates/daemon-lifecycle/)  ← SHARED!
├── observability-narration-core (bin/99_shared_crates/narration-core/)  ← SHARED!
├── rbee-operations (bin/99_shared_crates/rbee-operations/)  ← SHARED!
├── rbee-config (bin/15_queen_rbee_crates/rbee-config/)  ← SHARED!
├── rbee-heartbeat (bin/99_shared_crates/heartbeat/)  ← SHARED!
├── job-registry (bin/99_shared_crates/job-registry/)
├── queen-rbee-hive-registry (bin/15_queen_rbee_crates/hive-registry/)
├── queen-rbee-hive-lifecycle (bin/15_queen_rbee_crates/hive-lifecycle/)
└── queen-rbee-scheduler (bin/15_queen_rbee_crates/scheduler/)
```

### rbee-hive Dependencies

```
rbee-hive (bin/20_rbee_hive/)
├── rbee-heartbeat (bin/99_shared_crates/heartbeat/)  ← SHARED!
└── observability-narration-core (bin/99_shared_crates/narration-core/)  ← SHARED!
```

### Cross-Binary Shared Crates

**These crates are used by MULTIPLE binaries:**

1. **observability-narration-core** → ALL 3 binaries
2. **daemon-lifecycle** → keeper, queen
3. **rbee-operations** → keeper, queen
4. **rbee-config** → keeper, queen
5. **rbee-heartbeat** → queen, hive

## The Cascade Problem

### Scenario: Edit `observability-narration-core`

```bash
# Edit shared crate used by ALL binaries
vim bin/99_shared_crates/narration-core/src/lib.rs

# What SHOULD happen:
# ✅ Rebuild rbee-keeper (uses narration-core)
# ✅ Rebuild queen-rbee (uses narration-core)
# ✅ Rebuild rbee-hive (uses narration-core)

# What ACTUALLY happens with current xtask:
# ❌ No rebuild (only checks bin/00_rbee_keeper/ directory)
# ❌ Runs stale binary with old narration-core
# ❌ Silent bugs or runtime errors
```

### Scenario: Edit `daemon-lifecycle`

```bash
# Edit shared crate used by keeper and queen
vim bin/99_shared_crates/daemon-lifecycle/src/lib.rs

# What SHOULD happen:
# ✅ Rebuild rbee-keeper (uses daemon-lifecycle)
# ✅ Rebuild queen-rbee (uses daemon-lifecycle)
# ⏭️  Skip rbee-hive (doesn't use daemon-lifecycle)

# What ACTUALLY happens:
# ❌ No rebuild for keeper
# ❌ No rebuild for queen
# ❌ Both run with stale daemon-lifecycle
```

## The Solution: Cargo.toml-Based Dependency Tracking

### Step 1: Parse Cargo.toml Dependencies

```rust
use cargo_toml::Manifest;

pub struct DependencyTracker {
    workspace_root: PathBuf,
}

impl DependencyTracker {
    /// Get all local path dependencies for a binary
    pub fn get_local_deps(&self, binary_name: &str) -> Result<Vec<PathBuf>> {
        let cargo_toml = self.workspace_root
            .join("bin")
            .join(self.binary_dir(binary_name))
            .join("Cargo.toml");
        
        let manifest = Manifest::from_path(&cargo_toml)?;
        
        let mut deps = Vec::new();
        
        // Parse [dependencies]
        for (name, dep) in manifest.dependencies {
            if let Some(path) = dep.detail().and_then(|d| d.path.as_ref()) {
                let dep_path = self.workspace_root.join(path);
                deps.push(dep_path);
            }
        }
        
        Ok(deps)
    }
    
    fn binary_dir(&self, binary_name: &str) -> &str {
        match binary_name {
            "rbee-keeper" => "00_rbee_keeper",
            "queen-rbee" => "10_queen_rbee",
            "rbee-hive" => "20_rbee_hive",
            _ => panic!("Unknown binary: {}", binary_name),
        }
    }
}
```

### Step 2: Recursive Dependency Resolution

```rust
impl DependencyTracker {
    /// Get ALL dependencies (including transitive) for a binary
    pub fn get_all_deps(&self, binary_name: &str) -> Result<Vec<PathBuf>> {
        let mut all_deps = Vec::new();
        let mut visited = HashSet::new();
        
        self.collect_deps_recursive(binary_name, &mut all_deps, &mut visited)?;
        
        Ok(all_deps)
    }
    
    fn collect_deps_recursive(
        &self,
        binary_name: &str,
        all_deps: &mut Vec<PathBuf>,
        visited: &mut HashSet<String>,
    ) -> Result<()> {
        if visited.contains(binary_name) {
            return Ok(());
        }
        visited.insert(binary_name.to_string());
        
        let deps = self.get_local_deps(binary_name)?;
        
        for dep_path in deps {
            all_deps.push(dep_path.clone());
            
            // Recursively check this dependency's dependencies
            if let Some(dep_name) = dep_path.file_name().and_then(|n| n.to_str()) {
                self.collect_deps_recursive(dep_name, all_deps, visited)?;
            }
        }
        
        Ok(())
    }
}
```

### Step 3: Updated needs_rebuild Logic

```rust
fn needs_rebuild(workspace_root: &Path, binary_name: &str) -> Result<bool> {
    let binary_path = workspace_root.join(format!("target/debug/{}", binary_name));
    
    // If binary doesn't exist, definitely need to build
    if !binary_path.exists() {
        return Ok(true);
    }
    
    let binary_time = std::fs::metadata(&binary_path)?.modified()?;
    
    // NEW: Get ALL dependencies (including transitive)
    let tracker = DependencyTracker::new(workspace_root.to_path_buf());
    let all_deps = tracker.get_all_deps(binary_name)?;
    
    // Check binary's own source directory
    let binary_dir = workspace_root.join("bin").join(binary_dir_name(binary_name));
    if check_dir_newer(&binary_dir, binary_time)? {
        return Ok(true);
    }
    
    // NEW: Check ALL dependency directories
    for dep_path in all_deps {
        if check_dir_newer(&dep_path, binary_time)? {
            println!("🔨 Dependency changed: {}", dep_path.display());
            return Ok(true);
        }
    }
    
    Ok(false)
}
```

## Example: Full Dependency Check

### rbee-keeper Dependency Tree

```
rbee-keeper needs rebuild if ANY of these changed:
├── bin/00_rbee_keeper/src/**/*.rs
├── bin/00_rbee_keeper/Cargo.toml
├── bin/99_shared_crates/daemon-lifecycle/**/*.rs
├── bin/99_shared_crates/daemon-lifecycle/Cargo.toml
├── bin/99_shared_crates/narration-core/**/*.rs
├── bin/99_shared_crates/narration-core/Cargo.toml
├── bin/99_shared_crates/timeout-enforcer/**/*.rs
├── bin/99_shared_crates/timeout-enforcer/Cargo.toml
├── bin/99_shared_crates/rbee-operations/**/*.rs
├── bin/99_shared_crates/rbee-operations/Cargo.toml
├── bin/15_queen_rbee_crates/rbee-config/**/*.rs
└── bin/15_queen_rbee_crates/rbee-config/Cargo.toml
```

### queen-rbee Dependency Tree (Partial)

```
queen-rbee needs rebuild if ANY of these changed:
├── bin/10_queen_rbee/src/**/*.rs
├── bin/10_queen_rbee/Cargo.toml
├── bin/99_shared_crates/daemon-lifecycle/**/*.rs  ← SHARED with keeper!
├── bin/99_shared_crates/narration-core/**/*.rs  ← SHARED with keeper!
├── bin/99_shared_crates/rbee-operations/**/*.rs  ← SHARED with keeper!
├── bin/15_queen_rbee_crates/rbee-config/**/*.rs  ← SHARED with keeper!
├── bin/99_shared_crates/heartbeat/**/*.rs
├── bin/99_shared_crates/job-registry/**/*.rs
├── bin/15_queen_rbee_crates/hive-registry/**/*.rs
├── bin/15_queen_rbee_crates/hive-lifecycle/**/*.rs
└── bin/15_queen_rbee_crates/scheduler/**/*.rs
```

## The Preemptive Rebuild Problem

### User's Concern: False Negatives

**Scenario:**
```bash
# 1. Edit shared crate
vim bin/99_shared_crates/narration-core/src/lib.rs

# 2. Rebuild keeper (triggers because narration-core changed)
./rbee queen start
# ✅ Keeper rebuilds (sees narration-core change)
# ✅ Keeper binary timestamp: 10:05:00

# 3. Later, try to rebuild queen
cargo build --bin queen-rbee
# ❓ Does queen see narration-core as changed?
# ❓ narration-core timestamp: 10:00:00 (original edit)
# ❓ queen binary timestamp: 09:00:00 (old)
# ✅ YES! 10:00 > 09:00 → rebuild triggered
```

**Verdict:** This is NOT a problem! Cargo's mtime-based tracking works correctly.

### BUT: Cargo Incremental Compilation Cache

**Real Problem:**
```bash
# 1. Edit shared crate
vim bin/99_shared_crates/narration-core/src/lib.rs

# 2. Build keeper
cargo build --bin rbee-keeper
# ✅ Recompiles narration-core
# ✅ Updates target/debug/deps/libnarration_core-*.rlib
# ✅ Links keeper

# 3. Build queen (without our auto-build check)
cargo build --bin queen-rbee
# ✅ Sees narration-core .rlib is newer
# ✅ Recompiles queen
# ✅ Links queen

# Conclusion: Cargo handles this correctly!
```

**However, our auto-build wrapper might skip the build entirely:**

```bash
# 1. Edit shared crate
vim bin/99_shared_crates/narration-core/src/lib.rs

# 2. Run keeper via xtask
./rbee queen start
# ❌ xtask checks: bin/00_rbee_keeper/ vs target/debug/rbee-keeper
# ❌ No changes in bin/00_rbee_keeper/ → SKIP BUILD
# ❌ Never invokes cargo build
# ❌ Runs stale keeper binary

# 3. Later, queen tries to spawn
# ❌ Queen binary is also stale (never rebuilt)
# ❌ Both running old narration-core code
```

## Solution: Trust Cargo, Check Dependencies

### Key Insight

**Cargo is smart.** We don't need to rebuild everything preemptively.

**Our job:** Detect when a rebuild is needed and invoke `cargo build`.  
**Cargo's job:** Figure out what to recompile (incremental compilation).

### Updated Strategy

```rust
fn needs_rebuild(workspace_root: &Path, binary_name: &str) -> Result<bool> {
    let binary_path = workspace_root.join(format!("target/debug/{}", binary_name));
    
    if !binary_path.exists() {
        return Ok(true);
    }
    
    let binary_time = std::fs::metadata(&binary_path)?.modified()?;
    
    // Check binary's source directory
    let binary_dir = workspace_root.join("bin").join(binary_dir_name(binary_name));
    if check_dir_newer(&binary_dir, binary_time)? {
        return Ok(true);
    }
    
    // Check ALL local dependencies (from Cargo.toml)
    let tracker = DependencyTracker::new(workspace_root.to_path_buf());
    let all_deps = tracker.get_all_deps(binary_name)?;
    
    for dep_path in all_deps {
        if check_dir_newer(&dep_path, binary_time)? {
            return Ok(true);
        }
    }
    
    Ok(false)
}
```

**Result:**
- If ANY dependency changed → invoke `cargo build`
- Cargo sees the change → recompiles affected crates
- Binary is up-to-date

## Implementation Checklist

### Phase 1: Fix xtask (Immediate)

- [ ] Add `cargo_toml` dependency to xtask
- [ ] Implement `DependencyTracker` in xtask
- [ ] Update `needs_rebuild()` to check dependencies
- [ ] Test: Edit shared crate → verify rebuild triggered

### Phase 2: Add to auto-build Crate

- [ ] Create `auto-build` crate with `DependencyTracker`
- [ ] Reuse in xtask
- [ ] Reuse in keeper (for queen auto-build)
- [ ] Reuse in queen (for hive auto-build)

### Phase 3: Testing

- [ ] Unit test: Edit narration-core → all 3 binaries rebuild
- [ ] Unit test: Edit daemon-lifecycle → keeper + queen rebuild
- [ ] Unit test: Edit heartbeat → queen + hive rebuild
- [ ] Integration test: Full cascade rebuild

## Dependencies Needed

```toml
# xtask/Cargo.toml
[dependencies]
cargo_toml = "0.18"  # NEW: Parse Cargo.toml files
```

## Performance Considerations

### Concern: Checking Many Directories

**Worst case:** rbee-keeper has ~10 dependencies
- Each dependency check: ~1-5ms (file system metadata)
- Total overhead: ~10-50ms

**Verdict:** Negligible compared to build time (seconds to minutes)

### Optimization: Cache Dependency Graph

```rust
// Cache the dependency graph (only parse Cargo.toml once)
lazy_static! {
    static ref DEPENDENCY_CACHE: Mutex<HashMap<String, Vec<PathBuf>>> = 
        Mutex::new(HashMap::new());
}
```

## Related Issues

1. **Workspace dependencies:** Some crates use `{ workspace = true }` syntax
   - Need to resolve workspace-level dependencies
   - Parse root `Cargo.toml` `[workspace.dependencies]`

2. **Dev dependencies:** Should we check dev-dependencies?
   - Probably NO (only affects tests, not binary)
   - Keep it simple: only `[dependencies]`

3. **Build scripts:** Some crates have `build.rs`
   - These can affect compilation
   - Should we check `build.rs` mtime?
   - Recommendation: YES, include in source directory check

## Conclusion

**Current xtask is broken.** It only checks the binary's own source directory.

**Fix:** Parse `Cargo.toml` to find ALL local dependencies, check them recursively.

**Trust Cargo:** Once we invoke `cargo build`, Cargo handles incremental compilation correctly.

**No preemptive rebuilds needed:** Cargo's dependency tracking works. We just need to detect WHEN to invoke cargo.
