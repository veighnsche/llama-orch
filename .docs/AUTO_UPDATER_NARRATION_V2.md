# Auto-Updater Narration Format V2

**Version:** 2.0  
**Date:** October 26, 2025  
**Status:** ✅ SPECIFICATION (Ready for implementation)

---

## Objective

Render auto-update steps with fewer, clearer lines, grouped by phase, using the `n!()` macro and consistent action names. Include short per-phase summaries with detail togglable via `--verbose` or `RBEE_VERBOSE=1`.

---

## Global Rules

### 1. Action Naming Convention

- **Phase banners:** `phase_*` (e.g., `phase_init`, `phase_deps`)
- **Regular actions:** Stable snake_case, ≤2 words (e.g., `find_workspace`, `parse_deps`, `check_rebuild`)
- **Summary actions:** Always `summary` at end of each phase

### 2. Message Format

- **Two-line format only:** Header (bold) + body message
- **No extra blank lines** or chatty text
- Format handled automatically by `format_message()`

### 3. Emoji Standards

**Phase banners:**
- 🚧 Init
- 🧭 Workspace
- 📦 Dependencies
- 🛠️ Build state
- 🔍 File scans
- 📑 Decision

**Status indicators:**
- ✅ Success/done
- ⚠️ Warning
- ❌ Failure
- ⏳ Waiting/in-progress

### 4. Batching & Summarization

- **Collapse repetitive lines** into batch operations
- **Show counts** instead of individual items
- **Detail behind verbose flag:** `--verbose` or `RBEE_VERBOSE=1`

### 5. Context Propagation

If `job_id` exists, wrap entire flow in `with_narration_context` so every line inherits it automatically.

---

## Phase Structure

### Phase 1: Init

**Purpose:** Initialize auto-updater configuration

**Actions:**
```rust
n!("phase_init", "🚧 Initializing auto-updater for {}", binary_name);
n!("init", "Mode: {} · Binary: {} · Source: {}", mode, binary_name, source_dir);
n!("summary", "✅ Init ok · {}ms", elapsed_ms);
```

**Example output:**
```
[auto-update        ] phase_init          
🚧 Initializing auto-updater for rbee-keeper

[auto-update        ] init                
Mode: debug · Binary: rbee-keeper · Source: bin/00_rbee_keeper

[auto-update        ] summary             
✅ Init ok · 2ms
```

---

### Phase 2: Workspace

**Purpose:** Detect workspace root

**Actions:**
```rust
n!("phase_workspace", "🧭 Workspace detection");
n!("find_workspace", "Searching for workspace root · depth: {}", depth);
n!("workspace_found", "Workspace: {}", workspace_path);
n!("summary", "✅ Workspace ok · {}ms", elapsed_ms);
```

**Example output:**
```
[auto-update        ] phase_workspace     
🧭 Workspace detection

[auto-update        ] find_workspace      
Searching for workspace root · depth: 3

[auto-update        ] workspace_found     
Workspace: /home/user/Projects/llama-orch

[auto-update        ] summary             
✅ Workspace ok · 5ms
```

---

### Phase 3: Dependencies

**Purpose:** Discover and parse dependency tree

**Actions (normal mode):**
```rust
n!("phase_deps", "📦 Dependency discovery");
n!("parse_deps", "Scanning root crate: {}", root_crate_path);
n!("collect_tomls", "Queued Cargo.toml files: {}", toml_count);
n!("parse_batch", "Parsed {} deps · {} local path · {} transitive", 
    total_deps, local_deps, transitive_deps);
n!("summary", "✅ Deps ok · {}ms", elapsed_ms);
```

**Actions (verbose mode - additional):**
```rust
// For each crate, emit:
n!("parse_detail", "{} · local={} · transitive={}", 
    crate_rel_path, local_count, transitive_count);
```

**Example output (normal):**
```
[auto-update        ] phase_deps          
📦 Dependency discovery

[auto-update        ] parse_deps          
Scanning root crate: bin/00_rbee_keeper

[auto-update        ] collect_tomls       
Queued Cargo.toml files: 9

[auto-update        ] parse_batch         
Parsed 21 deps · 12 local path · 9 transitive

[auto-update        ] summary             
✅ Deps ok · 118ms
```

**Example output (verbose - additional lines):**
```
[auto-update        ] parse_detail        
bin/99_shared_crates/daemon-lifecycle · local=3 · transitive=8

[auto-update        ] parse_detail        
bin/99_shared_crates/narration-core · local=0 · transitive=5

[auto-update        ] parse_detail        
bin/05_rbee_keeper_crates/queen-lifecycle · local=2 · transitive=4
```

---

### Phase 4: Build State

**Purpose:** Check binary existence and modification time

**Actions:**
```rust
n!("phase_build", "🛠️ Build state");
n!("check_rebuild", "Binary: {} · mtime: {}", binary_path, mtime_display);
n!("find_binary", "Mode={} · found={}", build_mode, binary_name);
n!("summary", "✅ Build state ok · {}ms", elapsed_ms);
```

**mtime_display format:**
- ISO 8601 timestamp: `2025-10-26T20:30:45Z`
- Or seconds since epoch: `1729975845`

**Example output:**
```
[auto-update        ] phase_build         
🛠️ Build state

[auto-update        ] check_rebuild       
Binary: target/debug/rbee-keeper · mtime: 2025-10-26T20:30:45Z

[auto-update        ] find_binary         
Mode=debug · found=rbee-keeper

[auto-update        ] summary             
✅ Build state ok · 8ms
```

---

### Phase 5: File Scans

**Purpose:** Check source file freshness

**Actions:**
```rust
n!("phase_scan", "🔍 Source freshness checks");

// For each unique directory (deduplicated):
n!("is_dir_newer", "{} · files={} · newer={}", rel_dir, file_count, newer_count);

n!("summary", "Scanned {} dirs · {} files · newer={} · {}ms", 
    dir_count, total_files, total_newer, elapsed_ms);
```

**Important:**
- **Deduplicate directories** - each path appears exactly once
- **Compute counts first** before emitting
- **One line per directory** with aggregated stats

**Example output:**
```
[auto-update        ] phase_scan          
🔍 Source freshness checks

[auto-update        ] is_dir_newer        
bin/99_shared_crates/narration-core · files=63 · newer=0

[auto-update        ] is_dir_newer        
bin/05_rbee_keeper_crates/queen-lifecycle · files=12 · newer=0

[auto-update        ] is_dir_newer        
bin/99_shared_crates/daemon-lifecycle · files=18 · newer=2

[auto-update        ] is_dir_newer        
bin/00_rbee_keeper · files=8 · newer=0

[auto-update        ] summary             
Scanned 12 dirs · 210 files · newer=2 · 84ms
```

---

### Phase 6: Decision

**Purpose:** Determine if rebuild is needed

**Actions (up-to-date):**
```rust
n!("phase_decision", "📑 Rebuild decision");
n!("up_to_date", "✅ {} is up-to-date", binary_name);
n!("summary", "✅ No rebuild needed · {}ms", elapsed_ms);
```

**Actions (needs rebuild):**
```rust
n!("phase_decision", "📑 Rebuild decision");
n!("needs_rebuild", "⚠️ Rebuild required · {}", reason);
n!("summary", "⚠️ Rebuild needed · {}ms", elapsed_ms);
```

**Reason format examples:**
- `"bin/99_shared_crates/daemon-lifecycle has 2 newer files"`
- `"Binary not found"`
- `"3 directories have newer files"`

**Example output (up-to-date):**
```
[auto-update        ] phase_decision      
📑 Rebuild decision

[auto-update        ] up_to_date          
✅ rbee-keeper is up-to-date

[auto-update        ] summary             
✅ No rebuild needed · 3ms
```

**Example output (needs rebuild):**
```
[auto-update        ] phase_decision      
📑 Rebuild decision

[auto-update        ] needs_rebuild       
⚠️ Rebuild required · bin/99_shared_crates/daemon-lifecycle has 2 newer files

[auto-update        ] summary             
⚠️ Rebuild needed · 3ms
```

---

## Complete Flow Example

### Normal Mode (No Verbose)

```
[auto-update        ] phase_init          
🚧 Initializing auto-updater for rbee-keeper

[auto-update        ] init                
Mode: debug · Binary: rbee-keeper · Source: bin/00_rbee_keeper

[auto-update        ] summary             
✅ Init ok · 2ms

[auto-update        ] phase_workspace     
🧭 Workspace detection

[auto-update        ] find_workspace      
Searching for workspace root · depth: 3

[auto-update        ] workspace_found     
Workspace: /home/user/Projects/llama-orch

[auto-update        ] summary             
✅ Workspace ok · 5ms

[auto-update        ] phase_deps          
📦 Dependency discovery

[auto-update        ] parse_deps          
Scanning root crate: bin/00_rbee_keeper

[auto-update        ] collect_tomls       
Queued Cargo.toml files: 9

[auto-update        ] parse_batch         
Parsed 21 deps · 12 local path · 9 transitive

[auto-update        ] summary             
✅ Deps ok · 118ms

[auto-update        ] phase_build         
🛠️ Build state

[auto-update        ] check_rebuild       
Binary: target/debug/rbee-keeper · mtime: 2025-10-26T20:30:45Z

[auto-update        ] find_binary         
Mode=debug · found=rbee-keeper

[auto-update        ] summary             
✅ Build state ok · 8ms

[auto-update        ] phase_scan          
🔍 Source freshness checks

[auto-update        ] is_dir_newer        
bin/99_shared_crates/narration-core · files=63 · newer=0

[auto-update        ] is_dir_newer        
bin/05_rbee_keeper_crates/queen-lifecycle · files=12 · newer=0

[auto-update        ] is_dir_newer        
bin/99_shared_crates/daemon-lifecycle · files=18 · newer=2

[auto-update        ] is_dir_newer        
bin/00_rbee_keeper · files=8 · newer=0

[auto-update        ] summary             
Scanned 12 dirs · 210 files · newer=2 · 84ms

[auto-update        ] phase_decision      
📑 Rebuild decision

[auto-update        ] needs_rebuild       
⚠️ Rebuild required · bin/99_shared_crates/daemon-lifecycle has 2 newer files

[auto-update        ] summary             
⚠️ Rebuild needed · 3ms
```

**Total:** 24 lines (6 phases × ~4 lines each)

---

## Implementation Guide

### 1. Action Map

Replace current ad-hoc action names with the exact set defined above:

**Phase actions:**
- `phase_init`
- `phase_workspace`
- `phase_deps`
- `phase_build`
- `phase_scan`
- `phase_decision`

**Regular actions:**
- `init`
- `find_workspace`
- `workspace_found`
- `parse_deps`
- `collect_tomls`
- `parse_batch`
- `parse_detail` (verbose only)
- `check_rebuild`
- `find_binary`
- `is_dir_newer`
- `up_to_date`
- `needs_rebuild`
- `summary`

### 2. Phase Functions

Wrap each major section in a helper function:

```rust
async fn phase_init(config: &Config) -> Result<()> {
    let start = Instant::now();
    
    n!("phase_init", "🚧 Initializing auto-updater for {}", config.binary_name);
    
    // ... phase logic ...
    n!("init", "Mode: {} · Binary: {} · Source: {}", 
        config.mode, config.binary_name, config.source_dir);
    
    let elapsed = start.elapsed().as_millis();
    n!("summary", "✅ Init ok · {}ms", elapsed);
    
    Ok(())
}

async fn phase_workspace() -> Result<PathBuf> {
    let start = Instant::now();
    
    n!("phase_workspace", "🧭 Workspace detection");
    
    // ... workspace detection logic ...
    n!("find_workspace", "Searching for workspace root · depth: {}", depth);
    
    let workspace = find_workspace_root()?;
    n!("workspace_found", "Workspace: {}", workspace.display());
    
    let elapsed = start.elapsed().as_millis();
    n!("summary", "✅ Workspace ok · {}ms", elapsed);
    
    Ok(workspace)
}

// Similar for other phases...
```

### 3. Batch Parsing

**Collect first, then emit:**

```rust
async fn phase_deps(root_crate: &Path) -> Result<DependencyTree> {
    let start = Instant::now();
    
    n!("phase_deps", "📦 Dependency discovery");
    n!("parse_deps", "Scanning root crate: {}", root_crate.display());
    
    // Collect all Cargo.toml paths
    let toml_files = collect_cargo_tomls(root_crate)?;
    n!("collect_tomls", "Queued Cargo.toml files: {}", toml_files.len());
    
    // Parse all dependencies
    let mut total_deps = 0;
    let mut local_deps = 0;
    let mut transitive_deps = 0;
    let mut details = Vec::new();
    
    for toml_path in &toml_files {
        let (local, trans) = parse_cargo_toml(toml_path)?;
        total_deps += 1;
        local_deps += local.len();
        transitive_deps += trans.len();
        
        // Store for verbose output
        details.push((toml_path.clone(), local.len(), trans.len()));
    }
    
    // Emit batch summary
    n!("parse_batch", "Parsed {} deps · {} local path · {} transitive", 
        total_deps, local_deps, transitive_deps);
    
    // Emit details if verbose
    if is_verbose() {
        for (path, local, trans) in details {
            let rel_path = path.strip_prefix(workspace).unwrap_or(&path);
            n!("parse_detail", "{} · local={} · transitive={}", 
                rel_path.display(), local, trans);
        }
    }
    
    let elapsed = start.elapsed().as_millis();
    n!("summary", "✅ Deps ok · {}ms", elapsed);
    
    Ok(dependency_tree)
}
```

### 4. Scan Deduplication

**Deduplicate and aggregate before emitting:**

```rust
async fn phase_scan(dirs: &[PathBuf], binary_mtime: SystemTime) -> Result<ScanResult> {
    let start = Instant::now();
    
    n!("phase_scan", "🔍 Source freshness checks");
    
    // Deduplicate directories
    let mut dir_stats: HashMap<PathBuf, (usize, usize)> = HashMap::new();
    
    for dir in dirs {
        if dir_stats.contains_key(dir) {
            continue; // Skip duplicates
        }
        
        let (file_count, newer_count) = scan_directory(dir, binary_mtime)?;
        dir_stats.insert(dir.clone(), (file_count, newer_count));
    }
    
    // Emit one line per unique directory
    let mut total_files = 0;
    let mut total_newer = 0;
    
    for (dir, (files, newer)) in &dir_stats {
        let rel_dir = dir.strip_prefix(workspace).unwrap_or(dir);
        n!("is_dir_newer", "{} · files={} · newer={}", 
            rel_dir.display(), files, newer);
        
        total_files += files;
        total_newer += newer;
    }
    
    let elapsed = start.elapsed().as_millis();
    n!("summary", "Scanned {} dirs · {} files · newer={} · {}ms", 
        dir_stats.len(), total_files, total_newer, elapsed);
    
    Ok(ScanResult { total_files, total_newer })
}
```

### 5. Decision Gate

**Compute reason before emitting:**

```rust
async fn phase_decision(scan_result: &ScanResult, binary_name: &str) -> Result<Decision> {
    let start = Instant::now();
    
    n!("phase_decision", "📑 Rebuild decision");
    
    let decision = if scan_result.total_newer == 0 {
        n!("up_to_date", "✅ {} is up-to-date", binary_name);
        Decision::UpToDate
    } else {
        // Find which directory has newer files for reason
        let reason = if let Some((dir, newer)) = find_first_newer_dir() {
            format!("{} has {} newer files", dir.display(), newer)
        } else {
            format!("{} directories have newer files", scan_result.dirs_with_newer)
        };
        
        n!("needs_rebuild", "⚠️ Rebuild required · {}", reason);
        Decision::NeedsRebuild(reason)
    };
    
    let elapsed = start.elapsed().as_millis();
    let status = if matches!(decision, Decision::UpToDate) { "✅ No" } else { "⚠️" };
    n!("summary", "{} rebuild needed · {}ms", status, elapsed);
    
    Ok(decision)
}
```

### 6. Context Propagation

**Wrap entire flow if job_id exists:**

```rust
pub async fn check_and_rebuild(
    binary_name: &str,
    source_dir: &str,
    job_id: Option<&str>,
) -> Result<bool> {
    // Create context if job_id provided
    let ctx = job_id.map(|jid| NarrationContext::new().with_job_id(jid));
    
    let impl_fn = async {
        // All phases run here - every n!() gets the context
        phase_init(&config).await?;
        let workspace = phase_workspace().await?;
        let deps = phase_deps(&root_crate).await?;
        let binary = phase_build(&binary_name, &workspace).await?;
        let scan = phase_scan(&deps.directories, binary.mtime).await?;
        let decision = phase_decision(&scan, binary_name).await?;
        
        Ok(matches!(decision, Decision::NeedsRebuild(_)))
    };
    
    // Execute with or without context
    if let Some(ctx) = ctx {
        with_narration_context(ctx, impl_fn).await
    } else {
        impl_fn.await
    }
}
```

### 7. Verbose Flag

**Check environment or config:**

```rust
fn is_verbose() -> bool {
    std::env::var("RBEE_VERBOSE")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false)
}

// Or from CLI args:
struct AutoUpdateConfig {
    verbose: bool,
    // ... other fields
}
```

---

## Before vs After Comparison

### Before (Old Pattern)

```
[auto_update] parse_cargo_toml 📄 Parsing /home/user/Projects/llama-orch/bin/99_shared_crates/daemon-lifecycle/Cargo.toml
[auto_update] parse_cargo_toml 📄 Parsing /home/user/Projects/llama-orch/bin/99_shared_crates/auto-update/Cargo.toml
[auto_update] parse_cargo_toml 📄 Parsing /home/user/Projects/llama-orch/bin/99_shared_crates/narration-core/Cargo.toml
[auto_update] parse_cargo_toml 📄 Parsing /home/user/Projects/llama-orch/bin/05_rbee_keeper_crates/queen-lifecycle/Cargo.toml
[auto_update] parse_cargo_toml 📄 Parsing /home/user/Projects/llama-orch/bin/99_shared_crates/timeout-enforcer/Cargo.toml
[auto_update] parse_cargo_toml 📄 Parsing /home/user/Projects/llama-orch/bin/99_shared_crates/job-registry/Cargo.toml
[auto_update] parse_cargo_toml 📄 Parsing /home/user/Projects/llama-orch/bin/15_queen_rbee_crates/hive-registry/Cargo.toml
[auto_update] parse_cargo_toml 📄 Parsing /home/user/Projects/llama-orch/bin/99_shared_crates/ssh-config/Cargo.toml
[auto_update] parse_cargo_toml 📄 Parsing /home/user/Projects/llama-orch/bin/00_rbee_keeper/Cargo.toml
[auto_update] check_directory 🔍 Checking /home/user/Projects/llama-orch/bin/99_shared_crates/narration-core
[auto_update] check_directory 🔍 Checking /home/user/Projects/llama-orch/bin/05_rbee_keeper_crates/queen-lifecycle
[auto_update] check_directory 🔍 Checking /home/user/Projects/llama-orch/bin/99_shared_crates/daemon-lifecycle
...
```

**Problems:**
- 🔴 Repetitive per-file lines (9+ lines for parsing)
- 🔴 No clear phase boundaries
- 🔴 No summaries or totals
- 🔴 Hard to scan for key information
- 🔴 Mixed emoji placement

### After (New Pattern)

```
[auto-update        ] phase_deps          
📦 Dependency discovery

[auto-update        ] parse_deps          
Scanning root crate: bin/00_rbee_keeper

[auto-update        ] collect_tomls       
Queued Cargo.toml files: 9

[auto-update        ] parse_batch         
Parsed 21 deps · 12 local path · 9 transitive

[auto-update        ] summary             
✅ Deps ok · 118ms

[auto-update        ] phase_scan          
🔍 Source freshness checks

[auto-update        ] is_dir_newer        
bin/99_shared_crates/narration-core · files=63 · newer=0

[auto-update        ] is_dir_newer        
bin/05_rbee_keeper_crates/queen-lifecycle · files=12 · newer=0

[auto-update        ] is_dir_newer        
bin/99_shared_crates/daemon-lifecycle · files=18 · newer=2

[auto-update        ] summary             
Scanned 12 dirs · 210 files · newer=2 · 84ms
```

**Benefits:**
- ✅ Batched operations (1 line vs 9+ lines)
- ✅ Clear phase boundaries with banners
- ✅ Summaries with totals and timing
- ✅ Easy to scan for key information
- ✅ Consistent emoji usage
- ✅ Detail available via verbose flag

---

## Acceptance Checklist

- [ ] No repetitive per-file "Parsing …" lines unless `--verbose`
- [ ] Exactly six phases, each with `phase_*` and `summary`
- [ ] One line per directory for freshness scans (deduplicated)
- [ ] Final decision appears under `phase_decision` as `up_to_date` or `needs_rebuild`
- [ ] All lines emitted via `n!()`
- [ ] Headers + messages match V2 format (two-line, bold header)
- [ ] Emojis used consistently per specification
- [ ] Context propagation works if `job_id` provided
- [ ] Verbose flag controls detail level
- [ ] All action names match specification exactly

---

## Testing

### Test Cases

1. **Normal mode (no verbose):**
   - Should show ~24 lines total (6 phases × ~4 lines)
   - No `parse_detail` lines
   - Batch summaries only

2. **Verbose mode:**
   - Should show additional `parse_detail` lines
   - One per Cargo.toml file parsed

3. **Up-to-date binary:**
   - Should show `up_to_date` action
   - Summary: "✅ No rebuild needed"

4. **Needs rebuild:**
   - Should show `needs_rebuild` action with reason
   - Summary: "⚠️ Rebuild needed"

5. **With job_id:**
   - All narrations should include job_id in context
   - Should route to SSE if registered

6. **Without job_id:**
   - Should work normally
   - Output to stderr only

### Manual Testing

```bash
# Normal mode
cargo build --bin rbee-keeper

# Verbose mode
RBEE_VERBOSE=1 cargo build --bin rbee-keeper

# With job_id (in queen-rbee context)
# Should see narrations in SSE stream
```

---

## Migration Path

### Step 1: Update Action Names

Replace all current action names with the standardized set:
- `parse_cargo_toml` → `parse_batch` (with batching)
- `check_directory` → `is_dir_newer`
- Add phase actions: `phase_init`, `phase_workspace`, etc.

### Step 2: Add Phase Functions

Create wrapper functions for each phase that:
- Emit `phase_*` at entry
- Perform phase logic
- Emit `summary` at exit with timing

### Step 3: Implement Batching

- Collect items first
- Emit counts/summaries
- Add verbose detail behind flag

### Step 4: Add Context Support

- Accept optional `job_id` parameter
- Wrap in `with_narration_context` if provided

### Step 5: Test & Verify

- Run with and without verbose
- Check line counts
- Verify phase structure
- Test with job_id

---

## Summary

The Auto-Updater Narration Format V2 provides:

✅ **Clear phase structure** - 6 distinct phases with banners  
✅ **Batched operations** - No repetitive per-file lines  
✅ **Summaries with timing** - Each phase shows elapsed time  
✅ **Consistent action names** - Standardized snake_case  
✅ **Emoji standards** - Clear visual indicators  
✅ **Verbose control** - Detail on demand  
✅ **Context propagation** - Automatic job_id routing  
✅ **V2 format compliance** - Uses `n!()` macro throughout  

**Result:** ~24 lines (normal) vs 50+ lines (old), with better clarity and scannability.
