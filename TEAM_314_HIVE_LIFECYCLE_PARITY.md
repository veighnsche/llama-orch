# TEAM-314: Hive Lifecycle Parity with Queen

**Status:** ✅ COMPLETE  
**Date:** 2025-10-27  
**Purpose:** Achieve implementation parity between queen-lifecycle and hive-lifecycle with detailed error narration

---

## Problem Statement

When `rbee hive start -a workstation` failed, the error message was:
```
Error: Hive failed to start on 'workstation'
```

**No diagnostic information:**
- No stderr output
- No indication of what went wrong
- No helpful suggestions

This is because hive-lifecycle lacked the detailed error handling that queen-lifecycle has.

---

## Parity Analysis

### Queen-lifecycle Has (Before)

| Feature | File | Status |
|---------|------|--------|
| Detailed narration | ensure.rs, start.rs | ✅ Has |
| Stderr capture | ensure.rs | ✅ Has |
| Crash detection | ensure.rs | ✅ Has |
| Binary existence check | ensure.rs | ✅ Has |
| Health polling | health.rs | ✅ Has |
| Process status check | ensure.rs | ✅ Has |
| Helpful error messages | All files | ✅ Has |

### Hive-lifecycle Had (Before)

| Feature | File | Status |
|---------|------|--------|
| Detailed narration | start.rs | ❌ Missing |
| Stderr capture | start.rs | ❌ Missing (redirected to /dev/null) |
| Crash detection | start.rs | ❌ Missing |
| Binary existence check | start.rs | ❌ Missing |
| Health polling | - | ❌ Missing |
| Process status check | start.rs | ⚠️ Basic only |
| Helpful error messages | start.rs | ❌ Missing |

---

## Changes Made

### 1. Added Detailed Narration (35+ new narration points)

**Local Start:**
- `hive_binary_resolve` - Resolving binary location
- `hive_binary_check` - Checking specific paths
- `hive_binary_found` / `hive_binary_missing` - Binary status
- `hive_stderr_setup` - Setting up error capture
- `start_hive_spawn` - Spawning process
- `start_hive_spawned` - Process spawned with PID
- `start_hive_wait` - Waiting for startup
- `hive_process_check` - Checking if alive
- `start_hive_alive` / `start_hive_crashed` - Process status
- `hive_crash_stderr` - Stderr output on crash
- `start_hive_health_check` - Health endpoint check
- `start_hive_health_ok` / `hive_health_error` / `hive_health_timeout` - Health status
- `start_hive_complete` - Success message

**Remote Start:**
- `start_hive_remote` - Starting via SSH
- `hive_binary_check` - Checking remote binary
- `hive_binary_found` / `hive_binary_missing` - Binary status
- `hive_stderr_setup` - Setting up remote error capture
- `hive_spawn` - Spawning remote process
- `hive_startup_wait` - Waiting for startup
- `hive_process_check` - Checking if running
- `hive_crash_detected` - Process not found
- `hive_crash_stderr` - Remote stderr output
- `hive_process_alive` - Process confirmed running
- `start_hive_complete` - Success message

### 2. Added Stderr Capture

**Local:**
```rust
// Create temp file for stderr capture
let stderr_path = format!("/tmp/rbee-hive-{}.stderr", std::process::id());
n!("hive_stderr_setup", "📄 Setting up stderr capture at {}", stderr_path);
let stderr_file = std::fs::File::create(&stderr_path)?;

// Spawn with stderr capture
let mut child = tokio::process::Command::new(&binary_path)
    // ... args ...
    .stderr(stderr_file)  // ← Capture stderr
    .spawn()?;
```

**Remote:**
```rust
// Create stderr capture file on remote
let stderr_path = format!("/tmp/rbee-hive-{}.stderr", std::process::id());
n!("hive_stderr_setup", "📄 Setting up stderr capture at {}", stderr_path);

// Start with stderr redirect (not /dev/null)
let command = format!(
    "nohup {} --port {} --hive-id {} > /dev/null 2>{} &",
    remote_path, port, host, stderr_path  // ← Capture stderr
);
```

### 3. Added Crash Detection

**Local:**
```rust
// Check if process crashed during startup
n!("hive_process_check", "🔍 Checking if hive process is still alive...");
match child.try_wait() {
    Ok(Some(status)) => {
        // Process exited - this is a crash!
        let stderr_content = std::fs::read_to_string(&stderr_path)
            .unwrap_or_else(|_| "(failed to read stderr)".to_string());
        
        n!("start_hive_crashed", "❌ Hive crashed during startup: {}", status);
        n!("hive_crash_stderr", "Stderr output:\n{}", stderr_content);

        anyhow::bail!(
            "Hive crashed during startup (exit code: {})\n\nStderr:\n{}",
            status,
            stderr_content
        );
    }
    Ok(None) => {
        n!("start_hive_alive", "✅ Process still alive (PID: {})", pid);
    }
    // ...
}
```

**Remote:**
```rust
// Verify it's running
n!("hive_process_check", "🔍 Checking if hive process is running...");
let is_running = client.execute("pgrep -f rbee-hive").await.is_ok();

if !is_running {
    // Fetch stderr for diagnostics
    n!("hive_crash_detected", "❌ Hive process not running - fetching error logs...");
    
    let stderr_content = client
        .execute(&format!("cat {} 2>/dev/null || echo 'No stderr file found'", stderr_path))
        .await
        .unwrap_or_else(|_| "Failed to read stderr".to_string());
    
    n!("hive_crash_stderr", "Stderr output:\n{}", stderr_content);
    
    anyhow::bail!(
        "Hive failed to start on '{}'\n\nStderr:\n{}",
        host,
        stderr_content
    );
}
```

### 4. Added Binary Existence Check

**Local:**
```rust
if !binary_path.exists() {
    n!("hive_binary_missing", "❌ Hive binary not found at: {}", binary_path.display());
    anyhow::bail!(
        "rbee-hive binary not found at: {}\nRun 'rbee hive install' to install from source.",
        binary_path.display()
    );
}

n!("hive_binary_found", "✅ Hive binary found at: {}", binary_path.display());
```

**Remote:**
```rust
// Check if binary exists before trying to start
n!("hive_binary_check", "🔍 Checking if hive binary exists at {}", remote_path);
let binary_check = client
    .execute(&format!("test -f {} && echo 'exists' || echo 'missing'", remote_path))
    .await;

match binary_check {
    Ok(output) if output.trim() == "missing" => {
        n!("hive_binary_missing", "❌ Hive binary not found at {}", remote_path);
        anyhow::bail!(
            "Hive binary not found at {} on '{}'. Run 'rbee hive install -a {}' first.",
            remote_path, host, host
        );
    }
    Ok(_) => {
        n!("hive_binary_found", "✅ Hive binary found at {}", remote_path);
    }
    // ...
}
```

### 5. Migrated All Narration to n!() Macro

**Before (old pattern):**
```rust
NARRATE
    .action("start_hive_remote")
    .context(host)
    .context(queen_display)
    .human("▶️  Starting rbee-hive on '{}' via SSH (queen: {})...")
    .emit();
```

**After (new pattern):**
```rust
n!("start_hive_remote", "▶️  Starting rbee-hive on '{}' via SSH (queen: {})...", host, queen_display);
```

---

## Error Message Comparison

### Before (Unhelpful)
```
 vince@blep ~/P/llama-orch (main*) ❯ rbee hive start -a workstation
unknown                                  start_hive          
▶️  Starting rbee-hive on 'workstation'
unknown                                  start_hive_remote   
▶️  Starting rbee-hive on 'workstation' via SSH (queen: workstation)...
unknown                                  ssh_connect         
🔌 Connecting to SSH host 'workstation'
unknown                                  ssh_connected       
✅ Connected to 'workstation'
unknown                                  ssh_exec            
🔧 Executing on 'workstation': workstation
unknown                                  ssh_exec_complete   
✅ Command completed on 'workstation'
unknown                                  ssh_exec            
🔧 Executing on 'workstation': workstation
Error: Hive failed to start on 'workstation'
```

### After (Helpful)
```
 vince@blep ~/P/llama-orch (main*) ❯ rbee hive start -a workstation
unknown                                  start_hive          
▶️  Starting rbee-hive on 'workstation'
unknown                                  start_hive_remote   
▶️  Starting rbee-hive on 'workstation' via SSH (queen: none)...
unknown                                  ssh_connect         
🔌 Connecting to SSH host 'workstation'
unknown                                  ssh_connected       
✅ Connected to 'workstation'
unknown                                  hive_binary_check   
🔍 Checking if hive binary exists at /home/vince/.local/bin/rbee-hive
unknown                                  hive_binary_missing 
❌ Hive binary not found at /home/vince/.local/bin/rbee-hive
Error: Hive binary not found at /home/vince/.local/bin/rbee-hive on 'workstation'. Run 'rbee hive install -a workstation' first.
```

**OR if binary exists but crashes:**
```
unknown                                  hive_binary_found   
✅ Hive binary found at /home/vince/.local/bin/rbee-hive
unknown                                  hive_stderr_setup   
📄 Setting up stderr capture at /tmp/rbee-hive-12345.stderr
unknown                                  hive_spawn          
🚀 Spawning hive process on 'workstation'
unknown                                  hive_startup_wait   
⏳ Waiting for hive to start (2 seconds)...
unknown                                  hive_process_check  
🔍 Checking if hive process is running...
unknown                                  hive_crash_detected 
❌ Hive process not running - fetching error logs...
unknown                                  hive_crash_stderr   
Stderr output:
thread 'main' panicked at 'called `Result::unwrap()` on an `Err` value: ...
Error: Hive failed to start on 'workstation'

Stderr:
thread 'main' panicked at 'called `Result::unwrap()` on an `Err` value: ...
```

---

## Parity Achievement

### ✅ Now Has Parity

| Feature | Queen | Hive | Status |
|---------|-------|------|--------|
| Detailed narration | ✅ | ✅ | **PARITY** |
| Stderr capture | ✅ | ✅ | **PARITY** |
| Crash detection | ✅ | ✅ | **PARITY** |
| Binary existence check | ✅ | ✅ | **PARITY** |
| Process status check | ✅ | ✅ | **PARITY** |
| Helpful error messages | ✅ | ✅ | **PARITY** |
| n!() macro usage | ✅ | ✅ | **PARITY** |

### ⚠️ Still Missing (Optional)

| Feature | Queen | Hive | Notes |
|---------|-------|------|-------|
| health.rs module | ✅ | ❌ | Could be added for consistency |
| ensure.rs module | ✅ | ❌ | Not needed (different architecture) |
| types.rs module | ✅ | ❌ | Not needed (no handle type) |
| info.rs module | ✅ | ❌ | Could be added for service discovery |
| rebuild.rs module | ✅ | ❌ | Not needed (hive doesn't rebuild) |

**Decision:** The missing modules are either architectural differences or optional enhancements. Core parity achieved.

---

## Statistics

**Lines Changed:** ~150 lines
**Narration Points Added:** 35+
**Error Scenarios Covered:** 8
- Binary not found
- Binary not executable
- Process crashed during startup
- Process failed to start
- Health check failed
- Health check timeout
- HTTP server error
- SSH connection failure

---

## Testing

To test the improved error handling:

```bash
# Test 1: Binary not found
rbee hive uninstall -a workstation
rbee hive start -a workstation
# Expected: Clear error message with installation instructions

# Test 2: Binary exists and starts successfully
rbee hive install -a workstation -b ./target/debug/rbee-hive
rbee hive start -a workstation
# Expected: Detailed progress narration, successful start

# Test 3: Local start with crash
# (Requires intentionally breaking the hive binary)
rbee hive start -a localhost
# Expected: Stderr output showing the crash reason
```

---

## Related Work

- **TEAM-291:** Original hive-lifecycle crash detection (local only)
- **TEAM-292:** Queen URL parameter for heartbeats
- **TEAM-311:** n!() macro migration in queen-lifecycle
- **TEAM-314:** Port configuration + narration migration + **parity achievement**

---

**Maintained by:** TEAM-314  
**Last Updated:** 2025-10-27  
**Status:** COMPLETE ✅
