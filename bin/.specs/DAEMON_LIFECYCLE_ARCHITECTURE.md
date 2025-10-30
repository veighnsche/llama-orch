# daemon-lifecycle Architecture: Feature Flags vs Separate Crates

**Date:** Oct 30, 2025  
**Status:** 🏗️ ARCHITECTURE DECISION  
**Related:** `CGROUP_INTEGRATION_PLAN.md`

---

## 🎯 Problem Statement

We have **3 different execution modes** for daemon lifecycle management:

1. **SSH Mode** - Execute commands on remote machines via SSH/SCP
2. **Local Mode** - Execute commands directly on localhost (bypass SSH overhead)
3. **cgroup Mode** - Start/stop processes in cgroup v2 slices (Linux-specific)

**Question:** Should we use:
- **Option A:** Feature flags in one crate?
- **Option B:** Three separate crates?
- **Option C:** One crate with runtime detection?

---

## 📊 Current State Analysis

### **Current Implementation**

**Crate:** `bin/99_shared_crates/daemon-lifecycle/`

**Current Features:**
```toml
[features]
tauri = ["dep:specta"]  # Only for Tauri bindings
```

**Current Architecture:**
```rust
// SSH is always available
pub async fn ssh_exec(ssh_config: &SshConfig, command: &str) -> Result<String> {
    // Runtime detection: Bypass SSH for localhost
    if ssh_config.is_localhost() {
        return local_exec(command).await;  // Direct execution
    }
    
    // SSH execution
    Command::new("ssh")...
}
```

**Key Insight:** Already uses **runtime detection** for SSH vs Local!

### **File Structure**

```
daemon-lifecycle/
├── src/
│   ├── lib.rs              # SshConfig with is_localhost()
│   ├── start.rs            # Start daemon
│   ├── stop.rs             # Stop daemon
│   ├── shutdown.rs         # Shutdown daemon
│   ├── install.rs          # Install binary
│   ├── uninstall.rs        # Uninstall binary
│   ├── build.rs            # Build binary
│   ├── rebuild.rs          # Rebuild + hot-reload
│   ├── status.rs           # Check status
│   └── utils/
│       ├── ssh.rs          # SSH execution (calls local.rs for localhost)
│       ├── local.rs        # Direct execution (no SSH)
│       ├── poll.rs         # Health polling
│       ├── binary.rs       # Binary detection
│       └── serde.rs        # Serialization helpers
```

**Observation:** SSH and Local are **already integrated** with runtime detection!

---

## 🔍 Analysis: Three Execution Modes

### **1. SSH Mode**

**Purpose:** Execute commands on remote machines

**Implementation:**
```rust
// Uses: ssh, scp commands
Command::new("ssh")
    .arg("-p").arg(port)
    .arg(format!("{}@{}", user, hostname))
    .arg(command)
```

**Dependencies:**
- External: `ssh`, `scp` binaries (must be installed)
- Rust: `tokio::process::Command`

**Platform:** Cross-platform (Linux, macOS, Windows with OpenSSH)

---

### **2. Local Mode**

**Purpose:** Bypass SSH overhead for localhost operations

**Implementation:**
```rust
// Direct execution
Command::new("sh")
    .arg("-c")
    .arg(command)
```

**Dependencies:**
- External: `sh` shell
- Rust: `tokio::process::Command`

**Platform:** Cross-platform (Linux, macOS, Windows with WSL/Git Bash)

**Current Integration:**
```rust
// Already integrated via runtime detection
if ssh_config.is_localhost() {
    return local_exec(command).await;  // Automatic!
}
```

---

### **3. cgroup Mode**

**Purpose:** Start/stop processes in cgroup v2 slices for monitoring

**Implementation:**
```rust
// Linux with systemd: systemd-run command
Command::new("systemd-run")
    .arg("--user")
    .arg("--scope")
    .arg("--slice=rbee.slice")
    .arg(format!("--unit={}-{}", service, instance))
    .arg(binary_path)

// Linux without systemd: Manual cgroup management
// Create cgroup: mkdir -p /sys/fs/cgroup/rbee.slice/{service}/{instance}
// Add PID: echo $PID > /sys/fs/cgroup/rbee.slice/{service}/{instance}/cgroup.procs

// macOS: No cgroup support - track PIDs in memory
// Windows: No cgroup support - track PIDs in memory
```

**Dependencies:**
- External (Linux): `systemd-run` binary (optional) OR manual cgroup access
- External (macOS/Windows): None (PID tracking only)
- Rust: `tokio::process::Command`

**Platform Support:**
- **Linux with systemd:** ✅ Full cgroup support via systemd-run
- **Linux without systemd:** ✅ Manual cgroup management (requires root or cgroup delegation)
- **macOS:** ⚠️ No cgroup - PID tracking only (no resource limits)
- **Windows:** ⚠️ No cgroup - PID tracking only (no resource limits)

**Fallback:** Always falls back to `nohup` + PID tracking if cgroup unavailable

---

## 🎨 Architecture Options

### **Option A: Feature Flags** ❌ **NOT RECOMMENDED**

```toml
[features]
default = ["ssh", "local"]
ssh = []
local = []
cgroup = []
```

**Pros:**
- Compile-time optimization (exclude unused code)
- Smaller binary size

**Cons:**
- ❌ **Breaks runtime flexibility** - Can't switch modes at runtime
- ❌ **Complicates builds** - Need different builds for different modes
- ❌ **User confusion** - Which features to enable?
- ❌ **Testing nightmare** - Need to test all feature combinations
- ❌ **Not needed** - All modes use same dependencies (`tokio::process::Command`)

**Verdict:** ❌ **Overkill for this use case**

---

### **Option B: Three Separate Crates** ❌ **NOT RECOMMENDED**

```
daemon-lifecycle-ssh/       # SSH mode only
daemon-lifecycle-local/     # Local mode only
daemon-lifecycle-cgroup/    # cgroup mode only
```

**Pros:**
- Clear separation of concerns
- Independent versioning

**Cons:**
- ❌ **Massive code duplication** - 90% of code is shared
- ❌ **Maintenance nightmare** - Fix bugs in 3 places
- ❌ **User confusion** - Which crate to use?
- ❌ **No runtime switching** - Can't mix modes
- ❌ **Violates DRY principle**

**Verdict:** ❌ **Terrible idea**

---

### **Option C: One Crate with Runtime Detection** ✅ **RECOMMENDED**

```
daemon-lifecycle/
├── All modes in one crate
├── Runtime detection (already implemented for SSH/Local)
├── Optional cgroup support via config
└── Automatic fallback (cgroup → nohup)
```

**Pros:**
- ✅ **Already implemented** for SSH/Local (just extend it)
- ✅ **Runtime flexibility** - Switch modes dynamically
- ✅ **Single codebase** - Fix bugs once
- ✅ **User-friendly** - One crate, simple API
- ✅ **Automatic fallback** - Works everywhere
- ✅ **No feature flag complexity**

**Cons:**
- Slightly larger binary (includes all modes)
- But: Difference is negligible (~50 KB)

**Verdict:** ✅ **Best approach**

---

## ✅ Recommended Architecture

### **One Crate with Runtime Detection**

**Strategy:** Extend existing runtime detection pattern to include cgroup mode.

### **Implementation Plan**

#### **1. Add CgroupConfig (Optional)**

```rust
// In start.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpDaemonConfig {
    pub daemon_name: String,
    pub health_url: String,
    // ... existing fields ...
    
    /// Optional: cgroup configuration
    /// If present, use systemd-run; if absent, use nohup
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cgroup_config: Option<CgroupConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CgroupConfig {
    /// Service name (e.g., "llm", "hive")
    pub service: String,
    
    /// Instance identifier (e.g., "8080", "main")
    pub instance: String,
    
    /// Optional: CPU limit (e.g., "200%" = 2 cores)
    pub cpu_limit: Option<String>,
    
    /// Optional: Memory limit (e.g., "4G")
    pub memory_limit: Option<String>,
}
```

#### **2. Update start.rs - Runtime Detection**

```rust
pub async fn start_daemon(start_config: StartConfig) -> Result<u32> {
    // ... find binary ...
    
    // Build start command based on config
    let start_cmd = if let Some(cgroup) = &daemon_config.cgroup_config {
        // Try cgroup mode (systemd-run)
        if is_systemd_available(&ssh_config).await? {
            build_systemd_run_command(binary_path, args, cgroup)
        } else {
            // Fallback to nohup
            tracing::warn!("systemd-run not available, falling back to nohup");
            build_nohup_command(binary_path, args)
        }
    } else {
        // No cgroup config - use nohup (old way)
        build_nohup_command(binary_path, args)
    };
    
    // Execute command (via SSH or local)
    let pid_output = ssh_exec(ssh_config, &start_cmd).await?;
    
    // ... rest of function ...
}

// Helper: Check if systemd-run is available
async fn is_systemd_available(ssh_config: &SshConfig) -> Result<bool> {
    let check_cmd = "which systemd-run && systemctl --user status";
    match ssh_exec(ssh_config, check_cmd).await {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

// Helper: Build systemd-run command
fn build_systemd_run_command(
    binary_path: &str,
    args: &str,
    cgroup: &CgroupConfig,
) -> String {
    let mut cmd = format!(
        "systemd-run --user --scope --slice=rbee.slice --unit={}-{}",
        cgroup.service, cgroup.instance
    );
    
    // Add resource limits if specified
    if let Some(cpu_limit) = &cgroup.cpu_limit {
        cmd.push_str(&format!(" --property=CPUQuota={}", cpu_limit));
    }
    if let Some(mem_limit) = &cgroup.memory_limit {
        cmd.push_str(&format!(" --property=MemoryMax={}", mem_limit));
    }
    
    // Add binary and args
    cmd.push_str(&format!(" {} {}", binary_path, args));
    cmd.push_str(" & echo $!");
    
    cmd
}

// Helper: Build nohup command (fallback)
fn build_nohup_command(binary_path: &str, args: &str) -> String {
    if args.is_empty() {
        format!("nohup {} > /dev/null 2>&1 & echo $!", binary_path)
    } else {
        format!("nohup {} {} > /dev/null 2>&1 & echo $!", binary_path, args)
    }
}
```

#### **3. Update stop.rs - Runtime Detection**

```rust
pub async fn stop_daemon(stop_config: StopConfig) -> Result<()> {
    // ... try HTTP shutdown first ...
    
    // Fallback to SSH-based shutdown
    let shutdown_config = ShutdownConfig {
        daemon_name: daemon_name.to_string(),
        shutdown_url: shutdown_url.to_string(),
        health_url: health_url.to_string(),
        ssh_config: ssh_config.clone(),
        job_id: stop_config.job_id.clone(),
        cgroup_config: stop_config.cgroup_config.clone(),  // Pass cgroup config
    };
    
    shutdown_daemon(shutdown_config).await?;
    Ok(())
}
```

#### **4. Update shutdown.rs - Runtime Detection**

```rust
pub async fn shutdown_daemon(shutdown_config: ShutdownConfig) -> Result<()> {
    let daemon_name = &shutdown_config.daemon_name;
    let ssh_config = &shutdown_config.ssh_config;
    
    // Choose shutdown method based on cgroup config
    if let Some(cgroup) = &shutdown_config.cgroup_config {
        // Try systemd stop first
        if is_systemd_available(ssh_config).await? {
            let unit_name = format!("{}-{}.scope", cgroup.service, cgroup.instance);
            let stop_cmd = format!("systemctl --user stop {}", unit_name);
            
            match ssh_exec(ssh_config, &stop_cmd).await {
                Ok(_) => {
                    n!("systemd_stop", "✅ Stopped via systemd");
                    return Ok(());
                }
                Err(e) => {
                    n!("systemd_stop_failed", "⚠️  systemd stop failed: {}", e);
                    // Fall through to cgroup-based kill
                }
            }
        }
        
        // Fallback: Kill all PIDs in cgroup
        shutdown_via_cgroup(ssh_config, cgroup).await?;
    } else {
        // No cgroup config - use pkill (old way)
        shutdown_via_pkill(ssh_config, daemon_name).await?;
    }
    
    Ok(())
}

// Helper: Shutdown via cgroup
async fn shutdown_via_cgroup(
    ssh_config: &SshConfig,
    cgroup: &CgroupConfig,
) -> Result<()> {
    let cgroup_path = format!(
        "/sys/fs/cgroup/rbee.slice/{}/{}",
        cgroup.service, cgroup.instance
    );
    
    // SIGTERM all PIDs in cgroup
    let sigterm_cmd = format!(
        "cat {}/cgroup.procs | xargs kill -TERM 2>/dev/null || true",
        cgroup_path
    );
    ssh_exec(ssh_config, &sigterm_cmd).await?;
    
    // Wait 5s
    tokio::time::sleep(Duration::from_secs(5)).await;
    
    // SIGKILL all PIDs in cgroup
    let sigkill_cmd = format!(
        "cat {}/cgroup.procs | xargs kill -KILL 2>/dev/null || true",
        cgroup_path
    );
    ssh_exec(ssh_config, &sigkill_cmd).await?;
    
    Ok(())
}

// Helper: Shutdown via pkill (fallback)
async fn shutdown_via_pkill(
    ssh_config: &SshConfig,
    daemon_name: &str,
) -> Result<()> {
    // ... existing pkill logic ...
}
```

### **API Examples**

#### **Example 1: SSH Mode (Remote)**

```rust
let ssh = SshConfig::new("192.168.1.100".to_string(), "vince".to_string(), 22);
let config = HttpDaemonConfig::new("rbee-hive", "http://192.168.1.100:7835")
    .with_args(vec!["--port".to_string(), "7835".to_string()]);

let pid = start_daemon(StartConfig {
    ssh_config: ssh,
    daemon_config: config,
    job_id: None,
}).await?;
```

#### **Example 2: Local Mode (Automatic)**

```rust
let ssh = SshConfig::localhost();  // Automatic detection!
let config = HttpDaemonConfig::new("rbee-hive", "http://localhost:7835")
    .with_args(vec!["--port".to_string(), "7835".to_string()]);

let pid = start_daemon(StartConfig {
    ssh_config: ssh,  // Will use local_exec() automatically
    daemon_config: config,
    job_id: None,
}).await?;
```

#### **Example 3: cgroup Mode (Linux)**

```rust
let ssh = SshConfig::localhost();
let config = HttpDaemonConfig::new("rbee-worker", "http://localhost:8080")
    .with_args(vec!["--port".to_string(), "8080".to_string()])
    .with_cgroup(CgroupConfig {
        service: "llm".to_string(),
        instance: "8080".to_string(),
        cpu_limit: Some("200%".to_string()),  // 2 cores
        memory_limit: Some("4G".to_string()),
    });

let pid = start_daemon(StartConfig {
    ssh_config: ssh,
    daemon_config: config,
    job_id: None,
}).await?;

// If systemd-run not available, automatically falls back to nohup
```

---

## 🔄 Runtime Detection Flow

```
start_daemon()
    ↓
Check: cgroup_config present?
    ├─ YES → Check: Platform + systemd availability
    │         ├─ Linux + systemd → Use systemd-run (full cgroup)
    │         ├─ Linux + no systemd → Use manual cgroup (if possible)
    │         ├─ macOS → Use nohup + PID tracking
    │         └─ Windows → Use nohup + PID tracking
    └─ NO  → Use nohup (old way)
    ↓
Execute via ssh_exec()
    ↓
Check: is_localhost()?
    ├─ YES → Use local_exec() (direct)
    └─ NO  → Use ssh (remote)
```

**Key Insight:** Three levels of runtime detection:
1. **Platform detection** - Linux, macOS, Windows
2. **cgroup availability** - systemd-run, manual cgroup, or PID tracking only
3. **SSH vs local** - Based on hostname (already implemented)

---

## 📦 Dependencies

### **Current Dependencies**

```toml
[dependencies]
tokio = { version = "1.0", features = ["full"] }
reqwest = { version = "0.11", features = ["rustls-tls"] }
anyhow = "1.0"
serde = { version = "1.0", features = ["derive"] }
# ... other deps ...
```

### **No New Dependencies Needed!**

All three modes use the same core dependency: `tokio::process::Command`

**External dependencies:**
- SSH mode: `ssh`, `scp` binaries (user must install)
- Local mode: `sh` shell (always available)
- cgroup mode: `systemd-run` binary (optional, auto-detected)

---

## ✅ Decision Matrix

| Criteria | Feature Flags | Separate Crates | Runtime Detection |
|----------|---------------|-----------------|-------------------|
| **Code Duplication** | Medium | High ❌ | Low ✅ |
| **Maintenance** | Medium | Hard ❌ | Easy ✅ |
| **User Experience** | Confusing | Confusing ❌ | Simple ✅ |
| **Runtime Flexibility** | No ❌ | No ❌ | Yes ✅ |
| **Testing Complexity** | High ❌ | High ❌ | Low ✅ |
| **Binary Size** | Smallest | Medium | Slightly larger |
| **Already Implemented** | No | No | Yes (SSH/Local) ✅ |
| **Fallback Support** | No ❌ | No ❌ | Yes ✅ |

**Winner:** ✅ **Runtime Detection** (Option C)

---

## 🎯 Final Recommendation

### **Use One Crate with Runtime Detection**

**Rationale:**
1. ✅ **Already proven** - SSH/Local runtime detection works great
2. ✅ **User-friendly** - One crate, simple API, automatic fallback
3. ✅ **Maintainable** - Single codebase, fix bugs once
4. ✅ **Flexible** - Switch modes at runtime
5. ✅ **Backward compatible** - Old code still works
6. ✅ **No new dependencies** - Uses existing `tokio::process::Command`

### **Implementation Steps**

1. ✅ **Keep existing SSH/Local runtime detection** (already works)
2. ✅ **Add `CgroupConfig` to `HttpDaemonConfig`** (optional field)
3. ✅ **Add `is_systemd_available()` helper** (runtime check)
4. ✅ **Update `start_daemon()`** - Add cgroup branch with fallback
5. ✅ **Update `stop_daemon()`** - Pass cgroup config through
6. ✅ **Update `shutdown_daemon()`** - Add cgroup-aware shutdown
7. ✅ **Add helper functions** - `build_systemd_run_command()`, etc.
8. ✅ **Test on Linux with/without systemd** - Verify fallback works

### **No Feature Flags Needed**

```toml
# Cargo.toml stays simple
[features]
tauri = ["dep:specta"]  # Only for Tauri bindings
```

**Why?** All modes use the same dependencies. Feature flags would add complexity without benefit.

---

## 🌍 Cross-Platform Support

### **Platform Detection**

```rust
// Detect platform at runtime
fn detect_platform() -> Platform {
    if cfg!(target_os = "linux") {
        Platform::Linux
    } else if cfg!(target_os = "macos") {
        Platform::MacOS
    } else if cfg!(target_os = "windows") {
        Platform::Windows
    } else {
        Platform::Unknown
    }
}

enum Platform {
    Linux,
    MacOS,
    Windows,
    Unknown,
}
```

### **Platform-Specific Behavior**

#### **Linux**

**Option 1: systemd-run (Preferred)**
```bash
systemd-run --user --scope --slice=rbee.slice --unit=llm-8080 \
  ./rbee-worker --port 8080 & echo $!
```
- ✅ Full cgroup support
- ✅ Resource limits (CPU, memory)
- ✅ Automatic cleanup
- ⚠️ Requires systemd + user session

**Option 2: Manual cgroup v2 (No systemd)**
```bash
# Create cgroup hierarchy
mkdir -p /sys/fs/cgroup/rbee.slice/llm/8080

# Enable controllers
echo "+cpu +memory +io" > /sys/fs/cgroup/rbee.slice/cgroup.subtree_control
echo "+cpu +memory +io" > /sys/fs/cgroup/rbee.slice/llm/cgroup.subtree_control

# Set resource limits (optional)
echo "200000" > /sys/fs/cgroup/rbee.slice/llm/8080/cpu.max  # 200% = 2 cores
echo "4294967296" > /sys/fs/cgroup/rbee.slice/llm/8080/memory.max  # 4GB

# Start process
nohup ./rbee-worker --port 8080 > /dev/null 2>&1 & echo $!

# Add PID to cgroup
echo $PID > /sys/fs/cgroup/rbee.slice/llm/8080/cgroup.procs
```
- ✅ Full cgroup v2 support without systemd
- ✅ Resource limits work
- ✅ Can read stats from cgroup files
- ⚠️ Requires cgroup delegation (`/etc/systemd/system/user@.service.d/delegate.conf`)
- ⚠️ Manual cleanup needed (remove cgroup dirs on stop)

**Option 3: cgroupfs-mount + manual management**
```bash
# One-time setup (as root or with sudo)
mount -t cgroup2 none /sys/fs/cgroup
mkdir -p /sys/fs/cgroup/rbee.slice
chown -R $USER:$USER /sys/fs/cgroup/rbee.slice

# Then use Option 2 commands
```
- ✅ Works on any Linux with cgroup v2 kernel support
- ✅ No systemd required
- ⚠️ Requires initial setup as root

**Option 4: /proc-based monitoring (Universal fallback)**
```bash
# Start process normally
nohup ./rbee-worker --port 8080 > /dev/null 2>&1 & echo $!

# Monitor via /proc/$PID/
# - /proc/$PID/stat for CPU
# - /proc/$PID/status for memory
# - /proc/$PID/io for I/O
```
- ✅ Works everywhere (even without cgroup)
- ✅ Can still get resource stats
- ❌ No resource limits
- ❌ No automatic grouping
- ⚠️ Must track PIDs manually

#### **macOS**

**Option 1: launchd (macOS native process manager)**
```bash
# Create launchd plist file
cat > ~/Library/LaunchAgents/com.rbee.worker.8080.plist <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.rbee.worker.8080</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/rbee-worker</string>
        <string>--port</string>
        <string>8080</string>
    </array>
    <key>RunAtLoad</key>
    <false/>
    <key>KeepAlive</key>
    <false/>
    <key>StandardOutPath</key>
    <string>/dev/null</string>
    <key>StandardErrorPath</key>
    <string>/dev/null</string>
    <!-- Resource limits -->
    <key>SoftResourceLimits</key>
    <dict>
        <key>NumberOfProcesses</key>
        <integer>10</integer>
        <key>ResidentSetSize</key>
        <integer>4294967296</integer> <!-- 4GB -->
    </dict>
</dict>
</plist>
EOF

# Load and start
launchctl load ~/Library/LaunchAgents/com.rbee.worker.8080.plist
launchctl start com.rbee.worker.8080

# Get PID
launchctl list | grep com.rbee.worker.8080 | awk '{print $1}'
```
- ✅ Native macOS process management
- ✅ Resource limits (memory, process count)
- ✅ Automatic restart on crash
- ✅ Can query status via `launchctl list`
- ❌ No cgroup-style monitoring
- ⚠️ More complex than nohup

**Option 2: /proc-style monitoring via `ps` and `top`**
```bash
# Start process
nohup ./rbee-worker --port 8080 > /dev/null 2>&1 & echo $!

# Monitor resources via ps
ps -p $PID -o pid,ppid,%cpu,%mem,rss,vsz,comm

# Or use top for continuous monitoring
top -pid $PID -l 1 -stats pid,cpu,mem,command
```
- ✅ Works everywhere
- ✅ Can get CPU%, memory stats
- ❌ No resource limits
- ❌ Must poll periodically (not continuous like cgroup)

**Option 3: Activity Monitor API (programmatic)**
```rust
// Use sysinfo crate for cross-platform process monitoring
use sysinfo::{System, SystemExt, ProcessExt, PidExt};

let mut sys = System::new_all();
sys.refresh_all();

if let Some(process) = sys.process(Pid::from_u32(pid)) {
    let cpu_usage = process.cpu_usage();  // %
    let memory = process.memory();         // bytes
    let disk_usage = process.disk_usage(); // read/write bytes
}
```
- ✅ Programmatic access to process stats
- ✅ Cross-platform (works on Linux, macOS, Windows)
- ✅ Can monitor CPU, memory, disk I/O
- ❌ No resource limits
- ❌ Must poll periodically

#### **Windows**

**Option 1: Windows Job Objects (native resource limits)**
```rust
// Create Job Object with limits
use windows::Win32::System::JobObjects::*;
use windows::Win32::Foundation::*;

unsafe {
    // Create job
    let job = CreateJobObjectW(None, None)?;
    
    // Set memory limit
    let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
    limits.ProcessMemoryLimit = 4 * 1024 * 1024 * 1024; // 4GB
    
    SetInformationJobObject(
        job,
        JobObjectExtendedLimitInformation,
        &limits as *const _ as *const _,
        std::mem::size_of_val(&limits) as u32,
    )?;
    
    // Start process and assign to job
    let process = Command::new("rbee-worker.exe")
        .arg("--port").arg("8080")
        .spawn()?;
    
    AssignProcessToJobObject(job, process.as_raw_handle())?;
}
```
- ✅ Native Windows resource limits
- ✅ Memory limits, CPU affinity, I/O limits
- ✅ Can query job statistics
- ✅ Automatic cleanup when job closed
- ❌ Requires Windows-specific code
- ⚠️ More complex than simple process spawn

**Option 2: WMI (Windows Management Instrumentation)**
```powershell
# Start process
Start-Process -NoNewWindow -FilePath "rbee-worker.exe" -ArgumentList "--port 8080" -PassThru

# Monitor via WMI
Get-WmiObject Win32_Process -Filter "ProcessId = $PID" | Select-Object ProcessId, Name, WorkingSetSize, UserModeTime, KernelModeTime

# Or use Get-Process
Get-Process -Id $PID | Select-Object Id, CPU, WorkingSet, VirtualMemorySize
```
- ✅ Built-in Windows monitoring
- ✅ Can get CPU, memory, I/O stats
- ❌ No resource limits
- ❌ Must poll periodically

**Option 3: Performance Counters (programmatic)**
```rust
// Use sysinfo crate (cross-platform)
use sysinfo::{System, SystemExt, ProcessExt, PidExt};

let mut sys = System::new_all();
sys.refresh_all();

if let Some(process) = sys.process(Pid::from_u32(pid)) {
    let cpu_usage = process.cpu_usage();  // %
    let memory = process.memory();         // bytes
    let disk_usage = process.disk_usage(); // read/write bytes
}
```
- ✅ Programmatic access to process stats
- ✅ Cross-platform (same code as macOS/Linux)
- ✅ Can monitor CPU, memory, disk I/O
- ❌ No resource limits
- ❌ Must poll periodically

### **Implementation Strategy**

```rust
pub async fn start_daemon(start_config: StartConfig) -> Result<u32> {
    let platform = detect_platform();
    let daemon_config = &start_config.daemon_config;
    
    // Build start command based on platform + config
    let start_cmd = match (&daemon_config.cgroup_config, platform) {
        // Linux with cgroup config
        (Some(cgroup), Platform::Linux) => {
            if is_systemd_available(&start_config.ssh_config).await? {
                // Option 1: systemd-run (preferred)
                build_systemd_run_command(binary_path, args, cgroup)
            } else if can_use_manual_cgroup(&start_config.ssh_config).await? {
                // Option 2: Manual cgroup
                build_manual_cgroup_command(binary_path, args, cgroup)
            } else {
                // Option 3: Fallback to nohup
                tracing::warn!("cgroup not available, falling back to nohup");
                build_nohup_command(binary_path, args)
            }
        }
        
        // macOS or Windows with cgroup config (warn + fallback)
        (Some(_), Platform::MacOS | Platform::Windows) => {
            tracing::warn!(
                "cgroup not supported on {:?}, falling back to nohup + PID tracking",
                platform
            );
            build_nohup_command(binary_path, args)
        }
        
        // No cgroup config - use nohup everywhere
        (None, _) => {
            build_nohup_command(binary_path, args)
        }
    };
    
    // Execute command
    let pid_output = ssh_exec(&start_config.ssh_config, &start_cmd).await?;
    let pid: u32 = pid_output.trim().parse()?;
    
    // Store PID for tracking (especially important for macOS/Windows)
    if daemon_config.cgroup_config.is_some() {
        store_pid_for_tracking(&daemon_config.daemon_name, pid).await?;
    }
    
    Ok(pid)
}

// Helper: Check if manual cgroup is available
async fn can_use_manual_cgroup(ssh_config: &SshConfig) -> Result<bool> {
    let check_cmd = "test -w /sys/fs/cgroup/rbee.slice && echo 'yes' || echo 'no'";
    match ssh_exec(ssh_config, check_cmd).await {
        Ok(output) if output.trim() == "yes" => Ok(true),
        _ => Ok(false),
    }
}

// Helper: Build manual cgroup command
fn build_manual_cgroup_command(
    binary_path: &str,
    args: &str,
    cgroup: &CgroupConfig,
) -> String {
    format!(
        "mkdir -p /sys/fs/cgroup/rbee.slice/{}/{} && \
         nohup {} {} > /dev/null 2>&1 & \
         PID=$! && \
         echo $PID > /sys/fs/cgroup/rbee.slice/{}/{}/cgroup.procs && \
         echo $PID",
        cgroup.service, cgroup.instance,
        binary_path, args,
        cgroup.service, cgroup.instance
    )
}

// Helper: Store PID for tracking (macOS/Windows)
async fn store_pid_for_tracking(daemon_name: &str, pid: u32) -> Result<()> {
    // Store in memory or file for later retrieval
    // Used by cgroup-monitor to track workers on non-Linux platforms
    Ok(())
}
```

### **Worker Discovery: Cross-Platform**

#### **Linux (Preferred)**
```rust
// Use cgroup-monitor crate
let monitor = CgroupMonitor::new()?;
let workers = monitor.collect_workers().await?;
// Returns: Vec<WorkerStats> with full resource usage
```

#### **macOS/Windows (Fallback)**
```rust
// Use PID tracking + process polling
let tracker = PidTracker::new()?;
let workers = tracker.collect_workers().await?;
// Returns: Vec<WorkerStats> with basic info (no resource usage)

// WorkerStats on macOS/Windows:
// - worker_id, service, instance, port: ✅ Available
// - cpu_pct, rss_mb, io_r_mb_s: ❌ Not available (or requires polling)
// - cgroup: ❌ Not available
```

### **Platform Comparison**

| Feature | Linux + systemd | Linux (no systemd) | macOS | Windows |
|---------|-----------------|-------------------|-------|---------|
| **cgroup monitoring** | ✅ Full | ⚠️ Manual | ❌ No | ❌ No |
| **Resource limits** | ✅ Yes | ⚠️ Manual | ❌ No | ❌ No |
| **Worker discovery** | ✅ Automatic | ✅ Automatic | ⚠️ PID tracking | ⚠️ PID tracking |
| **Resource stats** | ✅ Full | ✅ Full | ❌ Limited | ❌ Limited |
| **Automatic cleanup** | ✅ Yes | ⚠️ Manual | ❌ Manual | ❌ Manual |

**Recommendation:** Linux with systemd is the primary deployment target. macOS/Windows supported for development only.

---

## 📝 Summary

**Question:** Feature flags, separate crates, or one crate?

**Answer:** ✅ **One crate with runtime detection**

**Why?**
- Already implemented for SSH/Local (just extend it)
- User-friendly (automatic fallback)
- Maintainable (single codebase)
- Flexible (runtime switching)
- No new dependencies
- **Cross-platform by default** (Linux, macOS, Windows)

**Implementation:** 
- Add optional `cgroup_config` field
- Detect platform at runtime (Linux/macOS/Windows)
- Detect systemd availability on Linux
- Automatic fallback chain:
  - Linux: systemd-run → manual cgroup → nohup
  - macOS/Windows: nohup + PID tracking

**Result:** One crate that works everywhere, with automatic optimization for each platform.

### **Platform Support Matrix**

| Platform | cgroup Support | Worker Discovery | Resource Stats | Deployment |
|----------|----------------|------------------|----------------|------------|
| **Linux + systemd** | ✅ Full | ✅ Automatic | ✅ Full | 🎯 Primary |
| **Linux (no systemd)** | ⚠️ Manual | ✅ Automatic | ✅ Full | ✅ Supported |
| **macOS** | ❌ No | ⚠️ PID tracking | ❌ Limited | 🔧 Dev only |
| **Windows** | ❌ No | ⚠️ PID tracking | ❌ Limited | 🔧 Dev only |

**Recommendation:** 
- **Production:** Linux with systemd (full cgroup support)
- **Development:** macOS/Windows (basic PID tracking)
- **Fallback:** Linux without systemd (manual cgroup or nohup)

---

## 🔗 Related Documents

- `CGROUP_INTEGRATION_PLAN.md` - Full cgroup integration plan
- `HEARTBEAT_ARCHITECTURE.md` - Why we need cgroup monitoring
- `daemon-lifecycle/src/utils/ssh.rs` - Existing runtime detection pattern

---

**Decision:** ✅ **Proceed with runtime detection in single crate**

**Next Steps:** 
1. Implement platform detection (Linux/macOS/Windows)
2. Add optional `cgroup_config` to `HttpDaemonConfig`
3. Implement systemd-run support (Linux only)
4. Implement manual cgroup fallback (Linux without systemd)
5. Implement PID tracking for macOS/Windows
6. Update `cgroup-monitor` crate to support cross-platform worker discovery

---

## 🎯 Key Takeaways

### **For Linux (Production)**
- ✅ Full cgroup support via systemd-run
- ✅ Automatic worker discovery via cgroup enumeration
- ✅ Full resource stats (CPU, RAM, I/O, VRAM)
- ✅ Resource limits (CPU quota, memory max)
- ✅ Automatic cleanup on daemon stop

### **For macOS/Windows (Development)**
- ⚠️ No cgroup support (not available on these platforms)
- ⚠️ PID tracking only (store PIDs in memory)
- ⚠️ Limited resource stats (requires polling)
- ⚠️ Manual cleanup needed
- ✅ Still works for development/testing

### **Fallback Strategy**
```
Linux:
  1. Try systemd-run (preferred)
  2. Try manual cgroup (if writable)
  3. Fall back to nohup + PID tracking

macOS/Windows:
  1. Use nohup + PID tracking (only option)
  2. Log warning about limited monitoring
```

### **No Feature Flags Needed**
- All platforms use same dependencies
- Runtime detection handles everything
- Automatic fallback ensures it works everywhere
- Single codebase, single binary
