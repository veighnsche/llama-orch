# 96_lifecycle - Daemon Lifecycle Management

**Category for all daemon lifecycle-related crates**

---

## 🎯 Overview

This category contains crates for managing daemon processes across different execution contexts.

**Why separate category?**
- All crates are about daemon lifecycle
- Shared naming convention (no need to repeat "daemon" in every name)
- Clear separation from other shared utilities

---

## 📦 Crates

### **1. lifecycle-local** (Keeper → Queen)
**Purpose:** Start/stop local daemons (no SSH, optional monitoring)  
**Used by:** rbee-keeper starting local queen-rbee  
**Operations:** start, stop, shutdown, status, install, uninstall, build, rebuild  
**Dependencies:** health-poll  

### **2. lifecycle-ssh** (Keeper → Hive)
**Purpose:** Start/stop remote daemons via SSH (no monitoring)  
**Used by:** rbee-keeper managing remote hives  
**Operations:** start, stop, shutdown, status, install, uninstall, build, rebuild  
**Dependencies:** health-poll, SSH  

### **3. lifecycle-monitored** (Hive → Worker)
**Purpose:** Start/stop local processes with monitoring  
**Used by:** rbee-hive managing workers  
**Operations:** start, stop, shutdown, status (NO install/uninstall/build)  
**Dependencies:** health-poll, process-monitor  

### **4. health-poll** (Shared utility)
**Purpose:** HTTP health check polling with exponential backoff  
**Used by:** All three lifecycle crates  
**Operations:** poll_health  
**Dependencies:** reqwest, tokio  

---

## 🔄 Dependency Graph

```
lifecycle-local ──┐
                  ├──> health-poll
lifecycle-ssh ────┤
                  │
lifecycle-monitored ──> health-poll
                     └──> process-monitor (99_shared_crates)
```

---

## 🎨 Design Principles

### **1. Focused Crates**
Each crate serves ONE use case:
- `lifecycle-local`: Local daemon management
- `lifecycle-ssh`: Remote daemon management
- `lifecycle-monitored`: Worker process management

### **2. No Optional Features**
- No `Option<SshConfig>` - use the right crate
- No `Option<MonitorConfig>` - monitoring is required or not needed
- Clear API, no confusion

### **3. Minimal Dependencies**
- `lifecycle-local`: No SSH, no monitoring
- `lifecycle-ssh`: SSH only, no monitoring
- `lifecycle-monitored`: Monitoring only, no SSH

### **4. Operation Separation**
- **Full lifecycle** (local, ssh): install, uninstall, build, rebuild, start, stop, status
- **Runtime only** (monitored): start, stop, status (NO install/uninstall/build)

**Why?** Hive doesn't install workers - Keeper does that via `lifecycle-local`.

---

## 📋 Operation Matrix

| Operation | lifecycle-local | lifecycle-ssh | lifecycle-monitored |
|-----------|----------------|---------------|---------------------|
| **start** | ✅ Yes | ✅ Yes | ✅ Yes |
| **stop** | ✅ Yes | ✅ Yes | ✅ Yes |
| **shutdown** | ✅ Yes | ✅ Yes | ✅ Yes |
| **status** | ✅ Yes | ✅ Yes | ✅ Yes |
| **install** | ✅ Yes | ✅ Yes | ❌ No |
| **uninstall** | ✅ Yes | ✅ Yes | ❌ No |
| **build** | ✅ Yes | ✅ Yes | ❌ No |
| **rebuild** | ✅ Yes | ✅ Yes | ❌ No |

---

## 🚀 Usage Examples

### **Keeper → Queen (lifecycle-local)**

```rust
use lifecycle_local::{start_daemon, LocalConfig};

let config = LocalConfig::new("queen-rbee", "http://localhost:7833")
    .with_args(vec!["--port", "7833"]);

let pid = start_daemon(config).await?;
```

### **Keeper → Hive (lifecycle-ssh)**

```rust
use lifecycle_ssh::{start_daemon, SshConfig, RemoteConfig};

let ssh = SshConfig::new("192.168.1.100", "vince", 22);
let config = RemoteConfig::new("rbee-hive", "http://192.168.1.100:7835", ssh)
    .with_args(vec!["--port", "7835"]);

let pid = start_daemon(config).await?;
```

### **Hive → Worker (lifecycle-monitored)**

```rust
use lifecycle_monitored::{start_daemon, MonitoredConfig};
use process_monitor::MonitorConfig;

let config = MonitoredConfig::new("rbee-worker", "http://localhost:8080")
    .with_args(vec!["--port", "8080"])
    .with_monitor(MonitorConfig {
        group: "llm",
        instance: "8080",
        cpu_limit: Some("200%"),
        memory_limit: Some("4G"),
    });

let pid = start_daemon(config).await?;
```

---

## 🔧 Implementation Status

| Crate | Status | Priority |
|-------|--------|----------|
| **health-poll** | 📋 Stub | 🔥 High (needed by all) |
| **lifecycle-local** | 📋 Stub | 🔥 High |
| **lifecycle-ssh** | 📋 Stub | 🔥 High |
| **lifecycle-monitored** | 📋 Stub | ⚠️ Medium (needs process-monitor) |

---

## 📚 Migration from Old Structure

### **Old: daemon-lifecycle (monolithic)**

```
bin/99_shared_crates/daemon-lifecycle/
├── src/
│   ├── start.rs          # Mixed SSH/local/monitor logic
│   ├── stop.rs           # Mixed logic
│   ├── install.rs        # SSH logic
│   └── utils/
│       ├── ssh.rs
│       ├── local.rs
│       └── poll.rs
```

### **New: 96_lifecycle (focused crates)**

```
bin/96_lifecycle/
├── health-poll/          # Shared utility
├── lifecycle-local/      # Local only
├── lifecycle-ssh/        # SSH only
└── lifecycle-monitored/  # Monitoring only
```

**Migration steps:**
1. ✅ Create category structure
2. ✅ Create stubs for all crates
3. Extract `health-poll` from old crate
4. Split `start.rs` into three versions
5. Move `install.rs` to local/ssh only
6. Update all consumers (keeper, hive)

---

## 🎯 Next Steps

1. Create stub crates with Cargo.toml
2. Extract health-poll implementation
3. Implement lifecycle-local (simplest)
4. Implement lifecycle-ssh (reuse old code)
5. Implement lifecycle-monitored (new, uses process-monitor)
6. Update rbee-keeper to use lifecycle-local + lifecycle-ssh
7. Update rbee-hive to use lifecycle-monitored
8. Delete old daemon-lifecycle crate

---

## 📖 Documentation

Each crate has its own focused documentation:
- [health-poll](./health-poll/README.md) - Health polling utility
- [lifecycle-local](./lifecycle-local/README.md) - Local daemon management
- [lifecycle-ssh](./lifecycle-ssh/README.md) - Remote daemon management
- [lifecycle-monitored](./lifecycle-monitored/README.md) - Worker process management
