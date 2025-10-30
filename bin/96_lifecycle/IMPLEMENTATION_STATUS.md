# 96_lifecycle Implementation Status

**Date:** Oct 30, 2025  
**Status:** 📋 STUBS CREATED

---

## ✅ Created Structure

```
bin/96_lifecycle/
├── README.md                    # Category overview
├── IMPLEMENTATION_STATUS.md     # This file
├── health-poll/                 # ✅ Implemented
│   ├── Cargo.toml
│   └── src/lib.rs
├── lifecycle-local/             # 📋 Stub
│   ├── Cargo.toml
│   └── src/lib.rs
├── lifecycle-ssh/               # 📋 Stub
│   ├── Cargo.toml
│   └── src/lib.rs
└── lifecycle-monitored/         # 📋 Stub
    ├── Cargo.toml
    └── src/lib.rs
```

---

## 📊 Implementation Status

| Crate | Status | LOC | Functions | Priority |
|-------|--------|-----|-----------|----------|
| **health-poll** | ✅ Complete | 100 | 1 | 🔥 High |
| **lifecycle-local** | 📋 Stub | 85 | 6 | 🔥 High |
| **lifecycle-ssh** | 📋 Stub | 105 | 6 | 🔥 High |
| **lifecycle-monitored** | 📋 Stub | 75 | 3 | ⚠️ Medium |

---

## 🎯 Implementation Order

### **Phase 1: Foundation** (Week 1)

1. ✅ **health-poll** - Already implemented
   - HTTP health polling with exponential backoff
   - Used by all three lifecycle crates

2. **lifecycle-local** - Implement next
   - Start with `start_daemon()` and `stop_daemon()`
   - Then add `install_daemon()`, `uninstall_daemon()`
   - Finally add `build_daemon()`, `rebuild_daemon()`

### **Phase 2: Remote Management** (Week 2)

3. **lifecycle-ssh** - After lifecycle-local
   - Reuse patterns from lifecycle-local
   - Add SSH execution layer
   - Add SCP for binary upload

### **Phase 3: Monitoring** (Week 3)

4. **lifecycle-monitored** - After process-monitor exists
   - Depends on `process-monitor` crate (99_shared_crates)
   - Simpler than local/ssh (only 3 functions)
   - No install/uninstall/build operations

---

## 🔧 Dependencies

### **External Dependencies**

All crates use minimal dependencies:
- `anyhow` - Error handling
- `tokio` - Async runtime
- `tracing` - Logging
- `serde` - Serialization

### **Internal Dependencies**

```
health-poll (no deps)
    ↓
lifecycle-local → health-poll
    ↓
lifecycle-ssh → health-poll
    ↓
lifecycle-monitored → health-poll + process-monitor
```

---

## 📝 Function Inventory

### **health-poll** ✅

- [x] `poll_health()` - HTTP health check with backoff

### **lifecycle-local** 📋

- [ ] `start_daemon()` - Start local daemon
- [ ] `stop_daemon()` - Stop local daemon
- [ ] `install_daemon()` - Install binary to ~/.local/bin
- [ ] `uninstall_daemon()` - Remove binary
- [ ] `build_daemon()` - Build with cargo
- [ ] `rebuild_daemon()` - Build + hot-reload

### **lifecycle-ssh** 📋

- [ ] `start_daemon()` - Start remote daemon via SSH
- [ ] `stop_daemon()` - Stop remote daemon
- [ ] `install_daemon()` - SCP binary to remote
- [ ] `uninstall_daemon()` - SSH remove binary
- [ ] `build_daemon()` - Build locally
- [ ] `rebuild_daemon()` - Build + SCP + restart

### **lifecycle-monitored** 📋

- [ ] `start_daemon()` - Start with process-monitor
- [ ] `stop_daemon()` - Stop monitored process
- [ ] `get_status()` - Query process stats

---

## 🎨 API Examples

### **Local Daemon**

```rust
use lifecycle_local::{start_daemon, LocalConfig};

let config = LocalConfig::new("queen-rbee", "http://localhost:7833")
    .with_args(vec!["--port", "7833"]);

let pid = start_daemon(config).await?;
```

### **Remote Daemon**

```rust
use lifecycle_ssh::{start_daemon, SshConfig, RemoteConfig};

let ssh = SshConfig { hostname: "192.168.1.100", user: "vince", port: 22 };
let config = RemoteConfig::new("rbee-hive", "http://192.168.1.100:7835", ssh)
    .with_args(vec!["--port", "7835"]);

let pid = start_daemon(config).await?;
```

### **Monitored Process**

```rust
use lifecycle_monitored::{start_daemon, MonitoredConfig};
use process_monitor::MonitorConfig;

let config = MonitoredConfig::new(
    "rbee-worker",
    "http://localhost:8080",
    MonitorConfig {
        group: "llm",
        instance: "8080",
        cpu_limit: Some("200%"),
        memory_limit: Some("4G"),
    },
).with_args(vec!["--port", "8080"]);

let pid = start_daemon(config).await?;
```

---

## 🚀 Next Steps

1. **Implement lifecycle-local**
   - Extract code from old `daemon-lifecycle` crate
   - Focus on start/stop first
   - Add install/uninstall/build later

2. **Implement lifecycle-ssh**
   - Reuse lifecycle-local patterns
   - Add SSH/SCP layer
   - Test with remote hive

3. **Wait for process-monitor**
   - Implement process-monitor crate first
   - Then implement lifecycle-monitored

4. **Update consumers**
   - Update rbee-keeper to use lifecycle-local + lifecycle-ssh
   - Update rbee-hive to use lifecycle-monitored

---

## 📚 Related Documents

- [Category README](./README.md) - Overview of 96_lifecycle
- [process-monitor README](../99_shared_crates/process-monitor/README.md) - Cross-platform monitoring
- [Old daemon-lifecycle](../99_shared_crates/daemon-lifecycle/) - Source for migration

---

## ✅ Checklist

- [x] Create 96_lifecycle category
- [x] Create health-poll crate (implemented)
- [x] Create lifecycle-local stub
- [x] Create lifecycle-ssh stub
- [x] Create lifecycle-monitored stub
- [x] Write documentation
- [ ] Implement lifecycle-local
- [ ] Implement lifecycle-ssh
- [ ] Implement lifecycle-monitored
- [ ] Update rbee-keeper
- [ ] Update rbee-hive
- [ ] Delete old daemon-lifecycle crate
