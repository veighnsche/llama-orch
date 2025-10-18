# Component: Local vs Network Mode

**Purpose:** Distinguish between local and network deployment modes  
**Affects:** queen-rbee, rbee-keeper, rbee-hive  
**Status:** 🔴 NOT IMPLEMENTED - Needs design and implementation

## Overview

The rbee ecosystem must support two deployment modes:
1. **Local Mode** - All components on same machine
2. **Network Mode** - Components distributed across machines

## Local Mode

### Characteristics
- All components on same machine
- No SSH required
- Shared filesystem
- Low latency
- Simple setup

### Architecture
```
┌─────────────────────────────────────────┐
│ Local Machine (localhost)               │
│                                         │
│  rbee-keeper (CLI)                      │
│         │                               │
│         ▼                               │
│  queen-rbee (Orchestrator)              │
│         │                               │
│         ▼                               │
│  rbee-hive (Worker Pool)                │
│         │                               │
│         ▼                               │
│  llm-worker-rbee (Workers)              │
│                                         │
└─────────────────────────────────────────┘
```

### Configuration
```toml
[queen]
mode = "local"
host = "127.0.0.1"
port = 8080

[[hive]]
mode = "local"
port = 8081
```

### Detection
```rust
fn is_local_mode(host: &str) -> bool {
    host == "localhost" || host == "127.0.0.1" || host == "::1"
}
```

## Network Mode

### Characteristics
- Components on different machines
- SSH for remote management
- Separate filesystems
- Network latency
- Complex setup

### Architecture
```
┌──────────────────┐         ┌──────────────────┐
│ Control Machine  │         │ Worker Machine 1 │
│                  │         │                  │
│  rbee-keeper     │   SSH   │  rbee-hive       │
│       │          ├────────►│       │          │
│       ▼          │         │       ▼          │
│  queen-rbee      │  HTTP   │  llm-worker-rbee │
│                  ├────────►│                  │
└──────────────────┘         └──────────────────┘
                                      │
                             ┌────────┴────────┐
                             ▼                 ▼
                    ┌──────────────┐  ┌──────────────┐
                    │ Worker M2    │  │ Worker M3    │
                    │  rbee-hive   │  │  rbee-hive   │
                    └──────────────┘  └──────────────┘
```

### Configuration
```toml
[queen]
mode = "network"
host = "control.local"
port = 8080

[[hive]]
mode = "network"
hostname = "worker1.local"
port = 8081
ssh_key = "/home/user/.ssh/id_rsa"
ssh_user = "rbee"

[[hive]]
mode = "network"
hostname = "192.168.1.100"
port = 8081
ssh_key = "/home/user/.ssh/id_rsa"
ssh_user = "rbee"
```

### SSH Requirements
- SSH key-based authentication
- Remote machine has rbee-hive binary
- Network connectivity
- Firewall rules allow HTTP traffic

## Implementation Needs

### 1. Mode Detection (rbee-keeper)

```rust
// src/mode.rs
pub enum DeploymentMode {
    Local,
    Network,
}

pub fn detect_mode(config: &Config) -> DeploymentMode {
    // Check explicit config first
    if let Some(mode) = config.mode {
        return mode;
    }
    
    // Auto-detect from hostname
    if is_local_hostname(&config.queen.host) {
        DeploymentMode::Local
    } else {
        DeploymentMode::Network
    }
}

fn is_local_hostname(host: &str) -> bool {
    matches!(host, "localhost" | "127.0.0.1" | "::1")
}
```

### 2. Queen Lifecycle (rbee-keeper)

```rust
// Local mode: spawn queen directly
async fn start_queen_local(config: &QueenConfig) -> Result<()> {
    let child = Command::new("queen-rbee")
        .arg("--port").arg(config.port)
        .spawn()?;
    
    store_pid(child.id())?;
    wait_for_health(&config.url).await?;
    Ok(())
}

// Network mode: SSH to remote machine
async fn start_queen_network(config: &QueenConfig) -> Result<()> {
    let ssh = SshSession::connect(&config.host, &config.ssh_key)?;
    
    // Start queen remotely
    ssh.exec(&format!("queen-rbee --port {}", config.port))?;
    
    // Wait for health check over network
    wait_for_health(&config.url).await?;
    Ok(())
}
```

### 3. Hive Management (queen-rbee)

```rust
// Local hive: spawn process
async fn start_hive_local(config: &HiveConfig) -> Result<HiveId> {
    let child = Command::new("rbee-hive")
        .arg("--port").arg(config.port)
        .spawn()?;
    
    let hive_id = register_hive(HiveInfo {
        mode: HiveMode::Local,
        url: format!("http://127.0.0.1:{}", config.port),
        pid: Some(child.id()),
    }).await?;
    
    Ok(hive_id)
}

// Network hive: SSH to remote
async fn start_hive_network(config: &HiveConfig) -> Result<HiveId> {
    let ssh = SshSession::connect(&config.hostname, &config.ssh_key)?;
    
    // Check if rbee-hive binary exists
    if !ssh.file_exists("rbee-hive")? {
        // Upload binary
        ssh.upload_file("rbee-hive", "/usr/local/bin/rbee-hive")?;
    }
    
    // Start hive
    ssh.exec(&format!("rbee-hive --port {}", config.port))?;
    
    let hive_id = register_hive(HiveInfo {
        mode: HiveMode::Network,
        url: format!("http://{}:{}", config.hostname, config.port),
        ssh_info: Some(SshInfo {
            hostname: config.hostname.clone(),
            key_path: config.ssh_key.clone(),
        }),
    }).await?;
    
    Ok(hive_id)
}
```

### 4. Configuration Schema

```rust
#[derive(Deserialize)]
struct Config {
    mode: Option<DeploymentMode>,  // Explicit override
    queen: QueenConfig,
    hives: Vec<HiveConfig>,
}

#[derive(Deserialize)]
struct QueenConfig {
    mode: Option<DeploymentMode>,
    host: String,
    port: u16,
    ssh_key: Option<PathBuf>,  // For network mode
}

#[derive(Deserialize)]
struct HiveConfig {
    mode: Option<DeploymentMode>,
    hostname: String,
    port: u16,
    ssh_key: Option<PathBuf>,
    ssh_user: Option<String>,
}
```

## Testing Strategy

### Local Mode Tests
```bash
# Start everything locally
rbee-keeper start --mode local

# Verify all components on localhost
curl http://localhost:8080/v1/health  # queen
curl http://localhost:8081/v1/health  # hive
```

### Network Mode Tests
```bash
# Start queen locally, hive remotely
rbee-keeper start --mode network --config network.toml

# Verify components
curl http://localhost:8080/v1/health        # queen (local)
curl http://worker1.local:8081/v1/health    # hive (remote)
```

## Current Status

🔴 **NOT IMPLEMENTED**

**Gaps:**
- No mode detection logic
- No SSH client implementation
- No remote hive management
- No configuration schema for modes
- Hard-coded localhost assumptions

**Affected Files:**
- `bin/rbee-keeper/src/` - Needs mode detection
- `bin/queen-rbee/src/` - Needs SSH + hive lifecycle
- Configuration files - Need mode fields

## Priority

**P1 - High Priority**

Required for:
- Multi-machine deployments
- Remote GPU utilization
- Production deployments

## Related Components

- **rbee-keeper** - Must detect mode, start queen accordingly
- **queen-rbee** - Must manage local/remote hives
- **rbee-hive** - Works same in both modes (receives HTTP)

---

**Created by:** TEAM-096 | 2025-10-18  
**Status:** 🔴 DESIGN ONLY - Implementation needed
