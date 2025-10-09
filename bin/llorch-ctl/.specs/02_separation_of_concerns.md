# llorch-cli Separation of Concerns

**Status**: Draft  
**Version**: 0.1.0

---

## Overview

This document defines the strict separation of concerns between `llorch-cli` (developer tooling) and the runtime binaries (orchestratord, pool-managerd, worker-orcd/llorch-candled).

**Key Principle:** llorch-cli is TOOLING, not RUNTIME.

---

## Architectural Boundaries

### Runtime Binaries (bin/)

**Location:** `bin/{orchestratord,pool-managerd,llorch-candled}/`

**Responsibilities:**
- Daemon lifecycle management
- Inference execution
- Worker spawning/management
- Job scheduling and queueing
- State management (SQLite, persistent)
- API endpoints (HTTP servers)
- Metrics emission (Prometheus)
- Heartbeat protocols
- VRAM/GPU management
- Model loading into memory

**NOT responsible for:**
- Model downloading from HuggingFace
- Git operations
- Development workflow automation
- Build orchestration
- Test execution

**Communication:**
- HTTP APIs between components
- SSE streaming for inference
- Persistent state in SQLite/files

**Lifecycle:**
- Managed by systemd (Linux) or launchd (macOS)
- Long-running daemons
- Restart on failure
- Log to journald/syslog

---

### Developer Tooling (tools/llorch-cli/)

**Location:** `tools/llorch-cli/`

**Responsibilities:**
- Development workflow automation
- Model provisioning (download from HF)
- Git operations (clone, pull, submodules)
- Build orchestration (cargo build wrappers)
- Test execution (cargo test wrappers)
- Remote execution (SSH forwarding)
- Environment validation (doctor command)
- Configuration management

**NOT responsible for:**
- Running daemons
- Inference execution
- Worker lifecycle management
- Job scheduling
- State persistence
- API serving
- Metrics emission

**Communication:**
- Shell commands (git, cargo, hf)
- SSH for remote execution
- File system operations
- Process spawning (short-lived)

**Lifecycle:**
- Short-lived CLI invocations
- No persistent state
- Exit after command completion
- Output to stdout/stderr

---

## Responsibility Matrix

| Responsibility | llorch-cli | orchestratord | pool-managerd | worker-orcd |
|----------------|------------|---------------|---------------|-------------|
| **Development** |
| Git operations | ✅ | ❌ | ❌ | ❌ |
| Model download | ✅ | ❌ | ❌ | ❌ |
| Build binaries | ✅ | ❌ | ❌ | ❌ |
| Run tests | ✅ | ❌ | ❌ | ❌ |
| Environment check | ✅ | ❌ | ❌ | ❌ |
| **Runtime** |
| Daemon lifecycle | ❌ | ✅ | ✅ | ✅ |
| Job scheduling | ❌ | ✅ | ❌ | ❌ |
| Worker spawning | ❌ | ❌ | ✅ | ❌ |
| Inference execution | ❌ | ❌ | ❌ | ✅ |
| Model loading (VRAM) | ❌ | ❌ | ❌ | ✅ |
| State persistence | ❌ | ✅ | ✅ | ❌ |
| API serving | ❌ | ✅ | ✅ | ✅ |
| Metrics emission | ❌ | ✅ | ✅ | ✅ |
| **Provisioning** |
| Model download (HF) | ✅ | ❌ | ❌ | ❌ |
| Model caching (RAM) | ❌ | ❌ | ✅ | ❌ |
| Model loading (VRAM) | ❌ | ❌ | ❌ | ✅ |

---

## Command Boundaries

### llorch-cli Commands (Developer Workflow)

```bash
# Git operations (development)
llorch git status
llorch git pull
llorch git sync
llorch git submodules list

# Model provisioning (development)
llorch models catalog
llorch models download tinyllama
llorch models verify tinyllama

# Build operations (development)
llorch build worker cuda
llorch build orchestrator
llorch build pool-manager

# Test operations (development)
llorch test unit
llorch test integration
llorch test smoke

# Remote operations (development)
llorch git status --remote mac
llorch models download tinyllama --remote workstation

# Development utilities
llorch dev doctor
llorch dev setup
llorch dev check
```

### Runtime Binary Commands (Production/Runtime)

```bash
# Worker daemon (runtime)
llorch-candled \
  --worker-id worker-1 \
  --model /path/to/model.gguf \
  --gpu 0 \
  --port 8001 \
  --callback-url http://pool-manager:9200/workers/ready

# Pool manager daemon (runtime)
pool-managerd \
  --config /etc/llorch/pool-manager.toml \
  --orchestrator-url http://orchestrator:8080

# Orchestrator daemon (runtime)
orchestratord \
  --config /etc/llorch/orchestrator.toml \
  --bind 0.0.0.0:8080
```

---

## Data Flow Boundaries

### Development Time (llorch-cli)

```
Developer
    ↓ (CLI command)
llorch-cli
    ↓ (shell commands)
git / cargo / hf / ssh
    ↓ (file operations)
Filesystem / Remote host
```

**Characteristics:**
- Synchronous operations
- Short-lived processes
- Interactive output
- No persistent state
- No daemon management

### Runtime (orchestratord/pool-managerd/worker-orcd)

```
Client
    ↓ (HTTP POST /v2/tasks)
orchestratord
    ↓ (HTTP POST /pools/{id}/workers/spawn)
pool-managerd
    ↓ (process spawn)
worker-orcd
    ↓ (HTTP POST /execute)
Inference Engine (CUDA/Metal)
    ↓ (SSE stream)
Client
```

**Characteristics:**
- Asynchronous operations
- Long-running daemons
- Structured logging
- Persistent state (SQLite)
- Daemon lifecycle management

---

## Future Integration Points

### M0+: Worker Management (Future)

**llorch-cli MAY provide convenience wrappers:**
```bash
# Development/testing only
llorch worker start cuda --model tinyllama --gpu 0
llorch worker stop <id>
llorch worker list
llorch worker logs <id>
```

**Implementation:**
- Wrapper around `llorch-candled` binary
- For development/testing ONLY
- NOT for production use
- Delegates to systemd/launchd for production

### M1+: Pool Manager Management (Future)

**llorch-cli MAY provide convenience wrappers:**
```bash
# Development/testing only
llorch pool-manager start --config pool.toml
llorch pool-manager stop
llorch pool-manager status
```

**Implementation:**
- Wrapper around `pool-managerd` binary
- For development/testing ONLY
- NOT for production use
- Delegates to systemd/launchd for production

### M2+: Orchestrator Management (Future)

**llorch-cli MAY provide convenience wrappers:**
```bash
# Development/testing only
llorch orchestrator start --config orch.toml
llorch orchestrator stop
llorch orchestrator status
llorch orchestrator jobs list
```

**Implementation:**
- Wrapper around `orchestratord` binary
- For development/testing ONLY
- NOT for production use
- Delegates to systemd/launchd for production

**Key Distinction:**
- llorch-cli provides **development conveniences**
- Production deployments use **systemd/launchd directly**
- llorch-cli NEVER embeds daemon logic

---

## Anti-Patterns (What NOT to Do)

### ❌ Embedding Daemon Logic in CLI

```rust
// WRONG: Do not embed daemon logic in CLI
impl Commands {
    fn start_orchestrator(&self) -> Result<()> {
        // ❌ Do not implement orchestrator logic here
        let server = HttpServer::new(/* ... */);
        server.run().await?;  // ❌ CLI should not run servers
    }
}
```

**Why wrong:**
- CLI is short-lived, daemons are long-lived
- CLI has no state management
- CLI has no restart/recovery logic
- Violates separation of concerns

**Correct approach:**
```rust
// ✅ Correct: Spawn the daemon binary
impl Commands {
    fn start_orchestrator(&self) -> Result<()> {
        // ✅ Spawn the orchestratord binary
        Command::new("orchestratord")
            .arg("--config")
            .arg(&self.config_path)
            .spawn()?;
        
        println!("Orchestrator started (PID: {})", pid);
        println!("Use systemd/launchd for production");
    }
}
```

### ❌ Duplicating Runtime Logic

```rust
// WRONG: Do not duplicate worker logic
impl Commands {
    fn execute_inference(&self, prompt: &str) -> Result<()> {
        // ❌ Do not implement inference logic here
        let model = load_model(&self.model_path)?;
        let output = model.generate(prompt)?;
        println!("{}", output);
    }
}
```

**Why wrong:**
- Duplicates worker-orcd logic
- No state management
- No metrics emission
- No error recovery

**Correct approach:**
```rust
// ✅ Correct: Call the worker API
impl Commands {
    fn execute_inference(&self, prompt: &str) -> Result<()> {
        // ✅ Call worker-orcd HTTP API
        let client = reqwest::blocking::Client::new();
        let response = client
            .post(&format!("{}/execute", self.worker_url))
            .json(&InferenceRequest { prompt, /* ... */ })
            .send()?;
        
        // Stream SSE response
        for event in response.sse_events() {
            println!("{}", event.data);
        }
    }
}
```

### ❌ Managing Daemon Lifecycle

```rust
// WRONG: Do not manage daemon lifecycle
impl Commands {
    fn restart_orchestrator(&self) -> Result<()> {
        // ❌ Do not implement restart logic
        self.stop_orchestrator()?;
        thread::sleep(Duration::from_secs(1));
        self.start_orchestrator()?;
    }
}
```

**Why wrong:**
- No crash recovery
- No log rotation
- No resource limits
- systemd/launchd does this better

**Correct approach:**
```rust
// ✅ Correct: Delegate to systemd
impl Commands {
    fn restart_orchestrator(&self) -> Result<()> {
        // ✅ Use systemd for lifecycle management
        Command::new("systemctl")
            .arg("restart")
            .arg("orchestratord")
            .status()?;
        
        println!("Orchestrator restarted via systemd");
    }
}
```

---

## Testing Boundaries

### llorch-cli Tests

**Unit tests:**
- Config parsing
- Catalog parsing
- Command argument parsing
- Error handling
- Utility functions

**Integration tests:**
- Git operations (with test repo)
- Model download (with small test model)
- Build commands (with test crate)
- Remote execution (with mock SSH)

**NOT tested:**
- Daemon lifecycle
- Inference execution
- Job scheduling
- State persistence

### Runtime Binary Tests

**Unit tests:**
- API handlers
- State management
- Worker lifecycle
- Job scheduling

**Integration tests:**
- HTTP API contracts
- Worker spawning
- Inference execution
- State persistence

**NOT tested:**
- Git operations
- Model downloading
- Build orchestration

---

## Summary

**llorch-cli is:**
- ✅ Developer tooling
- ✅ Workflow automation
- ✅ Build orchestration
- ✅ Model provisioning
- ✅ Short-lived CLI invocations

**llorch-cli is NOT:**
- ❌ Runtime daemon
- ❌ Inference engine
- ❌ State manager
- ❌ API server
- ❌ Long-running process

**Key Takeaway:**
llorch-cli makes developers productive by automating workflows. Runtime binaries handle production inference workloads. These are separate concerns with separate implementations.

---

**Version**: 0.1.0  
**Last Updated**: 2025-10-09  
**Status**: Normative (MUST follow)

---

**End of Separation of Concerns**
