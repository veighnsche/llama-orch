# Git Clone + Build Installation Method

**Date:** Oct 24, 2025  
**Status:** Implemented in daemon-sync

---

## Overview

The daemon-sync crate now supports **git clone + cargo build** as the primary installation method for development. This is much better than downloading from GitHub releases during active development.

## Benefits

1. **Always up-to-date** - Clones latest code from your branch
2. **No release management** - No need to create GitHub releases during dev
3. **Feature flags** - Build workers with specific features (cuda, metal, cpu)
4. **Fast** - Shallow clone (--depth 1) for speed
5. **Flexible** - Can switch branches/tags easily

---

## Configuration

### Default (Git Clone)

By default, all installations use git clone from `git@github.com:veighnsche/llama-orch`:

```toml
[[hive]]
alias = "gpu-1"
hostname = "192.168.1.100"
ssh_user = "vince"
workers = [
    { type = "vllm", version = "latest", features = ["cuda"] },
    { type = "llama-cpp", version = "latest", features = ["metal"] },
]
```

### Explicit Git Method

You can explicitly specify the git method:

```toml
[[hive]]
alias = "gpu-1"
hostname = "192.168.1.100"
ssh_user = "vince"

# Explicit git install for hive
install_method = { git = { repo = "git@github.com:veighnsche/llama-orch", branch = "main" } }

workers = [
    { 
        type = "vllm", 
        version = "latest",
        features = ["cuda"],
        install_method = { git = { repo = "git@github.com:veighnsche/llama-orch", branch = "develop" } }
    },
]
```

### GitHub Release Method (Future)

For production deployments:

```toml
[[hive]]
alias = "prod-1"
hostname = "prod.example.com"
ssh_user = "deploy"
install_method = { release = { repo = "veighnsche/llama-orch", tag = "v0.1.0" } }

workers = [
    { 
        type = "vllm",
        version = "latest",
        install_method = { release = { repo = "veighnsche/llama-orch", tag = "v0.1.0" } }
    },
]
```

### Local Binary Method

Use existing binaries on remote system:

```toml
[[hive]]
alias = "custom"
hostname = "192.168.1.200"
ssh_user = "vince"
install_method = { local = { path = "/opt/rbee/rbee-hive" } }

workers = [
    { 
        type = "vllm",
        version = "latest",
        install_method = { local = { path = "/opt/rbee/workers/rbee-worker-vllm" } }
    },
]
```

---

## Worker Feature Flags

Workers can be built with different feature flags for different hardware:

### CUDA (NVIDIA GPUs)
```toml
{ type = "vllm", version = "latest", features = ["cuda"] }
```

### Metal (Apple Silicon)
```toml
{ type = "llama-cpp", version = "latest", features = ["metal"] }
```

### CPU Only
```toml
{ type = "llama-cpp", version = "latest", features = ["cpu"] }
```

### Multiple Features
```toml
{ type = "vllm", version = "latest", features = ["cuda", "flash-attn"] }
```

---

## How It Works

### Git Clone + Build Process

1. **Clone** - Shallow clone (--depth 1) to `~/.local/share/rbee/build`
2. **Build** - Run `cargo build --release --bin <binary>`
3. **Install** - Copy binary to appropriate location
4. **Verify** - Run `--version` to confirm installation

### Hive Installation

```bash
# On remote host via SSH:
rm -rf ~/.local/share/rbee/build
git clone --depth 1 --branch main git@github.com:veighnsche/llama-orch ~/.local/share/rbee/build
cd ~/.local/share/rbee/build
cargo build --release --bin rbee-hive
cp target/release/rbee-hive ~/.local/bin/rbee-hive
chmod +x ~/.local/bin/rbee-hive
```

### Worker Installation

```bash
# On remote host via SSH:
rm -rf ~/.local/share/rbee/build
git clone --depth 1 --branch main git@github.com:veighnsche/llama-orch ~/.local/share/rbee/build
cd ~/.local/share/rbee/build
cargo build --release --bin llm-worker-rbee --features cuda
cp target/release/llm-worker-rbee ~/.local/share/rbee/workers/rbee-worker-vllm
chmod +x ~/.local/share/rbee/workers/rbee-worker-vllm
```

---

## Implementation Details

### InstallMethod Enum

```rust
pub enum InstallMethod {
    Git {
        repo: String,
        branch: String,
    },
    Release {
        repo: String,
        tag: String,
    },
    Local {
        path: String,
    },
}
```

### Default

```rust
impl Default for InstallMethod {
    fn default() -> Self {
        InstallMethod::Git {
            repo: "git@github.com:veighnsche/llama-orch".to_string(),
            branch: "main".to_string(),
        }
    }
}
```

### Config Fields

**HiveConfig:**
- `install_method: InstallMethod` (default: git clone)

**WorkerConfig:**
- `install_method: InstallMethod` (default: git clone)
- `features: Vec<String>` (cargo feature flags)

---

## Example Configs

### Development Setup

```toml
# ~/.config/rbee/hives.conf

[[hive]]
alias = "dev-gpu"
hostname = "192.168.1.100"
ssh_user = "vince"
workers = [
    { type = "vllm", version = "latest", features = ["cuda"] },
    { type = "llama-cpp", version = "latest", features = ["cpu"] },
]

[[hive]]
alias = "dev-mac"
hostname = "192.168.1.101"
ssh_user = "vince"
workers = [
    { type = "llama-cpp", version = "latest", features = ["metal"] },
]
```

### Mixed Setup (Dev + Prod)

```toml
# Development hive - git clone
[[hive]]
alias = "dev"
hostname = "dev.local"
ssh_user = "vince"
workers = [
    { type = "vllm", version = "latest", features = ["cuda"] },
]

# Production hive - releases
[[hive]]
alias = "prod"
hostname = "prod.example.com"
ssh_user = "deploy"
install_method = { release = { repo = "veighnsche/llama-orch", tag = "v0.1.0" } }
workers = [
    { 
        type = "vllm",
        version = "latest",
        install_method = { release = { repo = "veighnsche/llama-orch", tag = "v0.1.0" } }
    },
]
```

---

## Usage

```bash
# Validate config
rbee validate

# Dry run to see what will happen
rbee sync --dry-run

# Install everything (git clone + build)
rbee sync

# Check status
rbee package-status
```

---

## Performance

**Git clone + build is slower than downloading releases**, but:
- Only happens once per installation
- Ensures you have latest code
- No need to manage releases during development
- Worth the extra time for development workflow

**Typical times:**
- Git clone: 5-10 seconds
- Cargo build (release): 2-5 minutes
- Total: ~3-6 minutes per component

**With concurrent installation:**
- 3 hives × 2 workers = 6 components
- Sequential: ~18-36 minutes
- Concurrent: ~3-6 minutes (same as 1 component!)

---

## Future Enhancements

1. **Docker method** - Pull and run containers
2. **Binary cache** - Cache built binaries locally
3. **Incremental builds** - Don't rebuild if code hasn't changed
4. **Build server** - Centralized build service

---

## Files Modified

- `bin/99_shared_crates/rbee-config/src/declarative.rs` (+60 lines)
  - Added `InstallMethod` enum
  - Added `install_method` field to HiveConfig
  - Added `install_method` and `features` fields to WorkerConfig

- `bin/99_shared_crates/daemon-sync/src/install.rs` (+250 lines)
  - Added `install_hive_from_git()`
  - Added `install_worker_from_git()`
  - Added `install_hive_from_release()`
  - Added `install_worker_from_release()`
  - Updated main install functions to use InstallMethod

---

**Git clone + build is now the default installation method for rbee!**
