# Declarative Configuration Analysis (Package Manager Pattern)

**Date:** Oct 23, 2025  
**Status:** PROPOSAL / ANALYSIS  
**Author:** TEAM-276

## Executive Summary

This document analyzes the shift from **imperative commands** to **declarative configuration** for rbee installation and management, following a package manager pattern (like `npm install`, `apt-get`, etc.).

**Current State:** Imperative commands (`rbee install-queen`, `rbee install-hive`)  
**Proposed State:** Declarative config + single install command (`rbee install`)

---

## Current Architecture (Imperative)

### User Workflow

```bash
# User must manually install each component
rbee install-queen
rbee start-queen

# For each hive
rbee install-hive --alias hive-1
rbee start-hive --alias hive-1

# For each worker on each hive
rbee install-worker --hive hive-1 --type vllm
rbee install-worker --hive hive-1 --type llama-cpp
```

**Problems:**
- ❌ Manual, error-prone
- ❌ No declarative state
- ❌ Hard to reproduce setup
- ❌ Sequential (slow)
- ❌ No "desired state" concept

---

## Proposed Architecture (Declarative)

### Configuration Files

#### 1. `~/.config/rbee/rbee.conf` (Queen Configuration)

```toml
# Queen configuration
[queen]
# Queen mode: "standalone" or "attached-hive"
mode = "standalone"  # or "attached-hive"

# If attached-hive, queen runs on same machine as a hive
# attached_hive_alias = "local-hive"  # Uncomment for attached mode

# Queen port
port = 8500

# Auto-start on system boot (optional)
auto_start = true
```

**Two modes:**
- **Standalone:** Queen runs separately, manages remote hives
- **Attached-hive:** Queen runs on same machine as a hive (for single-machine setups)

#### 2. `~/.config/rbee/hives.conf` (Hive & Worker Configuration)

```toml
# Hive 1: Remote GPU server
[[hive]]
alias = "gpu-server-1"
hostname = "192.168.1.100"
ssh_user = "vince"
ssh_port = 22
hive_port = 8600

# Workers to install on this hive
workers = ["vllm", "llama-cpp"]

# Optional: Pre-download models
# models = ["meta-llama/Llama-3-8b", "mistralai/Mistral-7B-v0.1"]

# Hive 2: Another GPU server
[[hive]]
alias = "gpu-server-2"
hostname = "192.168.1.101"
ssh_user = "vince"
ssh_port = 22
hive_port = 8600

workers = ["vllm", "comfyui"]

# Hive 3: Local hive (for attached mode)
[[hive]]
alias = "local-hive"
hostname = "localhost"
ssh_user = "vince"
ssh_port = 22
hive_port = 8600

workers = ["llama-cpp"]
```

**Key Features:**
- Declare all hives in one place
- Declare which workers each hive should have
- Optional: Declare which models to pre-download

#### 3. `~/.config/rbee/models.conf` (Optional - Model Configuration)

```toml
# Model catalog (optional)
# If specified, rbee will pre-download these models

[[model]]
name = "meta-llama/Llama-3-8b"
source = "huggingface"
# Pre-download to all hives
hives = ["gpu-server-1", "gpu-server-2"]

[[model]]
name = "mistralai/Mistral-7B-v0.1"
source = "huggingface"
# Pre-download to specific hive
hives = ["gpu-server-1"]

[[model]]
name = "stable-diffusion-v1-5"
source = "huggingface"
hives = ["gpu-server-2"]
```

**Optional:** Models can be downloaded on-demand instead of pre-configured.

---

## Package Manager Commands

### 1. `rbee install` (Main Command)

**Purpose:** Install everything declared in config files

```bash
# Install everything (queen + all hives + all workers)
rbee install

# Install specific components
rbee install --queen-only
rbee install --hives-only
rbee install --hive gpu-server-1
rbee install --workers-only

# Dry run (show what would be installed)
rbee install --dry-run

# Force reinstall
rbee install --force
```

**Behavior:**
1. Read `rbee.conf` and `hives.conf`
2. Validate configuration
3. Install queen (if needed)
4. Install hives **concurrently** via SSH
5. Install workers **concurrently** on each hive
6. (Optional) Pre-download models **concurrently**
7. Report success/failure for each component

**Output:**
```
📦 Installing rbee components...

✅ Queen: Already installed (standalone mode)
🔄 Hives: Installing 3 hives concurrently...
   ✅ gpu-server-1: Installed rbee-hive v0.1.0
   ✅ gpu-server-2: Installed rbee-hive v0.1.0
   ✅ local-hive: Installed rbee-hive v0.1.0

🔄 Workers: Installing workers concurrently...
   ✅ gpu-server-1: Installed vllm, llama-cpp
   ✅ gpu-server-2: Installed vllm, comfyui
   ✅ local-hive: Installed llama-cpp

✅ Installation complete! (12.3s)
```

### 2. `rbee uninstall` (Cleanup)

```bash
# Uninstall everything
rbee uninstall

# Uninstall specific components
rbee uninstall --queen-only
rbee uninstall --hive gpu-server-1
rbee uninstall --workers-only
```

### 3. `rbee sync` (Reconcile State)

**Purpose:** Ensure actual state matches declared state

```bash
# Check if actual state matches config
rbee sync --check

# Fix any drift (install missing, remove extra)
rbee sync

# Show what would change
rbee sync --dry-run
```

**Example:**
```
🔍 Checking state...

Queen: ✅ OK (standalone mode)
Hives:
  ✅ gpu-server-1: OK
  ⚠️  gpu-server-2: Missing worker 'comfyui'
  ❌ gpu-server-3: Not in config (extra hive)

🔧 Syncing...
  ✅ Installed comfyui on gpu-server-2
  ⚠️  Skipping gpu-server-3 (use --remove-extra to uninstall)

✅ Sync complete!
```

### 4. `rbee status` (Show Current State)

```bash
# Show status of all components
rbee status

# Show status of specific component
rbee status --hive gpu-server-1
```

**Output:**
```
📊 rbee Status

Queen: ✅ Running (standalone mode, port 8500)

Hives:
  ✅ gpu-server-1 (192.168.1.100:8600)
     Workers: vllm ✅, llama-cpp ✅
     VRAM: 16GB / 24GB (66%)
  
  ✅ gpu-server-2 (192.168.1.101:8600)
     Workers: vllm ✅, comfyui ✅
     VRAM: 20GB / 24GB (83%)
  
  ✅ local-hive (localhost:8600)
     Workers: llama-cpp ✅
     VRAM: 0GB / 24GB (0% - CPU only)
```

---

## Pros of Declarative Configuration

### 1. **Infrastructure as Code** ⭐⭐⭐

**Current (Imperative):**
```bash
# User must remember and execute commands
rbee install-hive --alias gpu-1 --host 192.168.1.100
rbee install-worker --hive gpu-1 --type vllm
# ... repeat for each component
```

**Declarative:**
```toml
# hives.conf - Version controlled!
[[hive]]
alias = "gpu-1"
hostname = "192.168.1.100"
workers = ["vllm"]
```

```bash
# Single command
rbee install
```

**Benefits:**
- ✅ **Version control** - Config files in git
- ✅ **Reproducible** - Same config = same setup
- ✅ **Documented** - Config IS documentation
- ✅ **Shareable** - Team members use same config

### 2. **Concurrent Installation** ⭐⭐⭐

**Current (Sequential):**
```bash
# Must install one at a time
rbee install-hive --alias hive-1  # 30s
rbee install-hive --alias hive-2  # 30s
rbee install-hive --alias hive-3  # 30s
# Total: 90 seconds
```

**Declarative (Concurrent):**
```bash
# Install all hives at once
rbee install
# Hive 1: 30s ┐
# Hive 2: 30s ├─ Parallel
# Hive 3: 30s ┘
# Total: 30 seconds
```

**Benefits:**
- ✅ **3x faster** for 3 hives
- ✅ **10x faster** for 10 hives
- ✅ **Scales** with number of hives

### 3. **Desired State Management** ⭐⭐⭐

**Current:**
- No concept of "desired state"
- User must manually track what's installed
- Drift detection is manual

**Declarative:**
```bash
# Config declares desired state
rbee sync --check
# ⚠️  gpu-server-2: Missing worker 'comfyui'

# Fix drift automatically
rbee sync
# ✅ Installed comfyui on gpu-server-2
```

**Benefits:**
- ✅ **Drift detection** - Know when actual ≠ desired
- ✅ **Self-healing** - `rbee sync` fixes drift
- ✅ **Idempotent** - Safe to run multiple times

### 4. **Easier Onboarding** ⭐⭐⭐

**Current:**
```bash
# New user must:
git clone https://github.com/you/llama-orch
cargo build --release
# ... 10+ manual steps
```

**Declarative:**
```bash
# New user:
curl -sSL https://rbee.dev/install.sh | sh  # Install rbee CLI
git clone https://github.com/you/rbee-config  # Get team config
cd rbee-config
rbee install  # Done!
```

**Benefits:**
- ✅ **One command** to set up everything
- ✅ **Team config** shared via git
- ✅ **No manual steps**

### 5. **Better Error Handling** ⭐⭐

**Current:**
```bash
# If hive-2 install fails, user must:
# 1. Notice the error
# 2. Manually retry
# 3. Remember which hives succeeded
```

**Declarative:**
```bash
rbee install
# ✅ hive-1: OK
# ❌ hive-2: SSH connection failed
# ✅ hive-3: OK

# Retry just the failed ones
rbee install --retry-failed
# ✅ hive-2: OK (retried)
```

**Benefits:**
- ✅ **Partial success** - Don't lose progress
- ✅ **Retry failed** - Easy to fix errors
- ✅ **Clear reporting** - Know what failed

### 6. **Queen Mode Configuration** ⭐⭐⭐

**Current:**
```bash
# User must choose at runtime
rbee start-queen --attached-hive
# or
rbee start-queen
```

**Declarative:**
```toml
# rbee.conf
[queen]
mode = "standalone"  # or "attached-hive"
```

**Benefits:**
- ✅ **Declared upfront** - No runtime decisions
- ✅ **Documented** - Mode is in config
- ✅ **Consistent** - Same mode every time

---

## Cons of Declarative Configuration

### 1. **Configuration Complexity** ⭐⭐

**Current (Simple):**
```bash
# Just run commands
rbee install-queen
```

**Declarative (More Complex):**
```toml
# Must learn TOML syntax
[queen]
mode = "standalone"

[[hive]]
alias = "gpu-1"
hostname = "192.168.1.100"
workers = ["vllm"]
```

**Challenges:**
- ❌ **Learning curve** - Users must learn config format
- ❌ **Syntax errors** - TOML parsing can fail
- ❌ **More files** - 2-3 config files vs 0

**Mitigation:**
- ✅ Provide `rbee init` to generate default config
- ✅ Validate config with helpful error messages
- ✅ Provide examples and templates

### 2. **Less Flexibility for One-Off Tasks** ⭐

**Current:**
```bash
# Easy to test one hive
rbee install-hive --alias test-hive --host 192.168.1.200
```

**Declarative:**
```toml
# Must edit config file
[[hive]]
alias = "test-hive"
hostname = "192.168.1.200"
workers = []
```

```bash
rbee install --hive test-hive
```

**Challenges:**
- ❌ **Extra step** - Must edit config first
- ❌ **Temporary changes** - Must remember to remove from config

**Mitigation:**
- ✅ Support imperative commands for one-offs: `rbee install-hive --alias test --host 192.168.1.200 --temp`
- ✅ `--temp` flag doesn't add to config

### 3. **Model Pre-Download Debate** ⭐⭐⭐

**Question:** Should models be in config?

**Option A: Models in Config**
```toml
[[model]]
name = "meta-llama/Llama-3-8b"
hives = ["gpu-server-1"]
```

**Pros:**
- ✅ Reproducible - Same models everywhere
- ✅ Pre-downloaded - Ready to use
- ✅ Documented - Know what's available

**Cons:**
- ❌ **Huge downloads** - 70GB+ per model
- ❌ **Slow install** - `rbee install` takes hours
- ❌ **Inflexible** - Can't easily try new models
- ❌ **Storage waste** - Models on all hives

**Option B: Models On-Demand (RECOMMENDED)**
```bash
# Models NOT in config
# Download when needed
rbee model download meta-llama/Llama-3-8b --hive gpu-server-1
```

**Pros:**
- ✅ **Fast install** - No model downloads
- ✅ **Flexible** - Download as needed
- ✅ **Storage efficient** - Only download what you use

**Cons:**
- ❌ First inference is slow (must download model first)

**Hybrid Approach (BEST):**
```toml
# Optional: Pre-download critical models
[[model]]
name = "meta-llama/Llama-3-8b"
hives = ["gpu-server-1"]
# Other models downloaded on-demand
```

### 4. **Config Drift** ⭐⭐

**Problem:** What if user manually installs something?

```bash
# User manually installs worker
ssh gpu-server-1 'rbee-hive install-worker --type comfyui'

# Now config doesn't match reality
rbee sync --check
# ⚠️  gpu-server-1: Extra worker 'comfyui' (not in config)
```

**Challenges:**
- ❌ **Manual changes** - Config becomes stale
- ❌ **Sync confusion** - What should `rbee sync` do?

**Mitigation:**
- ✅ `rbee sync --check` detects drift
- ✅ `rbee sync --adopt` adds manual changes to config
- ✅ `rbee sync --remove-extra` removes manual changes

### 5. **Migration Effort** ⭐

**Current users must:**
1. Create config files
2. Migrate existing setup
3. Learn new commands

**Challenges:**
- ❌ **Breaking change** - Old commands deprecated
- ❌ **Migration work** - Users must update

**Mitigation:**
- ✅ Provide `rbee migrate` command to generate config from current setup
- ✅ Support old commands with deprecation warnings
- ✅ Gradual migration path

---

## Implementation Strategy

### Phase 1: Config File Support (8-12 hours)

**Add config parsing:**
```rust
// bin/99_shared_crates/rbee-config/src/lib.rs

#[derive(Deserialize)]
pub struct RbeeConfig {
    pub queen: QueenConfig,
    pub hives: Vec<HiveConfig>,
    pub models: Option<Vec<ModelConfig>>,
}

#[derive(Deserialize)]
pub struct QueenConfig {
    pub mode: QueenMode,  // "standalone" or "attached-hive"
    pub port: u16,
    pub auto_start: bool,
}

#[derive(Deserialize)]
pub struct HiveConfig {
    pub alias: String,
    pub hostname: String,
    pub ssh_user: String,
    pub ssh_port: u16,
    pub hive_port: u16,
    pub workers: Vec<String>,
}
```

**Files:**
- `bin/99_shared_crates/rbee-config/src/declarative.rs` (new)
- Update `rbee-config` to support declarative config

### Phase 2: `rbee install` Command (12-16 hours)

**Add install command:**
```rust
// bin/00_rbee_keeper/src/commands/install.rs

pub async fn install_all(config: RbeeConfig) -> Result<()> {
    // 1. Install queen
    install_queen(&config.queen).await?;
    
    // 2. Install hives concurrently
    let hive_futures = config.hives.iter().map(|hive| {
        install_hive(hive)
    });
    futures::future::join_all(hive_futures).await;
    
    // 3. Install workers concurrently
    let worker_futures = config.hives.iter().flat_map(|hive| {
        hive.workers.iter().map(|worker| {
            install_worker(hive, worker)
        })
    });
    futures::future::join_all(worker_futures).await;
    
    Ok(())
}
```

**Features:**
- Concurrent installation
- Progress reporting
- Error handling with retry

### Phase 3: `rbee sync` Command (8-12 hours)

**Add sync command:**
```rust
// bin/00_rbee_keeper/src/commands/sync.rs

pub async fn sync(config: RbeeConfig, opts: SyncOptions) -> Result<()> {
    // 1. Query actual state
    let actual = query_actual_state().await?;
    
    // 2. Compare with desired state
    let diff = compare_states(&config, &actual);
    
    // 3. Apply changes
    if opts.check_only {
        print_diff(&diff);
    } else {
        apply_diff(&diff).await?;
    }
    
    Ok(())
}
```

### Phase 4: Migration Tool (4-6 hours)

**Add migrate command:**
```bash
# Generate config from current setup
rbee migrate --output ~/.config/rbee/

# Creates:
# - rbee.conf
# - hives.conf
# - models.conf (optional)
```

---

## Recommendation

### For v0.1.0 - v0.3.0: Imperative (Current)

**Keep imperative commands:**
- Simpler to implement
- Easier to debug
- Faster iteration

**Rationale:**
- Focus on core functionality
- Prove architecture works
- Gather user feedback

### For v0.4.0: Add Declarative (Hybrid)

**Add declarative config:**
- Support both imperative and declarative
- `rbee install` reads config
- Old commands still work

**Rationale:**
- Users want easier setup
- Infrastructure as code is valuable
- Concurrent installation is faster

### For v1.0.0: Declarative-First

**Make declarative primary:**
- Config files are main interface
- Imperative commands for one-offs
- Full sync support

**Rationale:**
- Production deployments prefer declarative
- Better for teams
- Industry standard (Kubernetes, Terraform, etc.)

---

## Model Configuration Decision

### Recommendation: On-Demand (No Models in Config)

**Rationale:**
1. **Models are huge** (70GB+) - Slow install
2. **Models change frequently** - Users experiment
3. **Models are per-inference** - Not infrastructure

**Instead:**
```bash
# Download models on-demand
rbee model download meta-llama/Llama-3-8b --hive gpu-server-1

# Or let first inference download automatically
rbee infer "Hello!" --model meta-llama/Llama-3-8b
# ⚠️  Model not found, downloading... (70GB, 5 minutes)
# ✅ Model downloaded
# 🤖 Response: "Hello! How can I help you?"
```

**Optional:** Allow models in config for critical production models:
```toml
# Optional: Pre-download critical models
[[model]]
name = "meta-llama/Llama-3-8b"
hives = ["gpu-server-1"]
pre_download = true  # Download during `rbee install`
```

---

## Example Workflows

### Workflow 1: New User Setup

```bash
# 1. Install rbee CLI
curl -sSL https://rbee.dev/install.sh | sh

# 2. Initialize config
rbee init
# ✅ Created ~/.config/rbee/rbee.conf
# ✅ Created ~/.config/rbee/hives.conf

# 3. Edit config (add your hives)
vim ~/.config/rbee/hives.conf

# 4. Install everything
rbee install
# ✅ Queen installed
# ✅ 3 hives installed concurrently
# ✅ 6 workers installed concurrently
# ✅ Done! (15 seconds)

# 5. Start everything
rbee start
# ✅ Queen started
# ✅ 3 hives started
```

### Workflow 2: Add New Hive

```bash
# 1. Edit config
vim ~/.config/rbee/hives.conf
# Add new hive entry

# 2. Install just the new hive
rbee install --hive new-hive
# ✅ new-hive installed

# 3. Verify
rbee status
# ✅ new-hive: Running
```

### Workflow 3: Team Setup

```bash
# 1. Clone team config repo
git clone https://github.com/company/rbee-config
cd rbee-config

# 2. Install everything
rbee install --config ./rbee.conf
# ✅ Team setup complete!

# 3. Everyone has same setup
```

### Workflow 4: Drift Detection

```bash
# 1. Someone manually installs worker
ssh gpu-server-1 'rbee-hive install-worker --type comfyui'

# 2. Detect drift
rbee sync --check
# ⚠️  gpu-server-1: Extra worker 'comfyui'

# 3. Fix drift (remove extra)
rbee sync --remove-extra
# ✅ Removed comfyui from gpu-server-1

# OR adopt the change
rbee sync --adopt
# ✅ Added comfyui to config
```

---

## Comparison Table

| Aspect | Imperative | Declarative | Winner |
|--------|-----------|-------------|--------|
| **Ease of Use (First Time)** | Easy | Medium | 🔧 Imperative |
| **Ease of Use (Ongoing)** | Medium | Easy | 📦 Declarative |
| **Speed (Single Hive)** | Fast | Fast | ⚖️ Tie |
| **Speed (Multiple Hives)** | Slow (sequential) | Fast (concurrent) | 📦 Declarative |
| **Reproducibility** | Poor | Excellent | 📦 Declarative |
| **Version Control** | No | Yes | 📦 Declarative |
| **Team Collaboration** | Hard | Easy | 📦 Declarative |
| **Drift Detection** | Manual | Automatic | 📦 Declarative |
| **One-Off Tasks** | Easy | Medium | 🔧 Imperative |
| **Complexity** | Low | Medium | 🔧 Imperative |
| **Industry Standard** | Old | Modern | 📦 Declarative |

**Score:** Declarative wins 8-3

---

## Conclusion

### Declarative Configuration is Better Long-Term

**Benefits:**
- ✅ Infrastructure as code
- ✅ Concurrent installation (3-10x faster)
- ✅ Desired state management
- ✅ Reproducible setups
- ✅ Team collaboration
- ✅ Industry standard

**Challenges:**
- ❌ More complex initially
- ❌ Learning curve
- ❌ Migration effort

### Recommended Path

1. **v0.1.0 - v0.3.0:** Imperative (current)
   - Focus on core functionality
   - Prove architecture

2. **v0.4.0:** Add declarative (hybrid)
   - Support both approaches
   - `rbee install` command
   - Concurrent installation

3. **v1.0.0:** Declarative-first
   - Config files primary
   - Full sync support
   - Production-ready

### Model Configuration

**Recommendation:** **On-demand, NOT in config**

**Rationale:**
- Models are huge (70GB+)
- Models change frequently
- Models are per-inference, not infrastructure

**Optional:** Allow pre-download for critical production models

---

## Next Steps

If you decide to implement declarative config:

1. **Create config schema** (rbee.conf, hives.conf)
2. **Implement `rbee install`** with concurrent installation
3. **Add `rbee sync`** for drift detection
4. **Provide `rbee init`** to generate default config
5. **Document migration** from imperative to declarative

**Start with hybrid approach** - support both imperative and declarative!
