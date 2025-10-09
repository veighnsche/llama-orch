# TEAM-022 Handoff: Infrastructure for Multi-Model Testing

**Date:** 2025-10-09  
**From:** TEAM-021 (Root Cause Analysis Team)  
**To:** TEAM-022 (Infrastructure Team)  
**Status:** ✅ CRITICAL BUG FIXED - Ready to build infrastructure

---

## Executive Summary

**TEAM-021 RESOLVED the Metal broadcasting bug!** 🎉

**Root Cause:** Cache pollution from warmup phase (OUR architectural bug, NOT Candle)  
**Solution:** Proper cache lifecycle management - reset cache before each inference request  
**Result:** llorch-candled works correctly on all backends (CPU, Metal, CUDA)

**THE REAL GOAL:** Multi-model testing across the entire infrastructure

**THE BLOCKER:** We don't have the infrastructure yet!

**Your Mission:**
1. **Port bash scripts to Rust** - llorch-ctl and pool-ctl binaries
2. **Build model catalog system** - Each pool manager tracks available models
3. **Implement pool management** - Model downloads, worker spawning
4. **Enable multi-model testing** - Test Llama, Mistral, Phi, Qwen across pools

---

## What TEAM-021 Fixed ✅

### The Bug (RESOLVED)

**Error:** `cannot broadcast [5, 5] to [1, 32, 5, 7]`

**Root Cause:** We added a warmup phase that polluted the KV cache, then reused the polluted cache for inference.

**Our Broken Flow:**
```rust
// 1. Warmup (pollutes cache)
warmup() {
    forward("Hello", position=0, &mut cache)  // Cache has 2 tokens
}

// 2. Inference (uses polluted cache)
execute("prompt") {
    forward("prompt", position=0, &mut cache)  // Cache STILL has warmup tokens!
    // Mask for 5 tokens, attention needs 5 + 2 = 7
    // ERROR: cannot broadcast [5, 5] to [1, 32, 5, 7]
}
```

**The Fix:**
```rust
// Reset cache before each inference request
pub async fn execute(&mut self, request: InferenceRequest) -> Result<()> {
    // Clear any warmup pollution
    self.cache = Cache::new(true, DType::F32, &self.config, &self.device)?;
    
    // Now inference starts clean
    forward(&input, 0, &mut self.cache)?;  // ✅ Works!
}
```

### What We Learned

**TEAM-019 and TEAM-020 were WRONG:**
- ❌ Candle does NOT have a Metal mask broadcasting bug
- ❌ No fork needed
- ❌ No workarounds needed
- ✅ Our code was not Candle-idiomatic

**Candle's Pattern (Correct):**
- Cache lives for ONE continuous generation
- No warmup phase
- No cache clearing between tokens
- Position tracking is cumulative (0, 5, 6, 7...)

**Our Pattern (Fixed):**
- Cache reset before each inference request
- Warmup is isolated (separate cache)
- Each request starts with clean cache
- Proper lifecycle management

### Files Changed

**Fixed:**
- `src/backend/models/llama.rs` - Added cache reset in execute()
- `src/backend/models/mod.rs` - Updated all model wrappers
- `.specs/TEAM_021_ROOT_CAUSE_ANALYSIS.md` - Documented findings

**Removed:**
- TEAM-019's workaround (cache recreation at position=0)
- TEAM-020's Candle fork (only had comments anyway)

**Validated:**
- ✅ CPU backend works
- ✅ Metal backend works (no broadcasting errors!)
- ✅ CUDA backend works
- ✅ Llama model tested on all backends

---

## The Real Problem: No Infrastructure Yet! 🚧

**Current State:**
- ✅ llorch-candled works (single worker, single model)
- ❌ No llorch-ctl (orchestrator CLI)
- ❌ No pool-ctl (pool manager CLI)
- ❌ No model catalog system
- ❌ No pool management
- ❌ Can't test multiple models across pools

**What We Have:**
- Bash scripts: `scripts/homelab/llorch-remote` (SSH wrapper)
- Manual model downloads
- Manual worker spawning
- No automation

**What We Need:**
- Rust binaries: `llorch-ctl` and `pool-ctl`
- Model catalog per pool
- Automated model downloads
- Automated worker spawning
- Multi-model orchestration

---

## Your Mission: Build the Infrastructure 🏗️

### Priority 0: Port Bash Scripts to Rust (FOUNDATION)

**Goal:** Create llorch-ctl and pool-ctl binaries to replace bash scripts

**Current Bash Script:** `scripts/homelab/llorch-remote`
```bash
#!/usr/bin/env bash
# SSH wrapper for remote pool management
# Usage: llorch-remote <host> <backend> <command>

HOST="$1"
BACKEND="$2"
COMMAND="$3"

ssh "$HOST" "cd ~/Projects/llama-orch && ./target/release/llorch-${BACKEND}-candled ..."
```

**Target Rust Implementation:**

#### Task 0.1: Create pool-ctl Binary

**Location:** `bin/pool-ctl/`

**Purpose:** Local pool management (runs on pool manager host)

**Commands:**
```bash
# Model management
llorch-pool models download <model>
llorch-pool models list
llorch-pool models catalog

# Worker management
llorch-pool worker spawn <backend> --model <model> --gpu <id>
llorch-pool worker list
llorch-pool worker stop <worker-id>

# Pool status
llorch-pool status
llorch-pool health
```

**Implementation:**
```rust
// bin/pool-ctl/src/main.rs
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "llorch-pool")]
#[command(about = "Pool manager control CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Models {
        #[command(subcommand)]
        action: ModelsAction,
    },
    Worker {
        #[command(subcommand)]
        action: WorkerAction,
    },
    Status,
}

#[derive(Subcommand)]
enum ModelsAction {
    Download { model: String },
    List,
    Catalog,  // Show model catalog with metadata
}

#[derive(Subcommand)]
enum WorkerAction {
    Spawn {
        backend: String,
        #[arg(long)]
        model: String,
        #[arg(long, default_value = "0")]
        gpu: u32,
    },
    List,
    Stop { worker_id: String },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Models { action } => handle_models(action),
        Commands::Worker { action } => handle_worker(action),
        Commands::Status => show_status(),
    }
}
```

#### Task 0.2: Create llorch-ctl Binary

**Location:** `bin/llorch-ctl/`

**Purpose:** Orchestrator control (runs on orchestrator host, commands pools via SSH)

**Commands:**
```bash
# Pool management (via SSH)
llorch pool models download <model> --host <host>
llorch pool models list --host <host>
llorch pool worker spawn <backend> --model <model> --host <host>

# Job management (future - HTTP to orchestratord)
llorch jobs submit --model <model> --prompt <prompt>
llorch jobs list
llorch jobs cancel <job-id>
```

**Implementation:**
```rust
// bin/llorch-ctl/src/main.rs
use clap::{Parser, Subcommand};
use std::process::Command;

#[derive(Parser)]
#[command(name = "llorch")]
#[command(about = "Orchestrator control CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Pool {
        #[command(subcommand)]
        action: PoolAction,
    },
}

#[derive(Subcommand)]
enum PoolAction {
    Models {
        #[command(subcommand)]
        action: ModelsAction,
        #[arg(long)]
        host: String,
    },
    Worker {
        #[command(subcommand)]
        action: WorkerAction,
        #[arg(long)]
        host: String,
    },
}

fn handle_pool_models(action: ModelsAction, host: &str) -> anyhow::Result<()> {
    // SSH to pool and run llorch-pool command
    let ssh_cmd = match action {
        ModelsAction::Download { model } => {
            format!("cd ~/Projects/llama-orch && llorch-pool models download {}", model)
        }
        ModelsAction::List => {
            "cd ~/Projects/llama-orch && llorch-pool models list".to_string()
        }
    };
    
    let status = Command::new("ssh")
        .arg(host)
        .arg(&ssh_cmd)
        .status()?;
    
    if !status.success() {
        anyhow::bail!("SSH command failed");
    }
    
    Ok(())
}
```

**Success Criteria:**
- [ ] pool-ctl binary compiles
- [ ] llorch-ctl binary compiles
- [ ] Can run commands locally (pool-ctl)
- [ ] Can run commands remotely via SSH (llorch-ctl)

---

### Priority 1: Model Catalog System

**Goal:** Each pool manager tracks available models with metadata

**Why:** Need to know what models are available on each pool before spawning workers

#### Task 1.1: Design Model Catalog Format

**Location:** `.test-models/catalog.json` (per pool)

**Format:**
```json
{
  "version": "1.0",
  "pool_id": "mac.home.arpa",
  "models": [
    {
      "id": "tinyllama",
      "name": "TinyLlama 1.1B Chat",
      "path": ".test-models/tinyllama",
      "format": "safetensors",
      "size_gb": 2.2,
      "architecture": "llama",
      "downloaded": true,
      "backends": ["cpu", "metal"],
      "metadata": {
        "repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "context_length": 2048,
        "vocab_size": 32000
      }
    },
    {
      "id": "qwen-0.5b",
      "name": "Qwen2.5 0.5B Instruct",
      "path": ".test-models/qwen-safetensors",
      "format": "safetensors",
      "size_gb": 1.0,
      "architecture": "qwen",
      "downloaded": false,
      "backends": ["cpu", "metal", "cuda"],
      "metadata": {
        "repo": "Qwen/Qwen2.5-0.5B-Instruct",
        "context_length": 32768,
        "vocab_size": 151936
      }
    }
  ]
}
```

#### Task 1.2: Implement Catalog Management in pool-ctl

**Commands:**
```bash
# Show catalog
llorch-pool models catalog

# Add model to catalog (before download)
llorch-pool models register <model-id> \
    --name "TinyLlama 1.1B" \
    --repo "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --architecture llama

# Update catalog after download
llorch-pool models update <model-id> --downloaded true

# Remove from catalog
llorch-pool models unregister <model-id>
```

**Implementation:**
```rust
// bin/pool-ctl/src/catalog.rs
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelCatalog {
    pub version: String,
    pub pool_id: String,
    pub models: Vec<ModelEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    pub path: PathBuf,
    pub format: String,
    pub size_gb: f64,
    pub architecture: String,
    pub downloaded: bool,
    pub backends: Vec<String>,
    pub metadata: serde_json::Value,
}

impl ModelCatalog {
    pub fn load() -> anyhow::Result<Self> {
        let path = PathBuf::from(".test-models/catalog.json");
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content)?)
    }
    
    pub fn save(&self) -> anyhow::Result<()> {
        let path = PathBuf::from(".test-models/catalog.json");
        std::fs::create_dir_all(".test-models")?;
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    pub fn list(&self) {
        println!("Model Catalog for {}", self.pool_id);
        println!("{:<15} {:<30} {:<10} {:<10}", "ID", "Name", "Downloaded", "Size");
        println!("{:-<70}", "");
        
        for model in &self.models {
            let status = if model.downloaded { "✅" } else { "❌" };
            println!("{:<15} {:<30} {:<10} {:.1} GB", 
                model.id, model.name, status, model.size_gb);
        }
    }
}
```

**Success Criteria:**
- [ ] Catalog format defined
- [ ] Catalog can be loaded/saved
- [ ] `llorch-pool models catalog` shows models
- [ ] Models can be registered/unregistered

---

### Priority 2: Automated Model Downloads

**Goal:** Implement `llorch-pool models download <model>` command

#### Task 2.1: Model Download Implementation

**Commands:**
```bash
# Download a model (updates catalog automatically)
llorch-pool models download tinyllama
llorch-pool models download qwen-0.5b
llorch-pool models download phi3
llorch-pool models download mistral
```

**Implementation:**
```rust
// bin/pool-ctl/src/commands/models.rs
use std::process::Command;

pub fn download_model(model_id: &str) -> anyhow::Result<()> {
    // Load catalog
    let mut catalog = ModelCatalog::load()?;
    
    // Find model entry
    let model = catalog.models.iter_mut()
        .find(|m| m.id == model_id)
        .ok_or_else(|| anyhow::anyhow!("Model {} not in catalog", model_id))?;
    
    if model.downloaded {
        println!("✅ Model {} already downloaded", model_id);
        return Ok(());
    }
    
    println!("📥 Downloading {} from {}", model.name, model.metadata["repo"]);
    
    // Download using huggingface-cli
    let status = Command::new("huggingface-cli")
        .args(&[
            "download",
            model.metadata["repo"].as_str().unwrap(),
            "--include", "*.safetensors", "*.json",
            "--local-dir", model.path.to_str().unwrap(),
        ])
        .status()?;
    
    if !status.success() {
        anyhow::bail!("Download failed");
    }
    
    // Update catalog
    model.downloaded = true;
    catalog.save()?;
    
    println!("✅ Model {} downloaded successfully", model_id);
    Ok(())
}
```

**Success Criteria:**
- [ ] `llorch-pool models download <model>` works
- [ ] Downloads SafeTensors format
- [ ] Updates catalog automatically
- [ ] Handles already-downloaded models

---

### Priority 3: Worker Spawning

**Goal:** Implement `llorch-pool worker spawn` command

#### Task 3.1: Worker Spawn Implementation

**Commands:**
```bash
# Spawn a worker for a specific model
llorch-pool worker spawn metal --model tinyllama --gpu 0
llorch-pool worker spawn cuda --model qwen-0.5b --gpu 0
llorch-pool worker spawn cpu --model phi3

# List running workers
llorch-pool worker list

# Stop a worker
llorch-pool worker stop worker-metal-0
```

**Implementation:**
```rust
// bin/pool-ctl/src/commands/worker.rs
use std::process::Command;

pub fn spawn_worker(backend: &str, model_id: &str, gpu: u32) -> anyhow::Result<()> {
    // Load catalog
    let catalog = ModelCatalog::load()?;
    
    // Find model
    let model = catalog.models.iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| anyhow::anyhow!("Model {} not found", model_id))?;
    
    if !model.downloaded {
        anyhow::bail!("Model {} not downloaded. Run: llorch-pool models download {}", 
            model_id, model_id);
    }
    
    // Check backend compatibility
    if !model.backends.contains(&backend.to_string()) {
        anyhow::bail!("Model {} doesn't support {} backend", model_id, backend);
    }
    
    let worker_id = format!("worker-{}-{}", backend, gpu);
    
    println!("🚀 Spawning worker:");
    println!("   ID: {}", worker_id);
    println!("   Backend: {}", backend);
    println!("   Model: {} ({})", model.name, model_id);
    println!("   GPU: {}", gpu);
    
    // Spawn llorch-candled
    let binary = format!("llorch-{}-candled", backend);
    let mut cmd = Command::new(&binary);
    cmd.args(&[
        "--worker-id", &worker_id,
        "--model", model.path.to_str().unwrap(),
        "--port", &format!("{}", 8000 + gpu),
    ]);
    
    if backend != "cpu" {
        cmd.args(&["--gpu", &gpu.to_string()]);
    }
    
    // Spawn as background process
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        unsafe {
            cmd.pre_exec(|| {
                libc::setsid();
                Ok(())
            });
        }
    }
    
    let child = cmd.spawn()?;
    
    println!("✅ Worker spawned (PID: {})", child.id());
    println!("   Endpoint: http://localhost:{}", 8000 + gpu);
    
    Ok(())
}
```

**Success Criteria:**
- [ ] `llorch-pool worker spawn` works
- [ ] Checks model is downloaded
- [ ] Checks backend compatibility
- [ ] Spawns worker as background process
- [ ] Worker registers with pool

---

### Priority 4: Multi-Model Testing Infrastructure

**Goal:** Enable testing multiple models across multiple pools

#### Task 4.1: Download All Test Models

**Models to Download:**
1. **TinyLlama** (already have) - 2.2GB
2. **Qwen 0.5B** (smallest) - 1GB
3. **Phi-3 Mini** (medium) - 5GB
4. **Mistral 7B** (largest) - 14GB

**Commands:**
```bash
# On each pool (mac.home.arpa, workstation.home.arpa)

# Register models in catalog
llorch-pool models register tinyllama \
    --name "TinyLlama 1.1B Chat" \
    --repo "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --architecture llama

llorch-pool models register qwen-0.5b \
    --name "Qwen2.5 0.5B Instruct" \
    --repo "Qwen/Qwen2.5-0.5B-Instruct" \
    --architecture qwen

llorch-pool models register phi3 \
    --name "Phi-3 Mini 4K Instruct" \
    --repo "microsoft/Phi-3-mini-4k-instruct" \
    --architecture phi

llorch-pool models register mistral \
    --name "Mistral 7B Instruct v0.2" \
    --repo "mistralai/Mistral-7B-Instruct-v0.2" \
    --architecture mistral

# Download models (start with smallest!)
llorch-pool models download qwen-0.5b    # 1GB - fastest
llorch-pool models download tinyllama    # 2.2GB
llorch-pool models download phi3         # 5GB
llorch-pool models download mistral      # 14GB - slowest
```

#### Task 4.2: Multi-Model Test Script

**Create:** `.docs/testing/test_all_models.sh`

```bash
#!/usr/bin/env bash
# Multi-model test across all pools
# Tests each model on each backend

set -euo pipefail

POOLS=("mac.home.arpa" "workstation.home.arpa")
MODELS=("tinyllama" "qwen-0.5b" "phi3" "mistral")

echo "Multi-Model Test Suite"
echo "======================"

for pool in "${POOLS[@]}"; do
    echo ""
    echo "Testing pool: $pool"
    echo "-------------------"
    
    # Get available backends for this pool
    if [[ "$pool" == "mac.home.arpa" ]]; then
        BACKENDS=("metal")
    else
        BACKENDS=("cuda")
    fi
    
    for backend in "${BACKENDS[@]}"; do
        for model in "${MODELS[@]}"; do
            echo ""
            echo "Testing $model on $backend @ $pool"
            
            # Spawn worker via llorch-ctl
            llorch pool worker spawn $backend \
                --model $model \
                --host $pool \
                --gpu 0
            
            sleep 5
            
            # Test inference
            curl -X POST http://$pool:8000/execute \
                -H "Content-Type: application/json" \
                -d '{"job_id":"test","prompt":"Hello","max_tokens":10}'
            
            # Stop worker
            llorch pool worker stop worker-$backend-0 --host $pool
            
            echo "✅ $model works on $backend @ $pool"
        done
    done
done

echo ""
echo "======================"
echo "All tests complete!"
```

**Success Criteria:**
- [ ] All 4 models downloaded on all pools
- [ ] Can spawn workers for each model
- [ ] Can test each model on each backend
- [ ] Test script runs successfully

**Test Order:** Qwen → Phi → Mistral (smallest to largest)

**For Each Model:**
```bash
# 1. CPU (fastest iteration)
cargo run --features cpu -- \
    --worker-id test-qwen \
    --model .test-models/qwen-safetensors \
    --port 8001

# Test inference
curl -X POST http://localhost:8001/execute \
    -H "Content-Type: application/json" \
    -d '{"job_id":"test","prompt":"Hello","max_tokens":10,"seed":42}'

# 2. Metal
./scripts/homelab/llorch-remote mac.home.arpa metal \
    --model .test-models/qwen-safetensors

# 3. CUDA
./scripts/homelab/llorch-remote workstation.home.arpa cuda \
    --model .test-models/qwen-safetensors
```

**Success Criteria:**
- [ ] Qwen works on CPU, Metal, CUDA
- [ ] Phi works on CPU, Metal, CUDA
- [ ] Mistral works on CPU, Metal, CUDA
- [ ] No broadcasting errors on any backend
- [ ] Tokens generated correctly

#### Task 1.3: Use Multi-Model Test Script

**Script:** `.docs/testing/test_multi_model.sh` (created by TEAM-020)

**Update Script:**
```bash
# Update model paths to SafeTensors
test_model "llama" "$MODELS_DIR/tinyllama"
test_model "mistral" "$MODELS_DIR/mistral-safetensors"
test_model "phi" "$MODELS_DIR/phi3-safetensors"
test_model "qwen" "$MODELS_DIR/qwen-safetensors"
```

**Run Tests:**
```bash
# Test all models on each backend
./test_multi_model.sh cpu
./test_multi_model.sh metal
./test_multi_model.sh cuda
```

**Expected Output:**
```
Testing multi-model support on metal backend
================================================

Testing llama...
✅ llama works on metal

Testing mistral...
✅ mistral works on metal

Testing phi...
✅ phi works on metal

Testing qwen...
✅ qwen works on metal

================================================
Test Results: 4 passed, 0 failed, 0 skipped
================================================
```

---

### Priority 2: Performance Benchmarks

**Goal:** Measure production performance for each model/backend combination

#### Task 2.1: Tokens/Second Benchmarks

**Create:** `.docs/testing/benchmark_performance.sh`

```bash
#!/usr/bin/env bash
# Performance benchmark script
# Measures tokens/sec for each model/backend

set -euo pipefail

BACKEND="${1:-cpu}"
MODEL="${2:-.test-models/tinyllama}"
ITERATIONS="${3:-10}"

echo "Benchmarking $MODEL on $BACKEND backend ($ITERATIONS iterations)"

# Start worker
./target/release/llorch-${BACKEND}-candled \
    --worker-id bench \
    --model "$MODEL" \
    --port 9876 \
    --callback-url http://localhost:9999 &
WORKER_PID=$!

sleep 5

# Benchmark
TOTAL_TOKENS=0
TOTAL_TIME=0

for i in $(seq 1 $ITERATIONS); do
    START=$(date +%s.%N)
    
    RESPONSE=$(curl -s -X POST http://localhost:9876/execute \
        -H "Content-Type: application/json" \
        -d '{"job_id":"bench-'$i'","prompt":"Write a story","max_tokens":100,"seed":42}')
    
    END=$(date +%s.%N)
    DURATION=$(echo "$END - $START" | bc)
    
    TOKENS=$(echo "$RESPONSE" | jq -r '.tokens_generated // 0')
    
    TOTAL_TOKENS=$((TOTAL_TOKENS + TOKENS))
    TOTAL_TIME=$(echo "$TOTAL_TIME + $DURATION" | bc)
    
    echo "Iteration $i: $TOKENS tokens in ${DURATION}s"
done

kill $WORKER_PID 2>/dev/null || true

# Calculate average
AVG_TOKENS_PER_SEC=$(echo "scale=2; $TOTAL_TOKENS / $TOTAL_TIME" | bc)

echo ""
echo "================================================"
echo "Average: $AVG_TOKENS_PER_SEC tokens/sec"
echo "Total: $TOTAL_TOKENS tokens in ${TOTAL_TIME}s"
echo "================================================"
```

**Run Benchmarks:**
```bash
# Benchmark each model on each backend
./benchmark_performance.sh cpu .test-models/tinyllama 10
./benchmark_performance.sh metal .test-models/tinyllama 10
./benchmark_performance.sh cuda .test-models/tinyllama 10

# Repeat for Qwen, Phi, Mistral
```

**Document Results:**
Create `.docs/PERFORMANCE_BENCHMARKS.md`:
```markdown
# Performance Benchmarks

## Llama (TinyLlama 1.1B)
- CPU: X tokens/sec
- Metal: Y tokens/sec
- CUDA: Z tokens/sec

## Qwen (0.5B)
- CPU: X tokens/sec
- Metal: Y tokens/sec
- CUDA: Z tokens/sec

## Phi (3.8B)
- CPU: X tokens/sec
- Metal: Y tokens/sec
- CUDA: Z tokens/sec

## Mistral (7B)
- CPU: X tokens/sec
- Metal: Y tokens/sec
- CUDA: Z tokens/sec
```

#### Task 2.2: Memory Usage Profiling

**Measure:**
- VRAM usage per model
- System RAM usage
- Memory leaks (long-running tests)

**Tools:**
```bash
# Metal (Mac)
sudo powermetrics --samplers gpu_power -i 1000 -n 10

# CUDA (Linux)
nvidia-smi --query-gpu=memory.used --format=csv -l 1

# System RAM
top -l 1 | grep llorch-candled
```

**Document:**
```markdown
# Memory Usage

## Llama (TinyLlama 1.1B)
- VRAM: X GB
- System RAM: Y GB

## Qwen (0.5B)
- VRAM: X GB
- System RAM: Y GB

## Phi (3.8B)
- VRAM: X GB
- System RAM: Y GB

## Mistral (7B)
- VRAM: X GB
- System RAM: Y GB
```

---

### Priority 3: Stability Testing

**Goal:** Ensure production stability under load

#### Task 3.1: Extended Inference Runs

**Test:** Long-running inference sessions

```bash
# Run 1000 inference requests
for i in $(seq 1 1000); do
    curl -X POST http://localhost:8001/execute \
        -H "Content-Type: application/json" \
        -d '{"job_id":"stress-'$i'","prompt":"Test","max_tokens":50,"seed":'$i'}'
    
    if [ $((i % 100)) -eq 0 ]; then
        echo "Completed $i requests"
    fi
done
```

**Monitor:**
- [ ] No crashes
- [ ] No memory leaks
- [ ] Consistent performance
- [ ] No errors in logs

#### Task 3.2: Rapid Request Sequences

**Test:** Burst traffic handling

```bash
# 10 concurrent requests
for i in $(seq 1 10); do
    curl -X POST http://localhost:8001/execute \
        -H "Content-Type: application/json" \
        -d '{"job_id":"burst-'$i'","prompt":"Test","max_tokens":10}' &
done
wait
```

**Monitor:**
- [ ] All requests complete
- [ ] No race conditions
- [ ] Proper queueing
- [ ] No deadlocks

#### Task 3.3: Edge Case Testing

**Test Cases:**
1. **Very long prompts** (>2000 tokens)
2. **Empty prompts** ("")
3. **Special characters** (unicode, emojis)
4. **Rapid cancel requests**
5. **Invalid model paths**
6. **Corrupted model files**

**Expected:**
- Graceful error handling
- No crashes
- Helpful error messages
- Proper cleanup

---

### Priority 4: Production Deployment Prep

#### Task 4.1: Update Documentation

**Files to Update:**

1. **MODEL_SUPPORT.md** - Add multi-model test results
```markdown
# Model Support

## Tested Models

| Model | CPU | Metal | CUDA | Notes |
|-------|-----|-------|------|-------|
| Llama (TinyLlama 1.1B) | ✅ | ✅ | ✅ | Fully tested |
| Qwen (0.5B) | ✅ | ✅ | ✅ | Fully tested |
| Phi (3.8B) | ✅ | ✅ | ✅ | Fully tested |
| Mistral (7B) | ✅ | ✅ | ✅ | Fully tested |
```

2. **README.md** - Update with production status
```markdown
## Production Status

✅ All backends validated (CPU, Metal, CUDA)
✅ Multi-model support (Llama, Qwen, Phi, Mistral)
✅ Performance benchmarks completed
✅ Stability testing passed
✅ Using upstream Candle 0.9 (no fork needed)
```

3. **ARCHITECTURE.md** - Document cache lifecycle
```markdown
## Cache Lifecycle Management

Each inference request gets a fresh cache to prevent pollution:

```rust
pub async fn execute(&mut self, request: InferenceRequest) -> Result<()> {
    // Reset cache before each request
    self.cache = Cache::new(true, DType::F32, &self.config, &self.device)?;
    
    // Execute inference with clean cache
    forward(&input, 0, &mut self.cache)?;
}
```

This ensures:
- No cache pollution between requests
- Proper mask shape matching
- Consistent behavior across backends
```

#### Task 4.2: Create Production Checklist

**File:** `.docs/PRODUCTION_CHECKLIST.md`

```markdown
# Production Deployment Checklist

## Pre-Deployment
- [ ] All backends tested (CPU, Metal, CUDA)
- [ ] All models tested (Llama, Qwen, Phi, Mistral)
- [ ] Performance benchmarks completed
- [ ] Stability testing passed
- [ ] Memory profiling completed
- [ ] Documentation updated
- [ ] Using upstream Candle 0.9 (no fork)

## Deployment
- [ ] Binary compiled with --release
- [ ] Model files downloaded and verified
- [ ] Environment variables configured
- [ ] Monitoring setup (logs, metrics)
- [ ] Health check endpoint working
- [ ] Graceful shutdown tested

## Post-Deployment
- [ ] Smoke tests passed
- [ ] Performance monitoring active
- [ ] Error tracking configured
- [ ] Backup/recovery tested
- [ ] Rollback plan documented
```

#### Task 4.3: Create Runbook

**File:** `.docs/RUNBOOK.md`

```markdown
# llorch-candled Production Runbook

## Starting the Worker

```bash
# CPU backend
./llorch-cpu-candled \
    --worker-id worker-cpu-1 \
    --model .test-models/tinyllama \
    --port 8001 \
    --callback-url http://orchestrator:8080

# Metal backend (Mac)
./llorch-metal-candled \
    --worker-id worker-metal-1 \
    --model .test-models/tinyllama \
    --port 8001 \
    --gpu 0

# CUDA backend (Linux)
./llorch-cuda-candled \
    --worker-id worker-cuda-1 \
    --model .test-models/tinyllama \
    --port 8001 \
    --gpu 0
```

## Health Check

```bash
curl http://localhost:8001/health
# Expected: {"status":"ok","model":"tinyllama","backend":"metal"}
```

## Common Issues

### Broadcasting Error
**Symptom:** `cannot broadcast [X, X] to [1, 32, X, Y]`
**Cause:** Cache pollution (should be fixed in latest version)
**Fix:** Restart worker, ensure using latest code

### Out of Memory
**Symptom:** Worker crashes, CUDA OOM error
**Cause:** Model too large for GPU
**Fix:** Use smaller model or add more VRAM

### Slow Inference
**Symptom:** Tokens/sec below expected
**Cause:** CPU backend, thermal throttling, or background processes
**Fix:** Use GPU backend, check system load
```

---

## Success Criteria

Your work is complete when:

### Phase 1: CLI Infrastructure (Week 1)
- [ ] **pool-ctl binary** - Compiles and runs locally
- [ ] **llorch-ctl binary** - Compiles and runs remotely via SSH
- [ ] **Basic commands work** - models list, worker list, status
- [ ] **Tested on all pools** - mac.home.arpa, workstation.home.arpa

### Phase 2: Model Catalog (Week 2)
- [ ] **Catalog format defined** - JSON schema documented
- [ ] **Catalog management** - register, unregister, list, update
- [ ] **Catalogs created** - One per pool with all 4 models registered
- [ ] **Catalog persistence** - Survives restarts

### Phase 3: Automation (Week 3)
- [ ] **Model downloads work** - `llorch-pool models download <model>`
- [ ] **Worker spawning works** - `llorch-pool worker spawn <backend> --model <model>`
- [ ] **Qwen downloaded** - On all pools (1GB each)
- [ ] **Qwen tested** - Works on Metal and CUDA

### Phase 4: Multi-Model Testing (Week 4) 🎯
- [ ] **All models downloaded** - TinyLlama, Qwen, Phi, Mistral on all pools
- [ ] **Test script created** - `.docs/testing/test_all_models.sh`
- [ ] **All models tested** - Each model on each backend
- [ ] **Results documented** - MODEL_SUPPORT.md updated with matrix

**THE ULTIMATE GOAL:**
```
Model Support Matrix

| Model | Metal (Mac) | CUDA (Workstation) | Notes |
|-------|-------------|---------------------|-------|
| TinyLlama 1.1B | ✅ | ✅ | Fully tested |
| Qwen 0.5B | ✅ | ✅ | Fully tested |
| Phi-3 Mini | ✅ | ✅ | Fully tested |
| Mistral 7B | ✅ | ✅ | Fully tested |
```

---

## Timeline

### Week 1: CLI Infrastructure (Foundation)
- **Day 1-2:** Create pool-ctl binary (skeleton + commands)
- **Day 3-4:** Create llorch-ctl binary (SSH wrapper)
- **Day 5:** Test CLIs locally and remotely

### Week 2: Model Catalog System
- **Day 1:** Design catalog format
- **Day 2-3:** Implement catalog management in pool-ctl
- **Day 4:** Test catalog on all pools
- **Day 5:** Register all models in catalogs

### Week 3: Model Downloads & Worker Spawning
- **Day 1-2:** Implement model download command
- **Day 3:** Download Qwen (1GB) on all pools
- **Day 4:** Implement worker spawn command
- **Day 5:** Test worker spawning with Qwen

### Week 4: Multi-Model Testing (THE GOAL!)
- **Day 1:** Download remaining models (TinyLlama, Phi, Mistral)
- **Day 2:** Create multi-model test script
- **Day 3:** Test all models on Metal (mac.home.arpa)
- **Day 4:** Test all models on CUDA (workstation.home.arpa)
- **Day 5:** Document results, celebrate! 🎉

---

## Resources

### Critical Files

**Implementation:**
- `src/backend/models/llama.rs` - Fixed cache lifecycle
- `src/backend/models/mod.rs` - Model factory
- `src/backend/device.rs` - Device initialization

**Testing:**
- `.docs/testing/test_multi_model.sh` - Multi-model test script
- `.docs/testing/benchmark_performance.sh` - Performance benchmarks (create this)

**Documentation:**
- `.specs/TEAM_021_ROOT_CAUSE_ANALYSIS.md` - Bug investigation findings
- `docs/MODEL_SUPPORT.md` - Model compatibility matrix
- `.docs/PERFORMANCE_BENCHMARKS.md` - Benchmark results (create this)
- `.docs/PRODUCTION_CHECKLIST.md` - Deployment checklist (create this)
- `.docs/RUNBOOK.md` - Operations guide (create this)

### Model Sources

**SafeTensors Downloads:**
- Qwen: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- Phi: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
- Mistral: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2

### Testing Tools

- `scripts/homelab/llorch-remote` - Remote testing CLI
- `curl` - HTTP testing
- `jq` - JSON parsing
- `bc` - Calculations for benchmarks

---

## Handoff Checklist

### TEAM-021 Completed ✅
- [x] Root cause identified (cache pollution, NOT Candle bug)
- [x] Bug fixed (cache reset before each request)
- [x] All backends validated (CPU, Metal, CUDA)
- [x] Llama tested on all backends
- [x] TEAM-019/020 assessment disproven
- [x] Documentation updated (ROOT_CAUSE_ANALYSIS.md)

### TEAM-022 TODO 🚀 (Infrastructure for Multi-Model Testing)

**Week 1: CLI Foundation**
- [ ] Create pool-ctl binary (local pool management)
- [ ] Create llorch-ctl binary (remote pool control via SSH)
- [ ] Test CLIs on all pools

**Week 2: Model Catalog**
- [ ] Design catalog format (JSON schema)
- [ ] Implement catalog management in pool-ctl
- [ ] Create catalogs on all pools
- [ ] Register all 4 models in catalogs

**Week 3: Automation**
- [ ] Implement model download command
- [ ] Implement worker spawn command
- [ ] Download Qwen (1GB) on all pools
- [ ] Test Qwen on Metal and CUDA

**Week 4: Multi-Model Testing (THE GOAL!)**
- [ ] Download remaining models (TinyLlama, Phi, Mistral)
- [ ] Create multi-model test script
- [ ] Test all models on all backends
- [ ] Document results in MODEL_SUPPORT.md
- [ ] Celebrate! 🎉

---

**Handoff completed:** 2025-10-09  
**From:** TEAM-021 (Root Cause Analysis)  
**To:** TEAM-022 (Infrastructure Team)  
**Status:** ✅ BUG FIXED - Ready to build infrastructure  
**Next action:** Build llorch-ctl and pool-ctl to enable multi-model testing

**CRITICAL SUCCESS:**
- ✅ Bug was in OUR code (cache pollution)
- ✅ Candle is fine (no fork needed)
- ✅ All backends work correctly
- ✅ Proper cache lifecycle implemented

**THE REAL GOAL:** Multi-model testing across the entire infrastructure

**THE PATH:**
1. Build pool-ctl (local pool management)
2. Build llorch-ctl (remote pool control)
3. Implement model catalog system
4. Automate model downloads
5. Automate worker spawning
6. Test all 4 models on all backends

**YOUR JOB:** Build the infrastructure skeleton so we can finally test multiple models! 🚀

---

**Signed:**  
TEAM-021  
2025-10-09T15:22:00+02:00
