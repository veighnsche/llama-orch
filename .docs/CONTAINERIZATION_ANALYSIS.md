# Containerization Analysis for rbee Architecture

**Date:** Oct 23, 2025  
**Status:** PROPOSAL / ANALYSIS  
**Author:** TEAM-276

## Executive Summary

This document analyzes the feasibility, benefits, and challenges of containerizing the rbee orchestration stack (queen, hive, workers) using Docker/Podman instead of bare-metal binaries.

**Current State:** Bare-metal binaries installed from git source  
**Proposed State:** Container images with pre-built binaries

## Current Architecture (Bare Metal)

```
rbee-keeper (bare metal)
    ↓ manages
queen-rbee (bare metal binary)
    ↓ manages (via SSH)
rbee-hive (bare metal binary on remote hosts)
    ↓ manages (local spawn)
llm-worker (bare metal binary)
    ↓ loads
model files (from ~/.cache/rbee/models/)
```

**Installation:**
- `cargo build` from git source
- Binaries copied to `~/.local/bin/` or `/usr/local/bin/`
- Manual dependency management

## Proposed Architecture (Containerized)

```
rbee-keeper (bare metal)
    ↓ manages
queen-rbee (container: ghcr.io/you/rbee-queen:latest)
    ↓ manages (via SSH + docker/podman)
rbee-hive (container: ghcr.io/you/rbee-hive:latest on remote hosts)
    ↓ manages (docker/podman spawn)
llm-worker (container: ghcr.io/you/rbee-worker-vllm:latest)
    ↓ has model pre-attached OR mounts model volume
model files (volume mount: /models)
```

**Installation:**
- `docker pull ghcr.io/you/rbee-queen:latest`
- Pre-built binaries in container
- Dependencies bundled

---

## Pros of Containerization

### 1. **Distribution & Installation** ⭐⭐⭐

**Current Problem:**
```bash
# User must:
git clone https://github.com/you/llama-orch
cd llama-orch
cargo build --release  # Takes 10+ minutes
cp target/release/queen-rbee ~/.local/bin/
# Repeat for each binary
```

**Container Solution:**
```bash
# User does:
docker pull ghcr.io/you/rbee-queen:latest  # 30 seconds
docker pull ghcr.io/you/rbee-hive:latest
docker pull ghcr.io/you/rbee-worker-vllm:latest
```

**Benefits:**
- ✅ **No build time** - Pre-built binaries
- ✅ **No Rust toolchain** required on user machines
- ✅ **Faster onboarding** - Pull image vs compile
- ✅ **Versioned releases** - `rbee-queen:v0.1.0`, `rbee-queen:latest`
- ✅ **Automated builds** - GitHub Actions builds on every release

### 2. **Dependency Management** ⭐⭐⭐

**Current Problem:**
- CUDA drivers must match on host
- System libraries (OpenSSL, etc.) must be present
- Different distros = different paths

**Container Solution:**
```dockerfile
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
# All dependencies bundled
COPY target/release/rbee-hive /usr/local/bin/
```

**Benefits:**
- ✅ **Isolated dependencies** - No conflicts with host
- ✅ **Reproducible environment** - Same everywhere
- ✅ **CUDA bundled** - No host driver mismatch
- ✅ **Cross-distro** - Works on Ubuntu, Fedora, Arch, etc.

### 3. **GPU Passthrough** ⭐⭐⭐

**Docker/Podman Support:**
```bash
# Docker with NVIDIA GPU
docker run --gpus all ghcr.io/you/rbee-worker-vllm:latest

# Podman with NVIDIA GPU
podman run --device nvidia.com/gpu=all ghcr.io/you/rbee-worker-vllm:latest
```

**Benefits:**
- ✅ **Native GPU support** - Docker/Podman handle passthrough
- ✅ **Multi-GPU** - `--gpus '"device=0,1"'`
- ✅ **Resource limits** - `--gpus 1` for single GPU
- ✅ **Isolation** - Workers don't interfere with each other

### 4. **Model Pre-Attachment** ⭐⭐⭐

**Option A: Baked-in Models (for specific use cases)**
```dockerfile
FROM ghcr.io/you/rbee-worker-vllm:base
# Bake model into image
COPY models/llama-3-8b /models/llama-3-8b
ENV MODEL_PATH=/models/llama-3-8b
```

**Option B: Volume Mounts (flexible)**
```bash
docker run -v ~/.cache/rbee/models:/models \
    ghcr.io/you/rbee-worker-vllm:latest \
    --model /models/llama-3-8b
```

**Benefits:**
- ✅ **Fast startup** - Model already present (Option A)
- ✅ **Flexibility** - Mount different models (Option B)
- ✅ **Shared models** - Multiple workers share volume
- ✅ **Immutable workers** - Model can't be corrupted

### 5. **Isolation & Security** ⭐⭐

**Benefits:**
- ✅ **Process isolation** - Workers can't interfere
- ✅ **Resource limits** - `--memory 16g --cpus 4`
- ✅ **Network isolation** - Custom networks
- ✅ **Read-only root** - `--read-only` for security

### 6. **Orchestration Integration** ⭐⭐

**Future Possibilities:**
```yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rbee-hive
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: hive
        image: ghcr.io/you/rbee-hive:latest
```

**Benefits:**
- ✅ **Kubernetes-ready** - If you want to scale
- ✅ **Docker Compose** - Easy multi-container setup
- ✅ **Podman Pods** - Grouped containers

---

## Cons of Containerization

### 1. **Complexity** ⭐⭐⭐

**Current (Simple):**
```bash
queen-rbee --port 8500
```

**Container (More Complex):**
```bash
docker run -d \
    --name queen-rbee \
    --network rbee-net \
    -p 8500:8500 \
    -v ~/.config/rbee:/root/.config/rbee \
    ghcr.io/you/rbee-queen:latest
```

**Challenges:**
- ❌ **More moving parts** - Docker daemon, networks, volumes
- ❌ **Debugging harder** - Must `docker exec` to inspect
- ❌ **Logs scattered** - `docker logs` vs direct stdout
- ❌ **Learning curve** - Users must know Docker/Podman

### 2. **SSH to Remote Containers** ⭐⭐⭐

**Current (Simple):**
```bash
# Queen SSHs to hive host
ssh user@hive-1 'rbee-hive --port 8600'
```

**Container (Tricky):**
```bash
# Queen SSHs to hive host, then runs container
ssh user@hive-1 'docker run -d --gpus all ghcr.io/you/rbee-hive:latest'
```

**Challenges:**
- ❌ **Docker must be installed** on all hive hosts
- ❌ **Docker socket permissions** - User must be in `docker` group
- ❌ **SSH + Docker** - Two layers of indirection
- ❌ **Container lifecycle** - Must track container IDs, not PIDs

### 3. **Model Storage** ⭐⭐

**Challenge: Where do models live?**

**Option A: Host Volume Mount**
```bash
# Models on host, mounted into container
docker run -v /data/models:/models rbee-worker-vllm:latest
```
- ✅ Flexible (can change models)
- ❌ Models must be on host filesystem
- ❌ Multiple workers = multiple mounts

**Option B: Baked into Image**
```dockerfile
COPY models/llama-3-8b /models/llama-3-8b
```
- ✅ Fast startup (model already there)
- ❌ Huge images (70GB+ for large models)
- ❌ Inflexible (can't change model without new image)

**Option C: Shared Volume**
```bash
# All workers share a volume
docker volume create rbee-models
docker run -v rbee-models:/models rbee-worker-vllm:latest
```
- ✅ Shared across workers
- ❌ Must populate volume first
- ❌ Volume management complexity

### 4. **Performance Overhead** ⭐

**Concerns:**
- ❌ **Container overhead** - Minimal but present (~1-2%)
- ❌ **GPU passthrough** - Slightly slower than bare metal
- ❌ **Network overhead** - Container networking adds latency
- ❌ **I/O overhead** - Volume mounts slower than direct filesystem

**Reality Check:**
- For LLM inference, these overheads are **negligible** (<1%)
- GPU compute dominates, not container overhead
- Network latency is already present (HTTP between services)

### 5. **Attached Hive Mode** ⭐⭐

**Current:**
```bash
# Hive runs on same machine as queen
queen-rbee --attached-hive
```

**Container Challenge:**
- ❌ **Container-to-container** - Queen container must spawn hive container
- ❌ **Docker-in-Docker** - Queen needs Docker socket access
- ❌ **Networking** - Containers must communicate

**Solution:**
```bash
# Queen container with Docker socket mounted
docker run -v /var/run/docker.sock:/var/run/docker.sock \
    ghcr.io/you/rbee-queen:latest --attached-hive
```
- ✅ Works, but requires Docker socket access
- ❌ Security concern (container can spawn containers)

### 6. **Development Workflow** ⭐⭐

**Current (Fast):**
```bash
cargo build
./target/debug/queen-rbee  # Instant feedback
```

**Container (Slower):**
```bash
docker build -t rbee-queen:dev .  # 2-5 minutes
docker run rbee-queen:dev
```

**Mitigation:**
- Use multi-stage builds
- Cache layers aggressively
- Keep dev workflow bare-metal, prod containerized

---

## Hybrid Approach (Recommended)

### Option 1: Bare Metal Dev, Container Prod

**Development:**
- Developers run bare-metal binaries
- Fast iteration, easy debugging
- `cargo build && ./target/debug/queen-rbee`

**Production:**
- Users pull container images
- No build required
- `docker pull && docker run`

**Benefits:**
- ✅ Best of both worlds
- ✅ Fast dev cycle
- ✅ Easy prod deployment

### Option 2: Container-Optional

**Support Both:**
```bash
# Bare metal
rbee-keeper start-queen

# Container
rbee-keeper start-queen --container
```

**Implementation:**
```rust
// queen-lifecycle/src/start.rs
pub async fn start_queen(config: StartConfig) -> Result<()> {
    if config.use_container {
        start_queen_container(config).await
    } else {
        start_queen_binary(config).await
    }
}
```

**Benefits:**
- ✅ User choice
- ✅ Gradual migration
- ✅ Fallback if containers fail

---

## Implementation Strategy

### Phase 1: Containerize Workers First ⭐⭐⭐

**Why Workers?**
- Most benefit (GPU isolation, model pre-attachment)
- Least complexity (no SSH, no remote management)
- Easiest to test

**Steps:**
1. Create `Dockerfile` for each worker type (vllm, llama-cpp)
2. Build images with GitHub Actions
3. Update `worker-lifecycle` to support container spawn
4. Test with local Podman/Docker

**Example Dockerfile:**
```dockerfile
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install vLLM
RUN pip3 install vllm

# Copy rbee worker binary
COPY target/release/rbee-worker-vllm /usr/local/bin/

EXPOSE 9000

ENTRYPOINT ["rbee-worker-vllm"]
```

### Phase 2: Containerize Hive ⭐⭐

**Challenges:**
- Must support SSH spawning
- Must support GPU passthrough
- Must spawn worker containers

**Steps:**
1. Create `Dockerfile` for rbee-hive
2. Update `hive-lifecycle` to support container spawn via SSH
3. Test remote container spawning

**SSH Container Spawn:**
```rust
// hive-lifecycle/src/start.rs
async fn start_hive_container(config: HiveStartConfig) -> Result<()> {
    let ssh_cmd = format!(
        "docker run -d --name rbee-hive-{} \
         --gpus all \
         -p {}:8600 \
         -v /var/run/docker.sock:/var/run/docker.sock \
         ghcr.io/you/rbee-hive:{}",
        config.alias,
        config.port,
        config.version.unwrap_or("latest")
    );
    
    ssh_client.execute(&ssh_cmd).await?;
    Ok(())
}
```

### Phase 3: Containerize Queen ⭐

**Challenges:**
- Must manage remote hive containers
- Must handle attached hive mode
- Needs Docker socket access

**Steps:**
1. Create `Dockerfile` for queen-rbee
2. Update `queen-lifecycle` to support container spawn
3. Test with rbee-keeper

### Phase 4: Distribution ⭐⭐⭐

**GitHub Actions:**
```yaml
name: Build and Push Images

on:
  release:
    types: [published]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Queen Image
        run: docker build -t ghcr.io/${{ github.repository }}/rbee-queen:${{ github.ref_name }} .
      
      - name: Push to GHCR
        run: docker push ghcr.io/${{ github.repository }}/rbee-queen:${{ github.ref_name }}
```

**Benefits:**
- ✅ Automated builds on release
- ✅ Versioned images
- ✅ Free hosting (GitHub Container Registry)

---

## Model Pre-Attachment Strategies

### Strategy 1: Base Image + Model Layers

**Base Image (Small, ~2GB):**
```dockerfile
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
RUN pip3 install vllm
COPY target/release/rbee-worker-vllm /usr/local/bin/
```

**Model Images (Large, 70GB+):**
```dockerfile
FROM ghcr.io/you/rbee-worker-vllm:base
COPY models/llama-3-70b /models/llama-3-70b
ENV MODEL_PATH=/models/llama-3-70b
```

**Usage:**
```bash
docker pull ghcr.io/you/rbee-worker-vllm:llama-3-70b
docker run --gpus all ghcr.io/you/rbee-worker-vllm:llama-3-70b
```

**Pros:**
- ✅ Model ready to go
- ✅ Fast startup
- ✅ Immutable

**Cons:**
- ❌ Huge images (70GB+)
- ❌ Long pull times
- ❌ Storage intensive

### Strategy 2: Volume Mounts (Recommended)

**Image (Small, ~2GB):**
```dockerfile
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
RUN pip3 install vllm
COPY target/release/rbee-worker-vllm /usr/local/bin/
```

**Usage:**
```bash
# Download model to host
huggingface-cli download meta-llama/Llama-3-70b --local-dir /data/models/llama-3-70b

# Mount into container
docker run --gpus all \
    -v /data/models:/models \
    ghcr.io/you/rbee-worker-vllm:latest \
    --model /models/llama-3-70b
```

**Pros:**
- ✅ Small images
- ✅ Flexible (change models)
- ✅ Shared across workers

**Cons:**
- ❌ Models must be on host
- ❌ Extra download step

### Strategy 3: Model Registry (Advanced)

**Separate Model Storage:**
```bash
# Model registry (S3, MinIO, etc.)
s3://rbee-models/llama-3-70b/

# Worker downloads on startup
docker run --gpus all \
    -e MODEL_REGISTRY=s3://rbee-models \
    -e MODEL_NAME=llama-3-70b \
    ghcr.io/you/rbee-worker-vllm:latest
```

**Pros:**
- ✅ Centralized model storage
- ✅ No host filesystem dependency
- ✅ Easy model updates

**Cons:**
- ❌ Slow startup (download on first run)
- ❌ Requires model registry infrastructure
- ❌ Network dependency

---

## Recommendation

### Short Term (v0.1.0 - v0.3.0): Bare Metal

**Rationale:**
- Simpler to implement
- Easier to debug
- Faster development iteration
- No container complexity

**Keep:**
- Current bare-metal architecture
- Git source installation
- Direct binary execution

### Medium Term (v0.4.0 - v0.6.0): Hybrid

**Rationale:**
- Users want easier installation
- Containers provide better isolation
- GPU passthrough is mature

**Add:**
- Container images for workers (most benefit)
- Optional container mode for hive
- Keep bare-metal as default

### Long Term (v1.0.0+): Container-First

**Rationale:**
- Production deployments prefer containers
- Kubernetes integration
- Better resource management

**Transition:**
- Container images as primary distribution
- Bare-metal still supported (for dev)
- Full Kubernetes support

---

## Decision Matrix

| Aspect | Bare Metal | Containers | Winner |
|--------|-----------|------------|--------|
| **Installation** | Slow (build) | Fast (pull) | 🐳 Containers |
| **Dependencies** | Manual | Bundled | 🐳 Containers |
| **GPU Support** | Native | Passthrough | ⚖️ Tie |
| **Debugging** | Easy | Harder | 🔧 Bare Metal |
| **Development** | Fast | Slow | 🔧 Bare Metal |
| **Distribution** | Git source | Images | 🐳 Containers |
| **Isolation** | None | Strong | 🐳 Containers |
| **Complexity** | Low | High | 🔧 Bare Metal |
| **SSH Remote** | Easy | Tricky | 🔧 Bare Metal |
| **Model Pre-attach** | N/A | Possible | 🐳 Containers |

**Score:** Containers win 6-4, but complexity is a major concern

---

## Conclusion

### Recommended Path Forward

1. **v0.1.0 - v0.3.0:** Stay bare-metal
   - Focus on core functionality
   - Prove the architecture works
   - Fast iteration

2. **v0.4.0:** Add container support for workers
   - Most benefit, least complexity
   - Optional feature (`--container` flag)
   - Test with real users

3. **v0.5.0:** Add container support for hive
   - Remote container spawning via SSH
   - GPU passthrough validation
   - Production testing

4. **v0.6.0:** Add container support for queen
   - Full stack containerized
   - Bare-metal still default
   - Migration guide

5. **v1.0.0:** Container-first
   - Containers as primary distribution
   - Pre-built images on GHCR
   - Bare-metal for dev only

### Key Takeaways

✅ **Containers solve real problems:**
- Easier distribution
- Better isolation
- GPU passthrough works well

❌ **Containers add complexity:**
- More moving parts
- Harder debugging
- SSH + Docker = tricky

🎯 **Hybrid approach is best:**
- Start bare-metal (v0.1.0)
- Add containers gradually (v0.4.0+)
- Let users choose

---

## Next Steps

If you decide to pursue containerization:

1. **Create Dockerfiles** for each component
2. **Set up GitHub Actions** for automated builds
3. **Add `--container` flag** to lifecycle crates
4. **Test GPU passthrough** with real workloads
5. **Document container usage** in README
6. **Gather user feedback** on installation experience

**Start with workers** - they benefit most and are easiest to containerize!
