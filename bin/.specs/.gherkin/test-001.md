# Test-001: Cross-Node Inference Request Flow (CORRECTED)

**Updated by:** TEAM-038 (aligned with queen-rbee orchestration and narration architecture)  
**Updated by:** TEAM-052 (added backend detection and registry schema enhancements)  
**Updated by:** TEAM-075 (added GPU FAIL FAST policy and comprehensive error handling)  
**Updated by:** TEAM-077 (updated naming conventions: rbee-hive, worker-rbee, milestone alignment)  
**Date:** 2025-10-11

**Milestone:** M0-M1 (Worker standalone + Pool Manager Lifecycle)  
**Status:** Reference document for BDD test implementation

---

## Topology

- **blep** = blep.home.arpa (with rbee-keeper and queen-rbee, can run workers on cpu)
- **workstation** = workstation.home.arpa (with rbee-hive and llm-worker-rbee, can run workers on cuda device 0, 1 and cpu)

**Component Naming (TEAM-077):**
- **queen-rbee** - Orchestrator daemon (not "orchestratord")
- **rbee-hive** - Pool manager daemon (not "pool-managerd")
- **worker-rbee** - Worker daemon (llm-worker-rbee, sd-worker-rbee, etc.)
- **rbee-keeper** - CLI/UI tool

**Note:** This test uses **workstation** node with **cuda** backend on **device 1**.

---

## Test Objective

On **blep**, I want to run inference on **workstation**:
- **Model:** hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
- **Prompt:** "write a short story"
- **Max tokens:** 20
- **Temperature:** 0.7
- **Backend:** cuda, device: 1

**Command:**
```bash
rbee-keeper infer --node workstation --model hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  --prompt "write a short story" --max-tokens 20 --temperature 0.7 --backend cuda --device 1
```

---

## Prerequisites: rbee-hive Registry Setup

**CRITICAL:** Before any inference can happen, the user must configure remote rbee-hive nodes through **rbee-keeper** (configuration mode).

### Error Handling: Configuration Errors

**EH-011a: Invalid SSH Key Path**
- **Trigger:** User provides non-existent SSH key path
- **Detection:** rbee-keeper validates file existence before sending to queen-rbee
- **Response:** Immediate error with clear message
- **Exit Code:** 1
- **Message:** "SSH key not found: /path/to/key"

**EH-011b: Duplicate Node Name**
- **Trigger:** User tries to add node with existing name
- **Detection:** queen-rbee checks registry for duplicate
- **Response:** Error with suggestion to update or remove
- **Exit Code:** 1
- **Message:** "Node 'name' already exists in registry"

### rbee-hive Registry Module (queen-rbee)

**queen-rbee** maintains a **rbee-hive Registry** (SQLite at `~/.rbee/beehives.db`):

```sql
CREATE TABLE beehives (
    node_name TEXT PRIMARY KEY,
    ssh_host TEXT NOT NULL,
    ssh_port INTEGER DEFAULT 22,
    ssh_user TEXT NOT NULL,
    ssh_key_path TEXT,
    git_repo_url TEXT NOT NULL,
    git_branch TEXT DEFAULT 'main',
    install_path TEXT NOT NULL,
    last_connected_unix INTEGER,
    last_error TEXT,
    status TEXT DEFAULT 'unknown',  -- unknown, reachable, unreachable
    backends TEXT,  -- TEAM-052: JSON array: ["cuda", "metal", "cpu"]
    devices TEXT    -- TEAM-052: JSON object: {"cuda": 2, "metal": 1, "cpu": 1}
);
```

### Configuration Flow: rbee-keeper setup

**User runs configuration command:**
```bash
rbee-keeper setup add-node \
  --name workstation \
  --ssh-host workstation.home.arpa \
  --ssh-user vince \
  --ssh-key ~/.ssh/id_ed25519 \
  --git-repo https://github.com/user/llama-orch.git \
  --git-branch main \
  --install-path ~/rbee
```

**rbee-keeper** sends configuration to **queen-rbee**:
```
POST http://localhost:8080/v2/registry/beehives/add
{
  "node_name": "workstation",
  "ssh_host": "workstation.home.arpa",
  "ssh_port": 22,
  "ssh_user": "vince",
  "ssh_key_path": "/home/vince/.ssh/id_ed25519",
  "git_repo_url": "https://github.com/user/llama-orch.git",
  "git_branch": "main",
  "install_path": "/home/vince/rbee",
  "backends": "[\"cuda\",\"cpu\"]",
  "devices": "{\"cuda\":2,\"cpu\":1}"
}
```

**queen-rbee** validates SSH connection:
```bash
ssh -i ~/.ssh/id_ed25519 vince@workstation.home.arpa "echo 'connection test'"
```

**If successful, queen-rbee saves to registry:**
```sql
INSERT INTO beehives (node_name, ssh_host, ssh_port, ssh_user, ssh_key_path, 
                      git_repo_url, git_branch, install_path, 
                      last_connected_unix, status, backends, devices)
VALUES ('workstation', 'workstation.home.arpa', 22, 'vince', '/home/vince/.ssh/id_ed25519',
        'https://github.com/user/llama-orch.git', 'main', '/home/vince/rbee',
        1728508603, 'reachable', '["cuda","cpu"]', '{"cuda":2,"cpu":1}');
```

**Narration:**
```
narrate("Testing SSH connection to workstation.home.arpa")
  → stdout → rbee-keeper shell
  → USER SEES: [queen-rbee] 🔌 Testing SSH connection to workstation.home.arpa

narrate("SSH connection successful! Saving to registry")
  → stdout → rbee-keeper shell
  → USER SEES: [queen-rbee] ✅ SSH connection successful! Node 'workstation' saved to registry
```

### Error Handling: SSH Connection Failures

**EH-001a: SSH Connection Timeout**
- **Trigger:** Remote host unreachable or network down
- **Detection:** SSH connection timeout after 10s
- **Retry:** 3 attempts with exponential backoff (0ms, 200ms, 400ms)
- **Response:** Error after all retries fail
- **Exit Code:** 1
- **Message:** "SSH connection failed after 3 attempts"

**EH-001b: SSH Authentication Failure**
- **Trigger:** Wrong SSH key or permissions
- **Detection:** SSH returns "Permission denied (publickey)"
- **Response:** Immediate error with suggestion
- **Exit Code:** 1
- **Message:** "SSH authentication failed: Permission denied"
- **Suggestion:** "Check SSH key permissions: chmod 600 ~/.ssh/id_ed25519"

**EH-001c: SSH Command Execution Failure**
- **Trigger:** rbee-hive binary not found on remote node
- **Detection:** SSH command returns "command not found"
- **Response:** Error with installation suggestion
- **Exit Code:** 1
- **Message:** "Failed to start rbee-hive: command not found"
- **Suggestion:** "Install rbee-hive: rbee-keeper setup install --node workstation"

### Backend Detection (TEAM-052)

**On the remote node, detect available backends:**
```bash
# On workstation.home.arpa
rbee-hive detect
```

**Output:**
```
Backend Detection Results:
==========================

Available backends: 2
  - cpu: 1 device(s)
  - cuda: 2 device(s)

Total devices: 3

Registry format:
  backends: ["cpu","cuda"]
  devices:  {"cpu":1,"cuda":2}
```

**These backend capabilities are stored in the registry during node registration.**

### Optional: Initial Installation

**User can trigger initial installation:**
```bash
rbee-keeper setup install --node workstation
```

**queen-rbee** performs installation via SSH:
```bash
ssh vince@workstation.home.arpa << 'EOF'
  cd ~/rbee
  git clone https://github.com/user/llama-orch.git .
  git checkout main
  cargo build --release --bin rbee-hive
  cargo build --release --bin llm-worker-rbee
EOF
```

**Narration:**
```
narrate("Cloning repository on workstation")
  → stdout → rbee-keeper shell
  → USER SEES: [queen-rbee] 📦 Cloning repository on workstation

narrate("Building rbee-hive and llm-worker-rbee")
  → stdout → rbee-keeper shell
  → USER SEES: [queen-rbee] 🔨 Building rbee-hive and llm-worker-rbee

narrate("Installation complete!")
  → stdout → rbee-keeper shell
  → USER SEES: [queen-rbee] ✅ Installation complete on workstation!
```

---

## Complete Flow with Narration Paths

### Phase 0: queen-rbee loads rbee-hive registry

**Before any inference, queen-rbee loads registry:**
```sql
SELECT * FROM beehives WHERE node_name = 'workstation';
```

**Result:**
```
node_name: workstation
ssh_host: workstation.home.arpa
ssh_user: vince
ssh_key_path: /home/vince/.ssh/id_ed25519
install_path: /home/vince/rbee
status: reachable
```

**If node not found in registry:**
```
ERROR: Node 'workstation' not found in rbee-hive registry.
Run: rbee-keeper setup add-node --name workstation ...
```

---

### Phase 1: rbee-keeper → queen-rbee

**rbee-keeper** (on blep) sends task to **queen-rbee** (on blep):
```
POST http://localhost:8080/v2/tasks
{
  "node": "workstation",
  "model": "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
  "prompt": "write a short story",
  "max_tokens": 20,
  "temperature": 0.7,
  "backend": "cuda",
  "device": 1
}
```

**Narration:** None yet (just HTTP request)

---

### Phase 2: queen-rbee → rbee-hive (SSH)

**Phase 2a: SSH Preflight Checks (NEW - M1)**

**Before starting rbee-hive, queen-rbee validates SSH connectivity:**

**SSH Preflight Checks:**
1. **SSH connection reachable** - Test basic connectivity
2. **SSH authentication works** - Verify key-based auth
3. **SSH command execution** - Test `echo 'test'` command
4. **Network latency acceptable** - Check latency < 100ms
5. **rbee-hive binary exists** - Verify binary at install_path

**Error Scenarios:**
- **EH-001a:** SSH connection timeout (covered above)
- **EH-001b:** SSH authentication failure (covered above)
- **EH-001c:** SSH command execution failure (covered above)

**Narration:**
```
narrate("Running SSH preflight checks for workstation")
  → stdout → rbee-keeper shell
  → USER SEES: [queen-rbee] 🔍 Running SSH preflight checks for workstation

narrate("✅ SSH connectivity: OK")
narrate("✅ SSH authentication: OK")
narrate("✅ Command execution: OK")
narrate("✅ Network latency: 15ms")
narrate("✅ rbee-hive binary: Found at /home/vince/rbee/target/release/rbee-hive")
  → USER SEES: [queen-rbee] ✅ All SSH preflight checks passed
```

**If any check fails:**
```
narrate("❌ SSH preflight failed: rbee-hive binary not found")
  → USER SEES: [queen-rbee] ❌ SSH preflight failed: rbee-hive binary not found
  → USER SEES: Suggestion: Install rbee-hive: rbee-keeper setup install --node workstation
  → Exit code: 1
```

---

**Phase 2b: Start rbee-hive via SSH**

**Error Handling: HTTP Connection Failures**

**EH-002a: rbee-hive HTTP Connection Timeout**
- **Trigger:** rbee-hive process crashed or not responding
- **Detection:** HTTP request timeout after 10s
- **Retry:** 3 attempts with exponential backoff
- **Response:** Error after all retries fail
- **Exit Code:** 1
- **Message:** "Cannot connect to rbee-hive on workstation"
- **Suggestion:** "Check rbee-hive logs: ssh workstation journalctl -u rbee-hive -n 50"

**EH-002b: rbee-hive Returns Malformed JSON**
- **Trigger:** rbee-hive buggy or corrupted
- **Detection:** JSON parse error
- **Response:** Immediate error with suggestion
- **Exit Code:** 1
- **Message:** "Invalid response from rbee-hive"
- **Suggestion:** "rbee-hive may be corrupted, try restarting: ssh workstation pkill rbee-hive"

**queen-rbee** looks up SSH details from rbee-hive registry and starts **rbee-hive** on workstation via SSH:
```bash
# Using registry data: ssh_user@ssh_host with ssh_key_path
ssh -i /home/vince/.ssh/id_ed25519 vince@workstation.home.arpa "cd /home/vince/rbee && ./target/release/rbee-hive daemon --port 9200"
```

**queen-rbee updates registry with last_connected_unix:**
```sql
UPDATE beehives 
SET last_connected_unix = 1728508603, status = 'reachable'
WHERE node_name = 'workstation';
```

**rbee-hive startup narration:**
```
narrate("rbee-hive starting on port 9200")
  → stdout → SSH tunnel → queen-rbee
  → queen-rbee → stdout → rbee-keeper shell
  → USER SEES: [rbee-hive] 🌅 Starting pool manager on port 9200
```

**rbee-hive HTTP server ready:**
```
narrate("HTTP server listening on 0.0.0.0:9200")
  → stdout → SSH tunnel → queen-rbee
  → queen-rbee → stdout → rbee-keeper shell
  → USER SEES: [http-server] 🚀 HTTP server ready on port 9200
```

---

### Phase 3: queen-rbee checks worker registry

**queen-rbee** queries **rbee-hive** worker registry:
```
GET http://workstation.home.arpa:9200/v1/workers/list
```

**Response:** Empty (no workers yet)

**Narration:** None (just HTTP query)

---

### Phase 3a: rbee-hive Preflight Checks (NEW - M1)

**Before spawning workers, queen-rbee validates rbee-hive readiness:**

**rbee-hive Preflight Checks:**
1. **HTTP API responding** - GET /v1/health returns 200
2. **Version compatibility** - rbee-hive version compatible with queen-rbee
3. **Worker binaries available** - Check worker_binaries catalog
4. **Backend catalog populated** - CUDA/Metal/CPU detected
5. **Sufficient resources** - RAM, disk space available

**Worker Binaries Catalog Check (NEW - M1):**
```
GET http://workstation.home.arpa:9200/v1/worker-binaries/list
```

**Response:**
```json
{
  "binaries": [
    {
      "worker_type": "llm-worker-rbee",
      "version": "0.1.0",
      "binary_path": "/home/vince/rbee/target/release/llm-worker-rbee",
      "installed_at_unix": 1728508000
    }
  ]
}
```

**Backend Catalog Check:**
```
GET http://workstation.home.arpa:9200/v1/backends/list
```

**Response:**
```json
{
  "backends": [
    {"name": "cpu", "available": true, "devices": 1},
    {"name": "cuda", "available": true, "devices": 2}
  ]
}
```

**Error Scenarios:**

**EH-019a: Worker Binary Not Installed**
- **Trigger:** Requested worker type not in binaries catalog
- **Detection:** Worker binary not found in catalog
- **Response:** FAIL FAST with installation suggestion
- **Exit Code:** 1
- **Message:** "Worker binary not found: llm-worker-rbee"
- **Suggestion:** "Install worker: rbee-keeper setup install --node workstation"

**EH-019b: rbee-hive Version Incompatible**
- **Trigger:** rbee-hive version too old/new
- **Detection:** Version check fails
- **Response:** Error with upgrade suggestion
- **Exit Code:** 1
- **Message:** "rbee-hive version incompatible: need >=0.1.0, have 0.0.9"
- **Suggestion:** "Upgrade rbee-hive: rbee-keeper setup install --node workstation"

**Narration:**
```
narrate("Running rbee-hive preflight checks")
  → stdout → rbee-keeper shell
  → USER SEES: [queen-rbee] 🔍 Running rbee-hive preflight checks

narrate("✅ HTTP API: Responding")
narrate("✅ Version: 0.1.0 (compatible)")
narrate("✅ Worker binaries: llm-worker-rbee found")
narrate("✅ Backends: cpu, cuda available")
narrate("✅ Resources: Sufficient")
  → USER SEES: [queen-rbee] ✅ All rbee-hive preflight checks passed
```

**If any check fails:**
```
narrate("❌ rbee-hive preflight failed: llm-worker-rbee not installed")
  → USER SEES: [queen-rbee] ❌ rbee-hive preflight failed: llm-worker-rbee not installed
  → USER SEES: Suggestion: Install worker: rbee-keeper setup install --node workstation
  → Exit code: 1
```

---

### Phase 4: queen-rbee → rbee-hive: Spawn worker

**queen-rbee** sends task to **rbee-hive**:
```
POST http://workstation.home.arpa:9200/v1/workers/spawn
{
  "model": "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
  "backend": "cuda",
  "device": 1
}
```

---

### Phase 5: rbee-hive checks model catalog (SQLite)

**rbee-hive** checks model catalog (SQLite at ~/.rbee/models.db):
```sql
SELECT local_path FROM models 
WHERE reference = 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF' AND provider = 'hf';
```

**Result:** Not found

**Narration:** None (internal SQLite query)

---

### Phase 6: rbee-hive downloads model

**Error Handling: Model Download Errors**

**EH-007a: Model Not Found on Hugging Face**
- **Trigger:** Model doesn't exist (404)
- **Detection:** HTTP 404 from Hugging Face
- **Response:** Immediate error
- **Exit Code:** 1
- **Message:** "Model not found: hf:NonExistent/FakeModel"
- **Suggestion:** "Check model reference on https://huggingface.co/"

**EH-007b: Model Repository is Private**
- **Trigger:** Model requires authentication (403)
- **Detection:** HTTP 403 from Hugging Face
- **Response:** Error with auth suggestion
- **Exit Code:** 1
- **Message:** "Access denied to model"
- **Suggestion:** "Provide HF token: export HF_TOKEN=your_token_here"

**EH-008a: Model Download Timeout**
- **Trigger:** Network very slow, no progress for 60s
- **Detection:** Stall detection (no bytes received in 60s)
- **Retry:** Up to 6 attempts with exponential backoff
- **Response:** Error after all retries fail
- **Exit Code:** 1
- **Message:** "Download timeout after 6 attempts"

**EH-008b: Model Download Connection Reset**
- **Trigger:** Network interruption during download
- **Detection:** "Connection reset by peer" error
- **Retry:** Up to 6 attempts with exponential backoff, resume from checkpoint
- **Response:** Error after all retries fail
- **Exit Code:** 1
- **Message:** "Download failed after 6 attempts"

**EH-008c: Downloaded Model Checksum Mismatch**
- **Trigger:** File corrupted during download
- **Detection:** SHA256 checksum verification fails
- **Retry:** Delete corrupted file, retry download
- **Response:** Error after all retries fail
- **Exit Code:** 1
- **Message:** "Checksum mismatch: file corrupted"

**rbee-hive** downloads model from Hugging Face:

**rbee-hive narration:**
```
narrate("Downloading model from Hugging Face")
  → stdout → SSH tunnel → queen-rbee
  → queen-rbee → SSE → rbee-keeper
  → USER SEES: [model-provisioner] 📦 Downloading model from Hugging Face
```

**Progress updates:**
```
narrate("Downloaded 1 MB / 5 MB (20%)")
  → stdout → SSH tunnel → queen-rbee
  → queen-rbee → SSE → rbee-keeper
  → USER SEES: [model-provisioner] Downloading... [████----] 20% (1 MB / 5 MB)
```

**Download complete:**
```
narrate("Model downloaded to /models/tinyllama-q4.gguf")
  → stdout → SSH tunnel → queen-rbee
  → queen-rbee → SSE → rbee-keeper
  → USER SEES: [model-provisioner] ✅ Model downloaded to /models/tinyllama-q4.gguf
```

---

### Phase 7: rbee-hive registers model in catalog

**rbee-hive** registers model in SQLite:
```sql
INSERT INTO models (id, provider, reference, local_path, size_bytes, downloaded_at_unix)
VALUES ('tinyllama-q4', 'hf', 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF', 
        '/models/tinyllama-q4.gguf', 5242880, 1728508603);
```

**Narration:** None (internal SQLite operation)

---

### Phase 8: rbee-hive worker preflight

**### GPU FAIL FAST Policy (TEAM-075)

**CRITICAL POLICY:** NO FALLBACK - FAIL FAST on GPU errors

**Policy Rules:**
- ❌ NO automatic backend fallback (GPU → CPU)
- ❌ NO graceful degradation
- ❌ NO CPU fallback on GPU failure
- ✅ FAIL FAST with exit code 1
- ✅ Clear error message with actionable suggestions
- ✅ User must explicitly choose backend

**Why:** Clear failure modes prevent silent degradation. User knows exactly what went wrong and how to fix it.

**Error Codes:**
- `CUDA_DEVICE_FAILED` - CUDA device initialization failed
- `GPU_VRAM_EXHAUSTED` - VRAM exhaustion
- `GPU_NOT_AVAILABLE` - GPU not available

**Exit Code:** 1 (FAIL FAST)

### Error Handling: Resource Errors

**EH-004a: Insufficient RAM**
- **Trigger:** Available RAM < required RAM (model_size * 1.2)
- **Detection:** RAM check before worker spawn
- **Response:** Immediate error with suggestions
- **Exit Code:** 1
- **Message:** "Insufficient RAM: need 6000 MB, have 4000 MB"
- **Suggestions:** "Close other applications, use smaller model (Q4 instead of Q8), try CPU backend"

**EH-004b: RAM Exhausted During Model Loading**
- **Trigger:** System OOM during model loading
- **Detection:** Worker process killed by OOM killer
- **Response:** Error with suggestion
- **Exit Code:** 1
- **Message:** "Out of memory during model loading"
- **Suggestion:** "Free up RAM and try again"

**EH-005a: VRAM Exhausted (GPU FAIL FAST)**
- **Trigger:** CUDA device VRAM < required VRAM
- **Detection:** VRAM check before model loading
- **Response:** FAIL FAST with exit code 1 (NO CPU fallback)
- **Exit Code:** 1
- **Message:** "Insufficient VRAM: need 4000 MB, have 2000 MB"
- **Suggestions:** "Use smaller quantized model (Q4_K_M instead of Q8_0), try CPU backend explicitly (--backend cpu), free VRAM"
- **Policy:** User must explicitly choose CPU backend, NO automatic fallback

**EH-006a: Insufficient Disk Space**
- **Trigger:** Free disk space < model size
- **Detection:** Disk space check before download
- **Response:** Immediate error
- **Exit Code:** 1
- **Message:** "Insufficient disk space: need 5000 MB, have 1000 MB"
- **Suggestion:** "Remove unused models: rbee-keeper models rm <model_name>"

**EH-006b: Disk Fills Up During Download**
- **Trigger:** Disk full mid-download
- **Detection:** Write error "No space left on device"
- **Response:** Cleanup partial download, error
- **Exit Code:** 1
- **Message:** "Disk full during download"
- **Suggestion:** "Free up disk space and try again"

**EH-009a: Backend Not Available (GPU FAIL FAST)**
- **Trigger:** Requested backend not installed
- **Detection:** Backend check fails
- **Response:** FAIL FAST with exit code 1 (NO automatic fallback)
- **Exit Code:** 1
- **Message:** "Backend not available: metal"
- **Suggestion:** "Available: [cpu, cuda]. Try: --backend cuda --device 0"
- **Policy:** User must explicitly choose available backend, NO automatic selection

**EH-009b: CUDA Not Installed (GPU FAIL FAST)**
- **Trigger:** CUDA backend requested but not installed
- **Detection:** CUDA driver check fails
- **Response:** FAIL FAST with exit code 1 (NO CPU fallback)
- **Exit Code:** 1
- **Message:** "CUDA backend not available"
- **Suggestion:** "Install CUDA: https://developer.nvidia.com/cuda-downloads OR use CPU explicitly: --backend cpu"
- **Policy:** User must fix CUDA installation or explicitly choose CPU

**rbee-hive** checks RAM:
```rust
let available_ram_mb = get_available_ram();  // 8000 MB
let required_ram_mb = model_size_mb * 1.2;   // 6000 MB

if available_ram_mb < required_ram_mb {
    return Err("Insufficient RAM");
}
```

**Narration:** None (internal check, only narrates if error)

**rbee-hive** checks CUDA backend:
```rust
if !cuda_available() {
    return Err("CUDA backend not available");
}
```

**Narration:** None (internal check, only narrates if error)

---

### Phase 9: rbee-hive spawns worker

**Error Handling: Worker Startup Errors**

**EH-012a: Worker Binary Not Found**
- **Trigger:** Worker binary doesn't exist at expected path
- **Detection:** Spawn fails immediately
- **Response:** Error with installation suggestion
- **Exit Code:** 1
- **Message:** "Worker binary not found: /path/to/llm-worker-rbee"
- **Suggestion:** "Install worker: rbee-keeper setup install --node workstation"

**EH-012b: Worker Port Already in Use**
- **Trigger:** Port occupied by another process
- **Detection:** Worker fails to bind port
- **Response:** Try next available port automatically
- **Exit Code:** 0 (success on alternate port)
- **Message:** "Port 8001 in use, trying 8002..."

**EH-012c: Worker Crashes During Startup**
- **Trigger:** Worker initialization fails (e.g., CUDA device not found)
- **Detection:** Worker process exits within 30s of spawn
- **Response:** Error with log suggestion
- **Exit Code:** 1
- **Message:** "Worker startup failed"
- **Suggestion:** "Check worker logs for details"

**rbee-hive** spawns **llm-worker-rbee** (worker-rbee daemon):
```bash
llm-worker-rbee \
  --model /models/tinyllama-q4.gguf \
  --backend cuda \
  --device 1 \
  --port 8001 \
  --api-key <worker_api_key>
```

**Worker startup narration (HTTP server NOT ready yet):**
```
worker narrate("Worker starting on port 8001")
  → stdout → rbee-hive captures
  → rbee-hive → SSE → queen-rbee
  → queen-rbee → stdout → rbee-keeper shell
  → USER SEES: [llm-worker-rbee] 🌅 Worker starting on port 8001
```

**Device initialization:**
```
worker narrate("Initialized CUDA device 1")
  → stdout → rbee-hive captures
  → rbee-hive → SSE → queen-rbee
  → queen-rbee → stdout → rbee-keeper shell
  → USER SEES: [device-manager] 🖥️ Initialized CUDA device 1
```

**Model loading:**
```
worker narrate("Loading model from /models/tinyllama-q4.gguf")
  → stdout → rbee-hive captures
  → rbee-hive → SSE → queen-rbee
  → queen-rbee → stdout → rbee-keeper shell
  → USER SEES: [model-loader] 📦 Loading model from /models/tinyllama-q4.gguf
```

**Model loaded:**
```
worker narrate("Model loaded! 669 MB in VRAM")
  → stdout → rbee-hive captures
  → rbee-hive → SSE → queen-rbee
  → queen-rbee → stdout → rbee-keeper shell
  → USER SEES: [model-loader] 🛏️ Model loaded! 669 MB cozy in VRAM!
```

**HTTP server starts:**
```
worker narrate("HTTP server listening on 0.0.0.0:8001")
  → stdout → rbee-hive captures
  → rbee-hive → SSE → queen-rbee
  → queen-rbee → stdout → rbee-keeper shell
  → USER SEES: [http-server] 🚀 HTTP server ready on port 8001
```

**Worker ready callback:**
```
worker → POST http://workstation.home.arpa:9200/v1/workers/ready
{
  "worker_id": "worker-abc123",
  "url": "http://workstation.home.arpa:8001",
  "model_ref": "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
  "backend": "cuda",
  "device": 1
}
```

**Narration:**
```
worker narrate("Calling rbee-hive ready callback")
  → stdout → rbee-hive captures
  → rbee-hive → SSE → queen-rbee
  → queen-rbee → stdout → rbee-keeper shell
  → USER SEES: [llm-worker-rbee] 👋 Reporting ready to rbee-hive
```

---

### Phase 10: rbee-hive registers worker

**rbee-hive** updates in-memory registry:
```rust
registry.register(WorkerInfo {
    id: "worker-abc123",
    url: "http://workstation.home.arpa:8001",
    model_ref: "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    backend: "cuda",
    device: 1,
    state: WorkerState::Idle,
    last_activity: SystemTime::now(),
});
```

**Narration:** None (internal registry update)

---

### Phase 11: rbee-hive returns worker URL to queen-rbee

**rbee-hive** responds to queen-rbee:
```json
{
  "worker_id": "worker-abc123",
  "url": "http://workstation.home.arpa:8001",
  "state": "idle"
}
```

**Narration:** None (HTTP response)

---

### Phase 12: queen-rbee returns worker URL to rbee-keeper

**queen-rbee** responds to rbee-keeper:
```json
{
  "worker_url": "http://workstation.home.arpa:8001",
  "worker_id": "worker-abc123"
}
```

**Narration:** None (HTTP response)

---

### Phase 13: rbee-keeper → worker: Execute inference

**Error Handling: Inference Errors**

**EH-013a: Worker Crashes During Inference**
- **Trigger:** Worker process crashes unexpectedly
- **Detection:** SSE stream closes unexpectedly
- **Response:** Save partial results, error
- **Exit Code:** 1
- **Message:** "SSE stream closed unexpectedly - worker may have crashed"
- **Action:** rbee-hive removes worker from registry

**EH-013b: Worker Hangs During Inference**
- **Trigger:** Worker stops responding, no tokens for 60s
- **Detection:** Stall timeout (no tokens in 60s)
- **Response:** Cancel request, error
- **Exit Code:** 1
- **Message:** "Worker timeout - no response for 60s"

**EH-003a: Worker HTTP Connection Lost**
- **Trigger:** Network connection drops mid-inference
- **Detection:** Connection loss within 5s
- **Response:** Display partial results, error
- **Exit Code:** 1
- **Message:** "Worker connection lost - network may be down"

**EH-018a: Worker Busy (All Slots Occupied)**
- **Trigger:** Worker already processing request
- **Detection:** Worker returns 503 Service Unavailable
- **Retry:** 3 attempts with exponential backoff (1s, 2s, 4s)
- **Response:** Error after all retries fail
- **Exit Code:** 1
- **Message:** "Worker still busy after 3 retries"
- **Suggestions:** "Wait for current request, use different node, spawn additional worker"

**EH-016a: Worker Loading Timeout**
- **Trigger:** Model loading takes > 5 minutes
- **Detection:** Timeout expires during loading
- **Response:** Error with log suggestion
- **Exit Code:** 1
- **Message:** "Model loading timeout after 5 minutes (stuck at 28/32 layers)"
- **Suggestion:** "Check worker logs: ssh workstation tail -f ~/.rbee/logs/worker-abc123.log"

**Gap-G12: Request Cancellation**
- **Trigger:** User presses Ctrl+C or client disconnects
- **Detection:** SIGINT signal or SSE stream closure
- **Response:** Send DELETE /v1/inference/<request_id> to worker
- **Worker Action:** Stop token generation immediately, release slot, return to idle
- **Exit Code:** 130 (SIGINT)
- **Message:** "Request canceled, slot released"

**EH-015: Request Validation Errors**
- **Invalid model reference:** "Invalid model reference format: expected hf:org/repo or file:///path"
- **Invalid backend:** "Invalid backend: quantum. Valid: [cpu, cuda, metal]"
- **Device out of range:** "Device 5 not available. Available: [0, 1]"

**EH-017: Authentication Errors**
- **Missing API key:** "Missing API key for workstation.home.arpa"
- **Invalid API key:** "Invalid API key for workstation.home.arpa"

**rbee-keeper** sends inference request to **worker** (DIRECT, bypassing rbee-hive):
```
POST http://workstation.home.arpa:8001/execute
{
  "job_id": "job-123",
  "prompt": "write a short story",
  "max_tokens": 20,
  "temperature": 0.7
}
```

**Worker inference narration (HTTP server ACTIVE - uses SSE):**

**Inference start:**
```
worker narrate("Starting inference (prompt: 18 chars, max_tokens: 20)")
  → SSE → queen-rbee
  → queen-rbee → stdout → rbee-keeper shell
  → USER SEES: [candle-backend] 🚀 Starting inference (prompt: 18 chars, max_tokens: 20)
```

**Tokenization:**
```
worker narrate("Tokenized prompt (4 tokens)")
  → SSE → queen-rbee
  → queen-rbee → stdout → rbee-keeper shell
  → USER SEES: [tokenizer] 🍰 Tokenized prompt (4 tokens)
```

**Cache reset:**
```
worker narrate("Reset KV cache for fresh start")
  → SSE → queen-rbee
  → queen-rbee → stdout → rbee-keeper shell
  → USER SEES: [candle-backend] 🧹 Reset KV cache for fresh start
```

**Token generation (interleaved with tokens):**
```
SSE stream:
  event: token
  data: {"t":"Once","i":0}
  
  event: token
  data: {"t":" upon","i":1}
  
  event: narration
  data: {"actor":"candle-backend","action":"token_generate","human":"Generated 10 tokens","cute":"🎯"}
  
  event: token
  data: {"t":" a","i":2}
  
  ...
```

**rbee-keeper displays:**
- **Tokens → stdout:** `Once upon a time...`
- **Narration → stderr:** `[candle-backend] 🎯 Generated 10 tokens`

**Inference complete:**
```
worker narrate("Inference complete! 20 tokens in 150ms (133 tok/s)")
  → SSE → queen-rbee
  → queen-rbee → stdout → rbee-keeper shell
  → USER SEES: [candle-backend] 🎉 Inference complete! 20 tokens in 150ms (133 tok/s)
```

---

### Phase 14: Cascading Shutdown

**Error Handling: Shutdown Errors**

**EH-014a: Worker Ignores Shutdown Signal**
- **Trigger:** Worker doesn't respond to shutdown within 30s
- **Detection:** Shutdown timeout expires
- **Response:** Force-kill worker process
- **Action:** rbee-hive logs force-kill event
- **Message:** "Worker did not respond, force-killing"

**EH-014b: Graceful Shutdown with Active Request**
- **Trigger:** Shutdown command while request in progress
- **Response:** Worker sets state to "draining"
- **Action:** Reject new requests (503), wait for active request (max 30s), then exit
- **Exit Code:** 0

**Worker** transitions to idle state.

**rbee-keeper** exits (user got their result).

**Cascading shutdown sequence:**

**1. rbee-keeper exits:**
```
rbee-keeper completes inference
rbee-keeper displays final result
rbee-keeper sends SIGTERM to queen-rbee (if it spawned it)
rbee-keeper exits
```

**2. queen-rbee shuts down:**
```
queen-rbee receives SIGTERM
queen-rbee sends shutdown to all rbee-hive instances via SSH
queen-rbee exits
```

**3. rbee-hive shuts down:**
```
rbee-hive receives shutdown signal via SSH
rbee-hive sends POST http://workstation.home.arpa:8001/shutdown to all workers
rbee-hive waits for workers to exit
rbee-hive exits
```

**4. Worker shuts down:**

**Worker shutdown narration (HTTP server closing - uses stdout):**
```
worker narrate("Shutting down gracefully")
  → stdout → rbee-hive captures
  → rbee-hive → SSE → queen-rbee (if still connected)
  → queen-rbee → stdout → rbee-keeper shell (already exited, not seen)
```

**VRAM freed:**
```
worker narrate("Freeing 669 MB VRAM")
  → stdout → rbee-hive captures
  → rbee-hive logs (queen-rbee already exited, not relayed)
```

**Worker exits:**
```
worker narrate("Worker exiting")
  → stdout → rbee-hive captures
  → rbee-hive logs (queen-rbee already exited, not relayed)
worker process exits
```

**Final state:**
- rbee-keeper: exited
- queen-rbee: exited
- rbee-hive: exited
- worker: exited
- VRAM: freed (available for games/other apps)

**Note:** Shutdown narration is typically NOT seen by user because rbee-keeper has already exited. This is by design - user got their result and moved on.

---

## Critical Corrections Applied

### ❌ WRONG (Original)
- "pool manager dies, worker lives"
- "ctl adds the worker details is last seen alive in the worker registry"
- "ctl runs a health check"
- "ctl runs execute"
- "ctl streams tokens to stdout"

### ✅ CORRECT (Updated)
- **rbee-hive is persistent daemon** (but dies when queen-rbee shuts down)
- **rbee-hive maintains worker registry** (in-memory, not ctl)
- **queen-rbee orchestrates** (not ctl)
- **rbee-keeper sends execute directly to worker** (bypasses rbee-hive)
- **rbee-keeper displays tokens to stdout, narration to stderr**
- **Cascading shutdown:** rbee-keeper → queen-rbee → rbee-hive → workers
- **Worker does NOT stay alive** after rbee-keeper exits

### 🆕 NEW (TEAM-041 Addition)
- **queen-rbee maintains rbee-hive Registry** (SQLite at `~/.rbee/beehives.db`)
- **rbee-keeper has configuration mode** (`rbee-keeper setup add-node`, `rbee-keeper setup install`)
- **rbee-hive registry stores SSH connection details** (host, user, key path, install path, git repo)
- **queen-rbee validates SSH connections** before saving to registry
- **queen-rbee uses registry data** to establish SSH connections for inference tasks
- **rbee-keeper is NOT just for testing** - it's also the configuration tool for the entire system

---

## Narration Flow Summary

**Before HTTP server ready (worker startup):**
```
worker narrate() → stdout → rbee-hive captures → SSE → queen-rbee → stdout → user shell
```

**During HTTP server active (inference):**
```
worker narrate() → SSE → queen-rbee → stdout → user shell
```

**After HTTP server closing (shutdown):**
```
worker narrate() → stdout → rbee-hive captures → SSE → queen-rbee → stdout → user shell
```

**All narration ends up in user's shell. The transport is just plumbing.**

---

## Error Handling Summary

### Error Categories

**1. Network & Connectivity (EH-001, EH-002, EH-003)**
- SSH connection failures (timeout, auth, command execution)
- HTTP connection failures (timeout, malformed response)
- Connection loss during inference

**2. Resource Errors (EH-004, EH-005, EH-006)**
- Insufficient RAM, VRAM, disk space
- OOM during model loading
- Disk full during download

**3. Model & Backend (EH-007, EH-008, EH-009)**
- Model not found (404), private (403)
- Download failures (timeout, connection reset, checksum mismatch)
- Backend not available, CUDA not installed

**4. Configuration (EH-010, EH-011)**
- Node not in registry
- Invalid SSH key path, duplicate node name

**5. Process Lifecycle (EH-012, EH-013, EH-014)**
- Worker binary not found, port in use, startup crash
- Worker crash/hang during inference
- Shutdown timeout, force-kill

**6. Request Validation (EH-015)**
- Invalid model reference, backend, device number

**7. Timeouts (EH-016)**
- Request timeout, model loading timeout, inference timeout

**8. Authentication (EH-017)**
- Missing API key, invalid API key

**9. Concurrency (EH-018)**
- Worker busy, all slots occupied

### Error Response Format

All errors follow standardized format:
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "details": {
      "key": "value"
    }
  }
}
```

### HTTP Status Codes

- **400** Bad Request - Invalid input (EH-015)
- **401** Unauthorized - Authentication failure (EH-017)
- **404** Not Found - Model/resource not found (EH-007a, EH-010)
- **403** Forbidden - Access denied (EH-007b)
- **408** Request Timeout - Request exceeded timeout (EH-016)
- **499** Client Closed Request - Cancellation (Gap-G12)
- **503** Service Unavailable - Worker busy, queue full (EH-018)
- **507** Insufficient Storage - VRAM/disk exhausted (EH-005, EH-006)
- **500** Internal Server Error - Unexpected errors

### Retry Strategy

**Exponential Backoff with Jitter:**
- SSH connections: 3 attempts (0ms, 200ms, 400ms)
- HTTP connections: 3 attempts (100ms, 200ms, 400ms)
- Model downloads: 6 attempts (100ms, 200ms, 400ms, 800ms, 1600ms, 3200ms)
- Worker busy: 3 attempts (1s, 2s, 4s)

**Jitter:** Random factor 0.5-1.5x to avoid thundering herd

### Timeout Values

- **SSH connection:** 10s per attempt
- **HTTP request:** 10s total, 5s connect
- **Model download stall:** 60s no progress
- **Worker startup:** 30s
- **Model loading:** 5 minutes
- **Inference stall:** 60s no tokens
- **Graceful shutdown:** 30s before force-kill
- **Global test suite:** 5 minutes

### Cancellation (Gap-G12)

**Explicit Cancellation:**
```
DELETE /v1/inference/<request_id>
Response: 204 No Content (idempotent)
```

**Client Disconnect:**
- Worker detects SSE stream closure within 1s
- Stops token generation immediately
- Releases slot, returns to idle
- Logs cancellation event

**Ctrl+C:**
- rbee-keeper sends DELETE to worker
- Waits for acknowledgment (5s timeout)
- Exits with code 130 (SIGINT)

## Milestone Alignment (TEAM-077)

### M0 (v0.1.0) - Worker Haiku Test
**Goal:** Worker binary runs standalone
**Components:** worker-rbee only
**Scenarios:** Worker lifecycle, inference execution

### M1 (v0.2.0) - Pool Manager Lifecycle
**Goal:** rbee-hive can start/stop workers, hot-load models
**Components:** rbee-hive + worker-rbee
**Scenarios:** All scenarios in this document
**New Components Needed:**
- Worker binaries catalog (track which workers installed)
- SSH preflight checks (validate SSH before spawning)
- rbee-hive preflight checks (validate readiness)

### M2 (v0.3.0) - Orchestrator Scheduling
**Goal:** queen-rbee with Rhai scheduler
**Components:** queen-rbee + Rhai scheduler
**Status:** Documented but deferred

### M3 (v0.4.0) - Security & Platform
**Goal:** auth, audit logging, multi-tenancy
**Components:** auth-min, audit-logging, secrets-management
**Status:** Documented but deferred

## Feature File Mapping (TEAM-077)

This test-001.md document maps to multiple feature files:

**M1 Feature Files (14 files):**
- 010-ssh-registry-management.feature (Phase 0: SSH setup, node registry)
- 020-model-catalog.feature (Phase 5-7: Model download, catalog)
- 025-worker-provisioning.feature (Phase 3b: Build workers + binaries catalog) - NEW!
- 030-queen-rbee-worker-registry.feature (M1: Global worker registry - just HTTP endpoints!)
- 040-rbee-hive-worker-registry.feature (Phase 3, 10: Local worker registry)
- 050-ssh-preflight-validation.feature (Phase 2a: SSH checks) - NEW!
- 060-rbee-hive-preflight-validation.feature (Phase 3a: rbee-hive readiness) - NEW!
- 070-worker-resource-preflight.feature (Phase 8: Worker resources - RAM, VRAM, disk)
- 080-worker-rbee-lifecycle.feature (Phase 9: worker-rbee daemon)
- 090-rbee-hive-lifecycle.feature (Phase 2b, 14: rbee-hive daemon)
- 100-queen-rbee-lifecycle.feature (M1: queen-rbee daemon - standard lifecycle!)
- 110-inference-execution.feature (Phase 13: Inference handling)
- 120-input-validation.feature (Input validation)
- 130-cli-commands.feature (CLI commands)
- 140-end-to-end-flows.feature (All phases: E2E integration)

**M2+ Feature Files (7 files - documented but deferred):**
- ~~030-queen-rbee-worker-registry.feature~~ (MOVED TO M1)
- ~~100-queen-rbee-lifecycle.feature~~ (MOVED TO M1)
- 150-authentication.feature (M3 - auth-min)
- 160-audit-logging.feature (M3 - audit logging)
- 170-input-validation.feature (M3 - injection prevention)
- 180-secrets-management.feature (M3 - secure credentials)
- 190-deadline-propagation.feature (M3 - resource enforcement)
- 200-rhai-scheduler.feature (M2 - programmable routing)
- 210-queue-management.feature (M2 - job queue)

## Revision History

**TEAM-038** (2025-10-10): Corrected orchestration flow, narration architecture, and cascading shutdown  
**TEAM-041** (2025-10-10): Added rbee-hive Registry module, SSH setup flow, and rbee-keeper configuration mode  
**TEAM-061** (2025-10-10): Added comprehensive error handling scenarios and timeout specifications  
**TEAM-075** (2025-10-10): Added GPU FAIL FAST policy, removed all fallback chains, enforced clear error modes  
**TEAM-077** (2025-10-11): Updated naming conventions (rbee-hive, worker-rbee), added milestone alignment, mapped to feature files

**Status:** ✅ CORRECTED + ENHANCED + ERROR HANDLING + GPU FAIL FAST + MILESTONE ALIGNED