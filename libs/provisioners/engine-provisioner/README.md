# engine-provisioner

**Prepares and starts engine processes (llama.cpp, vLLM, etc.)**

`libs/provisioners/engine-provisioner` — Provisions inference engines from config and emits handoff files.

---

## What This Library Does

engine-provisioner provides **engine lifecycle management** for llama-orch:

- **Process management** — Start and stop engine processes
- **Configuration** — Normalize engine flags and settings
- **Handoff files** — Emit machine-readable metadata for orchestrator
- **Health checks** — Wait for engine readiness
- **GPU validation** — Fail fast if GPU unavailable
- **Graceful shutdown** — SIGTERM with timeout before SIGKILL

**Used by**: `pool-managerd` to start engines

---

## Usage

### CLI

```bash
# Provision engine from config
cargo run -p provisioners-engine-provisioner --bin engine-provisioner -- \
  --config requirements/llamacpp-3090-source.yaml

# Specify pool
cargo run -p provisioners-engine-provisioner --bin engine-provisioner -- \
  --config requirements/llamacpp-3090-source.yaml \
  --pool default
```

### Library

```rust
use provisioners_engine_provisioner::{EngineProvisioner, ProvisionConfig};

let config = ProvisionConfig {
    engine: "llamacpp".to_string(),
    pool_id: "default".to_string(),
    replica_id: "r0".to_string(),
    port: 8081,
    model_path: "/models/llama-3.1-8b.gguf".to_string(),
    flags: vec!["--parallel".to_string(), "1".to_string()],
};

let provisioner = EngineProvisioner::new(config);
provisioner.start().await?;
```

---

## Handoff File Format

### Location

`.runtime/engines/{pool_id}-{replica_id}.json`

Example: `.runtime/engines/default-r0.json`

### Format

```json
{
  "engine": "llamacpp",
  "engine_version": "b1234-cuda",
  "provisioning_mode": "source",
  "url": "http://127.0.0.1:8081",
  "pool_id": "default",
  "replica_id": "r0",
  "model": {
    "id": "local:/models/llama-3.1-8b-instruct-q4_k_m.gguf",
    "path": "/models/llama-3.1-8b-instruct-q4_k_m.gguf"
  },
  "flags": ["--parallel", "1", "--no-cont-batching", "--metrics"]
}
```

### Fields

- **engine** — Engine type (llamacpp, vllm, tgi)
- **engine_version** — Version string from `/version` endpoint
- **provisioning_mode** — How engine was provisioned (source, binary, container)
- **url** — HTTP endpoint for engine
- **pool_id** — Pool identifier
- **replica_id** — Replica identifier
- **model** — Model metadata (id, path)
- **flags** — Engine command-line flags

---

## Engine Flags

### llama.cpp Defaults

```bash
--parallel 1              # Single request at a time
--no-cont-batching        # Disable continuous batching
--metrics                 # Enable Prometheus metrics
--no-webui                # Disable web UI
--n-gpu-layers -1         # All layers on GPU
```

### Flag Normalization

Legacy flags are automatically normalized:

- `--gpu-layers` → `--n-gpu-layers`
- `--ngl` → `--n-gpu-layers`

---

## Health Checks

### Readiness Wait

```rust
provisioner.wait_ready(timeout).await?;
```

Polls engine health endpoint until:
- Returns 200 OK
- Or timeout expires

Treats 503 as transient during model load.

### Metrics Validation

If `--metrics` flag is set, validates `/metrics` endpoint is accessible.

---

## Graceful Shutdown

```rust
provisioner.stop().await?;
```

Shutdown sequence:
1. Send SIGTERM to process
2. Wait up to 5 seconds
3. Send SIGKILL if still running

---

## Testing

### Unit Tests

```bash
# Run all tests
cargo test -p provisioners-engine-provisioner -- --nocapture

# Run specific test
cargo test -p provisioners-engine-provisioner -- test_provision --nocapture
```

---

## Dependencies

### Internal

- `contracts/config-schema` — Configuration types
- `provisioners-model-provisioner` — Model staging

### External

- `tokio` — Async runtime, process management
- `serde` — Serialization
- `serde_json` — JSON handoff files
- `reqwest` — Health check HTTP requests

---

## Configuration

### Input

From `contracts/config-schema::Config`:

```yaml
pools:
  - id: default
    engine: llamacpp
    replicas: 1
    port: 8081
    model:
      id: local:/models/llama-3.1-8b.gguf
    flags:
      - --parallel
      - "1"
```

### Output

Handoff file at `.runtime/engines/default-r0.json`

---

## Known Limitations

### MVP Scope

- ✅ **Engine version capture** — Probes `/version` endpoint
- ✅ **Graceful shutdown** — SIGTERM with 5s timeout
- ✅ **Health checks** — Readiness wait with 503 handling
- ❌ **Restart on crash** — Not implemented (manual restart required)
- ❌ **PID files** — Not implemented

### Future Work

- Supervision loop for automatic restart
- PID file management
- Drain hooks for graceful shutdown
- Multi-engine support (vLLM, TGI)

---

## Specifications

Implements requirements from `.specs/00_llama-orch.md`:
- Engine provisioning
- Handoff file format
- GPU validation
- Health checks

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha
- **Maintainers**: @llama-orch-maintainers
