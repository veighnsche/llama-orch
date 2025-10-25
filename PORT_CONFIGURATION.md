# Port Configuration Reference

**Version:** 1.0  
**Last Updated:** 2025-10-25  
**Purpose:** Central registry of all port numbers used in the rbee ecosystem

---

## 🎯 Quick Port Reference - EDIT THESE VALUES

### Backend Services

| Service | Port | Description |
|---------|------|-------------|
| **queen-rbee** | `7833` | Orchestrator daemon |
| **rbee-hive** | `7844` | Hive daemon |
| **worker-pool** | `7855 - 7899` | LLM worker instances (dynamic) |

### Frontend Services

| Service | Port | Description |
|---------|------|-------------|
| **rbee-ui Storybook** | `6006` | Component library |
| **web-ui** | `5173` | Main web UI (React + Vite) |
| **commercial** | `7822` | Marketing site (Next.js) |
| **user-docs** | `7811` | Documentation (Next.js + Nextra) |

---

## ⚠️ IMPORTANT: How to Change Ports

**DO NOT edit code files directly!**

1. **Edit the tables above** - Change the port numbers
2. **Ask a junior developer** to update all the code files manually (see checklist below)
3. **Verify** - Test that all services start on the correct ports

This ensures consistency and prevents configuration drift.

---

---

## Detailed Information

### Backend Services

#### queen-rbee (Port 8500)
```rust
// bin/10_queen_rbee/src/main.rs
#[arg(short, long, default_value = "8500")]
port: u16,
```

**Endpoints:**
- `GET /health` - Health check
- `GET /v1/info` - Queen info (base_url, port, version)
- `GET /v1/build-info` - Build information
- `POST /v1/jobs` - Job submission
- `GET /v1/jobs/{job_id}/stream` - SSE stream for job events
- `POST /v1/hive-heartbeat` - Hive heartbeat endpoint
- `POST /v1/worker-heartbeat` - Worker heartbeat endpoint

**Configuration:**
- CLI: `queen-rbee -p 8500`
- rbee-keeper config: `~/.config/rbee/config.toml`
  ```toml
  queen_port = 8500
  ```

#### rbee-hive (Port 9000)
```rust
// bin/20_rbee_hive/src/main.rs
#[arg(short, long, default_value = "9000")]
port: u16,
```

**Endpoints:**
- `GET /health` - Health check
- `POST /v1/jobs` - Job submission (worker/model operations)
- `GET /v1/jobs/{job_id}/stream` - SSE stream for job events
- `GET /v1/capabilities` - Hive capabilities (GPU, CPU, workers)

**Configuration:**
- CLI: `rbee-hive -p 9000 --queen-url http://localhost:8500 --hive-id localhost`

#### llm-worker-rbee (Dynamic Ports)
```rust
// bin/30_llm_worker_rbee/src/main.rs
#[arg(long)]
port: u16,
```

**Port Assignment:** Dynamically assigned by `rbee-hive` when spawning workers.

**Typical Range:** `9001-9999` (not enforced, managed by hive)

**Endpoints:**
- `GET /health` - Worker health check
- `POST /v1/infer` - Inference endpoint
- `GET /v1/capabilities` - Worker capabilities

### Frontend Services

#### web-ui (Port 7833)
```json
// frontend/apps/web-ui/package.json
{
  "scripts": {
    "dev": "vite --port 7833"
  }
}
```

**URL:** `http://localhost:7833`  
**Framework:** Vite + React + TypeScript  
**Purpose:** Main web UI for rbee system management  
**Note:** Custom port (Vite default is 5173)

#### commercial (Port 7834)
```json
// frontend/apps/commercial/package.json
{
  "scripts": {
    "dev": "next dev -p 7834"
  }
}
```

**URL:** `http://localhost:7834`  
**Framework:** Next.js 15  
**Purpose:** Commercial marketing website  
**Note:** Custom port (Next.js default is 3000)

#### user-docs (Port 7835)
```json
// frontend/apps/user-docs/package.json
{
  "scripts": {
    "dev": "next dev -p 7835"
  }
}
```

**URL:** `http://localhost:7835`  
**Framework:** Next.js 15 + Nextra  
**Purpose:** User documentation site  
**Note:** Custom port (Next.js default is 3000)

#### rbee-ui Storybook (Port 7832)
```json
// frontend/packages/rbee-ui/package.json
{
  "scripts": {
    "storybook": "storybook dev -p 7832 --no-open"
  }
}
```

**URL:** `http://localhost:7832`  
**Framework:** Storybook 9  
**Purpose:** Component library documentation and testing  
**Note:** Custom port (Storybook default is 6006)

---

## Testing & Development

### Mock Services

| Service | Port | Usage |
|---------|------|-------|
| Mock Hive Server | Dynamic | Integration tests (xtask) |
| Test Workers | `8700` | Unit tests (example port) |

### Test Port Allocation
```rust
// xtask/src/integration/harness.rs
fn get_free_port() -> Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}
```

Tests use **dynamic port allocation** to avoid conflicts.

---

## Port Ranges Summary

| Range | Purpose | Notes |
|-------|---------|-------|
| `7832-7835` | Frontend dev servers | Sequential, easy to remember |
| `8500` | queen-rbee | Core orchestrator |
| `9000` | rbee-hive | Hive daemon |
| `9001-9999` | LLM workers | Dynamically assigned |

**Why 7832-7835?**
- Sequential and grouped together
- Avoids common conflicts (3000, 5173, 6006, 8080)
- Room for expansion (7836+)
- Managed by `frontend/kill-dev-servers.sh`

---

## 📝 Files to Update When Changing Ports

When you change a port in this document, update these files:

### Backend Services

**queen-rbee (Port 7833):**
- `bin/10_queen_rbee/src/main.rs` - Line 48: `default_value = "7833"`
- `bin/10_queen_rbee/src/http/info.rs` - Lines 37-38: hardcoded URLs
- `bin/00_rbee_keeper/src/config.rs` - Line 19: `default_queen_port()`
- `bin/20_rbee_hive/src/main.rs` - Line 46: `default_value = "http://localhost:7833"`
- `bin/20_rbee_hive/src/job_router.rs` - Line 146: hardcoded URL
- `bin/25_rbee_hive_crates/worker-lifecycle/src/lib.rs` - Line 44: example URL

**rbee-hive (Port 7844):**
- `bin/20_rbee_hive/src/main.rs` - Line 41: `default_value = "7844"`

### Frontend Services

**rbee-ui Storybook (Port 6006):**
- `frontend/packages/rbee-ui/package.json` - Lines 36, 43: `storybook dev -p 6006`
- `frontend/kill-dev-servers.sh` - Line 8: `PORTS=(6006 ...)`

**web-ui (Port 5173):**
- `frontend/apps/web-ui/package.json` - Lines 7, 10: `--port 5173`
- `frontend/kill-dev-servers.sh` - Line 8: `PORTS=(... 5173 ...)`

**commercial (Port 7822):**
- `frontend/apps/commercial/package.json` - Line 6: `next dev -p 7822`
- `frontend/kill-dev-servers.sh` - Line 8: `PORTS=(... 7822 ...)`

**user-docs (Port 7811):**
- `frontend/apps/user-docs/package.json` - Line 6: `next dev -p 7811`
- `frontend/kill-dev-servers.sh` - Line 8: `PORTS=(... 7811)`

### Frontend Packages (SDK Examples)

**rbee-react package:**
- `frontend/packages/rbee-react/README.md` - Line 32: Example RbeeClient URL
- `frontend/packages/rbee-react/src/hooks/useRbeeSDK.ts` - Line 20: JSDoc example
- `frontend/packages/rbee-react/src/hooks/useRbeeSDKSuspense.ts` - Line 21: JSDoc example

**rbee-sdk package:**
- `frontend/packages/rbee-sdk/test.html` - Lines 83, 90, 165: Test page examples
- `frontend/packages/rbee-sdk/TEAM_286_ALL_OPERATIONS_IMPLEMENTED.md` - Line 103: Documentation example

---

## Configuration Files

### Backend Configuration

#### rbee-keeper Config
```toml
# ~/.config/rbee/config.toml
queen_port = 8500
```

```rust
// bin/00_rbee_keeper/src/config.rs
fn default_queen_port() -> u16 {
    8500
}
```

### Frontend Configuration

#### .dxrc.json (DX Tooling)
```json
{
  "base_url": "http://localhost:7834",
  "workspace": {
    "commercial": {
      "url": "http://localhost:7834",
      "port": 7834
    },
    "storybook": {
      "url": "http://localhost:7832",
      "port": 7832
    }
  }
}
```

#### frontend/kill-dev-servers.sh
```bash
PORTS=(7832 7833 7834 7835)
```
Kills all frontend dev servers by process name and port.

---

## Hardcoded References

### ⚠️ Known Hardcoded Ports (Need Cleanup)

These locations have hardcoded port references that should be refactored to use configuration:

#### queen-rbee
```rust
// bin/10_queen_rbee/src/http/info.rs:37
base_url: "http://localhost:8500".to_string(),
port: 8500,
```

#### rbee-hive
```rust
// bin/20_rbee_hive/src/job_router.rs:146
let queen_url = "http://localhost:8500".to_string();
```

#### worker-lifecycle
```rust
// bin/25_rbee_hive_crates/worker-lifecycle/src/lib.rs:44
queen_url: "http://localhost:8500".to_string(),
```

**Action Item:** Refactor these to use configuration or environment variables.

---

## Environment Variables

### Supported Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `LLORCH_API_TOKEN` | API authentication token | None (required for non-loopback) |
| `LLORCH_RUN_ID` | Test run identifier | Generated UUID |
| `LLORCH_PROOF_DIR` | Proof bundle output directory | `.proof_bundle` |
| `LLORCH_BDD_FEATURE_PATH` | BDD test feature path | `tests/features` |

**Note:** Port configuration is currently **CLI-only**, not environment variables.

---

## Security Considerations

### Loopback-Only Binding

All services default to **localhost-only** (`127.0.0.1`) for security:

```rust
// bin/10_queen_rbee/src/main.rs:82
let addr = SocketAddr::from(([127, 0, 0, 1], args.port));
```

### Non-Loopback Binding Policy

```rust
// bin/98_security_crates/auth-min/src/policy.rs
pub fn enforce_startup_bind_policy(bind_addr: &str) -> Result<()> {
    // Non-loopback addresses REQUIRE LLORCH_API_TOKEN
    if !is_loopback_addr(bind_addr) && token.is_none() {
        return Err(AuthError::BindPolicyViolation(...));
    }
    Ok(())
}
```

**Supported Loopback Formats:**
- IPv4: `127.0.0.1`, `127.0.0.1:8500`
- IPv6: `::1`, `[::1]`, `[::1]:8500`
- Hostname: `localhost`, `localhost:8500`

---

## Port Conflicts

### Common Conflicts to Avoid

| Port | Common Service | Our Usage |
|------|---------------|-----------|
| `3000` | Next.js default | ❌ Avoided (using 7834, 7835) |
| `5173` | Vite default | ❌ Avoided (using 7833) |
| `6006` | Storybook default | ❌ Avoided (using 7832) |
| `7832-7835` | - | ✅ Frontend dev servers |
| `8080` | Common HTTP servers | ❌ Avoided |
| `8500` | - | ✅ queen-rbee |
| `9000` | - | ✅ rbee-hive |

### Checking for Port Conflicts

```bash
# Check if port is in use
lsof -i :8500
netstat -tuln | grep 8500

# Kill process on port
kill $(lsof -t -i:8500)

# Kill all frontend dev servers
./frontend/kill-dev-servers.sh
```

---

## Future Considerations

### Planned Changes

1. **Configuration Consolidation**
   - Move hardcoded ports to central config
   - Support environment variable overrides
   - Add port validation on startup

2. **Port Range Management**
   - Formalize worker port range (9001-9999)
   - Add port pool management in rbee-hive
   - Prevent port conflicts between workers

3. **Service Discovery**
   - Consider using port 0 (OS-assigned) for workers
   - Implement service registry for dynamic port discovery
   - Add health check port scanning

4. **Remote Deployment**
   - Support configurable bind addresses (0.0.0.0)
   - Enforce LLORCH_API_TOKEN for non-loopback
   - Add TLS/SSL support for production

---

## Quick Reference

### Start All Services

```bash
# Backend
queen-rbee -p 8500 &
rbee-hive -p 9000 --queen-url http://localhost:8500 --hive-id localhost &

# Frontend
cd frontend/packages/rbee-ui && pnpm dev &     # Port 7832 (Storybook)
cd frontend/apps/web-ui && pnpm dev &          # Port 7833
cd frontend/apps/commercial && pnpm dev &      # Port 7834
cd frontend/apps/user-docs && pnpm dev &       # Port 7835
```

### Health Checks

```bash
# Backend
curl http://localhost:8500/health  # queen-rbee
curl http://localhost:9000/health  # rbee-hive

# Frontend
curl http://localhost:7832         # rbee-ui Storybook
curl http://localhost:7833         # web-ui
curl http://localhost:7834         # commercial
curl http://localhost:7835         # user-docs
```

---

## Related Documentation

- [Architecture Overview](.arch/00_OVERVIEW_PART_1.md)
- [Configuration Guide](docs/CONFIGURATION.md)
- [Security Policy](bin/98_security_crates/auth-min/src/policy.rs)
- [Testing Guide](.docs/testing/)

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2025-10-25 | 1.0 | Initial port configuration document created |

---

**Maintained by:** rbee Core Team  
**Last Review:** 2025-10-25
