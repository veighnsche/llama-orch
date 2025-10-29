# Port Configuration Reference

**Version:** 2.0  
**Last Updated:** 2025-10-25  
**Purpose:** Central registry of all port numbers used in the rbee ecosystem

---

## 📊 Port Map Visualization

```
Backend APIs:
7833 ← queen-rbee (orchestrator)
7835 ← rbee-hive (hive manager)
8000 ← vllm-worker
8080 ← llm-worker
8188 ← comfy-worker

Frontend (Development):
5173 ← keeper GUI (Tauri)
6006 ← Storybook
7811 ← user-docs
7822 ← commercial
7834 ← queen UI (dev) → 7833/ (prod)
7836 ← hive UI (dev) → 7835/ (prod)
7837 ← llm-worker UI (dev) → 8080/ (prod)
7838 ← comfy-worker UI (dev) → 8188/ (prod)
7839 ← vllm-worker UI (dev) → 8000/ (prod)
```

---

## 🎯 Quick Port Reference - EDIT THESE VALUES

### Backend Services (HTTP APIs)

| Service | Port | Description |
|---------|------|-------------|
| **queen-rbee** | `7833` | Orchestrator daemon (HTTP API) |
| **rbee-hive** | `7835` | Hive daemon (HTTP API) |
| **llm-worker** | `8080` | LLM worker (HTTP API) |
| **comfy-worker** | `8188` | ComfyUI worker (HTTP API) |
| **vllm-worker** | `8000` | vLLM worker (HTTP API) |

### Frontend Services (Development)

| Service | Port | Description | Production |
|---------|------|-------------|------------|
| **rbee-keeper GUI** | `5173` | Keeper Tauri GUI (dev server) | Tauri app |
| **queen-rbee UI** | `7834` | Queen UI (dev server) | Hosted at `7833/` |
| **rbee-hive UI** | `7836` | Hive UI (dev server) | Hosted at `7835/` |
| **llm-worker UI** | `7837` | LLM worker UI (dev server) | Hosted at `8080/` |
| **comfy-worker UI** | `7838` | ComfyUI worker UI (dev server) | Hosted at `8188/` |
| **vllm-worker UI** | `7839` | vLLM worker UI (dev server) | Hosted at `8000/` |
| **rbee-ui Storybook** | `6006` | Component library | N/A |
| **commercial** | `7822` | Marketing site (Next.js) | Deployed |
| **user-docs** | `7811` | Documentation (Next.js + Nextra) | Deployed |
| **web-ui** | `5179` | OLD UI (DEPRECATED) | N/A |

---

## ⚠️ IMPORTANT: How to Change Ports

**DO NOT edit code files directly!**

1. **Edit the tables above** - Change the port numbers
2. **Ask a junior developer** to update all the code files manually (see checklist below)
3. **Verify** - Test that all services start on the correct ports

This ensures consistency and prevents configuration drift.

---

---

## 🏗️ Hierarchical UI Architecture

### Overview

The rbee system uses a **hierarchical, distributed UI architecture**:

```
┌─────────────────────────────────────┐
│ rbee-keeper (Tauri GUI)             │  Port 5173 (dev)
│ ├─ Sidebar (dynamic)                │
│ ├─ iframe: queen-rbee UI            │  → http://localhost:7833/ (prod)
│ ├─ iframe: hive UI (per hive)       │  → http://localhost:7835/ (prod)
│ └─ iframe: worker UI (per worker)   │  → http://localhost:8080/ (prod)
└─────────────────────────────────────┘
```

### Port Mapping: Development vs Production

| Component | Vite Dev Port | Backend URL | Dev Behavior | Prod Behavior |
|-----------|---------------|-------------|--------------|---------------|
| **Keeper GUI** | 5173 | Tauri app | Vite dev server | Tauri bundle |
| **Queen UI** | 7834 | `7833/` | Backend proxies to 7834 | Backend serves dist/ |
| **Hive UI** | 7836 | `7835/` | Backend proxies to 7836 | Backend serves dist/ |
| **LLM Worker UI** | 7837 | `8080/` | Backend proxies to 7837 | Backend serves dist/ |
| **ComfyUI Worker UI** | 7838 | `8188/` | Backend proxies to 7838 | Backend serves dist/ |
| **vLLM Worker UI** | 7839 | `8000/` | Backend proxies to 7839 | Backend serves dist/ |

**Key Principles:**
- **Development:** Backend proxies ROOT to Vite dev server (hot-reload works)
- **Production:** Backend serves static files from `dist/` (no Vite needed)
- **Same URL:** `http://localhost:7833/` works in both modes

### Development Workflow

**Terminal 1: Keeper GUI**
```bash
cd frontend/apps/00_rbee_keeper
pnpm dev  # Runs on port 5173
```

**Terminal 2: Queen UI**
```bash
cd frontend/apps/10_queen_rbee
pnpm dev  # Runs on port 7834
```

**Terminal 3: Hive UI**
```bash
cd frontend/apps/20_rbee_hive
pnpm dev  # Runs on port 7836
```

**Terminal 4: Worker UIs**
```bash
cd frontend/apps/30_llm_worker_rbee
pnpm dev  # Runs on port 7837
```

### Development vs Production

**Development Mode:**
Backend proxies ROOT requests to Vite dev servers (enables hot-reload):

```rust
// Example: queen-rbee proxies to Vite dev server
if dev_mode {
    // Proxy ROOT to http://localhost:7834 (Vite dev server)
    Router::new().fallback(proxy_to_vite)
} else {
    // Serve static files from dist/ at ROOT
    Router::new().fallback(serve_static_files)
}
```

**Production Mode:**
Backend serves static files from `dist/` at ROOT:

```rust
// API routes have priority, static files are fallback
let app = api_router.merge(static_router);
```

**URLs (same in both modes):**
- Queen: `http://localhost:7833/`
- Hive: `http://localhost:7835/`
- Worker: `http://localhost:8080/`

**See:** `.docs/ui/DEVELOPMENT_PROXY_STRATEGY.md` for implementation details

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
| `5173` | Keeper GUI dev server | Tauri frontend (Vite default) |
| `6006` | Storybook | Component library |
| `7811` | user-docs | Next.js documentation |
| `7822` | commercial | Next.js marketing site |
| `7833` | queen-rbee API | Backend HTTP API |
| `7834` | queen-rbee UI (dev) | Frontend dev server |
| `7835` | rbee-hive API | Backend HTTP API |
| `7836` | rbee-hive UI (dev) | Frontend dev server |
| `7837-7839` | Worker UIs (dev) | LLM, ComfyUI, vLLM dev servers |
| `8000` | vLLM worker API | Backend HTTP API |
| `8080` | LLM worker API | Backend HTTP API |
| `8188` | ComfyUI worker API | Backend HTTP API |

**Port Assignment Strategy:**
- **5173:** Keeper GUI (Vite default for Tauri)
- **6006:** Storybook (standard)
- **7811, 7822:** Existing Next.js apps
- **7833-7839:** rbee services (sequential, grouped)
- **8000, 8080, 8188:** Worker APIs (standard ports for each type)

---

## 📝 Files to Update When Changing Ports

When you change a port in this document, update these files:

### Backend Services

**queen-rbee (Port 7833):**
- `bin/10_queen_rbee/src/main.rs` - Default port argument (line 48)
- `bin/10_queen_rbee/src/http/info.rs` - Hardcoded URLs and port (lines 28, 37-38, 50-51)
- `bin/10_queen_rbee/src/http/build_info.rs` - Documentation example (line 10)
- `bin/10_queen_rbee/src/hive_forwarder.rs` - Documentation example (line 28)
- `bin/00_rbee_keeper/src/config.rs` - `default_queen_port()` (line 19)
- `bin/05_rbee_keeper_crates/queen-lifecycle/src/ensure.rs` - Extracted from base_url (line 142 comment)
- `bin/20_rbee_hive/src/main.rs` - Default queen URL (line 46)
- `bin/20_rbee_hive/src/job_router.rs` - Hardcoded URL (line 146)
- `bin/25_rbee_hive_crates/worker-lifecycle/src/lib.rs` - Documentation example (line 44)

**rbee-hive (Port 7835):**
- `bin/20_rbee_hive/src/main.rs` - Default port argument (line 41)
- `bin/10_queen_rbee/src/hive_forwarder.rs` - Documentation example (line 28)

**llm-worker (Port 8080):**
- `bin/30_llm_worker_rbee/src/main.rs` - Default port argument

**comfy-worker (Port 8188):**
- `bin/30_comfy_worker_rbee/src/main.rs` - Default port argument

**vllm-worker (Port 8000):**
- `bin/30_vllm_worker_rbee/src/main.rs` - Default port argument

### Documentation & Examples

**Note:** The following files contain port references in documentation, examples, and tests.
These should be updated when ports change to keep documentation accurate:

- `bin/99_shared_crates/narration-core/src/builder.rs` - Example code (uses 8080 in examples)
- `bin/98_security_crates/auth-min/src/policy.rs` - Example code and tests (uses 8080)
- `bin/30_llm_worker_rbee/src/http/server.rs` - Example code and tests (uses 8080)

### Frontend Services (Development Ports)

**rbee-keeper GUI (Port 5173):**
- `bin/00_rbee_keeper/ui/vite.config.ts` - Server port (line 10)
- `bin/00_rbee_keeper/tauri.conf.json` - devPath URL

**queen-rbee UI (Port 7834):**
- `bin/10_queen_rbee/ui/app/vite.config.ts` - Server port (line 9)

**rbee-hive UI (Port 7836):**
- `bin/20_rbee_hive/ui/app/vite.config.ts` - Server port (line 7)

**llm-worker UI (Port 7837):**
- `bin/30_llm_worker_rbee/ui/app/vite.config.ts` - Server port (line 7)

**comfy-worker UI (Port 7838):**
- `frontend/apps/30_comfy_worker_rbee/vite.config.ts` - Server port
- `frontend/apps/30_comfy_worker_rbee/package.json` - Dev script

**vllm-worker UI (Port 7839):**
- `frontend/apps/30_vllm_worker_rbee/vite.config.ts` - Server port
- `frontend/apps/30_vllm_worker_rbee/package.json` - Dev script

**rbee-ui Storybook (Port 6006):**
- `frontend/packages/rbee-ui/package.json` - Storybook dev script

**commercial (Port 7822):**
- `frontend/apps/commercial/package.json` - Next.js dev script

**user-docs (Port 7811):**
- `frontend/apps/user-docs/package.json` - Next.js dev script

**web-ui (Port 5179 - DEPRECATED):**
- `frontend/apps/web-ui/package.json` - Vite dev script

### SDK Packages

**queen-rbee-sdk:**
- `frontend/packages/10_queen_rbee/queen-rbee-sdk/src/index.ts` - API base URL

**rbee-hive-sdk:**
- `frontend/packages/20_rbee_hive/rbee-hive-sdk/src/index.ts` - API base URL

**llm-worker-sdk:**
- `frontend/packages/30_llm_worker_rbee/llm-worker-sdk/src/index.ts` - API base URL

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

**Backend:**
```bash
# Core services
queen-rbee -p 7833 &
rbee-hive -p 7835 --queen-url http://localhost:7833 --hive-id localhost &

# Workers (spawned by hive)
llm-worker --port 8080 &
comfy-worker --port 8188 &
vllm-worker --port 8000 &
```

**Frontend (Development):**
```bash
# Keeper GUI
cd frontend/apps/00_rbee_keeper && pnpm dev &  # Port 5173

# Component UIs
cd frontend/apps/10_queen_rbee && pnpm dev &   # Port 7834
cd frontend/apps/20_rbee_hive && pnpm dev &    # Port 7836
cd frontend/apps/30_llm_worker_rbee && pnpm dev &  # Port 7837

# Shared
cd frontend/packages/rbee-ui && pnpm dev &     # Port 6006 (Storybook)

# Existing apps
cd frontend/apps/commercial && pnpm dev &      # Port 7822
cd frontend/apps/user-docs && pnpm dev &       # Port 7811
```

**Or use turbo:**
```bash
cd frontend/apps
turbo dev  # Starts all apps in parallel
```

### Health Checks

**Backend APIs:**
```bash
curl http://localhost:7833/health  # queen-rbee
curl http://localhost:7835/health  # rbee-hive
curl http://localhost:8080/health  # llm-worker
curl http://localhost:8188/health  # comfy-worker
curl http://localhost:8000/health  # vllm-worker
```

**Frontend (Development):**
```bash
curl http://localhost:5173         # keeper GUI
curl http://localhost:7834         # queen-rbee UI
curl http://localhost:7836         # rbee-hive UI
curl http://localhost:7837         # llm-worker UI
curl http://localhost:6006         # rbee-ui Storybook
curl http://localhost:7822         # commercial
curl http://localhost:7811         # user-docs
```

**Frontend (Production):**
```bash
curl http://localhost:7833/        # queen-rbee UI (served by binary)
curl http://localhost:7835/        # rbee-hive UI (served by binary)
curl http://localhost:8080/        # llm-worker UI (served by binary)
```

---

## Related Documentation

- [UI Architecture](.docs/ui/00_UI_ARCHITECTURE_OVERVIEW.md) - Hierarchical UI system
- [Folder Structure](.docs/ui/FOLDER_STRUCTURE.md) - bin/ ↔ frontend/apps/ parity
- [Package Structure](.docs/ui/PACKAGE_STRUCTURE.md) - Specialized SDK packages
- [Architecture Overview](.arch/00_OVERVIEW_PART_1.md) - System architecture
- [Configuration Guide](docs/CONFIGURATION.md) - Configuration reference
- [Security Policy](bin/98_security_crates/auth-min/src/policy.rs) - Security rules

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2025-10-25 | 2.0 | Updated for hierarchical UI architecture (keeper, queen, hive, workers) |
| 2025-10-25 | 1.0 | Initial port configuration document created |

---

**Maintained by:** rbee Core Team  
**Last Review:** 2025-10-25

## Summary

**Total Ports Tracked:** 14

**Backend APIs:** 5 (queen, hive, 3 worker types)  
**Frontend Dev:** 9 (keeper, queen, hive, 3 workers, storybook, commercial, user-docs)  
**Deprecated:** 1 (web-ui)

**Key Principle:** Each binary hosts its own UI at ROOT (/) in production. API routes take priority via router merge order.
