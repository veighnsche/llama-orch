# Development Proxy Summary

**TEAM-293: Backend proxies to Vite dev servers during development**

## Quick Summary

**Problem:** In production, backends serve static files. In development, we want hot-reload.

**Solution:** Backends proxy `/ui` requests to Vite dev servers during development.

## How It Works

### Development Mode

```
User → http://localhost:7833/ui
       ↓
queen-rbee backend (port 7833)
       ↓ (proxy)
Vite dev server (port 7834)
       ↓
Hot-reload enabled ✅
```

### Production Mode

```
User → http://localhost:7833/ui
       ↓
queen-rbee backend (port 7833)
       ↓ (serve static)
dist/ directory
       ↓
Optimized bundle ✅
```

## Port Mapping

| Backend | API Port | Vite Dev Port | Proxy Target |
|---------|----------|---------------|--------------|
| queen-rbee | 7833 | 7834 | `http://localhost:7834` |
| rbee-hive | 7835 | 7836 | `http://localhost:7836` |
| llm-worker | 8080 | 7837 | `http://localhost:7837` |
| comfy-worker | 8188 | 7838 | `http://localhost:7838` |
| vllm-worker | 8000 | 7839 | `http://localhost:7839` |

## Implementation (Rust)

```rust
// bin/10_queen_rbee/src/main.rs

#[derive(clap::Parser)]
struct Args {
    #[arg(long)]
    dev_mode: bool,
}

fn ui_service(dev_mode: bool) -> Router {
    if dev_mode {
        // Proxy to Vite dev server
        proxy_to("http://localhost:7834")
    } else {
        // Serve static files
        ServeDir::new("../../../frontend/apps/10_queen_rbee/dist")
    }
}
```

## Usage

### Development

```bash
# Terminal 1: Start Vite dev server
cd frontend/apps/10_queen_rbee
pnpm dev  # Port 7834

# Terminal 2: Start backend in dev mode
cd bin/10_queen_rbee
cargo run -- --dev-mode

# Access (both work, backend proxies):
curl http://localhost:7833/ui  # Backend (proxies to 7834)
curl http://localhost:7834      # Vite (direct)
```

### Production

```bash
# Build frontend
cd frontend/apps/10_queen_rbee
pnpm build  # Creates dist/

# Start backend (no dev mode)
cd bin/10_queen_rbee
cargo run

# Access
curl http://localhost:7833/ui  # Backend serves dist/
```

## Benefits

✅ **Hot Reload:** Changes in React code instantly reflect  
✅ **Same URL:** `http://localhost:7833/ui` works in both modes  
✅ **No CORS:** Backend and frontend on same origin  
✅ **Fast Development:** Vite's HMR works seamlessly  
✅ **Production-like:** Same URL structure as production

## Environment Variable (Alternative)

```bash
# Development
DEV_MODE=1 cargo run

# Production
cargo run
```

## Complete Documentation

See `.docs/ui/DEVELOPMENT_PROXY_STRATEGY.md` for:
- Complete implementation examples
- Multiple proxy approaches
- Makefile helpers
- Troubleshooting guide

---

**Status:** 📋 STRATEGY DEFINED  
**Key Principle:** Same URL, different backend behavior (proxy vs static)
