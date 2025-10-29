# Queen UI Proxy Setup

**Date:** Oct 29, 2025  
**Status:** ✅ COMPLETE

---

## Overview

The queen-rbee binary now serves the UI differently based on build mode:

### Development Mode (`cargo run`)
- **Proxies** `/ui` requests to Vite dev server at `http://localhost:7834`
- Enables hot-reload and fast development
- No build step required

### Production Mode (`cargo build --release`)
- **Serves** embedded static files from `ui/app/dist/`
- Single binary distribution
- No external dependencies

---

## Port Configuration

According to `PORT_CONFIGURATION.md`:

| Service | Dev Port | Production URL |
|---------|----------|----------------|
| **Queen UI** | `7834` | `http://localhost:7833/ui` |
| **Queen API** | `7833` | `http://localhost:7833` |

---

## Development Workflow

### Terminal 1: Start Queen Binary
```bash
cd bin/10_queen_rbee
cargo run
```

### Terminal 2: Start Vite Dev Server
```bash
cd bin/10_queen_rbee/ui/app
npm run dev  # Runs on port 7834
```

### Access the UI
```bash
# Queen API
curl http://localhost:7833/health

# Queen UI (proxied to Vite)
open http://localhost:7833/ui
```

---

## Production Build

### Build Frontend
```bash
cd bin/10_queen_rbee/ui/app
npm run build  # Creates dist/
```

### Build Backend (with embedded UI)
```bash
cd bin/10_queen_rbee
cargo build --release
```

### Run Production Binary
```bash
./target/release/queen-rbee

# UI is served at /ui
open http://localhost:7833/ui
```

---

## Implementation Details

### `src/http/static_files.rs`

```rust
pub fn create_static_router() -> Router {
    #[cfg(debug_assertions)]
    {
        // Development: Proxy to Vite dev server
        Router::new().nest("/ui", Router::new().fallback(dev_proxy_handler))
    }
    
    #[cfg(not(debug_assertions))]
    {
        // Production: Serve embedded static files
        Router::new().nest("/ui", Router::new().fallback(static_handler))
    }
}
```

**Development Handler:**
- Proxies all `/ui/*` requests to `http://localhost:7834`
- Preserves headers and status codes
- Shows helpful error if Vite is not running

**Production Handler:**
- Serves files from `RustEmbed` (embedded at compile time)
- Falls back to `index.html` for SPA routing
- Returns 404 if file not found

---

## Error Handling

### Development Mode Errors

If Vite is not running:
```
Dev server not available. Start it with: cd bin/10_queen_rbee/ui/app && npm run dev

Error: connection refused
```

**Solution:** Start the Vite dev server in a separate terminal.

### Production Mode Errors

If UI files are not built:
```
404 - Not Found
```

**Solution:** Build the UI first with `npm run build`.

---

## Benefits

### Development
- ✅ Hot-reload works (Vite HMR)
- ✅ Fast iteration (no Rust rebuild needed)
- ✅ Same URL as production (`/ui`)
- ✅ No CORS issues

### Production
- ✅ Single binary distribution
- ✅ No external dependencies
- ✅ Fast startup (no Vite needed)
- ✅ Works offline

---

## Turbo Dev Integration

The root `package.json` includes a dev script for Queen:

```json
{
  "scripts": {
    "dev:queen": "turbo dev --filter=@rbee/queen-rbee-ui ..."
  }
}
```

This starts the Vite dev server on port 7834.

---

## Testing

### Test Development Mode
```bash
# Terminal 1: Start queen binary
cargo run

# Terminal 2: Start Vite
cd ui/app && npm run dev

# Terminal 3: Test
curl http://localhost:7833/ui
# Should proxy to Vite and return HTML
```

### Test Production Mode
```bash
# Build UI
cd ui/app && npm run build

# Build binary (release mode)
cargo build --release

# Run
./target/release/queen-rbee

# Test
curl http://localhost:7833/ui
# Should serve embedded files
```

---

## Summary

The Queen UI now has a **seamless development experience**:
- Same URL in dev and prod (`/ui`)
- Hot-reload in development
- Single binary in production
- No configuration needed

**The proxy setup is automatic based on build mode!** 🎉
