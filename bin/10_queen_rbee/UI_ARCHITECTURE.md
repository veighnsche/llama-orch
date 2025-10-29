# Queen UI Architecture

**Date:** Oct 29, 2025  
**Status:** ✅ CORRECT

---

## CRITICAL: UI is Served at ROOT Path

**URL:** `http://localhost:7833/`  
**NOT:** ~~`http://localhost:7833/ui`~~

The UI is served at the **root path** (`/`), not at `/ui`.

This is a **hard requirement**. API routes take priority via router merge order.

---

## How It Works

### Router Merge Order

```rust
// main.rs
let api_router = Router::new()
    .route("/health", get(handle_health))
    .route("/v1/*", /* all API routes */);

let static_router = create_static_router(); // Fallback for everything else

api_router.merge(static_router) // API routes have priority
```

### Development Mode (`cargo run`)

Backend proxies **root path** to Vite dev server:

```rust
// static_files.rs
#[cfg(debug_assertions)]
Router::new().fallback(dev_proxy_handler)

// Proxies http://localhost:7833/* → http://localhost:7834/*
```

- Vite runs on port `7834`
- Hot-reload works
- No build step needed

### Production Mode (`cargo build --release`)

Backend serves embedded static files at **root path**:

```rust
// static_files.rs
#[cfg(not(debug_assertions))]
Router::new().fallback(static_handler)

// Serves files from ui/app/dist/ embedded at compile time
```

- Single binary (no Vite needed)
- Files embedded via `RustEmbed`
- SPA routing via index.html fallback

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

# Queen UI (proxied to Vite in dev, or served from dist/ in prod)
open http://localhost:7833/
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

The `build.rs` script automatically builds the UI before Rust compilation.

### Run Production Binary
```bash
./target/release/queen-rbee

# UI is served at ROOT
open http://localhost:7833/
```

---

## URL Routing

### API Routes (Priority)
- `GET /health` → Health check
- `GET /v1/info` → Queen info
- `POST /v1/jobs` → Job submission
- `GET /v1/jobs/{job_id}/stream` → SSE stream
- etc.

### UI Routes (Fallback)
- `GET /` → index.html
- `GET /assets/*` → Static assets (JS, CSS, images)
- `GET /dashboard` → index.html (SPA routing)
- `GET /any-other-path` → index.html (SPA routing)

**How it works:** API routes are registered first, so they take priority. Everything else falls through to the UI fallback handler.

---

## Files

- **`src/main.rs`** - Router setup, merge order
- **`src/http/static_files.rs`** - UI serving logic (dev proxy + prod embedded)
- **`build.rs`** - Automatic UI build before Rust compilation
- **`ui/app/`** - Vite React app

---

## Related Services

Same architecture applies to:
- **rbee-hive:** `http://localhost:7835/` (not `/ui`)
- **llm-worker:** `http://localhost:8080/` (not `/ui`)
- **comfy-worker:** `http://localhost:8188/` (not `/ui`)
- **vllm-worker:** `http://localhost:8000/` (not `/ui`)

---

## Why Root Path?

1. **Simpler URLs** - `localhost:7833` instead of `localhost:7833/ui`
2. **Standard SPA behavior** - Most React/Vue apps serve at root
3. **No path prefix issues** - No need to configure base path in Vite/React Router
4. **Cleaner architecture** - API routes are namespaced (`/v1/*`), UI gets the rest

---

## Troubleshooting

### UI not loading in dev mode

**Error:** "Dev server not available"

**Solution:** Start Vite dev server:
```bash
cd bin/10_queen_rbee/ui/app && npm run dev
```

### UI not loading in prod mode

**Error:** "404 - Not Found"

**Solution:** Build UI first:
```bash
cd bin/10_queen_rbee/ui/app && npm run build
```

Check that `ui/app/dist/` exists and contains `index.html`.

---

**See also:**
- `src/http/static_files.rs` - Implementation with detailed comments
- `PORT_CONFIGURATION.md` - Port mapping reference
