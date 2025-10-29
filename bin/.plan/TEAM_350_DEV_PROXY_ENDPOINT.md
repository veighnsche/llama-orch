# TEAM-350: Development Proxy Endpoint `/dev`

**Status:** ✅ COMPLETE

## Problem

In development, we need the queen UI to use the Vite dev server (port 7834) for hot-reload, but in production it should use embedded static files. The previous approach of running turbo dev conflicted with build.rs.

## Solution

Created a `/dev` endpoint in queen-rbee that proxies to the Vite dev server.

### Architecture

**Development:**
```
rbee-keeper (port 5173)
  └─ iframe: http://localhost:7833/dev
       ↓
     queen-rbee backend (port 7833)
       ↓ /dev/* → Proxy
       ↓
     Vite dev server (port 7834)
```

**Production:**
```
rbee-keeper (Tauri app)
  └─ iframe: http://localhost:7833/
       ↓
     queen-rbee backend (port 7833)
       ↓ / → Embedded static files
```

## Implementation

### 1. Development Proxy Handler

**File:** `bin/10_queen_rbee/src/http/dev_proxy.rs`

```rust
pub async fn dev_proxy_handler(uri: Uri, req: Request) -> impl IntoResponse {
    // Strip /dev prefix
    let path = uri.path().strip_prefix("/dev").unwrap_or(uri.path());
    
    // Construct target URL for Vite dev server
    let vite_url = format!("http://localhost:7834{}{}", path, query);
    
    // Forward the request using reqwest
    let client = reqwest::Client::new();
    // ... proxy logic
}
```

**Features:**
- Strips `/dev` prefix before forwarding
- Preserves query strings
- Forwards headers
- Returns Vite response to client

### 2. Added Route

**File:** `bin/10_queen_rbee/src/main.rs`

```rust
.route("/dev/{*path}", get(http::dev_proxy_handler))
```

**Note:** Axum requires `{*path}` syntax for wildcard capture, not `*path`.

### 3. Updated rbee-keeper iframe

**File:** `bin/00_rbee_keeper/ui/src/pages/QueenPage.tsx`

```typescript
const isDev = import.meta.env.DEV
const queenUrl = isDev 
  ? "http://localhost:7833/dev"  // Dev: Proxy to Vite
  : "http://localhost:7833"       // Prod: Embedded files

return <iframe src={queenUrl} ... />
```

## URL Mapping

| Environment | rbee-keeper | iframe URL | Backend Serves |
|-------------|-------------|------------|----------------|
| **Development** | http://localhost:5173 | http://localhost:7833/dev | Proxy to :7834 |
| **Production** | Tauri app | http://localhost:7833/ | Embedded dist/ |

## Benefits

✅ **No turbo dev conflicts** - Only Vite dev server runs, no build.rs conflict
✅ **Hot reload works** - Changes in queen UI reflect immediately
✅ **Clean separation** - `/dev` for development, `/` for production
✅ **No code duplication** - Single endpoint handles both modes
✅ **Automatic** - Environment detection via `import.meta.env.DEV`

## Development Workflow

```bash
# Terminal 1: Start Vite dev server for queen UI
cd bin/10_queen_rbee/ui/app
pnpm dev  # Runs on port 7834

# Terminal 2: Start queen-rbee backend
cargo run --bin queen-rbee  # Runs on port 7833

# Terminal 3: Start rbee-keeper
cd bin/00_rbee_keeper/ui
pnpm dev  # Runs on port 5173

# Open http://localhost:5173
# Navigate to Queen page
# iframe loads http://localhost:7833/dev
# Backend proxies to http://localhost:7834
# Hot reload works! ✅
```

## Production Workflow

```bash
# Build everything
cargo build --release --bin queen-rbee

# build.rs automatically builds:
# 1. SDK (WASM)
# 2. React package
# 3. App (Vite build → dist/)

# Start queen-rbee
./target/release/queen-rbee

# Open rbee-keeper (Tauri app)
# iframe loads http://localhost:7833/
# Backend serves embedded dist/
# No Vite needed! ✅
```

## Files Changed

1. **bin/10_queen_rbee/src/http/dev_proxy.rs** - New proxy handler
2. **bin/10_queen_rbee/src/http/mod.rs** - Export dev_proxy_handler
3. **bin/10_queen_rbee/src/main.rs** - Add /dev/* route
4. **bin/00_rbee_keeper/ui/src/pages/QueenPage.tsx** - Environment-aware iframe URL

## Dependencies

- ✅ `reqwest` already in Cargo.toml (used for device detection)
- ✅ No new dependencies needed

## Testing

```bash
# 1. Start Vite dev server
cd bin/10_queen_rbee/ui/app && pnpm dev

# 2. Start queen-rbee
cargo run --bin queen-rbee

# 3. Test proxy endpoint directly
curl http://localhost:7833/dev
# Should return Vite dev server HTML

# 4. Start rbee-keeper
cd bin/00_rbee_keeper/ui && pnpm dev

# 5. Open http://localhost:5173
# Navigate to Queen page
# Should see queen UI with hot reload working
```

## Related

- PORT_CONFIGURATION.md - Port mapping documentation
- TEAM_350_BUILD_RS_REAL_FIX.md - build.rs now builds all packages
- TEAM_350_POSTMESSAGE_ORIGIN_FIX.md - postMessage origin fix

---

**TEAM-350 Signature:** Added /dev proxy endpoint for development hot-reload without turbo dev conflicts
