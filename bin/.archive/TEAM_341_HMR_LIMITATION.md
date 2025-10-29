# TEAM-341: Vite HMR Limitation

**Date:** Oct 29, 2025  
**Status:** ⚠️ LIMITATION (by design)

---

## Problem

Vite Hot Module Replacement (HMR) uses WebSocket connections for instant updates.

When accessing the Queen UI through `http://localhost:7833/`, the Axum proxy **cannot forward WebSocket connections** to the Vite dev server.

**Result:** Browser shows:
```
Firefox can't establish a connection to the server at ws://localhost:7833/?token=...
```

---

## Why This Happens

1. **Browser loads:** `http://localhost:7833/` (proxied through Axum)
2. **Vite client tries:** `ws://localhost:7833/` (WebSocket to queen)
3. **Queen proxy:** HTTP-only proxy, **can't upgrade to WebSocket**
4. **HMR broken:** No hot reload through proxy

---

## Solution

**Use Vite directly on port 7834 for development.**

### Development Workflow

**Terminal 1: Queen API**
```bash
cd bin/10_queen_rbee
cargo run
# Queen API: http://localhost:7833
```

**Terminal 2: Vite Dev Server**
```bash
cd bin/10_queen_rbee/ui/app
pnpm dev
# Vite UI: http://localhost:7834 ← USE THIS FOR DEVELOPMENT
```

**Access UI:** `http://localhost:7834/` ✅
- ✅ Hot reload works
- ✅ WebSocket connects
- ✅ Fast refresh
- ✅ Full Vite features

**Access through Queen:** `http://localhost:7833/` ❌
- ❌ No hot reload (WebSocket blocked)
- ✅ UI loads (HTTP proxy works)
- ⚠️ Manual refresh required

---

## When to Use Each

| Port | Use When | HMR |
|------|----------|-----|
| **7834** | UI development (React, styling, SDK) | ✅ Works |
| **7833** | API testing (need both UI + API together) | ❌ Blocked |
| **Production** | Single binary deployment | N/A (embedded) |

---

## Technical Details

### Why We Can't Fix It

Axum's proxy uses HTTP-only client (`reqwest`). WebSocket upgrade requires:
1. HTTP `Upgrade: websocket` header
2. TCP connection upgrade
3. Bidirectional streaming

**Axum can't do this** without:
- `tokio-tungstenite` (WebSocket client)
- Custom upgrade logic
- Connection state management
- Extra 1000+ LOC complexity

**NOT WORTH IT** for development-only feature.

---

## Alternative: Proxy WebSockets

**IF we wanted to fix it** (DON'T):

```rust
// Would need ~1000 LOC like this:
use tokio_tungstenite::connect_async;

async fn ws_proxy_handler(
    ws: WebSocketUpgrade,
    uri: Uri,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| async move {
        // Connect to Vite WS
        let (vite_ws, _) = connect_async("ws://localhost:7834").await?;
        
        // Bidirectional forwarding
        let (client_tx, client_rx) = socket.split();
        let (vite_tx, vite_rx) = vite_ws.split();
        
        // Forward client -> vite
        tokio::spawn(async move {
            // ...complex forwarding logic...
        });
        
        // Forward vite -> client
        tokio::spawn(async move {
            // ...complex forwarding logic...
        });
    })
}
```

**Complexity:** Not worth it for dev-only feature.

---

## User Guidance

### In Documentation

**Development Guide:**
```markdown
## Development Setup

1. Start Queen API:
   ```bash
   cargo run --bin queen-rbee
   ```

2. Start Vite dev server:
   ```bash
   cd ui/app && pnpm dev
   ```

3. **Open UI:** http://localhost:7834 (for hot reload)

**Note:** Use port 7834 directly for development. Port 7833 works but without hot reload.
```

### In UI Error Messages

If we detect WebSocket failure, show helpful message:
```
❌ Hot reload unavailable

For full development experience with hot reload:
→ Open http://localhost:7834 directly

Currently viewing through: http://localhost:7833
(API proxy - hot reload disabled)
```

---

## Conclusion

**Not a bug, it's a limitation by design.**

**Solution:** Use Vite directly on port 7834 for UI development.

**Production:** Single binary works perfectly (embedded files, no Vite needed).

---

**Files Changed:**
- `static_files.rs` - Added WebSocket blocking + helpful error message
- `UI_ARCHITECTURE.md` - Documented the limitation

**Status:** Working as intended ✅
