# TEAM-350: Quick Reference Card

## Port Configuration

```
┌─────────────────────────────────────────────────────────────┐
│ Component       │ Dev Port │ Prod Port      │ Status       │
├─────────────────────────────────────────────────────────────┤
│ Keeper UI       │ 5173     │ Tauri app      │ ✅ Working   │
│ Queen UI        │ 7834     │ 7833 (embed)   │ ✅ Working   │
│ Queen Backend   │ 7833     │ 7833           │ ✅ Working   │
│ Hive UI         │ 7836     │ 7835 (embed)   │ 🔜 Next      │
│ Hive Backend    │ 7835     │ 7835           │ 🔜 Next      │
│ Worker UI       │ 7837     │ 8080 (embed)   │ 🔜 Next      │
│ Worker Backend  │ 8080     │ 8080           │ 🔜 Next      │
└─────────────────────────────────────────────────────────────┘
```

## Development Workflow

```bash
# Terminal 1: Start Queen Vite dev server
cd bin/10_queen_rbee/ui/app
pnpm dev  # Port 7834

# Terminal 2: Start Queen backend
cargo run --bin queen-rbee  # Port 7833

# Terminal 3: Start Keeper
cd bin/00_rbee_keeper/ui
pnpm dev  # Port 5173

# Open http://localhost:5173
# Navigate to Queen page
# Hot reload works! ✅
```

## Key Code Patterns

### 1. Environment-Aware iframe URL

```typescript
// Pattern: Direct to Vite in dev, embedded in prod
const isDev = import.meta.env.DEV
const url = isDev 
  ? "http://localhost:7834"  // Dev: Vite
  : "http://localhost:7833"   // Prod: Embedded
```

### 2. Smart build.rs Skipping

```rust
// Pattern: Skip UI builds when Vite is running
let vite_dev_running = std::net::TcpStream::connect("127.0.0.1:7834").is_ok();
if vite_dev_running {
    println!("cargo:warning=⚡ Skipping UI builds");
    return;
}
```

### 3. Environment-Aware postMessage

```typescript
// Pattern: Detect by port, not import.meta.env.DEV
const isOnVite = window.location.port === '7834'
const parentOrigin = isOnVite
  ? 'http://localhost:5173'  // Dev: Keeper Vite
  : '*'                       // Prod: Tauri
```

### 4. Multi-Origin Message Listener

```typescript
// Pattern: Accept both dev and prod origins
const allowedOrigins = [
  "http://localhost:7833", // Prod
  "http://localhost:7834", // Dev
]
if (!allowedOrigins.includes(event.origin)) return
```

### 5. Type Mapping

```typescript
// Pattern: Explicit field mapping
const keeperEvent: KeeperType = {
  field1: queenEvent.fieldA,
  field2: queenEvent.fieldB || 'default',
  field3: extractFromFormatted(queenEvent.formatted),
}
```

## Common Pitfalls

### ❌ WRONG: Axum wildcard route
```rust
.route("/dev/*path", get(handler))  // PANICS!
```

### ✅ CORRECT: Axum wildcard route
```rust
.route("/dev/{*path}", get(handler))  // Works!
```

---

### ❌ WRONG: ANSI escape regex
```typescript
formatted.match(/\u001b\[1m([^\u001b]+)\u001b\[0m/)  // Doesn't match!
```

### ✅ CORRECT: ANSI escape regex
```typescript
formatted.match(/\x1b\[1m([^\x1b]+)\x1b\[0m/)  // Matches!
```

---

### ❌ WRONG: Port detection in iframe
```typescript
const isDev = import.meta.env.DEV  // Always true in Vite!
```

### ✅ CORRECT: Port detection in iframe
```typescript
const isOnVite = window.location.port === '7834'  // Reliable!
```

## Narration Flow

```
Backend (Rust)
  ↓ SSE JSON
Queen UI (iframe)
  ↓ postMessage
Keeper UI (parent)
  ↓ Type mapping
Narration Store
  ↓ React state
UI Display
```

## Debug Commands

```bash
# Check if Vite is running
nc -zv localhost 7834

# Check if Queen backend is running
curl http://localhost:7833/health

# Check narration SSE stream
curl http://localhost:7833/v1/jobs/{job_id}/stream

# Force rebuild UI
cd bin/10_queen_rbee/ui/app && pnpm build
```

## Console Log Checklist

**Development mode should show:**
```
🔧 [KEEPER UI] Running in DEVELOPMENT mode
🔧 [QUEEN UI] Running in DEVELOPMENT mode
🔧 [QUEEN] Running in DEBUG mode
[Queen] Sending narration to parent: {isQueenOnVite: true, ...}
[Keeper] Received narration from Queen: {...}
```

**Production mode should show:**
```
🚀 [KEEPER UI] Running in PRODUCTION mode
🚀 [QUEEN UI] Running in PRODUCTION mode
🚀 [QUEEN] Running in RELEASE mode
```

## Files to Modify for Hive/Worker

```
For Hive (ports 7836 dev, 7835 prod):
├── bin/25_rbee_hive/build.rs                    (port 7836)
├── bin/25_rbee_hive/src/main.rs                 (startup logs)
├── bin/25_rbee_hive/ui/app/src/App.tsx          (startup logs)
├── bin/25_rbee_hive/ui/packages/.../narrationBridge.ts  (port 7836)
└── bin/00_rbee_keeper/ui/src/pages/HivePage.tsx (iframe URL)
    bin/00_rbee_keeper/ui/src/utils/narrationListener.ts (add origins)

For Worker (ports 7837 dev, 8080 prod):
├── Same pattern, different ports
```

## Quick Test

```bash
# 1. Start all dev servers
pnpm run dev:queen  # Starts keeper + queen Vite

# 2. Start queen backend
cargo run --bin queen-rbee

# 3. Open http://localhost:5173
# 4. Navigate to Queen page
# 5. Press Test button in RHAI IDE
# 6. Check narration panel on right side

Expected: ✅ Narration appears with function names
```

---

**TEAM-350 Quick Reference** - Copy this for Hive/Worker implementation!
