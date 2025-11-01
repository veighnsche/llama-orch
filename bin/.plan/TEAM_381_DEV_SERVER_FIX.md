# TEAM-381: Dev Server Detection Fix + Hot Reload Strategy

**Date:** 2025-11-01  
**Status:** ✅ COMPLETE

## Problem 1: False Positive Dev Server Detection

### Issue
`rbee-hive` build script was detecting Vite dev server as running even when it wasn't, causing UI builds to be skipped incorrectly.

**Symptoms:**
```
warning: rbee-hive@0.1.0: ⚡ Vite dev server detected on port 7836 - SKIPPING ALL UI builds
```
But `lsof -i :7836` showed port was not in use.

### Root Cause
The build scripts used `TcpStream::connect()` to check if the dev server was running:

```rust
let vite_dev_running = std::net::TcpStream::connect("127.0.0.1:7836").is_ok();
```

**Problem:** TCP socket checks can give false positives due to:
- TIME_WAIT state from previous connections
- Stale sockets in the kernel
- OS-level socket caching

### Solution
Changed to HTTP-based detection using `curl`:

```rust
let vite_dev_running = Command::new("curl")
    .args(&["-s", "-o", "/dev/null", "-w", "%{http_code}", "http://127.0.0.1:7836"])
    .output()
    .ok()
    .and_then(|output| String::from_utf8(output.stdout).ok())
    .map(|code| code.starts_with('2') || code.starts_with('3')) // 2xx or 3xx response
    .unwrap_or(false);
```

**Why this works:**
- ✅ HTTP request requires actual server response
- ✅ No false positives from stale sockets
- ✅ Checks that Vite is actually serving content
- ✅ Falls back to `false` if curl fails or no response

### Files Changed
1. `bin/20_rbee_hive/build.rs` - Fixed port 7836 detection
2. `bin/10_queen_rbee/build.rs` - Fixed port 7834 detection

### Verification
```bash
# Without dev server running
cargo build -p rbee-hive
# ✅ Now correctly builds UI

# With dev server running
turbo dev --filter=@rbee/rbee-hive-ui &
cargo build -p rbee-hive
# ✅ Correctly skips UI build
```

---

## Problem 2: Hot Reload Strategy

### Question
> "Do we need nodemon in all the packages so that it hot reloads after changes?"

### Answer: NO - You Already Have Hot Reload!

**Current Setup (CORRECT):**

#### Frontend Hot Reload ✅
Your frontend packages already have hot reload via **Vite HMR** (Hot Module Replacement):

```json
// bin/20_rbee_hive/ui/app/package.json
{
  "scripts": {
    "dev": "vite"  // ← Vite provides HMR automatically
  }
}
```

**How it works:**
1. Run `turbo dev` (or `pnpm dev` in a package)
2. Vite watches your source files
3. On file change → Vite hot-reloads the module in the browser
4. **No page refresh needed** - state is preserved

**What you get:**
- ✅ Instant updates on `.tsx`, `.ts`, `.css` changes
- ✅ React Fast Refresh (preserves component state)
- ✅ WASM SDK rebuilds automatically
- ✅ Tailwind CSS hot reload

#### Backend Hot Reload ✅
Your Rust backend has hot reload via **cargo-watch**:

```bash
# Already works!
cargo watch -x 'run -p rbee-hive'
```

**How it works:**
1. `cargo-watch` monitors Rust source files
2. On file change → Rebuilds and restarts the binary
3. UI is served from the rebuilt binary

**What you get:**
- ✅ Automatic rebuild on `.rs` changes
- ✅ Automatic restart of the daemon
- ✅ Fresh UI embedded in the binary

### Why You DON'T Need Nodemon

**Nodemon is for Node.js projects** that don't have built-in hot reload. You don't need it because:

1. **Vite > Nodemon for Frontend**
   - Vite HMR is faster (milliseconds vs seconds)
   - Vite preserves React state
   - Vite is already configured

2. **cargo-watch > Nodemon for Backend**
   - cargo-watch is Rust-native
   - Handles incremental compilation
   - Already integrated with your workflow

3. **Turbo orchestrates everything**
   - `turbo dev` runs all dev servers in parallel
   - Handles dependencies between packages
   - Caches builds for speed

### Recommended Workflow

#### Development Mode (Hot Reload)
```bash
# Terminal 1: Run all frontend dev servers
turbo dev

# Terminal 2: Run Rust backend with auto-reload
cargo watch -x 'run -p rbee-hive'
```

**Result:**
- Frontend changes → Instant HMR in browser
- Backend changes → Auto-rebuild and restart
- WASM SDK changes → Auto-rebuild, HMR updates

#### Production Build
```bash
# Build everything (UI embedded in Rust binary)
cargo build --release -p rbee-hive
```

**Result:**
- UI is built and embedded in the binary
- Single binary with all assets
- No dev server needed

### Current Package Scripts

**Frontend packages already have `dev` scripts:**
```json
{
  "scripts": {
    "dev": "vite",           // ← HMR enabled
    "build": "vite build"    // ← Production build
  }
}
```

**Root package.json has convenience scripts:**
```json
{
  "scripts": {
    "dev:hive": "turbo dev --filter=@rbee/rbee-hive-ui ...",
    "dev:queen": "turbo dev --filter=@rbee/queen-rbee-ui ...",
    "dev:all": "turbo dev"
  }
}
```

### What About the Build Script Detection?

The build script detection is **intentional** and **correct**:

**When dev server is running:**
- `cargo build` skips UI build (uses dev server assets)
- Fast Rust-only rebuilds
- UI changes via HMR (no rebuild needed)

**When dev server is NOT running:**
- `cargo build` builds UI into `dist/`
- Embeds UI in Rust binary
- Production-ready single binary

This is the **best of both worlds**:
- Development: Fast HMR + Fast Rust rebuilds
- Production: Single binary with embedded UI

### Summary

**DO:**
- ✅ Use `turbo dev` for frontend hot reload (Vite HMR)
- ✅ Use `cargo watch` for backend hot reload
- ✅ Keep current build script detection logic

**DON'T:**
- ❌ Add nodemon (redundant with Vite + cargo-watch)
- ❌ Change the build script detection (it's working correctly now)
- ❌ Add extra watchers (you already have everything you need)

**Your current setup is optimal!** The only issue was the false positive detection, which is now fixed.

## Testing

### Test 1: Dev Server Not Running
```bash
# Stop all dev servers
pkill -f vite

# Build should compile UI
cargo build -p rbee-hive 2>&1 | grep "Building rbee-hive UI"
# Expected: "🔨 Building rbee-hive UI packages and app..."
```

### Test 2: Dev Server Running
```bash
# Start dev server
cd bin/20_rbee_hive/ui/app
pnpm dev &

# Build should skip UI
cargo build -p rbee-hive 2>&1 | grep "Vite dev server detected"
# Expected: "⚡ Vite dev server detected on port 7836 - SKIPPING ALL UI builds"
```

### Test 3: Hot Reload Works
```bash
# Terminal 1: Start dev server
turbo dev --filter=@rbee/rbee-hive-ui

# Terminal 2: Edit a file
echo "// test change" >> bin/20_rbee_hive/ui/app/src/App.tsx

# Browser: Should hot reload instantly (no page refresh)
```

## Impact

- **Developer Experience:** No more false positives, faster feedback loop
- **Build Reliability:** Accurate detection of dev server state
- **Hot Reload:** Already working perfectly, no changes needed
- **Production Builds:** Unaffected, still embed UI correctly
