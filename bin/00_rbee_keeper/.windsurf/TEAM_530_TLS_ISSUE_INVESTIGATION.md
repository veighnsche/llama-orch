# TEAM_530: TLS Issue Investigation - Tauri WebView Cannot Make HTTPS Requests

## Problem Statement

The Keeper UI (Tauri app) cannot make HTTPS fetch requests to external APIs (HuggingFace, CivitAI, GWC). All HTTPS requests fail with:

```
[Error] TLS support is not available
[Error] Fetch API cannot load https://huggingface.co/api/models?... due to access control checks.
[Error] Failed to load resource: TLS support is not available
```

This blocks the GWC adapter migration (TEAM_529) from being tested.

## Timeline of Investigation

### TEAM_529: Initial GWC Adapter Migration
- **Goal**: Migrate Keeper worker pages from Tauri commands to `@rbee/marketplace-core` adapters
- **Changes Made**:
  - Migrated `MarketplaceRbeeWorkers.tsx` to use `fetchGWCWorkers()`
  - Migrated `WorkerDetailsPage.tsx` to use `fetchGWCWorker()`
  - Updated filtering logic for GWC worker structure
- **Blocker**: TLS errors prevented testing

### TEAM_530: TLS Fix Attempts

#### Attempt 1: Add HTTPS URLs to Tauri Remote Permissions
**File**: `/home/vince/Projects/rbee/bin/00_rbee_keeper/tauri.conf.json`

```json
"remote": {
  "urls": [
    "http://localhost:7843",
    "https://huggingface.co",
    "https://civitai.com",
    "https://gwc.rbee.dev"
  ]
}
```

**Result**: ❌ No effect. Remote URLs only control navigation, not fetch/XHR.

#### Attempt 2: Add tauri-plugin-http
**Files Modified**:
1. `/home/vince/Projects/rbee/bin/00_rbee_keeper/Cargo.toml` (line 80)
   ```toml
   tauri-plugin-http = "2"
   ```

2. `/home/vince/Projects/rbee/bin/00_rbee_keeper/src/main.rs` (line 102)
   ```rust
   .plugin(tauri_plugin_http::init())
   ```

3. `/home/vince/Projects/rbee/bin/00_rbee_keeper/tauri.conf.json` (lines 69-70)
   ```json
   "http:default",
   "http:allow-fetch"
   ```

**Build**: ✅ Successful (compiled tauri-plugin-http v2.5.4)
**Result**: ❌ No effect. TLS errors persist.

#### Attempt 3: Fix GWC URL Resolution for Tauri
**Files Modified**:
- `/home/vince/Projects/rbee/frontend/packages/marketplace-core/src/adapters/gwc/list.ts`
- `/home/vince/Projects/rbee/frontend/packages/marketplace-core/src/adapters/gwc/details.ts`

**Logic**: Detect Tauri environment with `window.__TAURI__` and use production URL instead of localhost.

```typescript
const GWC_API_BASE =
  process.env.NEXT_PUBLIC_GWC_API_URL ||
  process.env.VITE_GWC_API_URL ||
  (process.env.NODE_ENV === 'development' && typeof window !== 'undefined' && !(window as any).__TAURI__
    ? 'http://localhost:7811'
    : 'https://gwc.rbee.dev')
```

**Result**: ❌ No effect on TLS errors (but correct for future).

## Current State

### What Works
- ✅ Tauri app launches successfully
- ✅ Vite dev server connects
- ✅ UI renders correctly
- ✅ Tauri commands work (SSH list, hive status, etc.)

### What Doesn't Work
- ❌ Any HTTPS fetch/XHR request from WebView
- ❌ External API calls (HuggingFace, CivitAI, GWC)
- ❌ Font loading from Google Fonts CDN
- ❌ CSS loading from external CDNs

### Error Pattern
```
[Error] TLS support is not available
[Error] Fetch API cannot load https://... due to access control checks.
[Error] Failed to load resource: TLS support is not available
```

## Technical Context

### Tauri Version
- **Tauri**: v2.9.3
- **tauri-plugin-http**: v2.5.4
- **WebView**: webkit2gtk v2.0.1 (Linux)

### Architecture
```
┌─────────────────────────────────────┐
│  Keeper UI (Tauri)                  │
│  ┌───────────────────────────────┐  │
│  │  Vite Dev Server              │  │
│  │  http://localhost:5173        │  │
│  │  ┌─────────────────────────┐  │  │
│  │  │  React Components       │  │  │
│  │  │  - HFListPage.tsx       │  │  │
│  │  │  - MarketplaceRbeeWorkers│ │  │
│  │  │  - WorkerDetailsPage    │  │  │
│  │  └─────────────────────────┘  │  │
│  └───────────────────────────────┘  │
│                                     │
│  ┌───────────────────────────────┐  │
│  │  marketplace-core             │  │
│  │  - fetchHuggingFaceModels()   │  │
│  │  - fetchGWCWorkers()          │  │
│  │  - fetchCivitAIModels()       │  │
│  └───────────────────────────────┘  │
│              │                       │
│              │ fetch() ❌ TLS FAILS  │
│              ▼                       │
│  ┌───────────────────────────────┐  │
│  │  WebKit2GTK WebView           │  │
│  │  (No native TLS support?)     │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

### Marketplace-Core Context
**Important**: `@rbee/marketplace-core` serves **both**:
1. **Next.js marketplace** (SSR + client-side)
2. **Tauri Keeper UI** (client-side only)

The adapters make direct fetch() calls from the frontend, expecting the browser/WebView to handle TLS.

## Key Files to Review

### Tauri Configuration
1. **`/home/vince/Projects/rbee/bin/00_rbee_keeper/tauri.conf.json`**
   - Lines 52-74: Security configuration
   - Lines 59-61: Remote URLs (navigation only)
   - Lines 62-71: Permissions (including http:default, http:allow-fetch)

2. **`/home/vince/Projects/rbee/bin/00_rbee_keeper/Cargo.toml`**
   - Line 80: `tauri-plugin-http = "2"`
   - Line 29: `reqwest = { version = "0.11", features = ["json", "stream"] }`

3. **`/home/vince/Projects/rbee/bin/00_rbee_keeper/src/main.rs`**
   - Lines 97-103: Tauri builder with plugin registration
   - Line 102: `.plugin(tauri_plugin_http::init())`

### Marketplace Core Adapters
4. **`/home/vince/Projects/rbee/frontend/packages/marketplace-core/src/adapters/huggingface/list.ts`**
   - Lines 143-208: `fetchHuggingFaceModels()` function
   - Line 173: Direct fetch() call to HuggingFace API

5. **`/home/vince/Projects/rbee/frontend/packages/marketplace-core/src/adapters/gwc/list.ts`**
   - Lines 6-23: GWC_API_BASE URL resolution
   - Lines 66-105: `fetchGWCWorkers()` function
   - Line 73: Direct fetch() call to GWC API

6. **`/home/vince/Projects/rbee/frontend/packages/marketplace-core/src/adapters/civitai/list.ts`**
   - Similar pattern to HuggingFace adapter

### Keeper UI Pages
7. **`/home/vince/Projects/rbee/bin/00_rbee_keeper/ui/src/pages/huggingface/HFListPage.tsx`**
   - Lines 28-40: React Query hook calling fetchHuggingFaceModels()
   - This is where the TLS error originates

8. **`/home/vince/Projects/rbee/bin/00_rbee_keeper/ui/src/pages/MarketplaceRbeeWorkers.tsx`**
   - Lines 65-80: React Query hook calling fetchGWCWorkers()
   - TEAM_529 migration (uses GWC adapter)

9. **`/home/vince/Projects/rbee/bin/00_rbee_keeper/ui/src/pages/WorkerDetailsPage.tsx`**
   - Lines 29-43: React Query hook calling fetchGWCWorker()
   - TEAM_529 migration (uses GWC adapter)

## Hypotheses for Root Cause

### Hypothesis 1: WebKit2GTK TLS Configuration
**Likelihood**: 🔴 HIGH

WebKit2GTK on Linux might not have native TLS support enabled by default. The tauri-plugin-http might not be intercepting WebView fetch() calls.

**Evidence**:
- Error message: "TLS support is not available"
- Same error for fonts, CSS, and API calls
- Plugin compiled successfully but no effect

**Investigation Needed**:
- Check if WebKit2GTK needs system-level TLS libraries
- Verify if tauri-plugin-http actually intercepts WebView fetch()
- Check Tauri v2 documentation for WebView TLS requirements

### Hypothesis 2: tauri-plugin-http Scope Configuration
**Likelihood**: 🟡 MEDIUM

The HTTP plugin might need explicit URL scope configuration, not just permissions.

**Evidence**:
- Plugin is registered but not working
- Permissions added but no effect
- Tauri v2 might have changed scope configuration

**Investigation Needed**:
- Check tauri-plugin-http v2 documentation for scope configuration
- Look for examples of HTTP plugin usage in Tauri v2
- Check if there's a separate capability file needed

### Hypothesis 3: CSP (Content Security Policy) Blocking
**Likelihood**: 🟡 MEDIUM

The CSP might be blocking external HTTPS requests despite being set to null.

**Evidence**:
- Error mentions "access control checks"
- CSP is set to null in tauri.conf.json
- Might need explicit CSP directives

**Investigation Needed**:
- Check browser DevTools Network tab for CSP violations
- Try setting explicit CSP with connect-src directives
- Check if Tauri v2 has default CSP that overrides null

### Hypothesis 4: Vite Dev Server Proxy Needed
**Likelihood**: 🟢 LOW

Tauri WebView might not be able to make external requests in dev mode, requiring a Vite proxy.

**Evidence**:
- Works in Next.js (has server-side)
- Fails in Tauri (client-only)
- Common pattern for Electron/Tauri apps

**Investigation Needed**:
- Check if Next.js marketplace has similar issues
- Try setting up Vite proxy for external APIs
- Check if this is a dev-only issue

## Recommended Next Steps

### Step 1: Verify tauri-plugin-http Installation
```bash
cd /home/vince/Projects/rbee/bin/00_rbee_keeper
cargo tree | grep tauri-plugin-http
```

Expected: Should show tauri-plugin-http v2.5.4 in dependency tree.

### Step 2: Check WebKit2GTK TLS Support
```bash
# Check if WebKit2GTK has TLS support
ldd /usr/lib/libwebkit2gtk-4.0.so | grep -i tls
ldd /usr/lib/libwebkit2gtk-4.0.so | grep -i ssl

# Check system TLS libraries
ldconfig -p | grep -i ssl
ldconfig -p | grep -i tls
```

### Step 3: Enable Tauri Debug Logging
Add to `tauri.conf.json`:
```json
"app": {
  "withGlobalTauri": true
}
```

Check Rust logs for HTTP plugin initialization:
```bash
RUST_LOG=tauri=debug,tauri_plugin_http=debug ./rbee
```

### Step 4: Test with Tauri HTTP Client (Bypass WebView)
Create a test Tauri command that uses Rust's reqwest directly:

```rust
#[tauri::command]
async fn test_https_fetch() -> Result<String, String> {
    let response = reqwest::get("https://huggingface.co/api/models?limit=1")
        .await
        .map_err(|e| e.to_string())?;
    let text = response.text().await.map_err(|e| e.to_string())?;
    Ok(text)
}
```

If this works, the issue is WebView-specific, not system TLS.

### Step 5: Check Tauri v2 Migration Guide
Review: https://v2.tauri.app/plugin/http/

Look for:
- Required system dependencies
- WebView TLS configuration
- Scope/allowlist changes from v1 to v2

### Step 6: Try Alternative Approach - Proxy Through Tauri Commands
Instead of direct fetch() from WebView, proxy all API calls through Tauri commands:

```rust
#[tauri::command]
async fn fetch_huggingface_models(params: HFParams) -> Result<String, String> {
    let url = format!("https://huggingface.co/api/models?{}", params.to_query_string());
    let response = reqwest::get(&url).await.map_err(|e| e.to_string())?;
    let text = response.text().await.map_err(|e| e.to_string())?;
    Ok(text)
}
```

This bypasses WebView TLS entirely and uses Rust's reqwest (which has TLS).

## Alternative Solutions

### Option A: Proxy All External Requests Through Tauri Commands
**Pros**:
- Guaranteed to work (Rust reqwest has TLS)
- More control over requests
- Better error handling

**Cons**:
- Breaks marketplace-core abstraction
- Duplicates code (Tauri commands + adapters)
- More maintenance overhead

### Option B: Use Vite Dev Proxy
**Pros**:
- Minimal code changes
- Standard development pattern
- Works for dev mode

**Cons**:
- Doesn't fix production builds
- Adds complexity to dev setup
- Might not work for all APIs (CORS)

### Option C: Use Different WebView (if possible)
**Pros**:
- Might have better TLS support
- Could fix other issues too

**Cons**:
- Major change
- Platform-specific
- Might not be possible on Linux

## Related Issues

### Similar Problems in Tauri Community
Search for:
- "tauri v2 tls support not available"
- "tauri webkit2gtk https fetch"
- "tauri-plugin-http not working"

Check:
- https://github.com/tauri-apps/tauri/issues
- https://github.com/tauri-apps/plugins-workspace/issues

## Success Criteria

The issue is resolved when:
1. ✅ HuggingFace API calls succeed from Keeper UI
2. ✅ GWC API calls succeed from Keeper UI
3. ✅ CivitAI API calls succeed from Keeper UI
4. ✅ External fonts/CSS load correctly
5. ✅ No TLS errors in console

## Documentation for Next Team

### Quick Start Investigation
1. Read this document completely
2. Review the 9 key files listed above
3. Run Step 1-3 from Recommended Next Steps
4. Check Tauri v2 HTTP plugin documentation
5. If stuck, try Option A (proxy through Tauri commands)

### Important Context
- marketplace-core serves both Next.js and Tauri
- GWC adapter migration (TEAM_529) is blocked by this
- tauri-plugin-http is installed but not working
- WebView TLS is the likely culprit, not system TLS

### Contact Points
- Tauri Discord: https://discord.gg/tauri
- Tauri GitHub Issues: https://github.com/tauri-apps/tauri/issues
- WebKit2GTK Docs: https://webkitgtk.org/

---

**Created by**: TEAM_530
**Date**: 2025-11-23
**Status**: 🔴 BLOCKED - TLS errors persist after all attempted fixes
**Priority**: 🔥 HIGH - Blocks marketplace functionality in Keeper UI
