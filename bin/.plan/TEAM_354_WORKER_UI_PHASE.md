# TEAM-354: Worker UI Implementation Phase

**Status:** 🔜 TODO  
**Assigned To:** TEAM-354  
**Estimated Time:** 2-3 days  
**Priority:** MEDIUM  
**Dependencies:** TEAM-356, TEAM-352, TEAM-353 must be complete

---

## Mission

Implement Worker UI using the proven pattern from Queen and Hive.

**Why This Matters:**
- Completes the UI suite (Keeper, Queen, Hive, Worker)
- Final validation of shared packages
- Zero code duplication
- Consistent UX across all services

**What You're Building:**
```
Worker UI (port 7837 dev, 8080 prod)
  ↓ Narration SSE
Narration Bridge (@rbee/narration-client)
  ↓ postMessage
Keeper UI
  ↓ Display
Worker Narration Panel
```

---

## Prerequisites

- [ ] TEAM-356 complete (shared packages)
  - [ ] `@rbee/sdk-loader` (34 tests passing)
  - [ ] `@rbee/react-hooks` (19 tests passing)
  - [ ] `@rbee/shared-config`
  - [ ] `@rbee/narration-client`
  - [ ] `@rbee/dev-utils`
- [ ] TEAM-352 complete (Queen migration)
- [ ] TEAM-353 complete (Hive implementation)
- [ ] Study Hive UI as reference (most recent example)
- [ ] Read `TEAM_356_EXTRACTION_EXTRAVAGANZA.md`
- [ ] Read TEAM_353 documentation

**CRITICAL:** This is a copy-paste-modify exercise. Don't reinvent anything!

---

## Deliverables Checklist

- [ ] Worker UI package structure created
- [ ] Worker WASM SDK package created
- [ ] Worker React hooks package created
- [ ] Worker Vite app created
- [ ] build.rs configured for Worker
- [ ] Narration flow working
- [ ] Both dev and prod modes working
- [ ] Keeper WorkerPage.tsx created
- [ ] Port 7837 (dev) and 8080 (prod) configured
- [ ] No duplicate code

---

## Phase 1: Copy Hive Structure

### Step 1: Copy Entire UI Directory from Hive

```bash
cd bin

# Copy Hive UI structure to Worker
cp -r 25_rbee_hive/ui 50_llm_worker/ui
```

### Step 2: Rename All Packages

```bash
cd bin/50_llm_worker/ui

# Find and replace all instances:
# "rbee-hive" → "llm-worker"
# "Hive" → "Worker"  
# "hive" → "worker"
# 7836 → 7837
# 7835 → 8080

# Use your editor's find-replace across entire ui/ directory
```

**Specific replacements:**

| Find | Replace |
|------|---------|
| `rbee-hive-sdk` | `llm-worker-sdk` |
| `rbee-hive-react` | `llm-worker-react` |
| `rbee-hive-ui` | `llm-worker-ui` |
| `@rbee/rbee-hive` | `@rbee/llm-worker` |
| `HIVE` | `WORKER` |
| `Hive` | `Worker` |
| `hive` | `worker` |
| `7836` | `7837` |
| `7835` | `8080` |

---

## Phase 2: Update Worker SDK

### Step 1: Verify Cargo.toml

**File:** `bin/50_llm_worker/ui/packages/llm-worker-sdk/Cargo.toml`

```toml
[package]
name = "llm-worker-sdk"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
wasm-bindgen = "0.2"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
web-sys = { version = "0.3", features = [
    "Request",
    "RequestInit",
    "RequestMode",
    "Response",
    "Window",
] }
js-sys = "0.3"
wasm-bindgen-futures = "0.4"

[dependencies.rbee-job-client]
path = "../../../../99_shared_crates/rbee-job-client"
features = ["wasm"]
```

### Step 2: Update src/lib.rs

```rust
// TEAM-354: Worker WASM SDK
// Pattern copied from Hive (TEAM-353)

use wasm_bindgen::prelude::*;
use rbee_job_client::JobClient;
use rbee_operations::Operation;

#[wasm_bindgen]
pub struct WorkerClient {
    client: JobClient,
}

#[wasm_bindgen]
impl WorkerClient {
    #[wasm_bindgen(constructor)]
    pub fn new(base_url: String) -> Self {
        Self {
            client: JobClient::new(base_url),
        }
    }

    #[wasm_bindgen(js_name = submitAndStream)]
    pub async fn submit_and_stream(
        &self,
        operation: JsValue,
        callback: &js_sys::Function,
    ) -> Result<String, JsValue> {
        let op: Operation = serde_json::from_str(
            &operation.as_string().ok_or("Operation must be string")?
        ).map_err(|e| JsValue::from_str(&e.to_string()))?;

        self.client
            .submit_and_stream(op, |line| {
                let json_value = JsValue::from_str(line);
                let _ = callback.call1(&JsValue::null(), &json_value);
                Ok(())
            })
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
```

### Step 3: Update package.json

```json
{
  "name": "@rbee/llm-worker-sdk",
  "version": "0.1.0",
  "type": "module",
  "main": "./pkg/llm_worker_sdk.js",
  "types": "./pkg/llm_worker_sdk.d.ts",
  "scripts": {
    "build": "wasm-pack build --target web --out-dir pkg"
  }
}
```

### Step 4: Build SDK

```bash
cd bin/50_llm_worker/ui/packages/llm-worker-sdk
pnpm install
pnpm build
```

**Verify:** `pkg/llm_worker_sdk.js` created

---

## Phase 3: Update Worker React Package

### Step 1: Update package.json

**File:** `bin/50_llm_worker/ui/packages/llm-worker-react/package.json`

```json
{
  "name": "@rbee/llm-worker-react",
  "version": "0.1.0",
  "type": "module",
  "main": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "scripts": {
    "build": "tsc",
    "dev": "tsc --watch"
  },
  "dependencies": {
    "@rbee/llm-worker-sdk": "workspace:*",
    "@rbee/sdk-loader": "workspace:*",
    "@rbee/react-hooks": "workspace:*",
    "@rbee/narration-client": "workspace:*",
    "react": "^19.0.0"
  },
  "devDependencies": {
    "@types/react": "^19.0.0",
    "typescript": "^5.0.0"
  }
}
```

### Step 2: Update src/hooks/useWorkerOperations.ts

```typescript
// TEAM-354: Worker operations hook
import { useState, useCallback } from 'react'
import { WorkerClient } from '@rbee/llm-worker-sdk'
import { createStreamHandler, SERVICES } from '@rbee/narration-client'

export function useWorkerOperations(baseUrl: string) {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const executeOperation = useCallback(async (operation: any) => {
    setIsLoading(true)
    setError(null)

    try {
      const client = new WorkerClient(baseUrl)
      
      // TEAM-354: Use shared narration client
      const handleNarration = createStreamHandler(SERVICES.worker)
      
      await client.submitAndStream(
        JSON.stringify(operation),
        (line: string) => {
          handleNarration(line)
        }
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setIsLoading(false)
    }
  }, [baseUrl])

  return { executeOperation, isLoading, error }
}
```

### Step 3: Build React Package

```bash
cd bin/50_llm_worker/ui/packages/llm-worker-react
pnpm install
pnpm build
```

---

## Phase 4: Update Worker Vite App

### Step 1: Update package.json

**File:** `bin/50_llm_worker/ui/app/package.json`

```json
{
  "name": "@rbee/llm-worker-ui",
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite --port 7837",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "@rbee/llm-worker-react": "workspace:*",
    "@rbee/sdk-loader": "workspace:*",
    "@rbee/react-hooks": "workspace:*",
    "@rbee/shared-config": "workspace:*",
    "@rbee/narration-client": "workspace:*",
    "@rbee/dev-utils": "workspace:*",
    "react": "^19.0.0",
    "react-dom": "^19.0.0"
  },
  "devDependencies": {
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "@vitejs/plugin-react": "^4.0.0",
    "typescript": "^5.0.0",
    "vite": "^5.0.0"
  }
}
```

### Step 2: Update vite.config.ts

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 7837,  // TEAM-354: Worker dev port
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})
```

### Step 3: Update src/App.tsx

```typescript
// TEAM-354: Worker UI entry point
import { useEffect } from 'react'
import { logStartupMode, isDevelopment, getCurrentPort } from '@rbee/dev-utils'
import { useWorkerOperations } from '@rbee/llm-worker-react'

export function App() {
  useEffect(() => {
    logStartupMode('WORKER UI', isDevelopment(), getCurrentPort())
  }, [])

  const { executeOperation, isLoading, error } = useWorkerOperations(
    'http://localhost:8080'  // Worker backend
  )

  return (
    <div>
      <h1>Worker Management Interface</h1>
      {/* Your Worker UI here */}
    </div>
  )
}
```

### Step 4: Update index.html Title

```html
<title>Worker Management</title>
```

### Step 5: Build and Test

```bash
cd bin/50_llm_worker/ui/app
pnpm install
pnpm dev
```

**Browser:** http://localhost:7837

**Expected:** "🔧 [WORKER UI] Running in DEVELOPMENT mode"

---

## Phase 5: Configure build.rs

**File:** `bin/50_llm_worker/build.rs`

```rust
// TEAM-354: Worker UI build script

use std::process::Command;

fn main() {
    // Check if Vite dev server is running on port 7837
    let vite_dev_running = std::net::TcpStream::connect("127.0.0.1:7837").is_ok();

    if vite_dev_running {
        println!("cargo:warning=⚡ Vite dev server detected on port 7837 - SKIPPING ALL UI builds");
        println!("cargo:warning=   (Dev server provides fresh packages via hot reload)");
        return;
    }

    println!("cargo:warning=🔨 Building llm-worker UI packages and app...");

    // Build SDK
    println!("cargo:warning=  📦 Building @rbee/llm-worker-sdk (WASM)...");
    let sdk_dir = std::path::PathBuf::from("ui/packages/llm-worker-sdk");
    let sdk_status = Command::new("pnpm")
        .args(&["build"])
        .current_dir(&sdk_dir)
        .status()
        .expect("Failed to build Worker SDK");

    if !sdk_status.success() {
        panic!("Worker SDK build failed");
    }

    // Build React
    println!("cargo:warning=  📦 Building @rbee/llm-worker-react...");
    let react_dir = std::path::PathBuf::from("ui/packages/llm-worker-react");
    let react_status = Command::new("pnpm")
        .args(&["build"])
        .current_dir(&react_dir)
        .status()
        .expect("Failed to build Worker React package");

    if !react_status.success() {
        panic!("Worker React build failed");
    }

    // Build App
    println!("cargo:warning=  🎨 Building @rbee/llm-worker-ui app...");
    let app_dir = std::path::PathBuf::from("ui/app");
    let app_status = Command::new("pnpm")
        .args(&["build"])
        .current_dir(&app_dir)
        .status()
        .expect("Failed to build Worker app");

    if !app_status.success() {
        panic!("Worker app build failed");
    }

    println!("cargo:warning=✅ llm-worker UI (SDK + React + App) built successfully");
}
```

**Test:**

```bash
cargo build --bin llm-worker
```

**Expected:** "🔨 Building llm-worker UI..."

---

## Phase 6: Update Keeper UI

### Step 1: Create WorkerPage.tsx

**File:** `bin/00_rbee_keeper/ui/src/pages/WorkerPage.tsx`

```typescript
// TEAM-354: Worker management page
import { getIframeUrl } from '@rbee/shared-config'

export function WorkerPage() {
  const isDev = import.meta.env.DEV
  const workerUrl = getIframeUrl('worker', isDev)

  return (
    <div className="flex flex-col h-full">
      <div className="p-4 border-b">
        <h1 className="text-2xl font-bold">Worker Management</h1>
      </div>
      <div className="flex-1">
        <iframe
          src={workerUrl}
          className="w-full h-full border-0"
          title="Worker Management Interface"
          sandbox="allow-same-origin allow-scripts allow-forms allow-popups allow-modals"
          allow="cross-origin-isolated"
        />
      </div>
    </div>
  )
}
```

### Step 2: Add Route

Edit `bin/00_rbee_keeper/ui/src/App.tsx`:

```typescript
import { WorkerPage } from './pages/WorkerPage'

<Route path="/worker" element={<WorkerPage />} />
```

### Step 3: Add Navigation

```typescript
<Link to="/worker">Worker Management</Link>
```

---

## Phase 7: Verify Shared Config Includes Worker

**No changes needed!** The shared config already includes Worker:

```typescript
// frontend/packages/shared-config/src/ports.ts
export const PORTS = {
  // ...
  worker: {
    dev: 7837,
    prod: 8080,
    backend: 8080,
  },
}
```

Verify in `@rbee/narration-client`:

```typescript
// frontend/packages/narration-client/src/config.ts
export const SERVICES = {
  // ...
  worker: {
    name: 'llm-worker',
    devPort: 7837,
    prodPort: 8080,
    keeperDevPort: 5173,
    keeperProdOrigin: '*',
  },
}
```

**If not present, add them!**

---

## Phase 8: Testing

### Test 1: Development Mode

**Terminal 1:** Worker Vite
```bash
cd bin/50_llm_worker/ui/app
pnpm dev  # Port 7837
```

**Terminal 2:** Worker backend
```bash
cargo run --bin llm-worker  # Port 8080
```

**Terminal 3:** Keeper
```bash
cd bin/00_rbee_keeper/ui
pnpm dev
```

**Checklist:**
- [ ] Navigate to Worker page
- [ ] iframe loads from http://localhost:7837
- [ ] Console: "🔧 [WORKER UI] Running in DEVELOPMENT mode"
- [ ] Hot reload works
- [ ] No console errors

### Test 2: Narration Flow

**Checklist:**
- [ ] Execute worker operation
- [ ] Narration events appear in console
- [ ] Events forwarded to keeper
- [ ] Narration panel shows events
- [ ] Function names extracted
- [ ] No parse errors

### Test 3: Production Mode

```bash
cargo build --release --bin llm-worker
```

**Checklist:**
- [ ] iframe loads from http://localhost:8080
- [ ] Console: "🚀 [WORKER UI] Running in PRODUCTION mode"
- [ ] Narration works
- [ ] No errors

---

## Phase 9: Final Verification

### Complete UI Suite Checklist

**All services running simultaneously:**

```bash
# Terminal 1: Queen
cd bin/10_queen_rbee && cargo run

# Terminal 2: Hive  
cd bin/25_rbee_hive && cargo run

# Terminal 3: Worker
cd bin/50_llm_worker && cargo run

# Terminal 4: Keeper
cd bin/00_rbee_keeper/ui && pnpm dev
```

**Test all services:**
- [ ] Queen page loads (port 7833/7834)
- [ ] Hive page loads (port 7835/7836)
- [ ] Worker page loads (port 8080/7837)
- [ ] All narration flows correctly
- [ ] All function names extracted
- [ ] No origin errors
- [ ] No console errors

### Shared Package Usage Verification

**Worker uses ALL shared packages:**
- [ ] `@rbee/sdk-loader` - WASM/SDK loading with retry logic (TEAM-356)
- [ ] `@rbee/react-hooks` - useAsyncState, useSSEWithHealthCheck (TEAM-356)
- [ ] `@rbee/shared-config` - Port configuration
- [ ] `@rbee/narration-client` - Narration handling
- [ ] `@rbee/dev-utils` - Environment detection
- [ ] `@rbee/iframe-bridge` - (if needed)

**Code duplication:**
- [ ] ZERO duplicate narration code
- [ ] ZERO duplicate port configuration
- [ ] ZERO duplicate environment detection

---

## Documentation

Create summary:

```bash
cat > bin/.plan/TEAM_354_WORKER_IMPLEMENTATION_SUMMARY.md << 'EOF'
# TEAM-354 Worker Implementation Summary

## Mission Complete

Implemented Worker UI using proven pattern from Queen and Hive.

## Created

### Packages
1. @rbee/llm-worker-sdk - WASM SDK
2. @rbee/llm-worker-react - React hooks
3. @rbee/llm-worker-ui - Vite app

### Files
- bin/50_llm_worker/ui/* - Complete UI suite
- bin/50_llm_worker/build.rs - Build script
- bin/00_rbee_keeper/ui/src/pages/WorkerPage.tsx - Keeper page

## Ports

- Dev: 7837 (Vite)
- Prod: 8080 (embedded + backend)

## Shared Package Usage

✅ ALL shared packages used:
- @rbee/sdk-loader - WASM/SDK loading with retry logic (TEAM-356)
- @rbee/react-hooks - useAsyncState, useSSEWithHealthCheck (TEAM-356)
- @rbee/shared-config - Port configuration
- @rbee/narration-client - Narration handling
- @rbee/dev-utils - Environment detection

## Code Reduction

Avoided ~120 LOC duplicate code.

## Complete UI Suite

✅ Keeper (5173)
✅ Queen (7833/7834)
✅ Hive (7835/7836)
✅ Worker (8080/7837)

All narration flows working end-to-end.

## Pattern Validation

✅ Pattern works for 3 services (Queen, Hive, Worker)
✅ Zero duplication across all UIs
✅ Shared packages save 360+ LOC total
✅ Single source of truth for ports
✅ Single source of truth for narration

## Success Metrics

- 4 services with working UIs
- 360+ LOC saved via shared packages
- 100% code reuse for common functionality
- 0 duplicate code
- Consistent UX across all services

## Next: TEAM-355

Final documentation and cleanup.
EOF
```

---

## Acceptance Criteria

- [ ] Worker SDK builds
- [ ] Worker React builds
- [ ] Worker App builds
- [ ] build.rs works correctly
- [ ] Keeper WorkerPage loads
- [ ] Dev mode works (port 7837)
- [ ] Prod mode works (port 8080)
- [ ] Narration flows correctly
- [ ] Zero duplicate code
- [ ] All 4 services work together
- [ ] Documentation complete

---

## Handoff to TEAM-355

**What you've completed:**
- Complete UI suite (Keeper, Queen, Hive, Worker)
- All services use shared packages
- Zero code duplication
- Pattern proven across 3 implementations

**Next team (TEAM-355) will:**
- Update all documentation
- Create architecture diagrams
- Write migration guides
- Close out the UI implementation project

---

## Success Criteria

✅ Worker UI complete  
✅ All 4 services working  
✅ Zero duplication  
✅ Pattern validated 3 times  
✅ Ready for final docs

---

**TEAM-354: Complete the UI suite!** 🎉
