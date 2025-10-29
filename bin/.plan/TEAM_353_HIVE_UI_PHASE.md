# TEAM-353: Hive UI Implementation Phase

**Status:** 🔜 TODO  
**Assigned To:** TEAM-353  
**Estimated Time:** 2-3 days  
**Priority:** HIGH  
**Dependencies:** TEAM-351 and TEAM-352 must be complete

---

## Mission

Implement Hive UI from scratch using shared packages and Queen UI pattern.

**Why This Matters:**
- Proves shared packages eliminate duplication
- Establishes pattern for Worker UI
- Provides Hive management interface
- No code duplication from Queen

**What You're Building:**
```
Hive UI (port 7836 dev, 7835 prod)
  ↓ Narration SSE
Narration Bridge (@rbee/narration-client)
  ↓ postMessage
Keeper UI
  ↓ Display
Hive Narration Panel
```

---

## Prerequisites

- [ ] TEAM-351 complete (shared packages exist)
- [ ] TEAM-352 complete (Queen migration validates pattern)
- [ ] Read `TEAM_350_COMPLETE_IMPLEMENTATION_GUIDE.md`
- [ ] Read `TEAM_351_SHARED_PACKAGES_PHASE.md`
- [ ] Read `TEAM_352_QUEEN_MIGRATION_PHASE.md`
- [ ] Study Queen UI structure as reference

---

## Deliverables Checklist

- [ ] Hive UI package structure created
- [ ] Hive WASM SDK package created
- [ ] Hive React hooks package created
- [ ] Hive Vite app created
- [ ] build.rs configured for Hive
- [ ] Narration flow working (backend → UI → keeper)
- [ ] Both dev and prod modes working
- [ ] Keeper HivePage.tsx created
- [ ] Port 7836 (dev) and 7835 (prod) configured
- [ ] No duplicate code (all using shared packages)

---

## Phase 1: Create Package Structure

### Step 1: Create Directory Structure

```bash
cd bin/25_rbee_hive

mkdir -p ui/packages/rbee-hive-sdk/src
mkdir -p ui/packages/rbee-hive-react/src
mkdir -p ui/app/src
```

**Structure:**
```
bin/25_rbee_hive/
├── ui/
│   ├── packages/
│   │   ├── rbee-hive-sdk/           # WASM SDK
│   │   └── rbee-hive-react/         # React hooks
│   └── app/                         # Vite app
├── src/                             # Rust backend
└── build.rs                         # UI build script
```

---

## Phase 2: Create Hive WASM SDK

### Step 1: Create Package Structure

```bash
cd bin/25_rbee_hive/ui/packages/rbee-hive-sdk
```

### Step 2: Create Cargo.toml

```toml
[package]
name = "rbee-hive-sdk"
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

### Step 3: Create src/lib.rs

```rust
// TEAM-353: Hive WASM SDK
// Pattern copied from Queen SDK (TEAM-350 validated)

use wasm_bindgen::prelude::*;
use rbee_job_client::JobClient;
use rbee_operations::Operation;

#[wasm_bindgen]
pub struct HiveClient {
    client: JobClient,
}

#[wasm_bindgen]
impl HiveClient {
    #[wasm_bindgen(constructor)]
    pub fn new(base_url: String) -> Self {
        Self {
            client: JobClient::new(base_url),
        }
    }

    /// Submit operation and stream results
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

### Step 4: Create package.json

```json
{
  "name": "@rbee/rbee-hive-sdk",
  "version": "0.1.0",
  "type": "module",
  "main": "./pkg/rbee_hive_sdk.js",
  "types": "./pkg/rbee_hive_sdk.d.ts",
  "scripts": {
    "build": "wasm-pack build --target web --out-dir pkg"
  }
}
```

### Step 5: Build SDK

```bash
pnpm install
pnpm build
```

**Verify:** `pkg/` folder created with WASM files

---

## Phase 3: Create Hive React Hooks Package

### Step 1: Create package.json

```bash
cd bin/25_rbee_hive/ui/packages/rbee-hive-react
```

```json
{
  "name": "@rbee/rbee-hive-react",
  "version": "0.1.0",
  "type": "module",
  "main": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "scripts": {
    "build": "tsc",
    "dev": "tsc --watch"
  },
  "dependencies": {
    "@rbee/rbee-hive-sdk": "workspace:*",
    "@rbee/narration-client": "workspace:*",
    "react": "^18.2.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "typescript": "^5.0.0"
  }
}
```

### Step 2: Create tsconfig.json

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ES2020",
    "moduleResolution": "node",
    "jsx": "react-jsx",
    "declaration": true,
    "outDir": "./dist",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

### Step 3: Create src/hooks/useHiveOperations.ts

```typescript
// TEAM-353: Hive operations hook using shared narration client
import { useState, useCallback } from 'react'
import { HiveClient } from '@rbee/rbee-hive-sdk'
import { createStreamHandler, SERVICES } from '@rbee/narration-client'

export function useHiveOperations(baseUrl: string) {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const executeOperation = useCallback(async (operation: any) => {
    setIsLoading(true)
    setError(null)

    try {
      const client = new HiveClient(baseUrl)
      
      // TEAM-353: Use shared narration client (no duplicate code!)
      const handleNarration = createStreamHandler(SERVICES.hive)
      
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

### Step 4: Create src/index.ts

```typescript
export * from './hooks/useHiveOperations'
```

### Step 5: Build React Package

```bash
pnpm install
pnpm build
```

**Verify:** `dist/` folder created

---

## Phase 4: Create Hive Vite App

### Step 1: Create Vite Project

```bash
cd bin/25_rbee_hive/ui/app
```

### Step 2: Create package.json

```json
{
  "name": "@rbee/rbee-hive-ui",
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite --port 7836",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "@rbee/rbee-hive-react": "workspace:*",
    "@rbee/shared-config": "workspace:*",
    "@rbee/dev-utils": "workspace:*",
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@vitejs/plugin-react": "^4.0.0",
    "typescript": "^5.0.0",
    "vite": "^5.0.0"
  }
}
```

### Step 3: Create vite.config.ts

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 7836,  // TEAM-353: Hive dev port
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})
```

### Step 4: Create src/App.tsx

```typescript
// TEAM-353: Hive UI entry point
import { useEffect } from 'react'
import { logStartupMode, isDevelopment, getCurrentPort } from '@rbee/dev-utils'
import { useHiveOperations } from '@rbee/rbee-hive-react'

export function App() {
  useEffect(() => {
    // TEAM-353: Use shared dev-utils (no duplicate code!)
    logStartupMode('HIVE UI', isDevelopment(), getCurrentPort())
  }, [])

  const { executeOperation, isLoading, error } = useHiveOperations(
    'http://localhost:7835'  // Hive backend
  )

  return (
    <div>
      <h1>Hive Management Interface</h1>
      {/* Your Hive UI here */}
    </div>
  )
}
```

### Step 5: Create src/main.tsx

```typescript
import React from 'react'
import ReactDOM from 'react-dom/client'
import { App } from './App'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
```

### Step 6: Create index.html

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Hive Management</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

### Step 7: Build and Test Vite App

```bash
pnpm install
pnpm dev
```

**Browser:** Open http://localhost:7836

**Expected:** "🔧 [HIVE UI] Running in DEVELOPMENT mode"

---

## Phase 5: Configure build.rs

### Step 1: Create build.rs

```bash
cd bin/25_rbee_hive
```

Create `build.rs`:

```rust
// TEAM-353: Hive UI build script
// Pattern copied from Queen (TEAM-350)

use std::process::Command;

fn main() {
    // TEAM-353: Check if Vite dev server is running on port 7836
    let vite_dev_running = std::net::TcpStream::connect("127.0.0.1:7836").is_ok();

    if vite_dev_running {
        println!("cargo:warning=⚡ Vite dev server detected on port 7836 - SKIPPING ALL UI builds");
        println!("cargo:warning=   (Dev server provides fresh packages via hot reload)");
        println!("cargo:warning=   SDK, React, and App builds skipped");
        return;
    }

    println!("cargo:warning=🔨 Building rbee-hive UI packages and app...");

    // Step 1: Build the WASM SDK package
    println!("cargo:warning=  📦 Building @rbee/rbee-hive-sdk (WASM)...");
    let sdk_dir = std::path::PathBuf::from("ui/packages/rbee-hive-sdk");
    let sdk_status = Command::new("pnpm")
        .args(&["build"])
        .current_dir(&sdk_dir)
        .status()
        .expect("Failed to build Hive SDK");

    if !sdk_status.success() {
        panic!("Hive SDK build failed");
    }

    // Step 2: Build the React hooks package
    println!("cargo:warning=  📦 Building @rbee/rbee-hive-react...");
    let react_dir = std::path::PathBuf::from("ui/packages/rbee-hive-react");
    let react_status = Command::new("pnpm")
        .args(&["build"])
        .current_dir(&react_dir)
        .status()
        .expect("Failed to build Hive React package");

    if !react_status.success() {
        panic!("Hive React build failed");
    }

    // Step 3: Build the Vite app
    println!("cargo:warning=  🎨 Building @rbee/rbee-hive-ui app...");
    let app_dir = std::path::PathBuf::from("ui/app");
    let app_status = Command::new("pnpm")
        .args(&["build"])
        .current_dir(&app_dir)
        .status()
        .expect("Failed to build Hive app");

    if !app_status.success() {
        panic!("Hive app build failed");
    }

    println!("cargo:warning=✅ rbee-hive UI (SDK + React + App) built successfully");
}
```

### Step 2: Test Build Script

```bash
# With Vite running (should skip)
cd ui/app && pnpm dev &
cargo build --bin rbee-hive

# Without Vite (should build all)
# Stop Vite first
cargo clean
cargo build --bin rbee-hive
```

**Expected output:**
```
⚡ Vite dev server detected - SKIPPING
```
OR
```
🔨 Building rbee-hive UI packages and app...
  📦 Building SDK...
  📦 Building React...
  🎨 Building App...
✅ Built successfully
```

---

## Phase 6: Update Keeper UI

### Step 1: Create HivePage.tsx

```bash
cd bin/00_rbee_keeper/ui/src/pages
```

Create `HivePage.tsx`:

```typescript
// TEAM-353: Hive management page
import { getIframeUrl } from '@rbee/shared-config'

export function HivePage() {
  const isDev = import.meta.env.DEV
  const hiveUrl = getIframeUrl('hive', isDev)

  return (
    <div className="flex flex-col h-full">
      <div className="p-4 border-b">
        <h1 className="text-2xl font-bold">Hive Management</h1>
      </div>
      <div className="flex-1">
        <iframe
          src={hiveUrl}
          className="w-full h-full border-0"
          title="Hive Management Interface"
          sandbox="allow-same-origin allow-scripts allow-forms allow-popups allow-modals"
          allow="cross-origin-isolated"
        />
      </div>
    </div>
  )
}
```

### Step 2: Add Route to Keeper

Edit `bin/00_rbee_keeper/ui/src/App.tsx`:

```typescript
import { HivePage } from './pages/HivePage'

// Add route
<Route path="/hive" element={<HivePage />} />
```

### Step 3: Add Navigation Link

Add link to sidebar/navigation:

```typescript
<Link to="/hive">Hive Management</Link>
```

---

## Phase 7: Update Message Listener

### Step 1: Verify getAllowedOrigins() Includes Hive

**File:** `bin/00_rbee_keeper/ui/src/utils/narrationListener.ts`

Verify this code exists (from TEAM-352):

```typescript
import { getAllowedOrigins } from '@rbee/shared-config'

const allowedOrigins = getAllowedOrigins()
// Automatically includes Hive ports (7835, 7836)
```

**No changes needed!** Shared config already includes Hive.

### Step 2: Add Hive Message Handler

Add handler for Hive narration messages:

```typescript
if (event.data?.type === "NARRATION_EVENT") {
  const message = event.data as NarrationMessage
  
  // Handle both Queen and Hive narration
  if (message.source === 'queen-rbee' || message.source === 'rbee-hive') {
    const keeperEvent = mapToKeeperFormat(message.payload)
    useNarrationStore.getState().addEntry(keeperEvent)
  }
}
```

---

## Phase 8: Testing

### Test 1: Development Mode

**Terminal 1:** Start Hive Vite dev server
```bash
cd bin/25_rbee_hive/ui/app
pnpm dev  # Port 7836
```

**Terminal 2:** Start Hive backend
```bash
cargo run --bin rbee-hive  # Port 7835
```

**Terminal 3:** Start Keeper
```bash
cd bin/00_rbee_keeper/ui
pnpm dev  # Port 5173
```

**Browser:** http://localhost:5173

**Checklist:**
- [ ] Navigate to Hive page
- [ ] iframe loads from http://localhost:7836
- [ ] Console: "🔧 [HIVE UI] Running in DEVELOPMENT mode"
- [ ] Hot reload works (edit Hive code, see changes)
- [ ] No console errors

### Test 2: Narration Flow

Execute a Hive operation (when backend is ready):

**Checklist:**
- [ ] Narration events sent from Hive backend
- [ ] SSE stream receives JSON events
- [ ] Hive UI forwards to parent via postMessage
- [ ] Keeper receives narration
- [ ] Narration appears in keeper panel
- [ ] Function names extracted
- [ ] No parse errors

### Test 3: Production Mode

```bash
cargo build --release --bin rbee-hive
```

**Checklist:**
- [ ] iframe loads from http://localhost:7835 (embedded)
- [ ] Console: "🚀 [HIVE UI] Running in PRODUCTION mode"
- [ ] Narration works
- [ ] No console errors

---

## Phase 9: Verification

### Code Duplication Check

**Count lines that would be duplicate without shared packages:**

**If duplicated from Queen:**
- narrationBridge: ~100 LOC
- Port configuration: ~10 LOC
- Environment detection: ~15 LOC
- **Total duplicate:** ~125 LOC

**Actual code (using shared packages):**
- Import statements: ~5 LOC
- **Total:** ~5 LOC

**Savings:** ~120 LOC not written! (96% reduction)

### Shared Package Usage

**Verify using all packages:**
- [ ] `@rbee/shared-config` - ✅ Used for ports
- [ ] `@rbee/narration-client` - ✅ Used for narration
- [ ] `@rbee/dev-utils` - ✅ Used for logging

---

## Troubleshooting

### Issue: Vite port conflict

**Fix:**
```bash
# Check if port 7836 is in use
lsof -i :7836

# Kill process if needed
kill -9 <PID>
```

### Issue: WASM SDK not found

**Fix:**
```bash
cd bin/25_rbee_hive/ui/packages/rbee-hive-sdk
pnpm build
```

### Issue: Narration not appearing

**Debug:**
1. Check Hive backend sends JSON: `curl http://localhost:7835/v1/jobs/{id}/stream`
2. Check Hive UI console for narration events
3. Check Keeper console for received messages
4. Verify allowed origins includes 7835 and 7836

---

## Documentation

Create summary document:

```bash
cat > bin/.plan/TEAM_353_HIVE_IMPLEMENTATION_SUMMARY.md << 'EOF'
# TEAM-353 Hive Implementation Summary

## Created

### Packages
1. @rbee/rbee-hive-sdk - WASM SDK for Hive operations
2. @rbee/rbee-hive-react - React hooks for Hive
3. @rbee/rbee-hive-ui - Vite app for Hive management

### Files
1. bin/25_rbee_hive/ui/packages/rbee-hive-sdk/* - SDK package
2. bin/25_rbee_hive/ui/packages/rbee-hive-react/* - React package
3. bin/25_rbee_hive/ui/app/* - Vite app
4. bin/25_rbee_hive/build.rs - UI build script
5. bin/00_rbee_keeper/ui/src/pages/HivePage.tsx - Keeper page

## Shared Package Usage

✅ ALL shared packages used (zero duplication):
- @rbee/shared-config - Port configuration
- @rbee/narration-client - Narration handling
- @rbee/dev-utils - Environment detection

## Code Reduction

Avoided ~120 LOC of duplicate code by using shared packages.

## Ports

- Dev: 7836 (Vite dev server)
- Prod: 7835 (embedded in backend)
- Backend: 7835 (HTTP server)

## Testing Results

### Development Mode
- [x] Loads from Vite (7836)
- [x] Hot reload works
- [x] Narration flows correctly
- [x] No console errors

### Production Mode
- [x] Loads embedded (7835)
- [x] Narration works
- [x] No console errors

## Pattern Validation

✅ Same pattern as Queen
✅ No code duplication
✅ Both modes working
✅ Ready for Worker UI replication

## Next: TEAM-354

Implement Worker UI using exact same pattern.
EOF
```

---

## Acceptance Criteria

**All must pass:**

- [ ] Hive SDK builds successfully
- [ ] Hive React package builds successfully
- [ ] Hive Vite app builds successfully
- [ ] build.rs detects Vite and skips builds
- [ ] Keeper HivePage loads iframe correctly
- [ ] Development mode: iframe from port 7836
- [ ] Production mode: iframe from port 7835
- [ ] Narration flows: backend → UI → keeper
- [ ] Function names extracted correctly
- [ ] Zero duplicate code (all using shared packages)
- [ ] Hot reload works in dev mode
- [ ] Both modes tested and working

---

## Handoff to TEAM-354

**What you've built:**
- Complete Hive UI with zero duplication
- Validated shared packages save ~120 LOC
- Proven pattern works for second service

**Next team (TEAM-354) will:**
- Implement Worker UI
- Reuse exact same pattern
- Complete the UI suite

**Pattern to replicate:**
1. Create 3 packages (SDK, React, App)
2. Use all shared packages
3. Configure ports in build.rs and vite.config.ts
4. Add page to Keeper
5. Test dev and prod modes

---

## Success Criteria

✅ Hive UI complete and working  
✅ Zero code duplication  
✅ Pattern validated for Worker  
✅ Both modes tested  
✅ Documentation complete

---

**TEAM-353: Build Hive UI with zero duplication!** 🏗️
