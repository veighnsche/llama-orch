# TEAM-286: Phase 1 - WASM Setup

**Phase:** 1 of 4  
**Duration:** 1 day  
**Status:** 📋 **READY TO START**

---

## Goal

Set up the rbee-sdk crate for WASM compilation, add necessary dependencies, and configure wasm-pack for building.

---

## Prerequisites

- ✅ All required reading completed
- ✅ Understanding of existing shared crates
- ✅ wasm-pack installed

---

## Deliverables

1. **Updated Cargo.toml**
   - WASM dependencies added
   - Existing shared crates referenced
   - Correct crate-type configured

2. **Project Structure**
   - src/lib.rs - Main entry point
   - src/client.rs - RbeeClient wrapper
   - src/types.rs - Type conversions
   - src/utils.rs - JS ↔ Rust helpers

3. **Build Configuration**
   - wasm-pack configuration
   - Build scripts

4. **Basic Compilation**
   - Verify WASM builds
   - Check generated JavaScript

---

## Step-by-Step Implementation

### Step 1: Install wasm-pack (30 min)

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Verify installation
wasm-pack --version
```

---

### Step 2: Update Cargo.toml (1 hour)

**File:** `consumers/rbee-sdk/Cargo.toml`

```toml
# TEAM-286: Rust SDK with WASM bindings

[package]
name = "rbee-sdk"
version = "0.1.0"
edition = "2021"
authors = ["rbee Team"]
license = "GPL-3.0-or-later"
description = "Rust SDK for rbee - compiles to WASM for browser/Node.js"
repository = "https://github.com/your-org/llama-orch"

[lib]
crate-type = ["cdylib", "rlib"]  # cdylib = WASM, rlib = for tests

[dependencies]
# ============================================================================
# EXISTING SHARED CRATES (REUSE!)
# ============================================================================

# TEAM-286: Reuse job-client for all HTTP + SSE logic
job-client = { path = "../../bin/99_shared_crates/job-client" }

# TEAM-286: Reuse operations-contract for all operation types
operations-contract = { path = "../../bin/97_contracts/operations-contract" }

# TEAM-286: Reuse rbee-config for configuration
rbee-config = { path = "../../bin/99_shared_crates/rbee-config" }

# ============================================================================
# WASM DEPENDENCIES
# ============================================================================

# TEAM-286: Core WASM bindings
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"

# TEAM-286: Serde for JS ↔ Rust conversion
serde = { version = "1.0", features = ["derive"] }
serde-wasm-bindgen = "0.6"

# TEAM-286: JavaScript types
js-sys = "0.3"

# TEAM-286: Web APIs
web-sys = { version = "0.3", features = [
    "Window",
    "Headers",
    "Request",
    "RequestInit",
    "RequestMode",
    "Response",
    "console",
] }

# TEAM-286: Async runtime (required for WASM)
tokio = { version = "1", features = ["sync"], default-features = false }

# TEAM-286: Error handling
anyhow = "1.0"
thiserror = "1.0"

[dev-dependencies]
wasm-bindgen-test = "0.3"

[profile.release]
# TEAM-286: Optimize for small WASM size
opt-level = "z"     # Optimize for size
lto = true          # Link-time optimization
codegen-units = 1   # Better optimization
```

**Key Changes:**
- ✅ `crate-type = ["cdylib", "rlib"]` - Enables WASM compilation
- ✅ References to existing shared crates (job-client, operations-contract)
- ✅ WASM-specific dependencies (wasm-bindgen, js-sys, web-sys)
- ✅ Release profile optimized for small bundle size

---

### Step 3: Create Module Structure (1 hour)

**File:** `consumers/rbee-sdk/src/lib.rs`

```rust
// TEAM-286: Main library entry point for WASM SDK

#![warn(missing_docs)]

//! rbee SDK - Rust SDK that compiles to WASM
//!
//! This crate provides JavaScript/TypeScript bindings to the rbee system
//! by wrapping existing Rust crates (job-client, operations-contract, etc.)
//! and compiling to WASM.
//!
//! # Architecture
//!
//! ```text
//! job-client (existing) → rbee-sdk (thin wrapper) → WASM → JavaScript
//! ```
//!
//! # Usage (JavaScript)
//!
//! ```javascript
//! import { RbeeClient } from '@rbee/sdk';
//!
//! const client = new RbeeClient('http://localhost:8500');
//! await client.submitAndStream(operation, (line) => console.log(line));
//! ```

use wasm_bindgen::prelude::*;

// TEAM-286: Modules
mod client;
mod types;
mod utils;

// TEAM-286: Re-export main client
pub use client::RbeeClient;

// TEAM-286: Initialize panic hook for better error messages in browser
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}
```

**File:** `consumers/rbee-sdk/src/client.rs`

```rust
// TEAM-286: RbeeClient - thin wrapper around job-client

use wasm_bindgen::prelude::*;
use job_client::JobClient;

/// Main client for rbee operations
///
/// TEAM-286: This is a thin wrapper around the existing JobClient
/// from the job-client shared crate. We just add WASM bindings.
#[wasm_bindgen]
pub struct RbeeClient {
    /// TEAM-286: Reuse existing job-client!
    inner: JobClient,
}

#[wasm_bindgen]
impl RbeeClient {
    /// Create a new RbeeClient
    ///
    /// # Arguments
    /// * `base_url` - Base URL of queen-rbee (e.g., "http://localhost:8500")
    #[wasm_bindgen(constructor)]
    pub fn new(base_url: String) -> Self {
        // TEAM-286: Just wrap the existing JobClient
        Self {
            inner: JobClient::new(base_url),
        }
    }

    /// Get the base URL
    #[wasm_bindgen(getter)]
    pub fn base_url(&self) -> String {
        self.inner.base_url().to_string()
    }
}

// TEAM-286: Additional methods will be added in Phase 2
```

**File:** `consumers/rbee-sdk/src/types.rs`

```rust
// TEAM-286: Type conversions between Rust and JavaScript

use wasm_bindgen::prelude::*;
use operations_contract::Operation;

/// Convert JavaScript operation to Rust Operation
///
/// TEAM-286: Uses serde-wasm-bindgen for automatic conversion
pub fn js_to_operation(js_value: JsValue) -> Result<Operation, JsValue> {
    serde_wasm_bindgen::from_value(js_value)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse operation: {}", e)))
}

/// Convert Rust error to JavaScript error
pub fn error_to_js(error: anyhow::Error) -> JsValue {
    JsValue::from_str(&error.to_string())
}
```

**File:** `consumers/rbee-sdk/src/utils.rs`

```rust
// TEAM-286: Utility functions for WASM

use wasm_bindgen::prelude::*;

/// Log to browser console
///
/// TEAM-286: Helper for debugging
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);
}

/// Log macro for easier usage
#[macro_export]
macro_rules! console_log {
    ($($t:tt)*) => {
        $crate::utils::log(&format!($($t)*))
    };
}
```

---

### Step 4: Add Build Scripts (30 min)

**File:** `consumers/rbee-sdk/build-wasm.sh`

```bash
#!/bin/bash
# TEAM-286: Build script for WASM

set -e

echo "🔧 Building rbee-sdk for WASM..."

# Build for web (browser)
echo "📦 Building for web..."
wasm-pack build --target web --out-dir pkg/web

# Build for Node.js
echo "📦 Building for Node.js..."
wasm-pack build --target nodejs --out-dir pkg/nodejs

# Build for bundlers (webpack, vite, etc.)
echo "📦 Building for bundlers..."
wasm-pack build --target bundler --out-dir pkg/bundler

echo "✅ Build complete!"
echo ""
echo "Output:"
echo "  - pkg/web/     (for browser via <script type='module'>)"
echo "  - pkg/nodejs/  (for Node.js via require/import)"
echo "  - pkg/bundler/ (for webpack/vite/rollup)"
```

Make executable:
```bash
chmod +x consumers/rbee-sdk/build-wasm.sh
```

**File:** `consumers/rbee-sdk/.gitignore`

```
/target
/pkg
Cargo.lock
*.wasm
*.js
*.d.ts
node_modules/
```

---

### Step 5: Test Basic Compilation (1 hour)

```bash
cd consumers/rbee-sdk

# Build for web
wasm-pack build --target web

# Check generated files
ls -lh pkg/

# Expected output:
# rbee_sdk_bg.wasm      - WASM binary
# rbee_sdk.js           - JavaScript glue
# rbee_sdk.d.ts         - TypeScript types
# package.json          - npm package info
```

**Verify generated TypeScript types:**

```bash
cat pkg/rbee_sdk.d.ts
```

Should show:
```typescript
/* tslint:disable */
/* eslint-disable */
/**
* Main client for rbee operations
*/
export class RbeeClient {
  free(): void;
  constructor(base_url: string);
  readonly base_url: string;
}
```

---

### Step 6: Create Test HTML Page (30 min)

**File:** `consumers/rbee-sdk/test.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>rbee SDK Test</title>
</head>
<body>
    <h1>🐝 rbee SDK Test</h1>
    <div id="status">Loading...</div>
    
    <script type="module">
        // TEAM-286: Test WASM loading
        import init, { RbeeClient } from './pkg/rbee_sdk.js';
        
        async function main() {
            // Initialize WASM
            await init();
            
            // Create client
            const client = new RbeeClient('http://localhost:8500');
            
            // Test
            document.getElementById('status').innerText = 
                `✅ WASM loaded! Base URL: ${client.base_url}`;
                
            console.log('Client created:', client);
        }
        
        main().catch(console.error);
    </script>
</body>
</html>
```

**Test it:**
```bash
# Serve locally
python3 -m http.server 8000

# Open http://localhost:8000/test.html
# Should see: "✅ WASM loaded! Base URL: http://localhost:8500"
```

---

### Step 7: Add Basic Tests (30 min)

**File:** `consumers/rbee-sdk/tests/wasm.rs`

```rust
// TEAM-286: Basic WASM tests

#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;
use rbee_sdk::RbeeClient;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_client_creation() {
    let client = RbeeClient::new("http://localhost:8500".to_string());
    assert_eq!(client.base_url(), "http://localhost:8500");
}
```

**Run tests:**
```bash
wasm-pack test --headless --firefox
```

---

## Verification Checklist

After completing Phase 1:

- [ ] Cargo.toml has all dependencies
- [ ] References to job-client, operations-contract work
- [ ] wasm-pack build succeeds
- [ ] Generated files exist in pkg/
- [ ] TypeScript types are auto-generated
- [ ] Test HTML page loads WASM
- [ ] Basic test passes
- [ ] No compilation errors
- [ ] All TEAM-286 signatures added

---

## Common Issues

### Issue: "crate `job-client` is not found"

**Fix:** Ensure path is correct in Cargo.toml:
```toml
job-client = { path = "../../bin/99_shared_crates/job-client" }
```

### Issue: "cannot find type `JobClient` in crate `job_client`"

**Fix:** Check that job-client is properly exported in its lib.rs

### Issue: WASM file is huge (>1MB)

**Fix:** Build with release profile:
```bash
wasm-pack build --release --target web
```

---

## Next Phase

**Phase 2: Core Bindings** - Wrap job-client methods and expose to JavaScript.

---

**Created by:** TEAM-286  
**Date:** Oct 24, 2025  
**Estimated Duration:** 1 day
