# rbee-sdk

**Rust SDK that compiles to WASM for browser/Node.js**

> ✅ **Status:** Phase 2 in progress - `submit_and_stream()` implemented!  
> **Version:** 0.1.0 (functional, under development)

---

## Mission

Provide a production-ready TypeScript/JavaScript SDK for building web UIs and Node.js applications that interact with rbee (queen-rbee).

---

## Quick Start

### Prerequisites

1. **Install wasm-pack:**
   ```bash
   curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
   ```

2. **Build the WASM:**
   ```bash
   cd consumers/rbee-sdk
   wasm-pack build --target web
   ```

3. **Test it:**
   ```bash
   # Start a local server
   python3 -m http.server 8000
   
   # Open http://localhost:8000/test.html
   ```

### What's Implemented

**✅ Phase 1 Complete:**
- WASM project setup
- Dependencies configured
- Module structure in place

**✅ Phase 2 In Progress:**
- `RbeeClient` class
- `submitAndStream()` method - **WORKING!**
- `submit()` method - **WORKING!**
- Type conversions (JS ↔ Rust)

**📋 Next:**
- Operation builders
- Convenience methods
- All 17 operations

---

## Current API (Working!)

### Installation (when published)

```bash
npm install @rbee/sdk
```

### Basic Usage (Works Now!)

```javascript
import init, { RbeeClient } from '@rbee/sdk';

// Initialize WASM
await init();

const client = new RbeeClient('http://localhost:8500');

// Submit a job and stream results
const jobId = await client.submitAndStream(
  { operation: 'status' },
  (line) => console.log(line)
);

console.log('Job ID:', jobId);

// Or just submit without streaming
const jobId2 = await client.submit({ operation: 'hive_list' });
```

---

## Implementation Status

**✅ Phase 1 - Foundation (COMPLETE):**
- Cargo.toml configured for WASM
- Dependencies on shared crates (job-client, operations-contract)
- Module structure (client, types, utils)
- Compiles successfully

**✅ Phase 2 - Core Bindings (IN PROGRESS):**
- RbeeClient wrapper around JobClient
- submitAndStream() - **WORKING!**
- submit() - **WORKING!**
- Type conversions (JS ↔ Rust)
- Test HTML page

**📋 Phase 3 - All Operations (TODO):**
- Operation builders for all 17 operations
- Convenience methods
- Examples

**📋 Phase 4 - Publishing (TODO):**
- Build optimization
- npm package
- Documentation

---

## Architecture

### How It Works

```
Existing Rust Crates (REUSE!)
├── job-client (HTTP + SSE)
├── operations-contract (all types)
└── rbee-config
         ↓
    rbee-sdk (thin wrapper)
    ├── src/lib.rs
    ├── src/client.rs (wraps JobClient)
    ├── src/types.rs (JS ↔ Rust)
    └── src/utils.rs
         ↓
    wasm-pack build
         ↓
    pkg/
    ├── rbee_sdk.wasm (~150-250KB)
    ├── rbee_sdk.js (glue code)
    └── rbee_sdk.d.ts (TypeScript types!)
```

**Key Insight:** We reuse 90%+ of existing Rust code!

---

## Development

### Build Commands

```bash
# Build for web
wasm-pack build --target web

# Build for Node.js
wasm-pack build --target nodejs

# Build for bundlers
wasm-pack build --target bundler

# Build all targets
./build-wasm.sh

# Check Rust code
cargo check -p rbee-sdk
```

### Testing

```bash
# Start queen-rbee
cargo run --bin queen-rbee

# In another terminal, serve test.html
python3 -m http.server 8000

# Open http://localhost:8000/test.html
```

---

## License

GPL-3.0-or-later

---

## Documentation

- **TEAM_286_PLAN_OVERVIEW.md** - Master plan
- **TEAM_286_PHASE_1_FOUNDATION.md** - WASM setup (COMPLETE)
- **TEAM_286_PHASE_2_IMPLEMENTATION.md** - Core bindings (IN PROGRESS)
- **TEAM_286_PHASE_3_ALL_OPERATIONS.md** - All operations (TODO)
- **TEAM_286_PHASE_4_PUBLISHING.md** - Publishing (TODO)
- **TEAM_286_IMPLEMENTATION_SUMMARY.md** - Overview

## Why Rust + WASM?

- ✅ **Code reuse:** 90%+ from existing shared crates
- ✅ **Type safety:** Auto-generated TypeScript types
- ✅ **Zero drift:** Same code as backend
- ✅ **Fix once:** Bug fixes propagate everywhere
- ✅ **Performance:** Near-native speed

**vs TypeScript:** Would require rewriting everything, manual type sync, permanent duplication.

---

**Status:** `submit_and_stream()` is working! 🚀
