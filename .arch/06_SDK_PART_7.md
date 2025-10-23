# rbee SDK Architecture

**Version:** 1.0.0  
**Date:** October 23, 2025  
**Status:** Design Phase

---

## Overview

The **rbee-sdk** provides programmatic access to rbee infrastructure alongside the CLI. While **rbee-keeper** is the primary user interface (CLI), the SDK enables developers to integrate rbee into their applications.

**Key Principle:** SDK and CLI are **complementary**, not competing. Both are thin HTTP clients to queen-rbee.

---

## Architecture Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interfaces                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  rbee-keeper (CLI)              rbee-sdk (Library)          │
│  ├─ Human-friendly              ├─ Programmatic API          │
│  ├─ Interactive prompts         ├─ Type-safe bindings        │
│  ├─ Auto-start queen            ├─ Async/await               │
│  └─ Pretty output               └─ SSE streaming             │
│                                                               │
│         Both are thin HTTP clients                           │
│                     ↓                                         │
└─────────────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│              queen-rbee (HTTP API)                           │
│              All business logic lives here                   │
└─────────────────────────────────────────────────────────────┘
```

---

## SDK vs CLI Comparison

| Feature | rbee-keeper (CLI) | rbee-sdk (Library) |
|---------|-------------------|-------------------|
| **Purpose** | Human interaction | Programmatic integration |
| **Interface** | Command-line args | Rust/TypeScript API |
| **Output** | Pretty-printed, colored | Structured data (JSON) |
| **Auto-start** | Yes (queen lifecycle) | No (expects queen running) |
| **Prompts** | Interactive (e.g., "use localhost?") | No prompts (explicit config) |
| **Error handling** | User-friendly messages | Result types, error enums |
| **Streaming** | Prints to stdout | Returns async stream |
| **Use case** | Operators, DevOps | Applications, integrations |

**Shared:** Both use the same HTTP API, same Operation types, same SSE protocol.

---

## SDK Design Goals

### 1. Single-Source, Multi-Target

**Rust Core:**
- Native Rust library (`rbee-sdk`)
- Type-safe API
- Async/await with tokio

**WASM/TypeScript:**
- Compile Rust to WASM
- Generate TypeScript bindings
- Publish to npm as `@rbee/sdk`

**Why Single-Source?**
- ✅ Type safety guaranteed (Rust → TS)
- ✅ No manual sync between Rust and TS
- ✅ Bugs fixed in one place
- ✅ Same behavior in both languages

### 2. Mimics rbee-keeper Logic

**SDK should replicate keeper's HTTP client pattern:**

```rust
// rbee-keeper pattern (simplified)
let operation = Operation::WorkerSpawn { model, device };
submit_and_stream_job(queen_url, operation, |line| {
    println!("{}", line);
}).await?;

// rbee-sdk should provide similar API
let client = RbeeClient::new("http://localhost:8500");
let mut stream = client.worker_spawn(model, device).await?;
while let Some(event) = stream.next().await {
    println!("{}", event);
}
```

**Shared Logic:**
1. Serialize Operation to JSON
2. POST to `/v1/jobs`
3. Extract `job_id` from response
4. Connect to `/v1/jobs/{job_id}/stream`
5. Process SSE events

### 3. Type-Safe API

**Use shared types from `rbee-operations`:**

```rust
// SDK exposes same Operation enum
pub use rbee_operations::Operation;

// Client methods are type-safe wrappers
impl RbeeClient {
    pub async fn worker_spawn(
        &self,
        model: String,
        device: String,
    ) -> Result<JobStream> {
        let operation = Operation::WorkerSpawn { model, device };
        self.submit_and_stream(operation).await
    }
}
```

---

## SDK Structure

### Rust SDK (`consumers/rbee-sdk/`)

```
rbee-sdk/
├── src/
│   ├── lib.rs              # Public API exports
│   ├── client.rs           # HTTP client (mimics keeper's job_client)
│   ├── types.rs            # Re-exports from rbee-operations
│   ├── stream.rs           # SSE stream wrapper
│   └── error.rs            # Error types
├── Cargo.toml              # Features: native, wasm
└── README.md
```

**Key Dependencies:**
```toml
[dependencies]
rbee-operations = { path = "../../bin/99_shared_crates/rbee-operations" }
reqwest = { version = "0.11", features = ["json", "stream"] }
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
anyhow = "1"

# WASM support
wasm-bindgen = { version = "0.2", optional = true }
tsify = { version = "0.4", optional = true }

[features]
default = ["native"]
native = []
wasm = ["dep:tsify", "dep:wasm-bindgen"]
```

### TypeScript SDK (`consumers/rbee-sdk/ts/`)

**Generated from WASM:**
```
ts/
├── index.ts                # TypeScript wrapper
├── package.json            # npm package config
└── README.md
```

**Build Process:**
```bash
# Build WASM
cd consumers/rbee-sdk
wasm-pack build --target web --features wasm

# Publish to npm
cd pkg
npm publish --access public
```

---

## API Design

### Rust API

#### Client Creation

```rust
use rbee_sdk::RbeeClient;

// Connect to queen
let client = RbeeClient::new("http://localhost:8500");

// With custom timeout
let client = RbeeClient::builder()
    .base_url("http://localhost:8500")
    .timeout(Duration::from_secs(60))
    .build()?;
```

#### Hive Operations

```rust
// Install hive
client.hive_install("localhost", None).await?;

// Start hive
let mut stream = client.hive_start("localhost").await?;
while let Some(event) = stream.next().await {
    println!("{}", event);
}

// Stop hive
client.hive_stop("localhost").await?;

// List hives
let hives = client.hive_list().await?;
for hive in hives {
    println!("{}: {}", hive.alias, hive.status);
}
```

#### Worker Operations

```rust
// Spawn worker
let mut stream = client.worker_spawn("llama-3-8b", "GPU-0").await?;
while let Some(event) = stream.next().await {
    match event {
        JobEvent::Progress(msg) => println!("{}", msg),
        JobEvent::Complete(worker_id) => println!("Worker ID: {}", worker_id),
        JobEvent::Error(err) => eprintln!("Error: {}", err),
    }
}

// List workers
let workers = client.worker_list().await?;

// Delete worker
client.worker_delete("worker-123").await?;
```

#### Model Operations

```rust
// Download model
let mut stream = client.model_download("meta-llama/Llama-2-7b-chat-hf").await?;
while let Some(event) = stream.next().await {
    match event {
        JobEvent::Progress(msg) => println!("{}", msg),
        JobEvent::Complete(_) => println!("Download complete"),
        JobEvent::Error(err) => eprintln!("Error: {}", err),
    }
}

// List models
let models = client.model_list().await?;

// Delete model
client.model_delete("meta-llama/Llama-2-7b-chat-hf").await?;
```

#### Inference

```rust
// Run inference
let mut stream = client.infer("Hello, world!", "llama-3-8b").await?;
while let Some(token) = stream.next().await {
    print!("{}", token);
}
println!();
```

### TypeScript API

#### Client Creation

```typescript
import { RbeeClient } from '@rbee/sdk';

// Connect to queen
const client = new RbeeClient('http://localhost:8500');

// With custom timeout
const client = new RbeeClient({
  baseUrl: 'http://localhost:8500',
  timeout: 60000,
});
```

#### Hive Operations

```typescript
// Install hive
await client.hiveInstall('localhost');

// Start hive
const stream = await client.hiveStart('localhost');
for await (const event of stream) {
  console.log(event);
}

// Stop hive
await client.hiveStop('localhost');

// List hives
const hives = await client.hiveList();
for (const hive of hives) {
  console.log(`${hive.alias}: ${hive.status}`);
}
```

#### Worker Operations

```typescript
// Spawn worker
const stream = await client.workerSpawn('llama-3-8b', 'GPU-0');
for await (const event of stream) {
  if (event.type === 'progress') {
    console.log(event.message);
  } else if (event.type === 'complete') {
    console.log('Worker ID:', event.workerId);
  }
}

// List workers
const workers = await client.workerList();

// Delete worker
await client.workerDelete('worker-123');
```

#### Inference

```typescript
// Run inference
const stream = await client.infer('Hello, world!', 'llama-3-8b');
for await (const token of stream) {
  process.stdout.write(token);
}
console.log();
```

---

## Implementation Strategy

### Phase 1: Core Client (Rust)

**Goal:** Implement HTTP client that mimics rbee-keeper's job_client

**Tasks:**
1. Create `RbeeClient` struct
2. Implement `submit_and_stream()` method (copy from keeper)
3. Add convenience methods for each operation
4. Add error handling
5. Add tests

**Effort:** 8-12 hours

**Files:**
- `consumers/rbee-sdk/src/client.rs` - HTTP client
- `consumers/rbee-sdk/src/stream.rs` - SSE stream wrapper
- `consumers/rbee-sdk/src/error.rs` - Error types

**Pattern:**
```rust
// Reuse job-client pattern from keeper
pub struct RbeeClient {
    base_url: String,
    client: reqwest::Client,
}

impl RbeeClient {
    pub async fn submit_and_stream(
        &self,
        operation: Operation,
    ) -> Result<JobStream> {
        // 1. POST to /v1/jobs
        let response = self.client
            .post(format!("{}/v1/jobs", self.base_url))
            .json(&operation)
            .send()
            .await?;
        
        // 2. Extract job_id
        let job_id = response.json::<CreateJobResponse>().await?.job_id;
        
        // 3. Connect to SSE stream
        let stream_url = format!("{}/v1/jobs/{}/stream", self.base_url, job_id);
        let stream = self.client.get(stream_url).send().await?;
        
        // 4. Return stream wrapper
        Ok(JobStream::new(stream))
    }
}
```

### Phase 2: Convenience Methods (Rust)

**Goal:** Add type-safe wrappers for each operation

**Tasks:**
1. Add methods for hive operations
2. Add methods for worker operations
3. Add methods for model operations
4. Add methods for inference
5. Add documentation

**Effort:** 4-6 hours

**Pattern:**
```rust
impl RbeeClient {
    /// Spawn a worker on a specific device
    pub async fn worker_spawn(
        &self,
        model: impl Into<String>,
        device: impl Into<String>,
    ) -> Result<JobStream> {
        let operation = Operation::WorkerSpawn {
            model: model.into(),
            device: device.into(),
        };
        self.submit_and_stream(operation).await
    }
    
    // ... more methods
}
```

### Phase 3: WASM Bindings (TypeScript)

**Goal:** Compile Rust to WASM and generate TypeScript bindings

**Tasks:**
1. Add WASM feature flag
2. Add `tsify` annotations to types
3. Create TypeScript wrapper
4. Build with wasm-pack
5. Test in Node.js and browser

**Effort:** 6-8 hours

**Pattern:**
```rust
// Add tsify annotations
#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
#[derive(Serialize, Deserialize)]
pub struct HiveInfo {
    pub alias: String,
    pub status: String,
}
```

### Phase 4: Documentation & Examples

**Goal:** Comprehensive docs and examples

**Tasks:**
1. Write API documentation
2. Create Rust examples
3. Create TypeScript examples
4. Write integration guide
5. Update README

**Effort:** 4-6 hours

---

## Shared Infrastructure

### Reuse from rbee-keeper

**The SDK should reuse the same patterns:**

1. **Operation Enum** (`rbee-operations`)
   - SDK uses same Operation types
   - No duplication

2. **Job Client Pattern** (`job-client` crate)
   - SDK can use `rbee-job-client` crate directly
   - Or copy the pattern (if WASM compatibility needed)

3. **SSE Streaming**
   - Same SSE protocol
   - Same event format
   - Same `[DONE]` marker

**Dependency:**
```toml
[dependencies]
rbee-operations = { path = "../../bin/99_shared_crates/rbee-operations" }
rbee-job-client = { path = "../../bin/99_shared_crates/rbee-job-client" }  # Reuse!
```

---

## Use Cases

### 1. Rust Application Integration

```rust
// Example: Batch processing application
use rbee_sdk::RbeeClient;

#[tokio::main]
async fn main() -> Result<()> {
    let client = RbeeClient::new("http://localhost:8500");
    
    // Ensure hive is running
    client.hive_start("localhost").await?;
    
    // Process batch of prompts
    let prompts = vec!["Hello", "World", "Rust"];
    for prompt in prompts {
        let mut stream = client.infer(prompt, "llama-3-8b").await?;
        while let Some(token) = stream.next().await {
            print!("{}", token);
        }
        println!();
    }
    
    Ok(())
}
```

### 2. Node.js Application

```typescript
// Example: Web server with LLM integration
import express from 'express';
import { RbeeClient } from '@rbee/sdk';

const app = express();
const rbee = new RbeeClient('http://localhost:8500');

app.post('/api/chat', async (req, res) => {
  const { prompt, model } = req.body;
  
  res.setHeader('Content-Type', 'text/event-stream');
  
  const stream = await rbee.infer(prompt, model);
  for await (const token of stream) {
    res.write(`data: ${JSON.stringify({ token })}\n\n`);
  }
  
  res.end();
});

app.listen(3000);
```

### 3. Browser Application

```typescript
// Example: React component
import { RbeeClient } from '@rbee/sdk';
import { useState } from 'react';

function ChatComponent() {
  const [response, setResponse] = useState('');
  const client = new RbeeClient('http://localhost:8500');
  
  async function handleSubmit(prompt: string) {
    setResponse('');
    const stream = await client.infer(prompt, 'llama-3-8b');
    for await (const token of stream) {
      setResponse(prev => prev + token);
    }
  }
  
  return (
    <div>
      <input onSubmit={e => handleSubmit(e.target.value)} />
      <div>{response}</div>
    </div>
  );
}
```

### 4. Testing & Automation

```rust
// Example: Integration tests
#[tokio::test]
async fn test_worker_lifecycle() {
    let client = RbeeClient::new("http://localhost:8500");
    
    // Spawn worker
    let worker_id = client.worker_spawn("llama-3-8b", "CPU-0")
        .await?
        .wait_for_complete()
        .await?;
    
    // Verify worker exists
    let workers = client.worker_list().await?;
    assert!(workers.iter().any(|w| w.id == worker_id));
    
    // Delete worker
    client.worker_delete(&worker_id).await?;
    
    // Verify worker deleted
    let workers = client.worker_list().await?;
    assert!(!workers.iter().any(|w| w.id == worker_id));
}
```

---

## SDK vs CLI: When to Use Which?

### Use rbee-keeper (CLI) when:
- ✅ Human operator managing infrastructure
- ✅ Interactive workflows (prompts, confirmations)
- ✅ One-off commands
- ✅ DevOps scripts (bash/shell)
- ✅ Need auto-start queen

### Use rbee-sdk (Library) when:
- ✅ Programmatic integration
- ✅ Application needs LLM capabilities
- ✅ Batch processing
- ✅ Web services
- ✅ Automated testing
- ✅ Custom tooling

**Both are valid!** They serve different audiences.

---

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_client_creation() {
        let client = RbeeClient::new("http://localhost:8500");
        assert_eq!(client.base_url(), "http://localhost:8500");
    }
    
    #[tokio::test]
    async fn test_operation_serialization() {
        let op = Operation::WorkerSpawn {
            model: "llama-3-8b".to_string(),
            device: "GPU-0".to_string(),
        };
        let json = serde_json::to_string(&op).unwrap();
        assert!(json.contains("WorkerSpawn"));
    }
}
```

### Integration Tests

```rust
#[tokio::test]
#[ignore] // Requires running queen
async fn test_hive_lifecycle() {
    let client = RbeeClient::new("http://localhost:8500");
    
    // Install
    client.hive_install("test-hive", None).await.unwrap();
    
    // Start
    client.hive_start("test-hive").await.unwrap();
    
    // List
    let hives = client.hive_list().await.unwrap();
    assert!(hives.iter().any(|h| h.alias == "test-hive"));
    
    // Stop
    client.hive_stop("test-hive").await.unwrap();
    
    // Uninstall
    client.hive_uninstall("test-hive").await.unwrap();
}
```

### WASM Tests

```bash
# Test in Node.js
wasm-pack test --node

# Test in browser
wasm-pack test --headless --chrome
```

---

## Documentation Requirements

### API Documentation

**Rust:**
```rust
/// Client for interacting with rbee infrastructure
///
/// # Examples
///
/// ```no_run
/// use rbee_sdk::RbeeClient;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let client = RbeeClient::new("http://localhost:8500");
///     let hives = client.hive_list().await?;
///     Ok(())
/// }
/// ```
pub struct RbeeClient { ... }
```

**TypeScript:**
```typescript
/**
 * Client for interacting with rbee infrastructure
 * 
 * @example
 * ```typescript
 * const client = new RbeeClient('http://localhost:8500');
 * const hives = await client.hiveList();
 * ```
 */
export class RbeeClient { ... }
```

### User Guides

1. **Getting Started** - Installation and first steps
2. **API Reference** - Complete API documentation
3. **Examples** - Common use cases
4. **Migration Guide** - From CLI to SDK
5. **Troubleshooting** - Common issues

---

## Versioning & Releases

### Semantic Versioning

- **Major:** Breaking API changes
- **Minor:** New features (backward compatible)
- **Patch:** Bug fixes

### Release Process

1. Update version in `Cargo.toml`
2. Update CHANGELOG.md
3. Tag release: `git tag v0.1.0`
4. Publish to crates.io: `cargo publish`
5. Build WASM: `wasm-pack build`
6. Publish to npm: `npm publish`

### Compatibility

**SDK version should match queen-rbee API version:**

- SDK 0.1.x → queen-rbee 0.1.x
- SDK 0.2.x → queen-rbee 0.2.x

---

## Implementation Checklist

### Phase 1: Rust Core (8-12 hours)
- [ ] Create `RbeeClient` struct
- [ ] Implement `submit_and_stream()` method
- [ ] Add error handling
- [ ] Add unit tests
- [ ] Add documentation

### Phase 2: Convenience Methods (4-6 hours)
- [ ] Add hive operation methods
- [ ] Add worker operation methods
- [ ] Add model operation methods
- [ ] Add inference method
- [ ] Add integration tests

### Phase 3: WASM Bindings (6-8 hours)
- [ ] Add WASM feature flag
- [ ] Add `tsify` annotations
- [ ] Create TypeScript wrapper
- [ ] Build with wasm-pack
- [ ] Test in Node.js and browser

### Phase 4: Documentation (4-6 hours)
- [ ] Write API documentation
- [ ] Create Rust examples
- [ ] Create TypeScript examples
- [ ] Write user guide
- [ ] Update README

**Total Effort:** 22-32 hours

---

## Future Enhancements

### 1. Retry Logic

```rust
impl RbeeClient {
    pub fn with_retry(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }
}
```

### 2. Connection Pooling

```rust
impl RbeeClient {
    pub fn with_pool_size(mut self, size: usize) -> Self {
        self.pool_size = size;
        self
    }
}
```

### 3. Middleware Support

```rust
impl RbeeClient {
    pub fn with_middleware<M>(mut self, middleware: M) -> Self
    where
        M: Middleware + 'static
    {
        self.middleware.push(Box::new(middleware));
        self
    }
}
```

### 4. Batch Operations

```rust
impl RbeeClient {
    pub async fn batch_infer(
        &self,
        prompts: Vec<String>,
        model: &str,
    ) -> Result<Vec<String>> {
        // Parallel inference
    }
}
```

---

## References

**Related Documentation:**
- `.arch/01_COMPONENTS_PART_2.md` - rbee-keeper architecture
- `bin/00_rbee_keeper/src/job_client.rs` - HTTP client pattern
- `bin/99_shared_crates/rbee-operations/src/lib.rs` - Operation types
- `bin/99_shared_crates/rbee-job-client/src/lib.rs` - Shared job client

**External Resources:**
- [wasm-pack documentation](https://rustwasm.github.io/wasm-pack/)
- [tsify documentation](https://docs.rs/tsify/)
- [reqwest documentation](https://docs.rs/reqwest/)

---

**Status:** Design phase - ready for implementation

**Priority:** Medium (nice-to-have for programmatic access)

**Effort:** 22-32 hours total
