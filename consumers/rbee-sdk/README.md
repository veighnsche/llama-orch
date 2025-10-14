# rbee-sdk

**Rust SDK for rbee with optional WASM/npm support**

`consumers/rbee-sdk` — Type-safe client library for Rust and JavaScript/TypeScript.

---

## What This SDK Does

rbee-sdk provides **client library** for rbee:

- **Rust API** — Native Rust client
- **WASM bindings** — Optional WebAssembly for npm
- **Type-safe** — Generated from OpenAPI specs
- **Async** — Built on tokio/reqwest
- **Streaming** — SSE support for token streaming

**Used by**: Rust applications, Node.js apps, browsers

---

## Rust Usage

### Add Dependency

```toml
[dependencies]
rbee-sdk = "0.0.0"
```

### Create Client

```rust
use rbee_sdk::Client;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new("http://localhost:8080");
    
    // Use client
    Ok(())
}
```

### Enqueue Job

```rust
use rbee_sdk::{Client, EnqueueRequest};

let request = EnqueueRequest {
    prompt: "Hello, world!".to_string(),
    model: "llama-3.1-8b-instruct".to_string(),
    max_tokens: 100,
    seed: Some(42),
    ..Default::default()
};

let response = client.enqueue(request).await?;
println!("Job ID: {}", response.job_id);
```

### Stream Tokens

```rust
let mut stream = client.enqueue_stream(request).await?;

while let Some(event) = stream.next().await {
    match event? {
        TokenEvent::Token { text, .. } => print!("{}", text),
        TokenEvent::End { .. } => println!("\nDone!"),
        _ => {}
    }
}
```

---

## JavaScript/TypeScript Usage (WASM)

### Install

```bash
npm install @llama-orch/sdk
```

### Create Client

```typescript
import { Client } from '@llama-orch/sdk';

const client = new Client('http://localhost:8080');
```

### Enqueue Job

```typescript
const response = await client.enqueue({
  prompt: 'Hello, world!',
  model: 'llama-3.1-8b-instruct',
  maxTokens: 100,
  seed: 42,
});

console.log('Job ID:', response.jobId);
```

### Stream Tokens

```typescript
const stream = await client.enqueueStream({
  prompt: 'Hello, world!',
  model: 'llama-3.1-8b-instruct',
  maxTokens: 100,
});

for await (const event of stream) {
  if (event.type === 'token') {
    process.stdout.write(event.text);
  }
}
```

---

## Building WASM

### Build for npm

```bash
# Build WASM package
cd consumers/llama-orch-sdk
wasm-pack build --target web

# Publish to npm
cd pkg
npm publish
```

---

## Testing

### Rust Tests

```bash
# Run all tests
cargo test -p llama-orch-sdk -- --nocapture
```

### WASM Tests

```bash
# Run WASM tests
wasm-pack test --node
```

---

## Dependencies

### Internal

- `contracts/api-types` — Shared types

### External

- `reqwest` — HTTP client (Rust)
- `tokio` — Async runtime (Rust)
- `wasm-bindgen` — WASM bindings
- `serde` — Serialization

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha
- **Maintainers**: @llama-orch-maintainers
