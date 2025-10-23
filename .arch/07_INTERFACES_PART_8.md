# rbee Architecture Overview - Part 8: User Interfaces

**Version:** 1.0.0  
**Date:** October 23, 2025  
**Status:** Living Document

---

## Overview

rbee provides **multiple interfaces** for users and applications to interact with the system:

1. **rbee-keeper (CLI)** - Command-line interface for operators
2. **rbee-sdk (Library)** - Programmatic access for Rust and TypeScript
3. **rbee-web-ui (Dashboard)** - Web-based visual interface
4. **OpenAI Adapter** - Compatibility layer for OpenAI applications

**All interfaces communicate with queen-rbee via HTTP.**

---

## 1. rbee-sdk (Programmatic Interface)

### Purpose

Programmatic access to rbee infrastructure for application integration. Single-source Rust codebase that compiles to both native libraries and WASM for TypeScript.

### Architecture

```
Application (Rust/TypeScript) → @rbee/sdk → queen-rbee HTTP API
```

### Key Features

**Single-Source Design:**
- Rust core library
- Compiles to native (Rust apps)
- Compiles to WASM (TypeScript/JavaScript apps)
- TypeScript bindings auto-generated

**Type-Safe API:**
- Same Operation types as rbee-keeper
- Compile-time safety
- IDE auto-completion

**Cross-Platform:**
- Native: Linux, macOS, Windows
- WASM: Node.js, browsers

### API Examples

**Rust:**
```rust
use rbee_sdk::RbeeClient;

#[tokio::main]
async fn main() -> Result<()> {
    let client = RbeeClient::new("http://localhost:8500");
    
    // Spawn worker
    let mut stream = client.worker_spawn("llama-3-8b", "GPU-0").await?;
    while let Some(event) = stream.next().await {
        println!("{}", event);
    }
    
    // Run inference
    let mut stream = client.infer("Hello, world!", "llama-3-8b").await?;
    while let Some(token) = stream.next().await {
        print!("{}", token);
    }
    
    Ok(())
}
```

**TypeScript:**
```typescript
import { RbeeClient } from '@rbee/sdk';

const client = new RbeeClient('http://localhost:8500');

// Spawn worker
const stream = await client.workerSpawn('llama-3-8b', 'GPU-0');
for await (const event of stream) {
  console.log(event);
}

// Run inference
const inferStream = await client.infer('Hello, world!', 'llama-3-8b');
for await (const token of inferStream) {
  process.stdout.write(token);
}
```

### Use Cases

1. **Rust Applications** - Batch processing, custom tooling, integration tests
2. **Node.js Services** - Web servers with LLM, API gateways, background workers
3. **Browser Applications** - Interactive UIs, chat interfaces, real-time streaming
4. **Automation** - CI/CD pipelines, testing frameworks, monitoring tools

### Status

- **Location:** `consumers/rbee-sdk/`
- **Version:** 0.0.0 (design phase)
- **Implementation:** Stubs only (methods unimplemented)
- **Effort:** 22-32 hours to complete

---

## 2. rbee-web-ui (Web Dashboard)

### Purpose

Visual dashboard for managing rbee infrastructure through a web browser. Built with React/Next.js, uses TypeScript SDK for API communication.

### Architecture

```
Browser → rbee-web-ui (React) → @rbee/sdk (WASM) → queen-rbee HTTP API
```

### Key Features

**Dashboard Overview:**
- Real-time status cards (queen, hives, workers, models)
- System metrics and graphs
- Quick actions

**Management Views:**
- **Hives:** Install, start, stop, uninstall with live logs
- **Workers:** Spawn, list, stop across all hives
- **Models:** Download from HuggingFace with progress tracking
- **Inference:** Interactive playground with streaming output

**Monitoring:**
- GPU utilization graphs
- Memory usage tracking
- Token throughput metrics
- Error logs viewer

### Tech Stack

- **Next.js 15** - React framework with App Router
- **React 19** - UI library
- **TypeScript** - Type safety
- **@rbee/sdk** - WASM-compiled TypeScript client
- **@rbee/ui** - Shared component library (shadcn/ui)
- **Tailwind CSS** - Styling
- **SSE** - Server-Sent Events for streaming

### Project Structure

```
rbee-web-ui/
├── src/
│   ├── app/
│   │   ├── layout.tsx          # Root layout
│   │   ├── page.tsx            # Dashboard
│   │   ├── hives/              # Hive management
│   │   ├── workers/            # Worker management
│   │   ├── models/             # Model management
│   │   └── inference/          # Inference playground
│   ├── components/
│   │   ├── hive-list.tsx
│   │   ├── worker-list.tsx
│   │   ├── model-list.tsx
│   │   └── inference-form.tsx
│   ├── hooks/
│   │   ├── use-rbee-client.ts  # SDK client hook
│   │   ├── use-hives.ts
│   │   ├── use-workers.ts
│   │   └── use-models.ts
│   └── lib/
│       └── rbee.ts             # SDK configuration
└── package.json
```

### Status

- **Location:** `frontend/apps/rbee-web-ui/`
- **Version:** 0.1.0 (stub)
- **Runs on:** `http://localhost:3002`
- **Dependencies:** Requires `@rbee/sdk` implementation
- **Effort:** 44-60 hours (after SDK ready)

### Future: Embedded UI (RSX + HTMX)

**Alternative Approach:** Each binary serves its own specialized UI.

**Vision:**
- `http://queen.home.arpa` → Queen dashboard (embedded)
- `http://hive1.home.arpa` → Hive-specific UI (embedded)
- `http://hive1.home.arpa/workers/worker-123` → Worker UI (iframe)

**Benefits:**
- No separate frontend build
- Single binary deployment
- Visual feedback (spawn worker → iframe 404 → 200)
- Specialized views per binary

**Status:** Design phase, future exploration (42-70 hours)

---

## 3. OpenAI API Adapter

### Purpose

OpenAI-compatible HTTP API that translates OpenAI requests to rbee Operations. Enables existing OpenAI applications to work with rbee without modification.

### Architecture

```
OpenAI SDK (Python/JS/Rust) → /openai/v1/* → OpenAI Adapter → rbee Operations → queen-rbee
```

### Supported Endpoints

**POST /openai/v1/chat/completions**
- Chat completions (streaming and non-streaming)
- Converts messages array to single prompt
- Maps OpenAI parameters to rbee parameters

**GET /openai/v1/models**
- List available models in OpenAI format

**GET /openai/v1/models/{model}**
- Get model details

### Request Translation

**OpenAI Request:**
```json
{
  "model": "llama-3-8b",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 100,
  "stream": true
}
```

**Translated to rbee:**
```rust
Operation::Infer {
    prompt: "<|system|>\nYou are helpful.\n</|system|>\n\n
