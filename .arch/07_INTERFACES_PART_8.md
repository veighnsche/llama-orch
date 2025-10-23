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

**Management dashboard** for rbee infrastructure - catalog, deployments, monitoring.

**NOT for chat/inference** - for that, you connect directly to workers.

### The rbee Pattern: Direct-to-Worker

**How rbee is different:**

```
Traditional (centralized):
  Browser → Dashboard → Queen → Worker
             (all inference through dashboard)

rbee (direct-to-worker):
  Browser → Dashboard → Queen (management only)
  Browser → Worker (direct chat/inference)
          (connect directly to worker's frontend)
```

**Why direct-to-worker?**
- ✅ **Lower latency** - No proxy overhead
- ✅ **Specialized UIs** - Each worker type has optimized frontend
- ✅ **ComfyUI pattern** - Same approach as ComfyUI (workers serve their own UI)
- ✅ **Scalability** - Queen not bottleneck for inference traffic

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  rbee-web-ui (Management Dashboard)                     │
│  http://localhost:3002                                  │
│  ├─ Model Catalog (browse, filter, compatibility)      │
│  ├─ Deployments (spawn workers, manage replicas)       │
│  ├─ Monitoring (GPU usage, metrics)                    │
│  └─ [Open Chat] button → redirects to worker URL       │
└─────────────────────────────────────────────────────────┘
         ↓ Management operations only
┌─────────────────────────────────────────────────────────┐
│  queen-rbee (http://localhost:8500)                     │
│  ├─ Worker lifecycle (spawn, stop)                     │
│  ├─ Model management (download, delete)                │
│  └─ Health/status                                       │
└─────────────────────────────────────────────────────────┘

         Inference happens directly:
         
┌─────────────────────────────────────────────────────────┐
│  Browser connects DIRECTLY to worker                    │
│  http://localhost:9001 (worker-1: llama-3-8b)          │
│  ├─ Chat interface (specialized for LLM)               │
│  ├─ Parameter controls (temp, top_p, max_tokens)       │
│  └─ Streaming output (token-by-token)                  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Browser connects DIRECTLY to ComfyUI worker           │
│  http://localhost:9002 (worker-2: stable-diffusion)    │
│  ├─ ComfyUI workflow editor (native ComfyUI UI)       │
│  ├─ Node graph                                         │
│  └─ Image gallery                                      │
└─────────────────────────────────────────────────────────┘
```

### Key Features

**1. Hive Catalog (Model Repository)**

**Purpose:** Browse and spawn workers with models

**Features:**
- Model cards with metadata (size, format, architecture)
- **Hive compatibility checks** (VRAM requirements vs available cells)
- Filter by: size, format (GGUF, safetensors), architecture
- Search by name/description
- **Spawn Worker** button → deploys to hive

**UI Example:**
```
┌───────────────────────────────────────────────────────┐
│ 🐝 Qwen3-0.5B                    [Spawn Worker ▼]   │
├───────────────────────────────────────────────────────┤
│ Model: Qwen/Qwen2.5-0.5B-Instruct                    │
│ Format: GGUF (Q4_K_M)   Size: 352 MB                 │
│ Context: 32K tokens     Layers: 24                   │
│                                                       │
│ 🟢 Hive Ready                                        │
│    VRAM needed: ~680 MB per worker                   │
│                                                       │
│ Available Cells (GPUs):                              │
│   • Cell-0 (cuda:0, RTX 4090) - 22.4GB free ✅      │
│   • Cell-1 (cuda:1, RTX 3090) - 18.2GB free ✅      │
│                                                       │
│ [Text Generation] [Fast Inference] [Apache 2.0]     │
└───────────────────────────────────────────────────────┘
```

**Spawn Worker Modal:**
```
┌───────────────────────────────────────────────────────┐
│ Spawn Worker: Qwen3-0.5B                    [×]      │
├───────────────────────────────────────────────────────┤
│ Worker ID                                             │
│ ┌───────────────────────────────┐                   │
│ │ qwen3-worker-1 ✏️             │                   │
│ └───────────────────────────────┘                   │
│                                                       │
│ Worker Type                                          │
│ ┌───────────────────────────────┐                   │
│ │ llm-worker-rbee (Candle) ▼   │                   │
│ └───────────────────────────────┘                   │
│                                                       │
│ Assign to Cell: cuda:0 (Cell-0) ▼                   │
│ Quantization: Q4_K_M ▼                               │
│ Swarm Size: 1 worker (click for multi-worker)       │
│                                                       │
│ 🔧 Advanced (temperature, context, etc.)            │
│                                                       │
│           [Cancel]  [Spawn]                          │
└───────────────────────────────────────────────────────┘
```

**2. Worker Pool (Active Workers)**

**Purpose:** Manage active workers in the hive

**Features:**
- List all workers across hives
- Status indicators (🟢 active, 🟡 spawning, 🔴 failed, ⚫ idle)
- **Swarm** - Multiple workers of same model (load distribution)
- **[Connect →]** button - Opens worker's direct interface
- Stop/restart workers
- View worker logs
- Cell assignment (which GPU)

**UI Example:**
```
┌──────────────────────────────────────────────────────────────────────────────┐
│ 🐝 Worker Pool                              [Spawn Worker ▼]    [↻ Refresh]  │
├──────────────────────────────────────────────────────────────────────────────┤
│ [Search workers...] [Filter by hive ▼] [Filter by status ▼]  [↻]            │
├──────┬─────────────────────┬────────────────────┬──────────┬──────────────────┤
│  ✓   │ Worker              │ Model              │ Cell     │ Spawned      │ ⚙️│
├──────┼─────────────────────┼────────────────────┼──────────┼──────────────────┤
│  ×   │ qwen3-worker-1      │ Qwen3-0.5B        │ Cell-0   │ 2025-07-16   │ 🗑️│
│      │ 🟢 Active           │ (GGUF Q4_K_M)     │ cuda:0   │ 16:27:15     │   │
│      │ [Connect →]         │ Swarm: 1/1        │          │              │   │
├──────┼─────────────────────┼────────────────────┼──────────┼──────────────────┤
│  ×   │ llama3-worker-2     │ Llama-3.1-8B      │ Cell-1   │ 2025-07-16   │ 🗑️│
│      │ ⚫ Idle (5m)        │ (GGUF Q5_K_M)     │ cuda:1   │ 15:42:08     │   │
│      │ [Connect →]         │ Swarm: 1/1        │          │              │   │
├──────┼─────────────────────┼────────────────────┼──────────┼──────────────────┤
│  ×   │ sd-worker-3         │ SDXL-1.0          │ Cell-0   │ 2025-07-16   │ 🗑️│
│      │ 🟡 Spawning...      │ (safetensors)     │ cuda:0   │ 16:55:33     │   │
└──────┴─────────────────────┴────────────────────┴──────────┴──────────────────┘
```

**Terminology:**
- **Worker Pool** - All active workers (instead of "deployments")
- **Swarm** - Multiple workers of same model (instead of "replicas")
- **Cell** - GPU/compute unit (instead of "backend" or "GPU")
- **Connect** - Open worker interface (instead of "Open Chat")
- **Spawn** - Create worker (instead of "Deploy")

**Click [Connect →] → Opens worker's direct interface:**
- For LLM workers: `http://localhost:9001` (chat interface)
- For ComfyUI: `http://localhost:9002` (workflow editor)
- For Stable Diffusion: `http://localhost:9003` (generation interface)

**3. Worker Interface (Direct Connection)**

**Purpose:** Connect directly to worker's specialized interface

**URL:** `http://localhost:9001` (worker's own HTTP server)

**Features:**
- Direct worker connection (zero proxy overhead)
- Streaming output (token-by-token)
- Worker-specific controls
- Real-time metrics
- Session history

**Worker Interface Elements:**
- **Worker ID** - Identifies this worker instance
- **Cell Assignment** - Which GPU this worker uses
- **Model Info** - Model name, size, quantization
- **Control Panel** - Temperature, tokens, sampling params
- **Conversation** - Message history with streaming
- **Metrics** - Token count, generation speed, VRAM usage

**UI Example:**
```
┌───────────────────────────────────────────────────────────────────────────────┐
│ 🐝 Worker: qwen3-worker-1                        [Metrics]  [Settings]  [⚙️]  │
│ Cell-0 (cuda:0) • Model: Qwen3-0.5B (Q4_K_M) • VRAM: 680 MB / 24 GB          │
├─────────────────────────────────────┬─────────────────────────────────────────┤
│                                     │  🎛️ Controls                            │
│ 👤 You                             │  ┌───────────────────────────────────┐  │
│ tell me a joke                     │  │ Worker: qwen3-worker-1            │  │
│                                     │  │ Cell: Cell-0 (cuda:0)             │  │
│ 🐝 Worker Response                 │  └───────────────────────────────────┘  │
│ [streaming tokens...]              │                                         │
│ Okay, here's a light one:          │  Temperature: ●────────────  0.9        │
│ Why did the bee go to therapy?     │  Max Tokens:  ●────────────  4096       │
│ Because it had too many...          │  Top P:       ●────────────  0.95       │
│                                     │  Top K:       ●────────────  40         │
│                                     │  Repeat Penalty: ●─────────  1.1        │
│ 📊 Tokens: 45 generated            │  Seed: [865]  [🎲 Random]               │
│ ⚡ Speed: 32.5 tok/s               │                                         │
│                            [📋 Copy]│  [Stop Generation]                      │
├─────────────────────────────────────┤                                         │
│ │ Message this worker...          │                                         │
│ ╰──────────────────────────────[🐝]│                                         │
└─────────────────────────────────────┴─────────────────────────────────────────┘
```

**Key Features:**
- **Direct connection** - Worker serves its own UI (no proxy)
- **Cell visibility** - Shows which GPU worker uses
- **Streaming** - Token-by-token generation
- **Real-time metrics** - Token count, speed, VRAM
- **Worker identity** - Clear worker ID and cell assignment
- **Specialized** - UI optimized for worker type (LLM, image gen, etc.)

### Tech Stack

- **Next.js 15** - React framework with App Router
- **React 19** - UI library
- **TypeScript** - Type safety
- **@rbee/sdk** - WASM-compiled TypeScript client
- **@rbee/ui** - Shared component library (shadcn/ui)
- **Tailwind CSS** - Styling
- **SSE** - Server-Sent Events for streaming

### Project Structure (Management Dashboard)

```
rbee-web-ui/
├── src/
│   ├── app/
│   │   ├── layout.tsx          # Root layout
│   │   ├── page.tsx            # Hive overview
│   │   ├── catalog/            # Hive catalog (models)
│   │   │   └── page.tsx        # Browse models, spawn workers
│   │   ├── workers/            # Worker pool (active workers)
│   │   │   └── page.tsx        # List workers, [Connect] buttons
│   │   ├── cells/              # Cell monitoring (GPU status)
│   │   │   └── page.tsx        # Cell usage, VRAM, temperature
│   │   └── hives/              # Hive management
│   │       └── page.tsx        # Install, start, stop hives
│   ├── components/
│   │   ├── model-card.tsx      # Model card in catalog
│   │   ├── spawn-modal.tsx     # Spawn worker dialog
│   │   ├── worker-list.tsx     # Active workers list
│   │   ├── cell-status.tsx     # Cell (GPU) status card
│   │   ├── swarm-indicator.tsx # Swarm size display
│   │   └── cell-graphs.tsx     # Cell usage graphs
│   ├── hooks/
│   │   ├── use-rbee-client.ts  # SDK client hook
│   │   ├── use-catalog.ts      # Model catalog data
│   │   ├── use-workers.ts      # Active workers data
│   │   ├── use-cells.ts        # Cell (GPU) metrics
│   │   └── use-hives.ts        # Hive status
│   └── lib/
│       └── rbee.ts             # SDK configuration
└── package.json
```

### Worker Frontends (Separate from Dashboard)

**Each worker serves its own specialized frontend:**

```
bin/30_llm_worker_rbee/
├── src/
│   ├── main.rs                 # Worker HTTP server
│   ├── inference.rs            # LLM inference
│   └── frontend/               # Worker's own UI
│       ├── chat.html           # Chat interface (served at /)
│       ├── chat.js             # Client-side logic
│       └── chat.css            # Styling
└── Cargo.toml
```

**Worker HTTP Server (serves UI):**
```rust
// bin/30_llm_worker_rbee/src/main.rs
use axum::{Router, routing::{get, post}};

#[tokio::main]
async fn main() {
    let app = Router::new()
        // Serve chat UI at root
        .route("/", get(serve_chat_ui))
        
        // API endpoints
        .route("/v1/infer", post(handle_infer))
        .route("/v1/status", get(handle_status));
    
    // Worker listens on its own port
    let addr = format!("0.0.0.0:{}", port);
    axum::Server::bind(&addr.parse()?)
        .serve(app.into_make_service())
        .await?;
}

async fn serve_chat_ui() -> Html<&'static str> {
    Html(include_str!("frontend/chat.html"))
}
```

**Worker Chat UI (chat.html):**
```html
<!DOCTYPE html>
<html>
<head>
  <title>🐝 rbee Worker - {model_name}</title>
  <style>/* GPUStack-inspired styling */</style>
</head>
<body>
  <div class="chat-container">
    <div class="messages" id="messages"></div>
    <div class="input-area">
      <textarea id="prompt"></textarea>
      <button onclick="sendMessage()">Send</button>
    </div>
  </div>
  
  <div class="params-sidebar">
    <h3>Parameters</h3>
    <label>Temperature: <input type="range" id="temp" /></label>
    <label>Max Tokens: <input type="range" id="max_tokens" /></label>
    <!-- More parameters -->
  </div>
  
  <script>
    // Direct connection to THIS worker
    async function sendMessage() {
      const prompt = document.getElementById('prompt').value;
      const response = await fetch('/v1/infer', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({prompt, temperature: 0.9, max_tokens: 4096})
      });
      
      // Stream tokens
      const reader = response.body.getReader();
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        displayToken(new TextDecoder().decode(value));
      }
    }
  </script>
</body>
</html>
```

### ComfyUI Worker (Native UI)

**ComfyUI already has its own UI:**

```
bin/35_adapters/rbee-comfyui-adapter/
├── src/
│   ├── main.rs                 # Adapter
│   └── comfyui_subprocess.rs   # Spawn ComfyUI
└── Cargo.toml
```

**ComfyUI serves its native UI at its port:**
```rust
// Adapter just spawns ComfyUI, doesn't need to serve UI
let comfyui = Command::new("python")
    .arg("-m").arg("comfyui")
    .arg("--port").arg("9002")
    .spawn()?;

// ComfyUI serves its own UI at http://localhost:9002
// Dashboard [Open Chat] button links to it directly
```

### Implementation Workflow

**1. User spawns worker from catalog:**
```
Dashboard: Catalog → Select model → [Spawn Worker]
  ↓
Spawn modal: Choose cell (GPU), swarm size, worker type
  ↓
Dashboard sends: POST /v1/jobs (Operation::WorkerSpawn)
  ↓
Queen routes to appropriate Hive
  ↓
Hive spawns worker on assigned cell (port 9001)
  ↓
Worker starts HTTP server with embedded interface
  ↓
Dashboard updates: Worker Pool shows "qwen3-worker-1 🟢 Active"
```

**2. User connects to worker:**
```
Dashboard: Worker Pool → [Connect →] button
  ↓
Opens new tab: http://localhost:9001
  ↓
Browser connects DIRECTLY to worker (zero proxy)
  ↓
Worker serves its specialized interface
  ↓
User interacts directly with worker
  (LLM chat, ComfyUI workflows, image generation, etc.)
```

### Status

**Management Dashboard:**
- **Location:** `frontend/apps/rbee-web-ui/`
- **Version:** 0.1.0 (stub)
- **Runs on:** `http://localhost:3002`
- **Dependencies:** Requires `@rbee/sdk` implementation
- **Effort:** 44-60 hours (after SDK ready)

**Worker Frontends:**
- **LLM Worker:** `bin/30_llm_worker_rbee/src/frontend/`
- **Status:** Not implemented (M1/M2)
- **Effort:** 16-24 hours per worker type
- **ComfyUI:** Uses native ComfyUI UI (no additional work)

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
