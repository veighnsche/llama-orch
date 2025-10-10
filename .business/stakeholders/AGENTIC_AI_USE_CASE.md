# Agentic AI Use Case: Building AI Agents with rbee

**Date:** 2025-10-10  
**Project:** rbee (pronounced "are-bee")

---

## The Vision: Independence from Big AI Providers

**rbee (pronounced "are-bee") enables developers to build their own AI infrastructure and escape dependency on external AI providers.**

### The Core Problem

**You're building complex codebases with AI assistance (Claude, GPT-4, etc.).**

**But you're scared because:**
- **What if the AI changes?** Model updates break your workflow
- **What if they shut down?** Your codebase becomes unmaintainable
- **What if pricing changes?** $20/month becomes $200/month
- **What if terms change?** Commercial use restricted

**You've created a dependency you can't control.**

**Heavy, complicated codebases built with AI assistance are a ticking time bomb if you depend on external providers.**

### The Solution: rbee

**Build your own AI infrastructure using ALL your home network hardware:**

1. **rbee orchestrator** - OpenAI-compatible inference infrastructure
2. **llama-orch-utils** - TypeScript utilities for building AI agents

**The result:**
- ✅ No external dependencies
- ✅ Models never change without your permission
- ✅ Always available (your hardware, your uptime)
- ✅ Zero ongoing costs (electricity only)
- ✅ Complete control over your AI tooling

---

## Use Case 1: Build Your Own AI Coder

### The Problem

**Current state:**
- You use Claude/GPT-4 for code generation
- You pay $20-100/month per developer
- You depend on their availability and pricing
- You have no control over model changes

**The fear:**
- What if they change the model and it breaks your workflow?
- What if they shut down or change pricing?
- Your complex codebase becomes unmaintainable without AI

### The Solution: Build Your Own AI Coder

**Use rbee + llama-orch-utils to build AI coders that run on YOUR hardware:**

```bash
# 1. Start rbee infrastructure on your home network
rbee-keeper daemon start
rbee-keeper hive start --pool default
rbee-keeper worker start --gpu 0 --backend cuda  # Computer 1
rbee-keeper worker start --gpu 1 --backend cuda  # Computer 2
rbee-keeper worker start --gpu 0 --backend metal # Mac

# 2. Build your AI coder with llama-orch-utils
import { invoke, FileReader, FileWriter } from '@llama-orch/utils';

// Your AI coder that NEVER depends on external APIs
const code = await invoke({
  prompt: 'Generate TypeScript API from schema',
  model: 'llama-3.1-70b',  // Running on YOUR hardware
  maxTokens: 4000
});

await FileWriter.write('src/api.ts', code.text);

# 3. Optional: Use with Zed IDE
export OPENAI_API_BASE=http://localhost:8080/v1
export OPENAI_API_KEY=your-rbee-token
# Now Zed's AI agents run on YOUR infrastructure
```

**Benefits:**
- ✅ No external dependencies
- ✅ Models never change without your permission
- ✅ Always available (your hardware, your uptime)
- ✅ Zero ongoing costs (electricity only)
- ✅ Complete control over your AI tooling
- ✅ Full privacy (code never leaves your network)
- ✅ Use ALL your GPUs (across multiple computers)
- ✅ No rate limits
- ✅ **Cascading shutdown guarantee** - Ctrl+C cleanly shuts down all components (no orphaned processes)

**Reliability Guarantee:**
```
Developer presses Ctrl+C
  ↓
rbee-keeper (receives SIGTERM)
  ↓ sends SIGTERM to queen-rbee
queen-rbee (receives SIGTERM)
  ↓ SSH: kill rbee-hive on ALL nodes
rbee-hive (receives shutdown signal)
  ↓ HTTP POST /v1/admin/shutdown to ALL workers
llm-worker-rbee (unloads model, exits cleanly)
  ↓
System is completely clean, no orphaned processes
```

---

## Secondary Use Case: Building Custom AI Agents

### llama-orch-utils: TypeScript Utilities for Agentic AI

**What it is:**
- TypeScript/WASM library for building LLM pipelines
- Composable utilities for agentic workflows
- Built on rbee's task-based API

**Key features:**

### 1. File Operations
```typescript
import { FileReader, FileWriter } from '@llama-orch/utils/fs';

// Read file
const content = await FileReader.read('input.txt');

// Write file
await FileWriter.write('output.txt', content);

// Stream large files
const stream = FileReader.stream('large-file.txt');
for await (const chunk of stream) {
  console.log(chunk);
}
```

### 2. LLM Invocation
```typescript
import { invoke } from '@llama-orch/utils/llm';

const response = await invoke({
  prompt: 'Hello, world!',
  model: 'llama-3.1-8b-instruct',
  maxTokens: 100,
  seed: 42,
});

console.log(response.text);
```

### 3. Model Definitions
```typescript
import { defineModel } from '@llama-orch/utils/model';

const model = defineModel({
  name: 'llama-3.1-8b-instruct',
  maxTokens: 8192,
  temperature: 0.7,
});

const response = await model.invoke('Hello, world!');
```

### 4. Response Extraction
```typescript
import { extractJson, extractCode } from '@llama-orch/utils/orch';

// Extract JSON from response
const data = extractJson(response.text);

// Extract code blocks
const code = extractCode(response.text, 'typescript');
```

### 5. Prompt Management
```typescript
import { Message, Thread } from '@llama-orch/utils/prompt';

// Build thread
const thread = Thread.create()
  .addSystem('You are a helpful assistant')
  .addUser('What is 2+2?')
  .addAssistant('4')
  .addUser('What is 3+3?');

const response = await invoke({
  messages: thread.toMessages(),
  model: 'llama-3.1-8b-instruct',
});
```

---

## Use Cases: What You Can Build

### 1. Code Generation Agents

**Example: "Generate a full CRUD API from a database schema"**

```typescript
// 1. Read schema file
const schema = await FileReader.read('schema.sql');

// 2. Invoke LLM to generate code
const code = await model.invoke(`Generate TypeScript CRUD API for:\n${schema}`);

// 3. Extract and write code
const apiCode = extractCode(code, 'typescript');
await FileWriter.write('src/api.ts', apiCode);
```

### 2. Documentation Generators

**Example: "Generate docs from code"**

```typescript
// 1. Read all source files
const files = await FileReader.readDir('src/');

// 2. For each file, generate docs
for (const file of files) {
  const content = await FileReader.read(file.path);
  const docs = await model.invoke(`Generate markdown docs for:\n${content}`);
  await FileWriter.write(`docs/${file.name}.md`, docs.text);
}
```

### 3. Test Generators

**Example: "Generate tests from specs"**

```typescript
// 1. Read spec
const spec = await FileReader.read('spec.md');

// 2. Generate tests
const tests = await model.invoke(`Generate Jest tests for:\n${spec}`);

// 3. Write tests
const testCode = extractCode(tests, 'typescript');
await FileWriter.write('tests/spec.test.ts', testCode);
```

### 4. Code Review Agents

**Example: "Review PR and suggest improvements"**

```typescript
// 1. Read diff
const diff = await FileReader.read('changes.diff');

// 2. Invoke LLM for review
const review = await model.invoke(`Review this code:\n${diff}\n\nProvide:\n- Issues\n- Suggestions\n- Security concerns`);

// 3. Post review
console.log(review.text);
```

### 5. Refactoring Agents

**Example: "Refactor legacy code to modern patterns"**

```typescript
// 1. Read legacy code
const legacy = await FileReader.read('legacy.js');

// 2. Refactor with LLM
const refactored = await model.invoke(`Refactor to modern TypeScript:\n${legacy}`);

// 3. Write refactored code
const modernCode = extractCode(refactored, 'typescript');
await FileWriter.write('modern.ts', modernCode);
```

---

## The Task-Based API

**rbee's OpenAPI spec provides:**

### 1. Task Submission
```typescript
POST /v2/tasks
{
  "task_id": "uuid",
  "workload": "completion",
  "model_ref": "llama-3.1-8b-instruct",
  "prompt": "Hello, world!",
  "max_tokens": 100
}
```

**Response:**
```json
{
  "task_id": "uuid",
  "queue_position": 3,
  "predicted_start_ms": 420,
  "streams": {
    "sse": "http://localhost:8080/v2/tasks/uuid/events"
  }
}
```

### 2. SSE Streaming
```
GET /v2/tasks/{id}/events

event: started
data: {"queue_position":3,"predicted_start_ms":420}

event: token
data: {"t":"Hello","i":0}

event: token
data: {"t":" world","i":1}

event: metrics
data: {"tokens_budget_remaining":98}

event: end
data: {"tokens_out":2,"decode_ms":50}
```

### 3. Preparation Tracking
```json
{
  "preparation": {
    "steps": [
      {
        "id": "engine:llamacpp",
        "kind": "engine_provision",
        "description": "Building llama.cpp",
        "estimated_ms": 30000,
        "status": "running"
      },
      {
        "id": "model:hf:meta-llama/Llama-3.1-8B",
        "kind": "model_fetch",
        "description": "Downloading model",
        "bytes_total": 8000000000,
        "status": "pending"
      }
    ]
  }
}
```

**Benefits:**
- Real-time progress updates
- Human-readable narration
- Cost/time budget tracking
- Preparation step visibility

---

## Competitive Advantage

### vs. OpenAI/Anthropic APIs

| Feature | rbee + llama-orch-utils | OpenAI/Anthropic |
|---------|-------------------------|------------------|
| **Cost** | $0 (your hardware) | $20-100/month per dev |
| **Privacy** | Complete (local) | Limited (cloud) |
| **Customization** | Full control | Limited |
| **Rate limits** | None | Yes |
| **Agentic tools** | llama-orch-utils included | Build yourself |
| **OpenAPI spec** | Yes (task-based) | Yes (token-based) |

### vs. Local Inference (Ollama, llama.cpp)

| Feature | rbee + llama-orch-utils | Ollama/llama.cpp |
|---------|-------------------------|------------------|
| **Multi-GPU** | Yes (orchestrator) | Limited |
| **Task queuing** | Yes (built-in) | No |
| **SSE streaming** | Yes (with progress) | Basic |
| **Agentic tools** | llama-orch-utils | Build yourself |
| **OpenAPI spec** | Yes (detailed) | Basic |
| **Preparation tracking** | Yes (human-readable) | No |

---

## Revenue Model

### Target Customers

1. **Self-hosters** (Home/Lab mode - free, open source)
2. **Small teams** (5-10 devs) - €99/month
3. **Growing companies** (10-50 devs) - €299/month
4. **Enterprises** (50+ devs) - Custom pricing

### Value Proposition

**For self-hosters:**
- Zero cost
- Complete control
- Privacy guaranteed
- Use all your hardware

**For teams:**
- Cheaper than cloud APIs
- No rate limits
- EU-compliant (GDPR)
- Agentic tools included

**For enterprises:**
- Dedicated instances
- Custom SLAs
- White-label option
- Enterprise support

---

## Roadmap

### Month 1 (Now)
- ✅ OpenAI-compatible API
- ✅ Zed IDE integration works
- ✅ llama-orch-utils alpha
- 🔄 First paying customer (€200 MRR)

### Month 3
- 🔄 5 customers (€1,500 MRR)
- 🔄 llama-orch-utils beta
- 🔄 Documentation site
- 🔄 Example agentic workflows

### Month 6
- 🔄 20 customers (€6,000 MRR)
- 🔄 llama-orch-utils v1.0
- 🔄 Web UI for management
- 🔄 Marketplace (platform mode)

### Month 12
- 🔄 35 customers (€10,000 MRR)
- 🔄 Year 1 revenue: €70,000
- 🔄 Multi-modal (LLMs, SD, TTS, embeddings)
- 🔄 Agentic AI development platform

---

## Why This Matters

### The Shift to Agentic AI

**Before:**
- LLMs as completion APIs
- Developers write glue code
- No standard tooling

**After (with rbee):**
- LLMs as task executors
- llama-orch-utils provides standard tooling
- OpenAI-compatible + agentic extensions
- Run on YOUR hardware

### The Homelab Renaissance

**Trends:**
- More powerful consumer GPUs (RTX 4090, etc.)
- Cheaper homelab hardware
- Privacy concerns about cloud
- AI coding tools need inference

**rbee enables:**
- Use your homelab for AI coding
- Zero cloud costs
- Complete privacy
- Professional-grade orchestration

---

## Getting Started

### 1. Install rbee
```bash
# From source
git clone https://github.com/veighnsche/llama-orch
cd llama-orch
cargo build --release

# Start orchestrator
rbee-keeper daemon start
rbee-keeper hive start --pool default
rbee-keeper worker start --gpu 0 --backend cuda
```

### 2. Configure Zed IDE
```bash
export OPENAI_API_BASE=http://localhost:8080/v1
export OPENAI_API_KEY=your-rbee-token
```

### 3. Install llama-orch-utils
```bash
npm install @llama-orch/utils
```

### 4. Build your first agent
```typescript
import { invoke } from '@llama-orch/utils/llm';

const response = await invoke({
  prompt: 'Generate a hello world function',
  model: 'llama-3.1-8b-instruct',
});

console.log(response.text);
```

---

**rbee (pronounced "are-bee"): Agentic AI development on YOUR hardware** 🐝🤖

---

## Quick Reference

**Pronunciation:** rbee (pronounced "are-bee")  
**Target Audience:** Developers who build with AI but fear provider dependency  
**The Fear:** Complex codebases become unmaintainable if provider changes/shuts down  
**The Solution:** Build your own AI infrastructure using home network hardware  
**Key Components:** rbee orchestrator + llama-orch-utils (TypeScript library)  
**Key Advantage:** 11 shared crates already built (saves 5 days)  
**30-Day Plan:** Detailed execution plan to first customer (€200 MRR)  
**Year 1 Goal:** 35 customers, €10K MRR, €70K revenue

---

*For more information:*
- **Website:** https://rbee.dev
- **GitHub:** https://github.com/veighnsche/llama-orch
- **Docs:** https://rbee.dev/docs
