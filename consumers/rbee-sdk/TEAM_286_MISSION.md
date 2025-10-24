# TEAM-286: Modernize rbee-sdk

**Date:** Oct 24, 2025  
**Status:** 🚧 **READY TO START**  
**Previous Team:** TEAM-285 (Heartbeat system, generic registry)

---

## Mission

**Modernize the rbee-sdk to provide a full-featured TypeScript/JavaScript SDK for building web UIs (React, Vue, etc.) that communicate with queen-rbee.**

The SDK should mirror the functionality of `rbee-keeper` but be usable from browsers and Node.js applications.

---

## Context: What is rbee-sdk?

**rbee-sdk** is a **client library** that allows applications (web UIs, Node.js apps, Rust apps) to interact with the rbee system.

**Current State:** Ancient, non-functioning stubs from early design phase

**Goal:** Production-ready SDK with:
- Full TypeScript/JavaScript support
- SSE streaming for real-time updates
- Type-safe API based on current contracts
- React hooks (optional)
- Comprehensive examples

---

## Why This Matters

### Current Problem

**Web UI developers** need to:
1. Manually construct HTTP requests to queen-rbee
2. Parse SSE streams manually
3. Handle reconnection logic
4. Manage TypeScript types manually
5. Implement heartbeat monitoring from scratch

**This is error-prone and duplicates work!**

### With rbee-sdk

```typescript
import { RbeeClient } from '@rbee/sdk';

const client = new RbeeClient('http://localhost:8500');

// Submit inference job
const result = await client.infer({
  model: 'llama-3-8b',
  prompt: 'Hello, world!',
  stream: true
});

// Stream tokens
for await (const token of result.stream()) {
  console.log(token);
}

// Monitor heartbeats
const monitor = client.heartbeats.stream();
monitor.on('update', (data) => {
  console.log('Workers online:', data.workers_online);
});
```

**Simple, type-safe, production-ready!**

---

## Required Reading (CRITICAL - Read Before Planning!)

### 1. Current Architecture Documents

**MUST READ (in order):**

1. **`/bin/CONTRACT_DEPENDENCY_ANALYSIS.md`**
   - Understand the contract hierarchy
   - Learn what types are available
   - See how contracts relate to each other

2. **`/bin/97_contracts/operations-contract/src/lib.rs`**
   - ALL available operations
   - Request/response types
   - This defines the API surface

3. **`/bin/97_contracts/worker-contract/src/`**
   - WorkerInfo, WorkerStatus types
   - WorkerHeartbeat structure

4. **`/bin/97_contracts/hive-contract/src/`**
   - HiveInfo types
   - HiveHeartbeat structure

5. **`/bin/97_contracts/shared-contract/src/`**
   - Shared types (HealthStatus, OperationalStatus)
   - HeartbeatTimestamp

### 2. Current API Endpoints

**MUST READ:**

1. **`/bin/10_queen_rbee/src/main.rs`** (lines 144-158)
   - See ALL registered routes
   - Understand endpoint structure
   - Note: `/v1/` prefix for all API endpoints

2. **`/bin/10_queen_rbee/src/http/heartbeat_stream.rs`**
   - SSE heartbeat streaming implementation
   - HeartbeatSnapshot structure
   - Update frequency (5 seconds)

3. **`/bin/ADDING_NEW_OPERATIONS.md`**
   - Special endpoints section
   - SSE stream documentation
   - JavaScript examples

### 3. Reference Implementation (rbee-keeper)

**MUST STUDY:**

1. **`/bin/00_rbee_keeper/src/job_client.rs`**
   - How to submit jobs
   - How to stream SSE results
   - Error handling patterns

2. **`/bin/00_rbee_keeper/src/handlers/`**
   - How to construct operations
   - Parameter handling
   - CLI → Operation mapping

3. **`/bin/99_shared_crates/job-client/src/lib.rs`**
   - Generic job submission pattern
   - SSE streaming implementation
   - This is the Rust reference - port to TypeScript!

### 4. Heartbeat System (TEAM-285 Work)

**MUST READ:**

1. **`/bin/TEAM_285_HEARTBEAT_MONITOR_READY.md`**
   - Complete heartbeat system documentation
   - SSE format and examples
   - JavaScript client examples

2. **`/bin/10_queen_rbee/examples/heartbeat_monitor.html`**
   - Working example of SSE consumption
   - Error handling
   - Reconnection logic

### 5. Architecture Understanding

**RECOMMENDED:**

1. **`/bin/SCHEDULER_VS_REGISTRY_CLARIFICATION.md`**
   - Understand scheduler vs registry
   - Data layer vs business logic
   - Important for SDK design decisions

2. **`/bin/TEAM_285_MIGRATION_COMPLETE.md`**
   - Recent architecture changes
   - Generic registry pattern
   - Code organization principles

---

## Current API Surface (What SDK Must Support)

### HTTP Endpoints (from queen-rbee/src/main.rs)

```
GET  /health                      - Health check
GET  /v1/build-info               - Build information
POST /v1/worker-heartbeat         - Worker heartbeat (internal)
POST /v1/hive-heartbeat           - Hive heartbeat (internal)
GET  /v1/heartbeats/stream        - SSE: Live heartbeat monitor
POST /v1/jobs                     - Submit job (any operation)
GET  /v1/jobs/{job_id}/stream     - SSE: Stream job results
POST /v1/shutdown                 - Shutdown queen (admin)
```

### Operations (from operations-contract)

**System:**
- `Status` - Get live status of all hives and workers

**Hive:**
- `HiveList` - List all hives
- `HiveGet` - Get hive details
- `HiveStatus` - Check hive health
- `HiveRefreshCapabilities` - Refresh device capabilities

**Worker Process:**
- `WorkerSpawn` - Spawn new worker
- `WorkerProcessList` - List worker processes
- `WorkerProcessGet` - Get worker process details
- `WorkerProcessDelete` - Kill worker process

**Active Workers:**
- `ActiveWorkerList` - List active workers (from registry)
- `ActiveWorkerGet` - Get active worker details
- `ActiveWorkerRetire` - Retire active worker

**Models:**
- `ModelDownload` - Download model
- `ModelList` - List models
- `ModelGet` - Get model details
- `ModelDelete` - Delete model

**Inference:**
- `Infer` - Run inference (streaming or non-streaming)

---

## SDK Architecture (Recommended)

### Core Structure

```
rbee-sdk/
├── src/
│   ├── lib.rs                 - Rust core (optional, for WASM)
│   └── ...
├── ts/
│   ├── client.ts              - Main RbeeClient class
│   ├── operations.ts          - Operation builders
│   ├── types.ts               - TypeScript types (from contracts)
│   ├── sse.ts                 - SSE streaming utilities
│   ├── heartbeat.ts           - Heartbeat monitor
│   ├── errors.ts              - Error types
│   └── index.ts               - Public API exports
├── examples/
│   ├── basic.ts               - Basic usage
│   ├── streaming.ts           - Streaming inference
│   ├── heartbeat.ts           - Heartbeat monitoring
│   └── react-hooks.tsx        - React integration
├── tests/
│   └── ...
└── package.json
```

### Recommended Layers

**Layer 1: HTTP Client**
- Fetch API wrapper
- Request/response handling
- Error mapping

**Layer 2: SSE Streaming**
- EventSource wrapper
- Reconnection logic
- Event parsing

**Layer 3: Operations**
- Type-safe operation builders
- Parameter validation
- Request construction

**Layer 4: High-Level API**
- `RbeeClient` class
- Convenience methods
- React hooks (optional)

---

## Implementation Phases

### Phase 1: Foundation (Week 1)

**Goal:** Basic HTTP client and types

**Tasks:**
1. Set up TypeScript project structure
2. Generate types from contracts (manual or codegen)
3. Implement basic HTTP client
4. Implement error handling
5. Write unit tests

**Deliverables:**
- `RbeeClient` class with constructor
- Type definitions for all operations
- Error types
- Basic tests

### Phase 2: Job Submission (Week 1-2)

**Goal:** Submit jobs and get results

**Tasks:**
1. Implement `POST /v1/jobs` wrapper
2. Implement `GET /v1/jobs/{job_id}/stream` SSE client
3. Add operation builders (Infer, Status, etc.)
4. Handle streaming vs non-streaming
5. Integration tests

**Deliverables:**
- `client.submitJob(operation)` method
- `client.infer()` convenience method
- SSE streaming support
- Examples

### Phase 3: Heartbeat Monitoring (Week 2)

**Goal:** Real-time heartbeat monitoring

**Tasks:**
1. Implement `GET /v1/heartbeats/stream` SSE client
2. Add heartbeat event types
3. Create `HeartbeatMonitor` class
4. Handle reconnection
5. Examples

**Deliverables:**
- `client.heartbeats.stream()` method
- HeartbeatSnapshot types
- Reconnection logic
- React hook example

### Phase 4: All Operations (Week 2-3)

**Goal:** Support all operations

**Tasks:**
1. Implement all operation builders
2. Add convenience methods for common operations
3. Comprehensive examples
4. Documentation

**Deliverables:**
- Full operation coverage
- JSDoc documentation
- Example for each operation
- README with usage guide

### Phase 5: React Integration (Week 3)

**Goal:** React hooks for easy integration

**Tasks:**
1. Create `useRbeeClient()` hook
2. Create `useHeartbeat()` hook
3. Create `useInference()` hook
4. Example React app

**Deliverables:**
- React hooks package
- Example React dashboard
- Documentation

---

## Key Design Decisions

### 1. TypeScript-First

**Decision:** Write in TypeScript, compile to JavaScript

**Why:**
- Type safety for SDK consumers
- Better IDE support
- Easier to maintain

**How:**
- Use strict TypeScript
- Export both .ts and .d.ts files
- Provide source maps

### 2. No Rust/WASM (Initially)

**Decision:** Pure TypeScript/JavaScript, no WASM

**Why:**
- Simpler to develop and maintain
- Faster iteration
- WASM adds complexity without clear benefit
- Can add WASM later if needed

**How:**
- Remove WASM dependencies from Cargo.toml
- Focus on `ts/` directory
- Use standard npm tooling

### 3. Fetch API (Not Axios)

**Decision:** Use native Fetch API

**Why:**
- Built into browsers and Node.js 18+
- No external dependencies
- Smaller bundle size
- Standard API

**How:**
- Polyfill for older Node.js if needed
- Use AbortController for cancellation

### 4. EventSource for SSE

**Decision:** Use native EventSource API

**Why:**
- Built into browsers
- Automatic reconnection
- Standard API

**How:**
- Polyfill for Node.js (eventsource package)
- Wrapper class for convenience

### 5. Mirror rbee-keeper Structure

**Decision:** SDK should mirror rbee-keeper's patterns

**Why:**
- Proven patterns
- Consistency across clients
- Easier to maintain

**How:**
- Study rbee-keeper handlers
- Port patterns to TypeScript
- Keep same operation structure

---

## Type Generation Strategy

### Option 1: Manual Types (Recommended for v1)

**Pros:**
- Full control
- Can add JSDoc comments
- Can optimize for TypeScript

**Cons:**
- Manual sync with contracts
- More work

**How:**
1. Read contract files
2. Write TypeScript interfaces
3. Add JSDoc documentation
4. Keep in sync manually

### Option 2: Codegen from Rust

**Pros:**
- Automatic sync
- Less manual work

**Cons:**
- Complex tooling
- Generated code may not be idiomatic
- Harder to customize

**Recommendation:** Start with Option 1, consider Option 2 later

---

## Example API Design

### Basic Usage

```typescript
import { RbeeClient } from '@rbee/sdk';

// Create client
const client = new RbeeClient({
  baseUrl: 'http://localhost:8500',
  timeout: 30000, // 30 seconds
});

// Check health
const health = await client.health();
console.log('Queen is healthy:', health.status === 'ok');

// Get status
const status = await client.status();
console.log('Workers online:', status.workers_online);
console.log('Hives online:', status.hives_online);
```

### Inference (Streaming)

```typescript
// Submit inference job
const inference = await client.infer({
  model: 'llama-3-8b',
  prompt: 'Write a haiku about TypeScript',
  maxTokens: 100,
  stream: true,
});

// Stream tokens
for await (const event of inference.stream()) {
  if (event.type === 'token') {
    process.stdout.write(event.text);
  } else if (event.type === 'done') {
    console.log('\nGeneration complete!');
  }
}
```

### Heartbeat Monitoring

```typescript
// Start monitoring
const monitor = client.heartbeats.stream();

monitor.on('update', (snapshot) => {
  console.log('Workers:', snapshot.workers_online);
  console.log('Hives:', snapshot.hives_online);
  console.log('Worker IDs:', snapshot.worker_ids);
});

monitor.on('error', (error) => {
  console.error('Monitor error:', error);
});

monitor.on('reconnect', () => {
  console.log('Reconnected to heartbeat stream');
});

// Stop monitoring
monitor.close();
```

### React Integration

```typescript
import { useRbeeClient, useHeartbeat } from '@rbee/sdk/react';

function Dashboard() {
  const client = useRbeeClient('http://localhost:8500');
  const heartbeat = useHeartbeat(client);

  if (heartbeat.loading) return <div>Connecting...</div>;
  if (heartbeat.error) return <div>Error: {heartbeat.error}</div>;

  return (
    <div>
      <h1>rbee Dashboard</h1>
      <div>Workers Online: {heartbeat.data.workers_online}</div>
      <div>Hives Online: {heartbeat.data.hives_online}</div>
      <ul>
        {heartbeat.data.worker_ids.map(id => (
          <li key={id}>{id}</li>
        ))}
      </ul>
    </div>
  );
}
```

---

## Testing Strategy

### Unit Tests

**Test:**
- Type construction
- URL building
- Error handling
- Event parsing

**Tools:**
- Jest or Vitest
- TypeScript

### Integration Tests

**Test:**
- Actual HTTP requests to queen-rbee
- SSE streaming
- Reconnection logic

**Setup:**
- Start queen-rbee in test mode
- Use real endpoints
- Clean up after tests

### Example Tests

**Test:**
- All examples work
- Documentation is accurate

**How:**
- Run examples as part of CI
- Verify output

---

## Documentation Requirements

### README.md

**Must include:**
- Installation instructions
- Quick start example
- API overview
- Link to full docs

### API Documentation

**Must include:**
- JSDoc for all public APIs
- Type definitions
- Examples for each method
- Error handling guide

### Examples

**Must include:**
- Basic usage
- Streaming inference
- Heartbeat monitoring
- Error handling
- React integration

---

## Success Criteria

### Minimum Viable SDK (v0.1.0)

- ✅ TypeScript types for all operations
- ✅ HTTP client with error handling
- ✅ Job submission (POST /v1/jobs)
- ✅ SSE streaming (GET /v1/jobs/{job_id}/stream)
- ✅ Heartbeat monitoring (GET /v1/heartbeats/stream)
- ✅ Basic examples
- ✅ README with usage guide
- ✅ Published to npm

### Full-Featured SDK (v1.0.0)

- ✅ All operations supported
- ✅ React hooks
- ✅ Comprehensive examples
- ✅ Full API documentation
- ✅ Integration tests
- ✅ Example React dashboard

---

## Common Pitfalls to Avoid

### 1. Don't Reinvent rbee-keeper

**Problem:** Trying to design a completely different API

**Solution:** Mirror rbee-keeper's patterns, just in TypeScript

### 2. Don't Over-Engineer

**Problem:** Adding features that aren't needed yet

**Solution:** Start simple, add features as needed

### 3. Don't Ignore Contracts

**Problem:** Making up your own types

**Solution:** Use types from contracts, stay in sync

### 4. Don't Skip Error Handling

**Problem:** Assuming everything works

**Solution:** Handle all error cases, provide good error messages

### 5. Don't Forget Reconnection

**Problem:** SSE streams break and never recover

**Solution:** Implement automatic reconnection with backoff

---

## Questions to Answer Before Starting

### Architecture

1. **What package manager?** (npm, yarn, pnpm)
2. **What bundler?** (esbuild, rollup, webpack)
3. **What test framework?** (Jest, Vitest)
4. **Monorepo or single package?** (Recommend single for v1)

### Publishing

1. **Package name?** (@rbee/sdk, rbee-sdk, @llama-orch/sdk)
2. **npm registry?** (public npm, private registry)
3. **Versioning strategy?** (semver)

### React Hooks

1. **Separate package?** (@rbee/sdk-react)
2. **Peer dependency on React?** (Yes, recommended)

---

## Handoff Checklist

Before starting implementation:

- [ ] Read ALL required reading materials
- [ ] Understand current API endpoints
- [ ] Study rbee-keeper implementation
- [ ] Review contract types
- [ ] Test heartbeat monitor example
- [ ] Set up TypeScript project
- [ ] Choose tooling (bundler, tests, etc.)
- [ ] Create initial types
- [ ] Write first example
- [ ] Get feedback before proceeding

---

## Resources

### Code References

- **rbee-keeper:** `/bin/00_rbee_keeper/`
- **Contracts:** `/bin/97_contracts/`
- **job-client:** `/bin/99_shared_crates/job-client/`
- **Heartbeat example:** `/bin/10_queen_rbee/examples/heartbeat_monitor.html`

### Documentation

- **Architecture:** `/bin/CONTRACT_DEPENDENCY_ANALYSIS.md`
- **Operations:** `/bin/ADDING_NEW_OPERATIONS.md`
- **Heartbeat:** `/bin/TEAM_285_HEARTBEAT_MONITOR_READY.md`
- **Scheduler:** `/bin/SCHEDULER_VS_REGISTRY_CLARIFICATION.md`

### API Endpoints

- **Queen routes:** `/bin/10_queen_rbee/src/main.rs`
- **Heartbeat stream:** `/bin/10_queen_rbee/src/http/heartbeat_stream.rs`
- **Job router:** `/bin/10_queen_rbee/src/job_router.rs`

---

## Final Notes

**This is a critical piece of infrastructure!**

The SDK will be used by:
- Web UI developers
- Node.js applications
- Third-party integrations
- Internal tools

**Take time to get it right:**
- Read all documentation
- Study existing patterns
- Start simple
- Test thoroughly
- Document well

**Good luck, TEAM-286!** 🚀

---

**Prepared by:** TEAM-285  
**Date:** Oct 24, 2025  
**Next Team:** TEAM-286  
**Estimated Effort:** 2-3 weeks for MVP, 4-6 weeks for full v1.0
