# Programmable Scheduler Specification

**Status**: Draft  
**Version**: 0.1.0  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 1. Executive Summary

The orchestrator scheduler is designed as a **policy execution engine** that can run user-defined scheduling logic. This enables users to customize scheduling behavior while maintaining a high-performance, battle-tested default for platform mode.

### Core Concepts

1. **Platform Mode Scheduler** (Immutable, Built-in)
   - Written in Rhai, compiled into orchestratord binary
   - Optimized for performance by platform team
   - Immutable - users cannot modify
   - Always available as fallback/reference
   - Enforces capacity limits and may reject jobs with 429 when thresholds exceeded

2. **User-Defined Scheduler** (Rhai or YAML)
   - Home/Lab modes: Users can write custom Rhai scripts
   - YAML: Declarative configuration (compiled to Rhai internally)
   - Full access to system state via provided API
   - Sandboxed execution environment
   - Can define custom queue policies (unbounded queues, custom eviction, etc.)

3. **Web UI Mode** (Visual Scheduler Builder)
   - Visual policy editor
   - Generates Rhai or YAML output
   - Live preview and testing
   - Can clone platform scheduler as starting point

---

## 2. Scheduler Modes

### 2.1 Platform Mode (Production)

**Characteristics**:
- Uses built-in, immutable Rhai scheduler
- Optimized for multi-tenant marketplace
- Security > Performance
- No user customization allowed
- Maintained and updated by platform team
- Enforces queue capacity limits and may reject jobs with 429 when capacity thresholds are exceeded

**Rationale**: Platform needs control over scheduling to ensure fair resource allocation, security, and SLA compliance across all tenants.

### 2.2 Home/Lab Mode (Customizable)

**Characteristics**:
- User can choose: Platform scheduler OR custom Rhai/YAML
- Performance > Security
- Full scheduling control including queue policies (typically unbounded queues with custom eviction)
- Experimental features allowed
- User responsibility for correctness

**Use cases**:
- Research experiments with novel scheduling algorithms
- Domain-specific optimizations (e.g., batch processing, real-time inference)
- Learning and prototyping

### 2.3 Web UI Mode (Visual)

**Characteristics**:
- Browser-based visual policy builder
- Drag-and-drop scheduling rules
- Generates Rhai or YAML code
- Live simulation/preview
- Can import/export policies

**Target users**: Non-programmers, operators, policy designers

---

## 3. Scheduler Language Options

### 3.1 Rhai (Primary)

**Why Rhai?**
- ✅ Rust-native, zero FFI overhead (built for Rust)
- ✅ Rust-like syntax (familiar to codebase)
- ✅ Type-safe with optional typing
- ✅ Better error messages than Lua
- ✅ 0-indexed arrays (sane defaults)
- ✅ Built-in sandboxing
- ✅ Fast compilation and execution
- ✅ Platform scheduler written in Rhai (reference implementation)

**Example Rhai Scheduler**:
```rust
// User-defined scheduler
fn schedule(ctx) {
    let queue = ctx.queue;
    let pools = ctx.pools;
    
    // Custom logic: prioritize jobs with seed for determinism
    for job in queue {
        if job.seed != () {
            let worker = find_best_worker(pools, job.model_ref);
            if worker != () {
                return #{
                    action: "dispatch",
                    job_id: job.id,
                    worker_id: worker.id
                };
            }
        }
    }
    
    // Fallback to platform scheduler
    return platform::schedule(ctx);
}
```

### 3.2 YAML (Declarative Alternative)

**Why YAML?**
- ✅ Simpler for non-programmers
- ✅ Declarative rules easier to reason about
- ✅ Can be validated against schema
- ✅ Compiled to Rhai internally (no performance penalty)
- ✅ Suitable for simple queue policies and scheduling rules

**Example YAML Scheduler**:
```yaml
scheduler:
  name: "determinism-first"
  version: "1.0"
  
  rules:
    - name: "prioritize-seeded-jobs"
      condition:
        job.seed: { exists: true }
      action:
        dispatch:
          worker_selection: "least-loaded"
          model_match: required
      priority: 100
    
    - name: "batch-jobs-low-priority"
      condition:
        job.priority: "batch"
      action:
        dispatch:
          worker_selection: "most-vram-free"
      priority: 50
    
    - name: "fallback"
      condition: always
      action:
        use: "platform-scheduler"
      priority: 0
```

**YAML → Rhai Compilation**:
- YAML is parsed and compiled to Rhai at scheduler load time
- Validation ensures YAML conforms to schema
- Generated Rhai is cached for performance
- Queue behavior (capacity, rejection, eviction) can be defined in YAML and compiled to Rhai logic

---

## 4. Scheduler API (Context Object)

The scheduler receives a `ctx` object with read-only access to system state:

```rust
// Rhai context object
ctx = #{
  // Queue state
  queue: [
    #{ id: "job-1", priority: "interactive", model_ref: "llama-7b", seed: 42, ... },
    #{ id: "job-2", priority: "batch", model_ref: "llama-13b", seed: (), ... },
    ...
  ],
  
  // Pool states (from heartbeats)
  pools: [
    #{
      id: "pool-1",
      workers: [
        #{ id: "worker-1", status: "ready", model_ref: "llama-7b", vram_free: 8_000_000_000, ... },
        #{ id: "worker-2", status: "busy", model_ref: "llama-13b", vram_free: 0, ... },
      ],
      gpus: [
        #{ id: 0, vram_total: 24_000_000_000, vram_allocated: 16_000_000_000, ... },
        #{ id: 1, vram_total: 24_000_000_000, vram_allocated: 8_000_000_000, ... },
      ]
    },
    ...
  ],
  
  // Tenant quotas (platform mode only)
  tenants: #{
    "tenant-1": #{ id: "tenant-1", quota_vram: 48_000_000_000, quota_used: 32_000_000_000, ... },
    ...
  },
  
  // System metrics
  metrics: #{
    queue_depth: 15,
    total_workers: 8,
    busy_workers: 5,
    ...
  }
};

// Helper functions available as ctx methods
ctx.find_worker(model_ref, pool_id);
ctx.calculate_wait_time(job_id);
ctx.check_quota(tenant_id, vram_needed);
```

**Return value** (scheduler decision):
```rust
return #{
  action: "dispatch",  // or "wait", "evict", "start_worker"
  job_id: "job-1",
  worker_id: "worker-1",
  // Optional fields based on action
  evict_worker_id: "worker-3",  // for eviction
  start_worker: #{ model_ref: "llama-7b", pool_id: "pool-1", gpu_id: 0 }
};
```

---

## 5. Platform Scheduler (Reference Implementation)

The platform scheduler is the **gold standard** implementation, optimized for:
- Fair multi-tenant resource allocation
- SLA compliance (latency, throughput)
- VRAM utilization efficiency
- Determinism preservation where possible

**Location**: `bin/orchestratord-crates/scheduling/platform-scheduler.rhai`

**Features**:
- Priority-based scheduling (Interactive > Batch)
- Least-loaded worker selection
- Quota enforcement
- Eviction policies (LRU for both model hot-load cache and worker VRAM)
- Preemptive worker starts for queued models
- Queue capacity management: may reject jobs with 429 when capacity thresholds are exceeded

**Users can**:
- Use platform scheduler as-is (recommended)
- Copy and modify for custom needs (Home/Lab mode only)
- Study as reference for best practices

---

## 6. Web UI Scheduler Builder

### 6.1 Visual Policy Editor

**Components**:
1. **Rule Builder**: Drag-and-drop conditions and actions
2. **Live Preview**: Simulate scheduling with sample data
3. **Code View**: See generated Lua/YAML in real-time
4. **Template Library**: Pre-built policies (clone platform scheduler, batch-optimized, etc.)
5. **Validator**: Check policy correctness before deployment

### 6.2 Generated Output

**Rhai Output**:
```rust
// Generated by Web UI Scheduler Builder
// Policy: "Batch Processing Optimized"
// Created: 2025-10-03

fn schedule(ctx) {
    // Rule 1: Batch jobs to most-vram-free workers
    for job in ctx.queue {
        if job.priority == "batch" {
            let worker = ctx.find_worker_most_vram_free(job.model_ref);
            if worker != () {
                return #{
                    action: "dispatch",
                    job_id: job.id,
                    worker_id: worker.id
                };
            }
        }
    }
    
    // Fallback to platform scheduler
    return platform::schedule(ctx);
}
```

**YAML Output**:
```yaml
# Generated by Web UI Scheduler Builder
# Policy: "Batch Processing Optimized"
# Created: 2025-10-03

scheduler:
  name: "batch-optimized"
  rules:
    - name: "batch-to-most-vram"
      condition:
        job.priority: "batch"
      action:
        dispatch:
          worker_selection: "most-vram-free"
```

### 6.3 Web UI Architecture

```
┌─────────────────────────────────────────┐
│ Web UI (Vue.js)                         │
│                                         │
│  ┌─────────────┐  ┌──────────────────┐ │
│  │ Rule Builder│  │ Live Simulator   │ │
│  └─────────────┘  └──────────────────┘ │
│                                         │
│  ┌─────────────┐  ┌──────────────────┐ │
│  │ Code Editor │  │ Template Library │ │
│  └─────────────┘  └──────────────────┘ │
└─────────────────────────────────────────┘
              ↓ (generates)
        Lua or YAML file
              ↓ (upload/deploy)
┌─────────────────────────────────────────┐
│ Orchestratord                           │
│  - Validates scheduler                  │
│  - Compiles YAML → Lua                  │
│  - Loads into sandbox                   │
│  - Executes on schedule() calls         │
└─────────────────────────────────────────┘
```

---

## 7. Design Questions & Decisions Needed

### 7.1 Language Support

**Q1**: Should we support BOTH Rhai and YAML, or pick one?
- **Option A**: Rhai only (simpler, more powerful)
- **Option B**: YAML only (simpler for users, less flexible)
- **Option C**: Both (YAML compiles to Rhai) ← **CHOSEN**

**Q2**: Should Web UI generate Rhai, YAML, or both?
- **Option A**: Generate both, user chooses format ← **CHOSEN**
- **Option B**: Generate YAML only (simpler), compile to Rhai internally
- **Option C**: Generate Rhai only (most powerful)

**Decision needed**: Which option aligns with target users?

---

### 7.2 Platform Scheduler Mutability

**Q3**: Can users see platform scheduler source code?
- **Option A**: Yes, fully open (users can learn from it)
- **Option B**: No, binary only (protect IP)
- **Option C**: Obfuscated Lua (readable but not easily copied)

**Q4**: Can users copy platform scheduler in Home/Lab mode?
- **Option A**: Yes, provide `platform.get_source()` function
- **Option B**: No, but provide similar "starter template"
- **Option C**: Yes, but watermarked/attributed

**Decision needed**: Open source vs proprietary strategy?

---

### 7.3 Sandbox Security

**Q5**: What Rhai operations should be restricted?
- File I/O (definitely restricted)
- Network access (definitely restricted)
- OS operations (definitely restricted)
- Infinite loops (timeout after N seconds?)
- Memory limits (max heap size?)

**Q6**: How to handle scheduler crashes/errors?
- **Option A**: Fallback to platform scheduler automatically
- **Option B**: Fail job with error, require user fix
- **Option C**: Retry with exponential backoff, then fallback

**Decision needed**: Safety vs user control trade-off?

---

### 7.4 Scheduler Lifecycle

**Q7**: When is scheduler loaded/compiled?
- **Option A**: At orchestratord startup (static)
- **Option B**: Hot-reload on file change (dynamic)
- **Option C**: Per-request compilation (flexible but slow)

**Q8**: Can scheduler be updated without restart?
- **Option A**: Yes, hot-reload with validation
- **Option B**: No, requires orchestratord restart
- **Option C**: Yes, but only in Home/Lab mode

**Decision needed**: Operational flexibility vs stability?

---

### 7.5 Web UI Integration

**Q9**: Where does Web UI live?
- **Option A**: Separate frontend app (e.g., `frontend/bin/orchestrator-ui`)
- **Option B**: Embedded in orchestratord (serve static files)
- **Option C**: Standalone desktop app (Electron/Tauri)

**Q10**: How does Web UI communicate with orchestratord?
- **Option A**: Via SDK (same as clients)
- **Option B**: Direct API calls to orchestratord
- **Option C**: Special admin API endpoints

**Q11**: Does Web UI need real-time updates?
- **Option A**: Yes, WebSocket for live queue/pool state
- **Option B**: Yes, SSE for log streaming
- **Option C**: No, polling is sufficient

**Decision needed**: Architecture and deployment model?

---

### 7.6 Scheduler Testing & Validation

**Q12**: How to test custom schedulers?
- **Option A**: Dry-run mode (simulate without executing)
- **Option B**: Test harness with sample data
- **Option C**: Property-based testing framework

**Q13**: What validation is required before deployment?
- **Option A**: Syntax check only
- **Option B**: Semantic validation (ensure returns valid actions)
- **Option C**: Performance testing (ensure completes in <50ms)

**Decision needed**: Testing infrastructure requirements?

---

### 7.7 Scheduler Versioning & Migration

**Q14**: How to version schedulers?
- **Option A**: Semantic versioning in scheduler file
- **Option B**: Git-based versioning
- **Option C**: Database-backed version history

**Q15**: What happens when platform scheduler updates?
- **Option A**: Users on old version continue (no auto-update)
- **Option B**: Force migration with deprecation warnings
- **Option C**: Side-by-side (users can choose version)

**Decision needed**: Upgrade and compatibility strategy?

---

### 7.8 Multi-Tenancy Considerations

**Q16**: In platform mode, can tenants have custom schedulers?
- **Option A**: No, platform scheduler only (consistent behavior)
- **Option B**: Yes, but sandboxed per-tenant (isolation)
- **Option C**: Yes, but only for premium tiers

**Q17**: How to prevent scheduler from accessing other tenants' data?
- **Option A**: Context object filtered by tenant_id
- **Option B**: Separate scheduler instance per tenant
- **Option C**: Not applicable (platform mode uses immutable scheduler)

**Decision needed**: Multi-tenancy model for schedulers?

---

### 7.9 Performance & Scalability

**Q18**: What is acceptable scheduler execution time?
- **Option A**: <10ms (strict, may limit complexity)
- **Option B**: <50ms (balanced)
- **Option C**: <100ms (lenient, may impact latency)

**Q19**: How to handle slow schedulers?
- **Option A**: Timeout and fallback to platform scheduler
- **Option B**: Warn user, continue execution
- **Option C**: Reject scheduler at load time if too slow

**Q20**: Should scheduler be JIT-compiled?
- **Option A**: Rhai has built-in optimization (no separate JIT needed)
- **Option B**: Rhai compilation is fast enough for scheduler workloads
- **Option C**: N/A - Lua is deprecated

**Decision needed**: Performance requirements and enforcement?

---

### 7.10 Observability & Debugging

**Q21**: How to debug custom schedulers?
- **Option A**: Logging API (scheduler can emit logs)
- **Option B**: Debugger integration (step through Lua)
- **Option C**: Trace mode (record all decisions)

**Q22**: What metrics should scheduler emit?
- **Option A**: Execution time, decisions made, fallback count
- **Option B**: Custom metrics via API
- **Option C**: Both built-in and custom

**Decision needed**: Debugging and observability tools?

---

## 8. Proposed Architecture

### 8.1 Scheduler Engine (Rust)

```rust
// bin/orchestratord-crates/scheduling/src/engine.rs

pub struct SchedulerEngine {
    rhai: Engine,  // Rhai runtime
    platform_scheduler: String,  // Immutable, built-in
    user_scheduler: Option<String>,  // Loaded from file/DB
    mode: SchedulerMode,
}

pub enum SchedulerMode {
    Platform,  // Use platform scheduler only (enforces capacity limits)
    UserRhai,  // Use user-provided Rhai (custom queue policies)
    UserYaml,  // Use user-provided YAML (compiled to Rhai)
}

impl SchedulerEngine {
    pub fn schedule(&self, context: SchedulerContext) -> SchedulerDecision {
        match self.mode {
            SchedulerMode::Platform => self.run_platform_scheduler(context),
            SchedulerMode::UserRhai => self.run_user_scheduler(context)
                .unwrap_or_else(|e| {
                    warn!("User scheduler failed: {}, falling back to platform", e);
                    self.run_platform_scheduler(context)
                }),
            SchedulerMode::UserYaml => {
                let rhai_code = compile_yaml_to_rhai(&self.user_scheduler.unwrap());
                self.run_rhai(rhai_code, context)
            }
        }
    }
}
```

### 8.2 YAML Schema

```yaml
# scheduler-schema.yaml
scheduler:
  type: object
  required: [name, rules]
  properties:
    name:
      type: string
    version:
      type: string
    rules:
      type: array
      items:
        type: object
        required: [name, condition, action, priority]
        properties:
          name: { type: string }
          condition:
            oneOf:
              - { const: "always" }
              - type: object  # Complex condition
          action:
            oneOf:
              - dispatch: { worker_selection: enum }
              - wait: { duration_ms: integer }
              - evict: { worker_id: string }
              - use: { const: "platform-scheduler" }
          priority: { type: integer }
```

### 8.3 Web UI Components

**Frontend**: `frontend/bin/orchestrator-ui/` (Vue.js)
- Rule builder (drag-and-drop)
- Code editor (Monaco/CodeMirror)
- Live simulator (WebSocket to orchestratord)
- Template library (pre-built policies)

**Backend**: Orchestratord admin API
- `POST /v2/admin/scheduler/validate` - Validate Lua/YAML
- `POST /v2/admin/scheduler/deploy` - Deploy scheduler
- `GET /v2/admin/scheduler/simulate` - Dry-run with sample data
- `GET /v2/admin/scheduler/platform-source` - Get platform scheduler (if allowed)

---

## 9. Implementation Phases

### Phase 1: Core Scheduler Engine (M1)
- [ ] Rhai runtime integration
- [ ] Platform scheduler implementation (Rhai with capacity limits and 429 rejection)
- [ ] Sandbox security (restrict dangerous ops)
- [ ] Context object API
- [ ] Fallback mechanism

### Phase 2: User Schedulers (M1)
- [ ] Load Rhai from file
- [ ] Validation and error handling
- [ ] Hot-reload support (Home/Lab mode)
- [ ] Metrics and logging
- [ ] Support custom queue policies (unbounded queues, custom eviction)

### Phase 3: YAML Support (M2)
- [ ] YAML schema definition (including queue policy rules)
- [ ] YAML → Rhai compiler
- [ ] Validation against schema
- [ ] Template library

### Phase 4: Web UI (M2)
- [ ] Vue.js frontend scaffold
- [ ] Rule builder component
- [ ] Code editor integration
- [ ] Live simulator (WebSocket)
- [ ] Template library UI

### Phase 5: Advanced Features (M3)
- [ ] Scheduler versioning
- [ ] A/B testing (compare schedulers)
- [ ] Performance profiling
- [ ] Debugger integration

---

## 10. Open Questions Summary

**Critical decisions needed**:
1. Rhai only, YAML only, or both? → **DECIDED: Both (YAML compiles to Rhai)**
2. Platform scheduler open source or proprietary? → **DECISION NEEDED**
3. Web UI architecture (standalone, embedded, desktop)? → **DECISION NEEDED**
4. Scheduler execution timeout (10ms, 50ms, 100ms)? → **DECIDED: 50ms**
5. Hot-reload support (yes/no, which modes)? → **DECIDED: Yes in Home/Lab, No in Platform**
6. Multi-tenant custom schedulers (yes/no)? → **DECIDED: No, platform scheduler only**
7. Queue policy differences? → **DECIDED: Platform mode enforces capacity limits (may reject with 429); Home/Lab modes typically use unbounded queues with custom policies**

**Nice-to-have clarifications**:
7. Debugging tools (logging, debugger, trace mode)?
8. Testing framework (dry-run, property tests)?
9. Versioning strategy (semver, git, database)?
10. Web UI real-time updates (WebSocket, SSE, polling)?

---

## 11. Next Steps

1. **Review this document** and make decisions on open questions
2. **Prototype Lua integration** with simple scheduler
3. **Design YAML schema** with example policies
4. **Scaffold Web UI** in `frontend/bin/orchestrator-ui/`
5. **Implement platform scheduler** as reference
6. **Write specs** for each component (engine, compiler, UI)

---

**Specs References**:
- `bin/orchestratord-crates/scheduling/.specs/00_scheduling.md` (main scheduling spec)
- `bin/orchestratord-crates/scheduling/.specs/01_rhai_scheduler_runtime.md` (Rhai runtime)
- `bin/orchestratord-crates/scheduling/.specs/02_yaml_compiler.md` (YAML → Rhai)
- `frontend/bin/orchestrator-ui/.specs/00_scheduler_builder.md` (Web UI)

**Related**:
- Platform scheduler: `bin/orchestratord-crates/scheduling/platform-scheduler.rhai`
- YAML schema: `bin/orchestratord-crates/scheduling/scheduler-schema.yaml`
- Web UI: `frontend/bin/orchestrator-ui/` (to be created)
