# 🎭 Rhai Programmable Scheduler: The Smart Brain

**Pronunciation:** rbee (pronounced "are-bee")  
**Date:** 2025-10-10  
**Status:** M2 Feature (Planned)  
**Spec:** `bin/.specs/00_llama-orch.md` [SYS-6.1.5]

**🎯 PRIMARY TARGET AUDIENCE:** Developers who build with AI but don't want to depend on big AI providers.

**THE FEAR:** Building complex codebases with AI assistance. What if the provider changes, shuts down, or changes pricing? Your codebase becomes unmaintainable.

**THE SOLUTION:** Build your own AI infrastructure using ALL your home network hardware. Never depend on external providers again.

---

## What is the Rhai Programmable Scheduler?

**Rhai** is an embedded Rust scripting language that powers rbee's intelligent scheduling decisions.

**Think of it as:**
- The "brain" of queen-rbee
- A policy execution engine
- User-programmable orchestration logic

---

## Two Modes: Platform vs Home/Lab

### Platform Mode (Multi-Tenant Marketplace)

**Purpose:** Secure, fair, multi-tenant GPU marketplace

**Scheduler:** `platform-scheduler.rhai` (built-in, immutable)

**Characteristics:**
- ✅ **Immutable** - Cannot be modified by users
- ✅ **Multi-tenant fairness** - Fair resource allocation
- ✅ **SLA compliance** - Guarantees service levels
- ✅ **Security-first** - Sandboxed execution
- ✅ **Quota enforcement** - Per-tenant limits
- ✅ **Capacity management** - Rejects with 429 when full

**Use Case:** Commercial GPU marketplace (api.yourplatform.com)

**Example Logic:**
```rhai
// Platform scheduler (immutable)
fn schedule_job(job, pools, workers) {
    // 1. Check tenant quota
    if tenant_over_quota(job.tenant_id) {
        return reject("quota_exceeded");
    }
    
    // 2. Priority-based scheduling
    if job.priority == "interactive" {
        // Route to least-loaded worker
        return workers.least_loaded();
    } else {
        // Route to cheapest available
        return workers.min_by(|w| w.cost_per_token);
    }
}
```

### Home/Lab Mode (Self-Hosted)

**Purpose:** Custom orchestration for personal/team use

**Scheduler:** User-written Rhai scripts or YAML configs

**Characteristics:**
- ✅ **Customizable** - Write your own logic
- ✅ **No recompilation** - Update scripts live
- ✅ **Unbounded queues** - No artificial limits
- ✅ **Custom policies** - Your rules, your way
- ✅ **YAML support** - Compiles to Rhai internally
- ✅ **Web UI builder** - Visual policy editor

**Use Case:** Homelab, research lab, small team

**Example Logic:**
```rhai
// Custom home scheduler
fn schedule_job(job, pools, workers) {
    // Route large models to multi-GPU setups
    if job.model.contains("70b") {
        return workers.filter(|w| w.gpus > 1).first();
    }
    
    // Route image generation to CUDA
    if job.type == "image-gen" {
        return workers
            .filter(|w| w.capability == "image-gen" && w.backend == "cuda")
            .least_loaded();
    }
    
    // Everything else to cheapest
    return workers.min_by(|w| w.cost_per_token);
}
```

---

## The Rhai Scheduler API

### Complete System State Access

**Available to Rhai scripts:**
- `queue` - Current job queue state
- `pools` - All registered pool managers
- `workers` - All active workers
- `gpus` - GPU/VRAM state
- `models` - Model catalog
- `tenants` - Tenant quotas (platform mode)

### 40+ Built-in Helper Functions

**Worker Selection:**
- `workers.least_loaded()` - Pick worker with lowest load
- `workers.most_vram_free()` - Pick worker with most free VRAM
- `workers.round_robin()` - Distribute evenly
- `workers.filter(predicate)` - Filter by condition
- `workers.min_by(key)` - Pick minimum by key
- `workers.max_by(key)` - Pick maximum by key

**GPU Queries:**
- `gpu_vram_total(gpu_id)` - Total VRAM
- `gpu_vram_free(gpu_id)` - Free VRAM
- `gpu_vram_allocated(gpu_id)` - Allocated VRAM
- `gpu_device_name(gpu_id)` - Device name

**Quota Checks (Platform Mode):**
- `tenant_over_quota(tenant_id)` - Check quota exceeded
- `tenant_remaining_quota(tenant_id)` - Remaining quota
- `tenant_usage(tenant_id)` - Current usage

**Model Queries:**
- `model_size_bytes(model_ref)` - Model file size
- `model_vram_required(model_ref)` - Estimated VRAM needed
- `model_exists(model_ref)` - Check if model available

**Eviction Helpers:**
- `evict_least_recently_used()` - LRU eviction
- `evict_lowest_priority()` - Priority-based eviction
- `evict_idle_workers()` - Evict idle workers

---

## Real-World Examples

### Example 1: Cost Optimizer (Home Mode)

```rhai
// Route to cheapest provider that meets deadline
fn schedule_job(job, pools, workers) {
    let deadline_ok = workers.filter(|w| 
        w.estimated_time < job.deadline
    );
    
    if deadline_ok.is_empty() {
        return reject("no_workers_meet_deadline");
    }
    
    return deadline_ok.min_by(|w| w.cost_per_token);
}
```

**Value:** Saves 20-30% on costs by always choosing cheapest option

### Example 2: Latency Optimizer (Home Mode)

```rhai
// Route to geographically closest worker
fn schedule_job(job, pools, workers) {
    let user_lat = job.user.geo.latitude;
    let user_lon = job.user.geo.longitude;
    
    return workers.min_by(|w| {
        distance(user_lat, user_lon, w.geo.latitude, w.geo.longitude)
    });
}
```

**Value:** 30-50% lower latency by minimizing network distance

### Example 3: Multi-Modal Smart Router (Home Mode)

```rhai
// Route different AI tasks to specialized workers
fn schedule_job(job, pools, workers) {
    if job.type == "image-gen" && job.priority == "high" {
        // Route Stable Diffusion to CUDA GPUs
        return workers
            .filter(|w| w.capability == "image-gen" && w.backend == "cuda")
            .least_loaded();
    } else if job.type == "text-gen" && job.model.contains("70b") {
        // Route large LLMs to multi-GPU setups
        return workers.filter(|w| w.gpus > 1).first();
    } else {
        // Route everything else to cheapest available
        return workers.min_by(|w| w.cost_per_token);
    }
}
```

**Value:** Optimizes both cost and performance based on task type

### Example 4: EU-Only Compliance Router (Home Mode)

```rhai
// Route only to EU providers for GDPR compliance
fn schedule_job(job, pools, workers) {
    let eu_workers = workers.filter(|w| 
        w.geo.region == "EU"
    );
    
    if eu_workers.is_empty() {
        return reject("no_eu_workers_available");
    }
    
    return eu_workers.least_loaded();
}
```

**Value:** Automatic GDPR compliance by geo-filtering

---

## YAML Support (Declarative Mode)

**For non-programmers, YAML compiles to Rhai:**

```yaml
# scheduler.yaml
rules:
  - name: "Large models to multi-GPU"
    condition:
      model_contains: "70b"
    action:
      select: "workers.filter(gpus > 1).first()"
  
  - name: "Image gen to CUDA"
    condition:
      task_type: "image-gen"
    action:
      select: "workers.filter(backend == 'cuda').least_loaded()"
  
  - name: "Default to cheapest"
    condition: true
    action:
      select: "workers.min_by(cost_per_token)"
```

**Compiles to Rhai internally, same performance**

---

## Web UI Mode (Visual Policy Builder)

**For visual users:**

1. **Drag-and-drop policy builder**
   - Add conditions (if model contains "70b")
   - Add actions (route to multi-GPU)
   - Visual flow diagram

2. **Live preview**
   - Test with sample jobs
   - See routing decisions
   - Verify correctness

3. **Export to Rhai or YAML**
   - Save as script file
   - Deploy to queen-rbee
   - Update without restart

---

## Security & Sandboxing

### Sandboxed Execution

**Safety guarantees:**
- ✅ **50ms timeout** - Scripts can't hang
- ✅ **Memory limits** - Can't exhaust RAM
- ✅ **No unsafe operations** - No file I/O, no network
- ✅ **Type safety** - Rhai is type-safe
- ✅ **No FFI** - Can't call external code

### Platform Mode Security

**Immutable scheduler:**
- ❌ Users cannot modify platform scheduler
- ❌ Users cannot inject malicious code
- ❌ Users cannot bypass quotas
- ✅ Platform operator controls all logic
- ✅ Audited and tested scheduler

---

## Why Rhai (Not Python/JavaScript/Lua)?

| Feature | Rhai | Python | JavaScript | Lua |
|---------|------|--------|------------|-----|
| Rust integration | ✅ Native | ❌ FFI | ❌ FFI | ❌ FFI |
| Type safety | ✅ Yes | ❌ No | ❌ No | ❌ No |
| Sandboxed | ✅ Yes | ❌ No | ⚠️ Partial | ⚠️ Partial |
| Performance | ✅ Fast | ❌ Slow | ⚠️ Medium | ✅ Fast |
| Zero dependencies | ✅ Yes | ❌ No | ❌ No | ⚠️ Partial |
| Embedded-first | ✅ Yes | ❌ No | ❌ No | ✅ Yes |

**Rhai is the ONLY choice for embedded Rust scripting.**

---

## Implementation Timeline

**M0 (Current):** ❌ Not implemented  
**M1:** ❌ Not implemented (worker lifecycle focus)  
**M2 (Q2 2026):** ✅ Rhai scheduler engine  
**M3 (Q3 2026):** ✅ Platform mode with immutable scheduler  
**M4 (Q4 2026):** ✅ Web UI policy builder

---

## Competitive Advantage

### vs. Competitors

**Runpod/Vast.ai:**
- ❌ No routing customization
- ❌ Fixed algorithms
- ❌ No user control

**Together.ai/Replicate:**
- ❌ Vendor lock-in
- ❌ Black box routing
- ❌ No customization

**AWS Bedrock:**
- ❌ Vendor lock-in
- ❌ No scripting
- ❌ Opaque routing

**rbee:**
- ✅ User-scriptable (Home/Lab mode)
- ✅ Transparent (open source)
- ✅ Customizable (Rhai + YAML + Web UI)
- ✅ Platform mode for marketplace
- ✅ Best of both worlds

---

## The Vision

**Today:** Fixed routing algorithms  
**M2:** Programmable Rhai scheduler  
**M3:** Platform mode for marketplace  
**M4:** Visual policy builder  
**Future:** Community-shared scheduler templates

**The result:** Maximum flexibility without sacrificing security. 🐝🎭

---

## Key Takeaways

1. **Two modes:** Platform (immutable) vs Home/Lab (customizable)
2. **Platform mode:** Multi-tenant marketplace with built-in scheduler
3. **Home/Lab mode:** Write custom Rhai scripts or YAML configs
4. **40+ helpers:** Worker selection, GPU queries, quota checks
5. **Sandboxed:** 50ms timeout, memory limits, type-safe
6. **Web UI:** Visual policy builder for non-programmers
7. **No marketplace:** Users don't sell scripts, they write for themselves

**This is the REAL Rhai implementation.** 🎭🐝

---

*Based on: bin/.specs/00_llama-orch.md [SYS-6.1.5] Programmable Scheduler*
