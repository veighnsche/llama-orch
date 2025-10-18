# Kubernetes Drift Found in Codebase

**Date:** 2025-10-18  
**Found by:** TEAM-113  
**Status:** 🚨 CRITICAL

---

## 🚨 KUBERNETES DRIFT IN PRODUCTION CODE

### Location: `bin/rbee-hive/src/http/health.rs`

**Lines 8-9:**
```rust
//! - `GET /health/live` - Kubernetes liveness probe (TEAM-104)
//! - `GET /health/ready` - Kubernetes readiness probe (TEAM-104)
```

**Lines 78-142:**
```rust
/// TEAM-104: Liveness probe response
#[derive(Serialize)]
pub struct LivenessResponse {
    status: String,
}

/// TEAM-104: Readiness probe response
#[derive(Serialize)]
pub struct ReadinessResponse {
    status: String,
    workers_total: usize,
    workers_ready: usize,
}

/// TEAM-104: Handle GET /health/live
///
/// Kubernetes liveness probe - checks if the process is alive
/// Returns 200 OK if the service is running
pub async fn handle_liveness() -> Json<LivenessResponse> {
    debug!("Liveness probe requested");
    Json(LivenessResponse {
        status: "alive".to_string(),
    })
}

/// TEAM-104: Handle GET /health/ready
///
/// Kubernetes readiness probe - checks if the service is ready to accept traffic
/// Returns 200 OK if at least one worker is ready, 503 otherwise
pub async fn handle_readiness(
    State(state): State<crate::http::routes::AppState>,
) -> Result<Json<ReadinessResponse>, StatusCode> {
    // ... implementation
}
```

---

## 🔍 Analysis

### What TEAM-104 Did
- Added `/health/live` endpoint (Kubernetes liveness probe)
- Added `/health/ready` endpoint (Kubernetes readiness probe)
- Explicitly labeled them as "Kubernetes" probes
- Implemented Kubernetes-style readiness logic

### The Problem
**rbee is NOT running in Kubernetes!**

- These are patterns for apps running IN Kubernetes
- rbee IS the orchestrator, not the orchestrated
- We don't need Kubernetes probes
- This is the start of Kubernetes drift

### Why This Happened
TEAM-104 saw "health checks" and thought "Kubernetes patterns"

**The drift pattern:**
1. See "health checks" → think "Kubernetes"
2. Add "liveness" and "readiness" probes
3. Next: Add "startup" probes
4. Next: Add Kubernetes deployment patterns
5. Next: Require Kubernetes to run
6. Result: rbee becomes a Kubernetes plugin

---

## ✅ What We Actually Need

### Current: `/v1/health` (GOOD)
```rust
pub async fn handle_health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "alive".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        api_version: "v1".to_string(),
    })
}
```

**This is perfect!**
- Simple health check
- Returns version info
- No Kubernetes assumptions
- Works everywhere

### Don't Need: `/health/live` and `/health/ready`
**Why not:**
- rbee doesn't run in Kubernetes
- rbee IS the orchestrator
- Simple `/v1/health` is enough
- Adding complexity for no benefit

---

## 🚨 Recommendation

### Option 1: Remove Kubernetes Endpoints (RECOMMENDED)
**Delete:**
- `/health/live` endpoint
- `/health/ready` endpoint
- All "Kubernetes" references

**Keep:**
- `/v1/health` endpoint (works fine)

**Reason:** rbee is not a Kubernetes app

### Option 2: Keep But Rename (COMPROMISE)
**If we keep them:**
- Rename `/health/live` → `/v1/health/alive`
- Rename `/health/ready` → `/v1/health/workers`
- Remove all "Kubernetes" references
- Document as "rbee health checks" not "Kubernetes probes"

**Reason:** Useful info, but not Kubernetes-specific

---

## 📋 Other Drift Found

### In Documentation (Not Code)

**File:** `bin/.specs/edge-cases/test-001-professional.md`
- Lines 297-316: References "Kubernetes-style distinction"
- Suggests liveness vs readiness probes
- **Status:** Spec document, not production code

**File:** `bin/llm-worker-rbee/src/http/ready.rs`
- Likely has similar Kubernetes drift
- **Action:** Need to check this file

---

## 🎯 Root Cause

### Why TEAM-104 Added This

**Likely thought process:**
1. "We need health checks"
2. "Industry standard is Kubernetes probes"
3. "Let's add liveness and readiness"
4. "This makes us production-ready"

**What they missed:**
- rbee is NOT running in Kubernetes
- rbee IS the orchestrator
- Simple health check is enough
- Kubernetes patterns don't apply

---

## 🚫 What This Leads To

### The Drift Spiral

**Month 1: Health Probes**
- ✅ Added `/health/live` and `/health/ready`
- Justification: "Kubernetes best practices"

**Month 2: More Kubernetes Patterns**
- Add `/health/startup` probe
- Add Kubernetes deployment manifests
- Add Helm charts
- Justification: "Cloud-native deployment"

**Month 3: Kubernetes Required**
- rbee now expects to run in Kubernetes
- Setup requires Kubernetes cluster
- Users ask "Why not just use KubeFlow?"

**Month 4: Product Death**
- rbee is now a Kubernetes plugin
- Lost the "simple alternative" value
- Deprecated in favor of KubeFlow

---

## ✅ Action Items

### Immediate
1. [ ] Review `bin/llm-worker-rbee/src/http/ready.rs` for similar drift
2. [ ] Decide: Remove or rename Kubernetes endpoints
3. [ ] Update documentation to remove Kubernetes references

### Week 2 Team
1. [ ] Don't add more Kubernetes patterns
2. [ ] Question any "Kubernetes-style" suggestions
3. [ ] Remember: rbee IS the orchestrator

### Long Term
1. [ ] Add to coding standards: "No Kubernetes patterns"
2. [ ] Add to PR checklist: "Does this assume Kubernetes?"
3. [ ] Train teams: "rbee is the alternative to Kubernetes"

---

## 📊 Summary

### Kubernetes Drift Found
- ✅ `/health/live` endpoint (Kubernetes liveness probe)
- ✅ `/health/ready` endpoint (Kubernetes readiness probe)
- ✅ Explicit "Kubernetes" labeling in code
- ⚠️ Possibly more in llm-worker-rbee

### Impact
- 🟡 LOW (currently just 2 endpoints)
- 🔴 HIGH RISK (this is how drift starts)

### Recommendation
- 🔥 Remove Kubernetes endpoints
- ✅ Keep simple `/v1/health`
- 🚫 Block future Kubernetes patterns

---

**Status:** 🚨 DRIFT DETECTED  
**Severity:** 🟡 LOW (but HIGH RISK)  
**Action:** Remove Kubernetes patterns  
**Prevention:** Add to coding standards

---

**Found by:** TEAM-113  
**Date:** 2025-10-18  
**Warning:** This is how products die
