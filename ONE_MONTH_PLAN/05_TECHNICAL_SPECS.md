# Technical Specifications — Leveraging Existing Crates

**Date**: 2025-10-09  
**Status**: Implementation Guide

---

## Existing Shared Crates (Already Built!)

You have **11 shared crates** ready to use:

### 1. audit-logging ✅
**Location:** `bin/shared-crates/audit-logging/`  
**Purpose:** Immutable, tamper-evident audit trail for GDPR/SOC2/ISO 27001  
**Features:**
- 3 modes: Disabled (home lab), Local (single-node), Platform (marketplace)
- Flush modes: Immediate (compliance-safe), Batched, Hybrid
- 32 pre-defined event types
- Hash chain integrity
- Async, non-blocking

**Use in orchestratord:**
```rust
use audit_logging::{AuditLogger, AuditConfig, AuditMode, FlushMode};

// EU audit mode
let audit_logger = if env::var("LLORCH_EU_AUDIT")? == "true" {
    AuditLogger::new(AuditConfig {
        mode: AuditMode::Local {
            base_dir: PathBuf::from("/var/log/llorch/audit"),
        },
        service_id: "orchestratord".to_string(),
        rotation_policy: RotationPolicy::Daily,
        retention_policy: RetentionPolicy::default(),
        flush_mode: FlushMode::Immediate,  // Compliance-safe
    })?
} else {
    // Homelab mode - zero overhead
    AuditLogger::new(AuditConfig {
        mode: AuditMode::Disabled,
        service_id: "orchestratord".to_string(),
        ..Default::default()
    })?
};

// Emit events
audit_logger.emit(AuditEvent::TaskSubmitted {
    timestamp: Utc::now(),
    actor: ActorInfo {
        user_id: "customer-123".to_string(),
        ip: Some(extract_ip(&req)),
        auth_method: AuthMethod::BearerToken,
    },
    task_id: job_id.clone(),
    model_ref: req.model.clone(),
    prompt_length: req.prompt.len(),
    service_id: "orchestratord".to_string(),
})?;
```

### 2. auth-min ✅
**Location:** `bin/shared-crates/auth-min/`  
**Purpose:** Minimal authentication (bearer tokens, API keys)  
**Features:**
- Token fingerprinting (never log full tokens)
- Timing-safe comparisons
- Token validation

**Use in orchestratord:**
```rust
use auth_min::{fingerprint_token, validate_bearer_token};

// Fingerprint for audit logging
let token_fp = fingerprint_token(&bearer_token);
audit_logger.emit(AuditEvent::AuthSuccess {
    actor: ActorInfo {
        user_id: format!("token:{}", token_fp),  // Safe to log
        ..
    },
    ..
});

// Validate token
if !validate_bearer_token(&bearer_token, &expected_token) {
    return Err(AuthError::InvalidToken);
}
```

### 3. input-validation ✅
**Location:** `bin/shared-crates/input-validation/`  
**Purpose:** Sanitize user input to prevent log injection  
**Features:**
- String sanitization
- Path validation
- SQL injection prevention

**Use in orchestratord:**
```rust
use input_validation::sanitize_string;

// Always sanitize before logging
let safe_model = sanitize_string(&req.model)?;
let safe_prompt = sanitize_string(&req.prompt)?;

audit_logger.emit(AuditEvent::TaskSubmitted {
    model_ref: safe_model,  // Protected from log injection
    prompt_length: safe_prompt.len(),
    ..
})?;
```

### 4. secrets-management ✅
**Location:** `bin/shared-crates/secrets-management/`  
**Purpose:** Secure secret storage and retrieval  
**Features:**
- Environment variable loading
- File-based secrets
- Encryption at rest

**Use in orchestratord:**
```rust
use secrets_management::load_secret;

// Load API keys securely
let api_key = load_secret("LLORCH_API_KEY")?;
```

### 5. narration-core ✅
**Location:** `bin/shared-crates/narration-core/`  
**Purpose:** Developer-focused observability (separate from audit logging)  
**Features:**
- Structured logging
- Correlation IDs
- Performance tracking

**Use in orchestratord:**
```rust
use narration_core::{narrate, NarrationEvent};

// Developer observability (NOT audit)
narrate(NarrationEvent::JobDispatched {
    job_id: job_id.clone(),
    worker_id: worker.id.clone(),
    correlation_id: correlation_id.clone(),
    duration_ms: dispatch_duration.as_millis(),
});
```

### 6. deadline-propagation ✅
**Location:** `bin/shared-crates/deadline-propagation/`  
**Purpose:** Request deadline tracking  
**Features:**
- Deadline propagation across services
- Timeout enforcement

### 7. gpu-info ✅
**Location:** `bin/shared-crates/gpu-info/`  
**Purpose:** GPU information queries  
**Features:**
- NVML wrapper
- GPU detection
- VRAM queries

### 8. pool-registry-types ✅
**Location:** `bin/shared-crates/pool-registry-types/`  
**Purpose:** Pool registry data structures  
**Features:**
- Pool metadata types
- Worker registry types

### 9. orchestrator-core ✅
**Location:** `bin/shared-crates/orchestrator-core/` (stub)  
**Purpose:** Shared orchestrator logic  
**Status:** Needs implementation (Week 1)

### 10. pool-core ✅
**Location:** `bin/shared-crates/pool-core/` (stub)  
**Purpose:** Shared pool manager logic  
**Status:** Needs implementation (Week 1)

### 11. narration-macros ✅
**Location:** `bin/shared-crates/narration-macros/`  
**Purpose:** Macros for narration-core

---

## Week 1 Implementation (Using Existing Crates)

### Day 1: orchestratord

**Add dependencies:**
```toml
[dependencies]
# Existing shared crates
audit-logging = { path = "../shared-crates/audit-logging" }
auth-min = { path = "../shared-crates/auth-min" }
input-validation = { path = "../shared-crates/input-validation" }
narration-core = { path = "../shared-crates/narration-core" }
secrets-management = { path = "../shared-crates/secrets-management" }

# HTTP server
axum = "0.7"
tokio = { version = "1", features = ["full"] }
tower-http = { version = "0.5", features = ["cors", "trace"] }

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Utilities
uuid = { version = "1", features = ["v4", "serde"] }
chrono = "0.4"
anyhow = "1"
tracing = "0.1"
tracing-subscriber = "0.3"
```

**Implement with audit logging:**
```rust
// src/main.rs
use audit_logging::{AuditLogger, AuditConfig, AuditMode, AuditEvent, ActorInfo};
use auth_min::fingerprint_token;
use input_validation::sanitize_string;
use narration_core::narrate;

#[derive(Clone)]
struct AppState {
    jobs: Arc<Mutex<Vec<Job>>>,
    workers: Arc<Mutex<Vec<Worker>>>,
    audit_logger: Arc<AuditLogger>,  // Use existing crate!
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    
    // Initialize audit logger (existing crate!)
    let eu_audit = env::var("LLORCH_EU_AUDIT")
        .unwrap_or_else(|_| "false".to_string()) == "true";
    
    let audit_logger = if eu_audit {
        tracing::info!("🇪🇺 EU audit mode ENABLED");
        AuditLogger::new(AuditConfig {
            mode: AuditMode::Local {
                base_dir: PathBuf::from("/var/log/llorch/audit"),
            },
            service_id: "orchestratord".to_string(),
            rotation_policy: RotationPolicy::Daily,
            retention_policy: RetentionPolicy::default(),
            flush_mode: FlushMode::Immediate,
        })?
    } else {
        tracing::info!("🏠 Homelab mode (audit disabled)");
        AuditLogger::new(AuditConfig {
            mode: AuditMode::Disabled,
            service_id: "orchestratord".to_string(),
            ..Default::default()
        })?
    };
    
    let state = AppState {
        jobs: Arc::new(Mutex::new(Vec::new())),
        workers: Arc::new(Mutex::new(Vec::new())),
        audit_logger: Arc::new(audit_logger),
    };
    
    let app = Router::new()
        .route("/health", get(health))
        .route("/v2/tasks", post(submit_task))
        .route("/workers/register", post(register_worker))
        .with_state(state);
    
    // Start server
    axum::Server::bind(&"0.0.0.0:8080".parse()?)
        .serve(app.into_make_service())
        .await?;
    
    Ok(())
}

async fn submit_task(
    State(state): State<AppState>,
    Json(req): Json<TaskRequest>,
) -> Result<Json<TaskResponse>, AppError> {
    let job_id = Uuid::new_v4().to_string();
    
    // Sanitize input (existing crate!)
    let safe_model = sanitize_string(&req.model)?;
    let safe_prompt = sanitize_string(&req.prompt)?;
    
    // Audit log (existing crate!)
    state.audit_logger.emit(AuditEvent::TaskSubmitted {
        timestamp: Utc::now(),
        actor: ActorInfo {
            user_id: "anonymous".to_string(),  // TODO: Extract from auth
            ip: None,  // TODO: Extract from request
            auth_method: AuthMethod::None,
        },
        task_id: job_id.clone(),
        model_ref: safe_model.clone(),
        prompt_length: safe_prompt.len(),
        service_id: "orchestratord".to_string(),
    })?;
    
    // Narration (existing crate!)
    narrate(NarrationEvent::JobSubmitted {
        job_id: job_id.clone(),
        model: safe_model.clone(),
        correlation_id: Uuid::new_v4().to_string(),
    });
    
    // ... rest of logic
    
    Ok(Json(TaskResponse {
        job_id,
        status: "queued".to_string(),
    }))
}
```

**You get for FREE:**
- ✅ Audit logging (immutable, tamper-evident)
- ✅ Input sanitization (log injection prevention)
- ✅ Token fingerprinting (safe logging)
- ✅ Narration (developer observability)
- ✅ Secrets management (API keys)

---

## Week 2 Implementation (GDPR Endpoints)

### Day 9: GDPR Endpoints (Using audit-logging)

**The audit-logging crate already has query APIs!**

```rust
// src/gdpr.rs
use audit_logging::{AuditLogger, AuditQuery};

pub async fn gdpr_export(
    State(state): State<AppState>,
    Query(query): Query<ExportQuery>,
) -> Result<Json<ExportResponse>, AppError> {
    // Query audit logs (existing crate!)
    let audit_events = state.audit_logger.query(AuditQuery {
        actor: Some(query.user_id.clone()),
        start_time: None,  // All time
        end_time: None,
        event_types: vec![],  // All types
        limit: 10000,
    }).await?;
    
    // Get jobs
    let jobs = state.jobs.lock().unwrap()
        .iter()
        .filter(|j| j.user_id.as_ref() == Some(&query.user_id))
        .cloned()
        .collect();
    
    Ok(Json(ExportResponse {
        user_id: query.user_id,
        jobs,
        audit_events,
        created_at: Utc::now().to_rfc3339(),
    }))
}

pub async fn gdpr_delete(
    State(state): State<AppState>,
    Json(req): Json<DeleteRequest>,
) -> Result<Json<DeleteResponse>, AppError> {
    // Soft delete jobs
    let mut jobs = state.jobs.lock().unwrap();
    let mut deleted_count = 0;
    
    for job in jobs.iter_mut() {
        if job.user_id.as_ref() == Some(&req.user_id) {
            job.status = "deleted".to_string();
            deleted_count += 1;
        }
    }
    
    // Audit the deletion (existing crate!)
    state.audit_logger.emit(AuditEvent::GdprRightToErasure {
        timestamp: Utc::now(),
        customer_id: req.user_id.clone(),
        reason: req.reason.clone(),
        completed_at: Utc::now(),
        service_id: "orchestratord".to_string(),
    })?;
    
    Ok(Json(DeleteResponse {
        user_id: req.user_id,
        jobs_deleted: deleted_count,
        status: "deleted".to_string(),
    }))
}
```

**You get for FREE:**
- ✅ Audit log querying
- ✅ GDPR event types (already defined)
- ✅ Immutable audit trail
- ✅ Compliance-ready retention

---

## Simplified Implementation Plan

### Week 1: Foundation

**Day 1: orchestratord**
- ✅ Use `audit-logging` crate (already exists!)
- ✅ Use `auth-min` crate (already exists!)
- ✅ Use `input-validation` crate (already exists!)
- ✅ Use `narration-core` crate (already exists!)
- ⬜ Implement HTTP server (axum)
- ⬜ Implement job queue (in-memory)
- ⬜ Implement worker registry (in-memory)

**Day 2: pool-ctl**
- ⬜ Implement CLI (clap)
- ⬜ Implement model download (hf CLI wrapper)
- ⬜ Implement worker spawn

**Day 3: llorch-ctl**
- ⬜ Implement CLI (clap)
- ⬜ Implement SSH commands
- ⬜ Implement job submission

### Week 2: EU Compliance

**Day 8: Audit Toggle**
- ✅ Already implemented in `audit-logging` crate!
- ⬜ Wire up to orchestratord
- ⬜ Test Disabled vs Local modes

**Day 9: GDPR Endpoints**
- ✅ Query API already in `audit-logging` crate!
- ✅ GDPR event types already defined!
- ⬜ Implement export endpoint
- ⬜ Implement delete endpoint
- ⬜ Implement consent endpoint

**Day 10: Data Residency**
- ⬜ Add region to worker registration
- ⬜ Filter workers by region
- ⬜ Test EU-only enforcement

---

## What You DON'T Need to Build

### ❌ Audit Logging System
**Already exists:** `bin/shared-crates/audit-logging/`
- Immutable storage ✅
- Hash chain integrity ✅
- Query API ✅
- GDPR event types ✅
- Flush modes ✅

### ❌ Authentication System
**Already exists:** `bin/shared-crates/auth-min/`
- Token fingerprinting ✅
- Timing-safe comparison ✅
- Bearer token validation ✅

### ❌ Input Sanitization
**Already exists:** `bin/shared-crates/input-validation/`
- String sanitization ✅
- Log injection prevention ✅

### ❌ Secrets Management
**Already exists:** `bin/shared-crates/secrets-management/`
- Environment variables ✅
- File-based secrets ✅

### ❌ Observability System
**Already exists:** `bin/shared-crates/narration-core/`
- Structured logging ✅
- Correlation IDs ✅

---

## Updated Timeline (Faster!)

### Week 1: 3 days instead of 5
- Day 1: orchestratord (use existing crates)
- Day 2: pool-ctl
- Day 3: llorch-ctl
- Days 4-5: Integration + buffer

### Week 2: 2 days instead of 5
- Day 8: Wire up audit-logging (already exists!)
- Day 9: GDPR endpoints (use existing query API!)
- Days 10-12: Web UI
- Days 13-14: Polish

**You're ahead of schedule because you already have the hard parts built!**

---

## Key Takeaway

**You have 11 shared crates already built:**
1. audit-logging ✅ (895 lines of docs!)
2. auth-min ✅
3. input-validation ✅
4. secrets-management ✅
5. narration-core ✅
6. narration-macros ✅
7. deadline-propagation ✅
8. gpu-info ✅
9. pool-registry-types ✅
10. orchestrator-core (stub)
11. pool-core (stub)

**You DON'T need to build:**
- ❌ Audit logging system
- ❌ Authentication
- ❌ Input sanitization
- ❌ Secrets management
- ❌ Observability

**You ONLY need to build:**
- ✅ orchestratord HTTP server
- ✅ pool-ctl CLI
- ✅ llorch-ctl CLI
- ✅ Web UI
- ✅ Wire up existing crates

**This makes the 30-day plan VERY achievable. You're not starting from zero!**

---

**Version**: 1.0  
**Status**: EXECUTE  
**Last Updated**: 2025-10-09
