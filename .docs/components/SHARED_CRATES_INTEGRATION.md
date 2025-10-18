# Shared Crates Integration Guide

**Purpose:** How to integrate existing shared crates into rbee components  
**Audience:** Future engineering teams  
**Last Updated:** TEAM-096 | 2025-10-18

## Overview

**9 production-ready shared crates** are already implemented in `bin/shared-crates/` but **not yet integrated** into the main components (queen-rbee, rbee-hive, llm-worker-rbee, rbee-keeper).

This guide shows how to integrate them.

---

## Quick Reference

| Need | Use This Crate | Add to Components |
|------|---------------|-------------------|
| **Authentication** | `auth-min` | queen, hive, worker |
| **JWT tokens** | `jwt-guardian` | queen (user auth) |
| **Secrets** | `secrets-management` | all components |
| **Input validation** | `input-validation` | all HTTP endpoints |
| **Audit logs** | `audit-logging` | queen, hive |
| **Timeouts** | `deadline-propagation` | queen, hive, worker |
| **GPU detection** | `gpu-info` | ✅ Already used |
| **Model tracking** | `model-catalog` | ✅ Already used |

---

## Integration Examples

### 1. Add Authentication (auth-min)

**Scenario:** Secure queen-rbee ↔ rbee-hive communication

#### Step 1: Add Dependency

```toml
# bin/queen-rbee/Cargo.toml
[dependencies]
auth-min = { path = "../shared-crates/auth-min" }
```

#### Step 2: Load API Token

```rust
// bin/queen-rbee/src/main.rs
use auth_min::timing_safe_eq;

// Load expected token from environment
let expected_token = std::env::var("LLORCH_API_TOKEN")
    .expect("LLORCH_API_TOKEN required");

// Store in app state
struct AppState {
    expected_token: String,
    // ... other fields
}
```

#### Step 3: Add Middleware

```rust
// bin/queen-rbee/src/http/middleware.rs
use auth_min::{parse_bearer, timing_safe_eq, token_fp6};
use axum::{
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::Response,
};

pub async fn auth_middleware<B>(
    State(state): State<AppState>,
    req: Request<B>,
    next: Next<B>,
) -> Result<Response, StatusCode> {
    // Parse Bearer token
    let auth_header = req.headers()
        .get("authorization")
        .and_then(|h| h.to_str().ok());
    
    let token = parse_bearer(auth_header)
        .ok_or(StatusCode::UNAUTHORIZED)?;
    
    // Timing-safe comparison
    if !timing_safe_eq(token.as_bytes(), state.expected_token.as_bytes()) {
        let fp = token_fp6(&token);
        tracing::warn!(identity = %format!("token:{}", fp), "auth failed");
        return Err(StatusCode::UNAUTHORIZED);
    }
    
    // Success - log with fingerprint
    let fp = token_fp6(&token);
    tracing::info!(identity = %format!("token:{}", fp), "authenticated");
    
    Ok(next.run(req).await)
}
```

#### Step 4: Apply to Routes

```rust
// bin/queen-rbee/src/http/routes.rs
use axum::middleware;

let app = Router::new()
    .route("/v1/workers/spawn", post(handle_spawn_worker))
    .layer(middleware::from_fn_with_state(state.clone(), auth_middleware))
    .with_state(state);
```

---

### 2. Add Secrets Management (secrets-management)

**Scenario:** Load API tokens from files instead of environment variables

#### Step 1: Add Dependency

```toml
# bin/queen-rbee/Cargo.toml
[dependencies]
secrets-management = { path = "../shared-crates/secrets-management" }
```

#### Step 2: Load from File

```rust
// bin/queen-rbee/src/main.rs
use secrets_management::Secret;

// Load API token from file
let api_token = Secret::load_from_file("/etc/llorch/secrets/api-token")
    .expect("Failed to load API token");

// Store in app state
struct AppState {
    api_token: Secret,
    // ... other fields
}
```

#### Step 3: Use in Middleware

```rust
// bin/queen-rbee/src/http/middleware.rs
use secrets_management::Secret;

pub async fn auth_middleware<B>(
    State(state): State<AppState>,
    req: Request<B>,
    next: Next<B>,
) -> Result<Response, StatusCode> {
    let token = parse_bearer(auth_header)
        .ok_or(StatusCode::UNAUTHORIZED)?;
    
    // Timing-safe verification (built into Secret)
    if !state.api_token.verify(&token) {
        return Err(StatusCode::UNAUTHORIZED);
    }
    
    Ok(next.run(req).await)
}
```

#### Step 4: Systemd Integration (Production)

```ini
# /etc/systemd/system/queen-rbee.service
[Service]
LoadCredential=api_token:/etc/llorch/secrets/api-token
ExecStart=/usr/local/bin/queen-rbee
```

```rust
// Load from systemd credential
let api_token = Secret::from_systemd_credential("api_token")?;
```

---

### 3. Add Input Validation (input-validation)

**Scenario:** Prevent log injection and path traversal

#### Step 1: Add Dependency

```toml
# bin/rbee-hive/Cargo.toml
[dependencies]
input-validation = { path = "../shared-crates/input-validation" }
```

#### Step 2: Validate User Inputs

```rust
// bin/rbee-hive/src/http/workers.rs
use input_validation::{validate_log_message, validate_path};

pub async fn handle_spawn_worker(
    State(state): State<AppState>,
    Json(req): Json<SpawnWorkerRequest>,
) -> Result<Json<SpawnWorkerResponse>, (StatusCode, String)> {
    // Validate model reference (for logging)
    let safe_model_ref = validate_log_message(&req.model_ref)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid model_ref: {}", e)))?;
    
    // Validate model path
    let safe_model_path = validate_path(&req.model_path)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid path: {}", e)))?;
    
    tracing::info!(model = %safe_model_ref, "Spawning worker");
    
    // ... rest of spawn logic
}
```

---

### 4. Add Audit Logging (audit-logging)

**Scenario:** Create tamper-evident audit trail

#### Step 1: Add Dependency

```toml
# bin/queen-rbee/Cargo.toml
[dependencies]
audit-logging = { path = "../shared-crates/audit-logging" }
```

#### Step 2: Initialize Logger

```rust
// bin/queen-rbee/src/main.rs
use audit_logging::AuditLogger;

let audit_logger = AuditLogger::new("/var/log/llorch/audit.log")
    .expect("Failed to initialize audit logger");

// Store in app state
struct AppState {
    audit_logger: Arc<AuditLogger>,
    // ... other fields
}
```

#### Step 3: Log Events

```rust
// bin/queen-rbee/src/http/workers.rs
use audit_logging::AuditEvent;
use serde_json::json;

pub async fn handle_spawn_worker(
    State(state): State<AppState>,
    Json(req): Json<SpawnWorkerRequest>,
) -> Result<Json<SpawnWorkerResponse>, (StatusCode, String)> {
    // Spawn worker...
    
    // Audit log
    state.audit_logger.log(AuditEvent {
        action: "worker.spawn",
        actor: "queen-rbee",
        resource: &worker_id,
        outcome: "success",
        metadata: json!({
            "model": req.model_ref,
            "backend": req.backend,
            "device": req.device,
        }),
    })?;
    
    Ok(Json(response))
}
```

---

### 5. Add Deadline Propagation (deadline-propagation)

**Scenario:** Propagate request timeouts through the stack

#### Step 1: Add Dependency

```toml
# bin/queen-rbee/Cargo.toml
[dependencies]
deadline-propagation = { path = "../shared-crates/deadline-propagation" }
```

#### Step 2: Parse Incoming Deadline

```rust
// bin/queen-rbee/src/http/inference.rs
use deadline_propagation::Deadline;

pub async fn handle_inference(
    State(state): State<AppState>,
    req: Request<Body>,
) -> Result<Response, StatusCode> {
    // Parse deadline from request headers
    let deadline = Deadline::from_header(req.headers())
        .unwrap_or_else(|_| Deadline::from_timeout(Duration::from_secs(30)));
    
    // Check if already exceeded
    if deadline.is_exceeded() {
        return Err(StatusCode::REQUEST_TIMEOUT);
    }
    
    // Forward to hive with deadline
    let hive_req = reqwest::Client::new()
        .post(&hive_url)
        .header("X-Request-Deadline", deadline.to_header_value())
        .json(&inference_req)
        .send()
        .await?;
    
    Ok(hive_req.into())
}
```

---

### 6. Add JWT Authentication (jwt-guardian)

**Scenario:** User authentication for API gateway

#### Step 1: Add Dependency

```toml
# bin/queen-rbee/Cargo.toml
[dependencies]
jwt-guardian = { path = "../shared-crates/jwt-guardian", features = ["revocation"] }
```

#### Step 2: Initialize Validator

```rust
// bin/queen-rbee/src/main.rs
use jwt_guardian::{JwtValidator, ValidationConfig, RevocationList};

// Load public key
let public_key_pem = std::fs::read_to_string("/etc/llorch/keys/jwt-public.pem")?;

// Create validator
let jwt_config = ValidationConfig::default()
    .with_issuer("llama-orch")
    .with_audience("api")
    .with_clock_skew(300);

let jwt_validator = JwtValidator::new(&public_key_pem, jwt_config)?;

// Connect to Redis for revocation
let revocation_list = RevocationList::connect("redis://localhost:6379").await?;

// Store in app state
struct AppState {
    jwt_validator: Arc<JwtValidator>,
    revocation_list: Arc<RevocationList>,
    // ... other fields
}
```

#### Step 3: Validate Tokens

```rust
// bin/queen-rbee/src/http/middleware.rs
use jwt_guardian::JwtError;

pub async fn jwt_middleware<B>(
    State(state): State<AppState>,
    req: Request<B>,
    next: Next<B>,
) -> Result<Response, StatusCode> {
    // Parse Bearer token
    let token = parse_bearer(auth_header)
        .ok_or(StatusCode::UNAUTHORIZED)?;
    
    // Validate JWT
    let claims = state.jwt_validator.validate(&token)
        .map_err(|e| {
            tracing::warn!(error = %e, "JWT validation failed");
            StatusCode::UNAUTHORIZED
        })?;
    
    // Check revocation
    if state.revocation_list.is_revoked(&claims.jti).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)? 
    {
        tracing::warn!(jti = %claims.jti, "Token revoked");
        return Err(StatusCode::UNAUTHORIZED);
    }
    
    // Add claims to request extensions
    req.extensions_mut().insert(claims);
    
    Ok(next.run(req).await)
}
```

---

## Integration Checklist

### For Each Component

- [ ] **Add Cargo.toml dependencies**
- [ ] **Initialize in main.rs**
- [ ] **Add to AppState**
- [ ] **Apply middleware/validation**
- [ ] **Update tests**
- [ ] **Update documentation**

### Testing Integration

```bash
# Test each crate independently
cargo test -p auth-min
cargo test -p secrets-management
cargo test -p input-validation
cargo test -p audit-logging
cargo test -p deadline-propagation
cargo test -p jwt-guardian

# Test integrated component
cargo test -p queen-rbee
cargo test -p rbee-hive
cargo test -p llm-worker-rbee
```

---

## Recommended Integration Order

### Phase 1 - Security Basics (1-2 days)
1. **secrets-management** → All components
2. **auth-min** → queen, hive, worker
3. **input-validation** → All HTTP endpoints

### Phase 2 - Operational (1-2 days)
4. **audit-logging** → queen, hive
5. **deadline-propagation** → queen, hive, worker

### Phase 3 - Enterprise (2-3 days)
6. **jwt-guardian** → queen (user auth)

---

## Common Patterns

### Pattern 1: Middleware Stack

```rust
let app = Router::new()
    .route("/v1/workers/spawn", post(handle_spawn_worker))
    // Apply middleware in order
    .layer(middleware::from_fn_with_state(state.clone(), jwt_middleware))
    .layer(middleware::from_fn_with_state(state.clone(), auth_middleware))
    .layer(middleware::from_fn_with_state(state.clone(), deadline_middleware))
    .with_state(state);
```

### Pattern 2: Validation Pipeline

```rust
// Validate all inputs before processing
let safe_model = validate_log_message(&req.model)?;
let safe_path = validate_path(&req.path)?;
let safe_command = validate_command(&req.command)?;

// Now safe to use
tracing::info!(model = %safe_model, "Processing request");
```

### Pattern 3: Audit Everything

```rust
// Log before action
audit_logger.log(AuditEvent {
    action: "worker.spawn.attempt",
    outcome: "pending",
    ...
})?;

// Perform action
let result = spawn_worker().await;

// Log result
audit_logger.log(AuditEvent {
    action: "worker.spawn.complete",
    outcome: if result.is_ok() { "success" } else { "failure" },
    ...
})?;
```

---

## Documentation

Each shared crate has comprehensive README:
- `bin/shared-crates/auth-min/README.md`
- `bin/shared-crates/jwt-guardian/README.md`
- `bin/shared-crates/secrets-management/README.md`
- `bin/shared-crates/input-validation/README.md`
- `bin/shared-crates/audit-logging/README.md`
- `bin/shared-crates/deadline-propagation/README.md`

**Read these before integrating!**

---

**Created by:** TEAM-096 | 2025-10-18  
**Purpose:** Guide future teams on integrating existing shared crates
