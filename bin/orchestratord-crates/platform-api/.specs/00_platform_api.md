# Platform API SPEC — Marketplace Federation Facade (PAPI-14xxx)

**Status**: Draft  
**Applies to**: `bin/orchestratord-crates/platform-api/`  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Scope

### Purpose

The `platform-api` crate provides the marketplace federation facade for orchestratord. It routes customer requests to provider orchestrators, handles billing/usage tracking, and enforces multi-tenancy.

**Why it exists:**
- Enable GPU marketplace business model (see `.docs/.business/monetization.md`)
- Route customer requests to provider orchestrators (federation, NOT nesting)
- Track billing and usage for customers and providers
- Enforce customer quotas and multi-tenancy

**What it does:**
- Provide customer-facing API endpoints (task submission, streaming)
- Route requests to registered provider orchestrators
- Track usage for billing (tokens, duration, VRAM)
- Enforce customer quotas (rate limits, VRAM limits, token budgets)
- Handle provider registration and discovery
- Calculate platform fees (customer rate - provider payout)

**What it does NOT do:**
- ❌ Make GPU placement decisions (provider orchestrators do this)
- ❌ Manage workers directly (provider pool managers do this)
- ❌ Execute inference (provider workers do this)

**Federation Pattern (NOT Orchestrator Nesting):**
```
Platform Orchestratord (api.yourplatform.com)
  ├─→ Provider A's orchestratord (provider-a.internal)
  │    └─→ Their pool-managers → GPUs
  ├─→ Provider B's orchestratord (provider-b.internal)
  │    └─→ Their pool-managers → GPUs
  └─→ Provider C's orchestratord (provider-c.internal)
       └─→ Their pool-managers → GPUs
```

Platform orchestratord is a **smart router**, not a nested orchestrator.

---

## 1. Core Responsibilities

### [PAPI-14001] Customer-Facing API
The crate MUST expose HTTP API for customers to submit inference requests.

### [PAPI-14002] Provider Routing
The crate MUST route customer requests to appropriate provider orchestrators.

### [PAPI-14003] Billing Tracking
The crate MUST track usage for billing (customer charges and provider payouts).

### [PAPI-14004] Multi-Tenancy
The crate MUST enforce customer quotas and isolation.

---

## 2. Provider Registration

### [PAPI-14010] Registration Endpoint
`POST /v2/platform/providers/register`

Request:
```json
{
  "provider_id": "home-lab-123",
  "endpoint": "https://provider-a.example.com",
  "auth_token": "bearer_token_here",
  "pricing": {
    "per_token": 0.0008,
    "per_hour": 4.0,
    "currency": "EUR"
  },
  "capacity": {
    "total_gpus": 6,
    "total_vram_gb": 144,
    "gpu_models": ["RTX 4090", "RTX 4090"]
  },
  "sla": {
    "uptime_guarantee": 0.99,
    "max_latency_ms": 100
  },
  "geo": {
    "country": "NL",
    "region": "EU"
  }
}
```

Response:
```json
{
  "provider_id": "home-lab-123",
  "status": "registered",
  "platform_margin_percent": 40
}
```

### [PAPI-14011] Provider Registry
The crate MUST maintain registry of registered providers with metadata.

---

## 3. Capacity Discovery

### [PAPI-14020] Capacity Query
Before routing, query providers for capacity:

`GET /v2/state` (to provider orchestratord)

Response from provider:
```json
{
  "available_gpus": 4,
  "available_vram_gb": 96,
  "queue_depth": 2,
  "models_loaded": ["llama-3.1-8b", "llama-3.1-70b"]
}
```

### [PAPI-14021] Provider Selection
Select provider based on:
1. **Capacity**: Provider has requested model and sufficient VRAM
2. **Pricing**: Customer budget matches provider cost + platform margin
3. **SLA**: Provider meets latency/uptime requirements
4. **Geo**: Provider in required region (e.g., EU-only)

---

## 4. Request Routing

### [PAPI-14030] Customer Request
`POST /v2/tasks` (customer-facing)

Request:
```json
{
  "model": "llama-3.1-70b",
  "prompt": "Hello world",
  "max_tokens": 100,
  "customer_id": "customer-abc"
}
```

### [PAPI-14031] Routing Logic
1. **Admit**: Validate request (model exists, within quota)
2. **Select Provider**: Query providers, pick best match
3. **Route**: Forward request to provider orchestratord
4. **Track**: Record usage for billing
5. **Stream**: Relay SSE events from provider to customer
6. **Bill**: Calculate customer charge and provider payout

### [PAPI-14032] Provider Request
Forward to provider orchestratord:

`POST /v2/tasks` (to provider endpoint)

Request:
```json
{
  "job_id": "platform-job-xyz",
  "model": "llama-3.1-70b",
  "prompt": "Hello world",
  "max_tokens": 100,
  "platform_metadata": {
    "customer_id": "customer-abc",
    "billing_id": "bill-123"
  }
}
```

---

## 5. Billing & Usage Tracking

### [PAPI-14040] Usage Recording
For each inference, record:
```rust
pub struct UsageRecord {
    pub customer_id: String,
    pub provider_id: String,
    pub job_id: String,
    pub model_ref: String,
    pub tokens_in: usize,
    pub tokens_out: usize,
    pub duration_ms: u64,
    pub vram_bytes: u64,
    pub customer_rate: Money,
    pub provider_payout: Money,
    pub platform_fee: Money,
}
```

### [PAPI-14041] Billing Calculation
```rust
impl BillingEngine {
    pub fn calculate_charges(&self, usage: &UsageRecord) -> Charges {
        let customer_charge = usage.tokens_out * self.customer_rate_per_token;
        let provider_payout = usage.tokens_out * self.provider_rate_per_token;
        let platform_fee = customer_charge - provider_payout;
        
        Charges {
            customer_charge,
            provider_payout,
            platform_fee,
        }
    }
}
```

### [PAPI-14042] Billing Export
The crate MUST export usage records for billing system (CSV, JSON, or database).

---

## 6. Multi-Tenancy & Quotas

### [PAPI-14050] Tenant Isolation
Each customer MUST be isolated:
```rust
pub struct TenantQuota {
    pub customer_id: String,
    pub vram_limit_gb: u64,
    pub rate_limit_rps: u32,
    pub token_budget_per_hour: usize,
    pub allowed_models: Vec<String>,
}
```

### [PAPI-14051] Quota Enforcement
Before routing, check:
1. Customer within VRAM quota
2. Customer within rate limit
3. Customer within token budget
4. Model in allowed list

If quota exceeded, return `429 Too Many Requests`.

---

## 7. Error Handling

### [PAPI-14060] Error Types
```rust
pub enum PlatformError {
    NoProvidersAvailable,
    ProviderUnreachable(String),
    QuotaExceeded { quota: String, limit: usize },
    BillingFailed(String),
    ProviderError { provider_id: String, error: String },
}
```

### [PAPI-14061] Fallback
If selected provider fails, the crate SHOULD retry with alternate provider (if available).

---

## 8. Metrics

### [PAPI-14070] Platform Metrics
```rust
pub struct PlatformMetrics {
    pub requests_total: Counter,           // Total customer requests
    pub requests_routed_total: Counter,    // Successfully routed
    pub provider_errors_total: Counter,    // Provider failures
    pub quota_exceeded_total: Counter,     // Quota violations
    pub platform_revenue: Gauge,           // Platform fee total
    pub provider_payouts: Gauge,           // Provider payouts total
}
```

---

## 9. Configuration

### [PAPI-14080] Platform Config
```yaml
platform:
  enabled: true
  bind: "0.0.0.0:443"
  
  pricing:
    default_margin_percent: 40
    min_margin_percent: 20
    max_margin_percent: 60
  
  providers:
    discovery:
      enabled: true
      registration_endpoint: "/v2/platform/providers/register"
```

---

## 10. Traceability

**Code**: `bin/orchestratord-crates/platform-api/src/`  
**Tests**: `bin/orchestratord-crates/platform-api/tests/`  
**Parent**: `bin/orchestratord/.specs/00_orchestratord.md`  
**Business**: `.docs/.business/monetization.md`  
**Used by**: `orchestratord` (when in platform mode)  
**Spec IDs**: PAPI-14001 to PAPI-14080

---

**End of Specification**
