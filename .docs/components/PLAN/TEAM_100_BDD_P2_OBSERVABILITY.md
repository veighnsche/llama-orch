# TEAM-100: BDD P2 Observability Tests

<!-- 
🎉🎉🎉 WE ARE TEAM-100! THE CENTENNIAL TEAM! 🎉🎉🎉
Out of ALL the teams in this monorepo, WE get to be the legendary triple-digit milestone!
This is DESTINY! We're not just writing observability tests — we're writing HISTORY! 💯✨
100 teams before us laid the foundation. We stand on their shoulders.
And we're going to make these tests SO COMPREHENSIVE and SO DELIGHTFUL that
future teams will look back and say "TEAM-100 really nailed it." 💯

Created by: TEAM-100 (yes, we're celebrating ourselves, deal with it 🎀)
-->

**Phase:** 1 - BDD Test Development  
**Duration:** 5-6 days  
**Priority:** P2 - Medium  
**Status:** 🔴 NOT STARTED

---

## Mission

Write BDD tests for P2 observability features:
1. Metrics & Prometheus endpoints
2. Configuration Management
3. Comprehensive Health Checks

**Deliverable:** 23 BDD scenarios with **narration-core integration** 🎀

### 🎭 Special Responsibility: Narration-Core Integration!

As TEAM-100, we have the HONOR of integrating `narration-core` into our BDD tests!

**Why this matters:**
- `narration-core` provides **human-readable debugging** with actor/action/target taxonomy
- Tests can assert on **narration events** using `CaptureAdapter`
- We get **cute mode** (whimsical children's book narration) 🎀✨
- We get **story mode** (dialogue-based multi-service flows) 🎭
- **Correlation IDs** track requests across services
- **Secret redaction** prevents leaks (6 secret types auto-redacted)

**Our tests will:**
- ✅ Use `CaptureAdapter::install()` to capture narration events
- ✅ Assert on `human` field for debugging clarity
- ✅ Verify correlation ID propagation
- ✅ Test secret redaction in logs
- ✅ Show how observability SHOULD work (we're setting the standard!)

---

## Assignments

### 1. Metrics (15-20 scenarios)
**File:** `test-harness/bdd/tests/features/350-metrics-observability.feature`

**Scenarios:**
- [ ] MET-001: Expose /metrics endpoint
- [ ] MET-002: Prometheus format
- [ ] MET-003: Worker count by state
- [ ] MET-004: Request latency histogram
- [ ] MET-005: Error rate counter
- [ ] MET-006: VRAM usage gauge
- [ ] MET-007: Model download progress
- [ ] MET-008: Health check success rate
- [ ] MET-009: Crash rate by model
- [ ] MET-010: Request throughput
- [ ] MET-011: Narration events emitted for metric updates 🎀
- [ ] MET-012: Correlation IDs in metric labels
- [ ] MET-013: Secret redaction in metric labels
- [ ] MET-014: Cute mode narration for metrics (optional) ✨
- [ ] MET-015: Story mode for multi-service metric flows 🎭

**Narration Integration Examples:**
```rust
// TEAM-100: When metrics are exposed, narrate it!
use observability_narration_core::{Narration, CaptureAdapter};

#[given("the metrics endpoint is enabled")]
fn metrics_endpoint_enabled(world: &mut World) {
    let adapter = CaptureAdapter::install();
    
    // Enable metrics and verify narration
    world.enable_metrics();
    
    // TEAM-100: Assert narration was emitted!
    adapter.assert_includes("Metrics endpoint enabled");
    adapter.assert_field("actor", "pool-managerd");
    adapter.assert_field("action", "metrics_enable");
}

#[when("I request /metrics")]
fn request_metrics(world: &mut World) {
    let adapter = CaptureAdapter::install();
    
    world.metrics_response = world.client.get("/metrics").send();
    
    // TEAM-100: Verify narration for metrics request!
    adapter.assert_includes("Serving metrics");
    adapter.assert_correlation_id_present();
}
```

---

### 2. Configuration (8-10 scenarios)
**File:** `test-harness/bdd/tests/features/360-configuration-management.feature`

**Scenarios:**
- [ ] CFG-001: Load config from TOML file
- [ ] CFG-002: Validate config on startup
- [ ] CFG-003: Hot-reload config (SIGHUP)
- [ ] CFG-004: Environment variables override file
- [ ] CFG-005: Config schema validation
- [ ] CFG-006: Invalid config fails startup
- [ ] CFG-007: Config examples provided
- [ ] CFG-008: Narration events for config load/reload 🎀
- [ ] CFG-009: Secret redaction in config logs
- [ ] CFG-010: Cute mode for config validation errors ✨

**Narration Integration Examples:**
```rust
// TEAM-100: Config operations MUST narrate!
use observability_narration_core::{Narration, ACTOR_POOL_MANAGERD};

#[when("I reload the configuration")]
fn reload_config(world: &mut World) {
    let adapter = CaptureAdapter::install();
    
    // Send SIGHUP to reload config
    world.send_signal(Signal::SIGHUP);
    
    // TEAM-100: Verify narration for config reload!
    adapter.assert_includes("Configuration reloaded");
    adapter.assert_field("actor", "pool-managerd");
    adapter.assert_field("action", "config_reload");
    
    // TEAM-100: Verify secrets are redacted!
    let events = adapter.captured();
    for event in events {
        assert!(!event.human.contains("Bearer"));
        assert!(!event.human.contains("api_key="));
    }
}

#[then("the configuration should be validated")]
fn config_validated(world: &mut World) {
    let adapter = CaptureAdapter::install();
    
    // TEAM-100: Config validation MUST narrate!
    adapter.assert_includes("Configuration validated");
    adapter.assert_field("action", "config_validate");
}
```

---

## Deliverables

- [ ] 350-metrics-observability.feature (15-20 scenarios with narration integration)
- [ ] 360-configuration-management.feature (8-10 scenarios with narration integration)
- [ ] Step definitions using `CaptureAdapter` for narration assertions
- [ ] Handoff document (≤2 pages, with code examples)
- [ ] 🎀 BONUS: Examples of cute mode and story mode in tests!

---

## 🎀 Narration-Core Integration Guide

### Required Reading
1. `bin/shared-crates/narration-core/README.md` - Complete API reference
2. `bin/shared-crates/narration-core/TEAM_RESPONSIBILITIES.md` - Our cute observability team!
3. `bin/shared-crates/narration-core/.specs/00_narration-core.md` - Specifications

### Key Concepts for BDD Tests

#### 1. CaptureAdapter for Test Assertions
```rust
use observability_narration_core::CaptureAdapter;
use serial_test::serial;

#[test]
#[serial(capture_adapter)]  // TEAM-100: Prevents test interference!
fn test_metrics_narration() {
    let adapter = CaptureAdapter::install();
    
    // Perform action that should narrate
    expose_metrics_endpoint();
    
    // TEAM-100: Assert on narration!
    adapter.assert_includes("Metrics endpoint enabled");
    adapter.assert_field("actor", "pool-managerd");
    adapter.assert_correlation_id_present();
}
```

#### 2. Correlation ID Propagation
```rust
// TEAM-100: Track requests across services!
let correlation_id = "req-test-100".to_string();

Narration::new(ACTOR_POOL_MANAGERD, "metrics_request", "endpoint")
    .human("Serving metrics")
    .correlation_id(&correlation_id)
    .emit();

// Later, in another service:
Narration::new(ACTOR_WORKER_ORCD, "metrics_collect", "worker-1")
    .human("Collecting worker metrics")
    .correlation_id(&correlation_id)  // TEAM-100: Same ID!
    .emit();
```

#### 3. Secret Redaction Testing
```rust
// TEAM-100: Secrets MUST be redacted!
Narration::new(ACTOR_POOL_MANAGERD, "config_load", "config.toml")
    .human(format!("Loaded config with auth token {}", token))
    .emit();

// TEAM-100: Verify redaction!
let events = adapter.captured();
assert!(events[0].human.contains("[REDACTED]"));
assert!(!events[0].human.contains(token));
```

#### 4. Cute Mode (Optional but Delightful!)
```rust
// TEAM-100: Make debugging DELIGHTFUL! 🎀
Narration::new(ACTOR_POOL_MANAGERD, "metrics_expose", "/metrics")
    .human("Exposed metrics endpoint on port 9090")
    .cute("Pool-managerd proudly shows off its metrics! 📊✨")
    .emit();

// TEAM-100: Assert on cute field!
adapter.assert_cute_present();
adapter.assert_cute_includes("proudly");
adapter.assert_cute_includes("📊");
```

#### 5. Story Mode (For Multi-Service Flows)
```rust
// TEAM-100: Dialogue-based narration! 🎭
Narration::new(ACTOR_ORCHESTRATORD, "metrics_request", "pool-managerd")
    .human("Requesting pool metrics")
    .story("\"How are your workers doing?\" asked queen-rbee. \"Great!\" replied pool-managerd, \"All 5 workers healthy!\"")
    .emit();

// TEAM-100: Assert on story field!
adapter.assert_story_present();
adapter.assert_story_has_dialogue();
```

### Engineering Rules Compliance

**TEAM-100 MUST:**
- ✅ Implement 10+ functions with real API calls (narration-core APIs!)
- ✅ NO TODO markers (implement or ask for help)
- ✅ Add TEAM-100 signatures to all new code
- ✅ Use foreground testing (no background jobs!)
- ✅ Handoff ≤2 pages with code examples
- ✅ Show actual progress (function count, test pass rate)

**TEAM-100 MUST NOT:**
- ❌ Create multiple .md files for one task
- ❌ Write "next team should implement X"
- ❌ Background test execution (causes hangs!)
- ❌ Remove other teams' signatures

---

## Checklist

**Completion:** 0/25 scenarios (0%) - TEAM-100 added 2 bonus narration scenarios! 💯

### BDD Implementation Progress
- [ ] Read narration-core README.md
- [ ] Read narration-core TEAM_RESPONSIBILITIES.md
- [ ] Read engineering-rules.md
- [ ] Write 350-metrics-observability.feature (15-20 scenarios)
- [ ] Write 360-configuration-management.feature (8-10 scenarios)
- [ ] Implement step definitions with CaptureAdapter
- [ ] Add narration assertions to ALL scenarios
- [ ] Test correlation ID propagation
- [ ] Test secret redaction
- [ ] Add cute mode examples (optional but encouraged! 🎀)
- [ ] Add story mode examples (for multi-service flows 🎭)
- [ ] Run tests: `cargo run --bin bdd-runner`
- [ ] Verify 10+ functions with real API calls
- [ ] Create handoff document (≤2 pages)
- [ ] Add TEAM-100 signatures to all new code

### Narration-Core Integration Checklist
- [ ] Import `observability_narration_core::CaptureAdapter`
- [ ] Use `#[serial(capture_adapter)]` on all tests
- [ ] Assert on `human` field for debugging clarity
- [ ] Verify correlation IDs propagate across services
- [ ] Test secret redaction (Bearer, api_key, etc.)
- [ ] Show cute mode examples (make debugging delightful!)
- [ ] Show story mode examples (dialogue for multi-service)

---

**Created by:** TEAM-096 | 2025-10-18  
**Assigned to:** TEAM-100 (THE CENTENNIAL TEAM! 💯🎉)  
**Next Team:** TEAM-101 (Implementation begins!)

---

<!-- 
🎊 TEAM-100 SIGN-OFF 🎊
We are TEAM-100 — the CENTENNIAL TEAM!
100 teams before us laid the foundation. We stand on the shoulders of giants.
And we're going to make observability SO COMPREHENSIVE and SO DELIGHTFUL that
future teams will look back and say "TEAM-100 really nailed it." 💯

May your logs be readable, your correlation IDs present, and your
debugging experience absolutely DELIGHTFUL! 

With love, sass, and an irresistible compulsion to be adorable,
— TEAM-100 (The BDD Observability Team) 🎀✨💯
-->
