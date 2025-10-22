# Testing Priorities - Visual Guide

**Quick visual reference for testing priorities**

---

## Priority Matrix

```
┌─────────────────────────────────────────────────────────────────┐
│ PRIORITY 1: CRITICAL PATH (START HERE)                          │
├─────────────────────────────────────────────────────────────────┤
│ 🔴 SSE Channel Lifecycle                                        │
│    - Memory leaks, race conditions                              │
│    - 15-20 tests, 5-7 days                                      │
│                                                                  │
│ 🔴 Concurrent Access (Reasonable Scale: 5-10 concurrent)        │
│    - Job-registry, hive-registry                                │
│    - 20-30 tests, 7-10 days                                     │
│                                                                  │
│ 🔴 Stdio::null() Behavior (CRITICAL - E2E tests hang without)   │
│    - Daemon lifecycle                                           │
│    - 5-10 tests, 2-3 days                                       │
│                                                                  │
│ 🔴 Timeout Propagation                                          │
│    - All layers (keeper, queen, hive)                           │
│    - 15-20 tests, 5-7 days                                      │
│                                                                  │
│ 🔴 Resource Cleanup                                             │
│    - Disconnect, crash, timeout scenarios                       │
│    - 20-25 tests, 7-10 days                                     │
│                                                                  │
│ EFFORT: 40-60 days (1 dev) or 2-3 weeks (3 devs)               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ PRIORITY 2: MEDIUM PRIORITY                                     │
├─────────────────────────────────────────────────────────────────┤
│ 🟡 SSH Client (0% coverage)                                     │
│    - Pre-flight, TCP, handshake, auth, command                 │
│    - 15 tests, 5-7 days                                         │
│                                                                  │
│ 🟡 Binary Resolution                                            │
│    - Hive-lifecycle (config → debug → release)                 │
│    - 6 tests, 1-2 days                                          │
│                                                                  │
│ 🟡 Graceful Shutdown                                            │
│    - SIGTERM → wait → SIGKILL                                   │
│    - 4 tests, 2-3 days                                          │
│                                                                  │
│ 🟡 Capabilities Cache                                           │
│    - Hit, miss, refresh, staleness                             │
│    - 6 tests, 2-3 days                                          │
│                                                                  │
│ 🟡 Error Propagation                                            │
│    - All boundaries (keeper↔queen↔hive)                        │
│    - 25-30 tests, 7-10 days                                     │
│                                                                  │
│ EFFORT: 30-40 days (1 dev) or 2-3 weeks (3 devs)               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ PRIORITY 3: LOW PRIORITY                                        │
├─────────────────────────────────────────────────────────────────┤
│ 🟢 Format String Edge Cases                                     │
│    - Quotes, newlines, unicode, emojis                          │
│    - 5-10 tests, 1-2 days                                       │
│                                                                  │
│ 🟢 Table Formatting Edge Cases                                  │
│    - Nested objects, large arrays, overflow                     │
│    - 7-10 tests, 2-3 days                                       │
│                                                                  │
│ 🟢 Config Corruption Handling                                   │
│    - Truncated, invalid UTF-8, partial write                    │
│    - 4-6 tests, 1-2 days                                        │
│                                                                  │
│ 🟢 Correlation ID Validation                                    │
│    - UUID format, uniqueness                                    │
│    - 3-5 tests, 0.5-1 days                                      │
│                                                                  │
│ EFFORT: 20-30 days (1 dev) or 1-2 weeks (3 devs)               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Priority Map

```
HIGH PRIORITY (Start Here):
├─ SSH Client (0% coverage) ..................... 15 tests, 5-7 days
├─ Daemon Lifecycle (Stdio::null()) ............. 10 tests, 3-5 days
├─ Hive Registry (concurrent access) ............ 20 tests, 7-10 days
├─ Job Registry (concurrent access) ............. 15 tests, 5-7 days
├─ Narration (SSE channel lifecycle) ............ 20 tests, 7-10 days
└─ Keeper↔Queen Integration ..................... 40 tests, 30-40 days

MEDIUM PRIORITY:
├─ Hive Lifecycle (binary, health, cache) ....... 25 tests, 10-15 days
├─ Config Loading (edge cases) .................. 15 tests, 5-7 days
├─ Heartbeat (background tasks, retry) .......... 15 tests, 5-7 days
└─ Queen↔Hive Integration ....................... 30 tests, 25-35 days

LOW PRIORITY:
├─ Narration (format strings, tables) ........... 20 tests, 7-10 days
├─ Config (corruption handling) ................. 6 tests, 1-2 days
└─ Timeout Enforcer (countdown, TTY) ............ 5 tests, 1-2 days

NOT IMPLEMENTED (Don't Test Yet):
├─ Worker operations ............................ N/A
├─ Inference flow ............................... N/A
└─ Model provisioning ........................... N/A
```

---

## Reasonable Scale Guidelines

```
✅ NUC-FRIENDLY SCALE:
┌────────────────────────────────────┐
│ Concurrent Operations:    5-10     │
│ Jobs/Hives/Workers:       100      │
│ Payload Size:             1MB      │
│ Workers per Hive:         5        │
│ SSE Channels:             10       │
└────────────────────────────────────┘

❌ OVERKILL SCALE (Don't Do This):
┌────────────────────────────────────┐
│ Concurrent Operations:    100+     │
│ Jobs/Hives/Workers:       1000+    │
│ Payload Size:             10MB+    │
│ Workers per Hive:         50+      │
│ SSE Channels:             100+     │
└────────────────────────────────────┘
```

---

## Test Implementation Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Pick Component                                      │
├─────────────────────────────────────────────────────────────┤
│ Start with HIGH priority:                                   │
│ • SSH Client (0% coverage)                                  │
│ • Stdio::null() (CRITICAL)                                  │
│ • Concurrent access (job-registry, hive-registry)           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Read Component Docs                                 │
├─────────────────────────────────────────────────────────────┤
│ 1. Behavior inventory (TEAM-XXX document)                   │
│ 2. README.md in component folder                            │
│ 3. Existing tests (if any)                                  │
│ 4. TESTING_GUIDE.md (if exists)                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Write Tests                                         │
├─────────────────────────────────────────────────────────────┤
│ BDD Pattern (Gherkin):                                      │
│   Feature: SSH Connection Testing                           │
│     Scenario: SSH agent not running                         │
│       Given SSH_AUTH_SOCK is not set                        │
│       When I test SSH connection                            │
│       Then the result should be failure                     │
│                                                              │
│ Rust Pattern:                                               │
│   #[tokio::test]                                            │
│   async fn test_ssh_agent_not_running() { ... }            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: Run Tests                                           │
├─────────────────────────────────────────────────────────────┤
│ cargo test -p <crate-name>                                  │
│ cargo xtask bdd                                             │
│ cargo test --workspace                                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: Verify Coverage                                     │
├─────────────────────────────────────────────────────────────┤
│ ✓ All happy paths tested                                    │
│ ✓ All error paths tested                                    │
│ ✓ All edge cases tested                                     │
│ ✓ All concurrent scenarios tested                           │
│ ✓ All cleanup scenarios tested                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Critical Invariants Checklist

```
MUST TEST (These are CRITICAL):
┌────────────────────────────────────────────────────────────┐
│ ☐ job_id MUST propagate                                    │
│   Without it, narration doesn't reach SSE                  │
│                                                             │
│ ☐ [DONE] marker MUST be sent                               │
│   Keeper uses it to detect completion                      │
│                                                             │
│ ☐ Stdio::null() MUST be used                               │
│   Prevents pipe hangs in E2E tests                         │
│                                                             │
│ ☐ Timeouts MUST fire                                       │
│   Zero tolerance for hanging operations                    │
│                                                             │
│ ☐ Channels MUST be cleaned up                              │
│   Prevent memory leaks                                     │
└────────────────────────────────────────────────────────────┘
```

---

## Quick Decision Tree

```
Starting a new test?
│
├─ Is the feature IMPLEMENTED?
│  ├─ YES → Continue
│  └─ NO → STOP (don't test unimplemented features)
│
├─ Is the scale reasonable for a NUC?
│  ├─ YES (5-10 concurrent) → Continue
│  └─ NO (100+ concurrent) → STOP (reduce scale)
│
├─ Is it HIGH priority?
│  ├─ YES → Start immediately
│  └─ NO → Check if HIGH priority tests are done first
│
└─ Have you read the component docs?
   ├─ YES → Write the test!
   └─ NO → Read TEAM-XXX behavior inventory first
```

---

## Test Commands Quick Reference

```bash
# Run unit tests for a crate
cargo test -p <crate-name>

# Run BDD tests
cargo xtask bdd

# Run integration tests
cargo test --test <test-name>

# Run all tests
cargo test --workspace

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test <test-name>

# Run tests in parallel
cargo test -- --test-threads=4
```

---

## Common Pitfalls (Avoid These!)

```
❌ WRONG:
├─ Testing unimplemented features (worker ops, inference)
├─ Unrealistic scale (100+ concurrent, 1000+ jobs)
├─ Missing job_id propagation (events are dropped)
├─ Forgetting Stdio::null() (E2E tests hang)
└─ Not testing cleanup (memory leaks)

✅ RIGHT:
├─ Testing implemented features (hive ops, SSE, heartbeat)
├─ Reasonable scale (5-10 concurrent, 100 jobs)
├─ Including job_id in narration (events reach SSE)
├─ Using Stdio::null() for daemons (E2E tests work)
└─ Testing cleanup (normal, error, timeout, disconnect)
```

---

**For detailed information, see:**
- `TESTING_ENGINEER_GUIDE.md` - Complete guide (90 min)
- `TESTING_QUICK_START.md` - Quick start (5 min)
- `TESTING_GAPS_EXECUTIVE_SUMMARY.md` - Overview
