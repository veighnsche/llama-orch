# BDD Test Contradictions Analysis

**Generated:** 2025-10-18  
**By:** TEAM-112  
**Purpose:** Document actual contradictions found in BDD tests

---

## 🚨 CRITICAL CONTRADICTION: Port Allocation Conflict

### The Contradiction

**rbee-hive uses TWO DIFFERENT PORTS in different test files:**

| Port | Usage | Feature Files | Count |
|------|-------|---------------|-------|
| **9200** | rbee-hive HTTP server | 160-end-to-end-flows.feature, 910-full-stack-integration.feature, 920-integration-scenarios.feature | 3 files |
| **8081** | rbee-hive HTTP server | 040-worker-provisioning.feature, 080-rbee-hive-preflight-validation.feature, 140-input-validation.feature, 230-resource-management.feature, 300-authentication.feature, 340-deadline-propagation.feature | 6 files |

### Evidence

**Tests Using Port 9200:**
```gherkin
# 160-end-to-end-flows.feature line 50
And queen-rbee queries rbee-hive worker registry at "http://workstation.home.arpa:9200/v1/workers/list"

# 910-full-stack-integration.feature line 14
And rbee-hive is healthy at 'http://localhost:9200'

# 920-integration-scenarios.feature line 17
And rbee-hive-1 is running on port 9200
```

**Tests Using Port 8081:**
```gherkin
# 040-worker-provisioning.feature line 20
And rbee-hive is running at "http://localhost:8081"

# 140-input-validation.feature line 189
Given rbee-hive is running at "http://localhost:8081"

# 300-authentication.feature line 95
And rbee-hive is running with auth at "http://workstation.home.arpa:8081"
```

### Product Code Reality

**rbee-hive ACTUALLY uses port 9200:**

```rust
// bin/rbee-hive/src/http/metrics.rs line 51
let addr: std::net::SocketAddr = "127.0.0.1:9200".parse().unwrap();

// bin/rbee-hive/src/http/routes.rs line 115
let addr: std::net::SocketAddr = "127.0.0.1:9200".parse().unwrap();

// bin/rbee-hive/src/http/middleware/auth.rs line 113
server_addr: "127.0.0.1:9200".parse().unwrap(),
```

**Workers use port 8081+ (dynamic allocation):**

```rust
// bin/rbee-hive/src/http/workers.rs line 165
let mut port = 8081u16;  // Starting port for workers

// bin/rbee-hive/src/http/workers.rs line 174-180
// Find first unused port starting from 8081
while used_ports.contains(&port) {
    port += 1;
    if port > 9000 {
        return Err("No available ports (8081-9000 all in use)");
    }
}
```

### The Problem

**6 test files expect rbee-hive on port 8081, but:**
- Product code binds rbee-hive to port 9200
- Port 8081 is actually used for WORKERS, not rbee-hive
- This means 6 feature files have WRONG port assignments

### Impact

**Tests Affected:** ~50+ scenarios across 6 feature files

**Severity:** HIGH - Tests will fail to connect to rbee-hive

**Why Tests Still Pass:** 
- Step implementations likely use mocks/stubs
- Real HTTP connections not being made
- Tests passing for wrong reasons

---

## ⚠️ MEDIUM CONTRADICTION: Worker URL Format

### The Contradiction

**Worker URLs use INCONSISTENT port references:**

**In Product Code:**
```rust
// Workers are spawned on ports 8081+
// rbee-hive/src/http/workers.rs line 165
let mut port = 8081u16;

// But registry examples show:
url: "http://localhost:8081"  // First worker
url: "http://localhost:8082"  // Second worker (implied)
```

**In Tests:**
```gherkin
# 160-end-to-end-flows.feature line 64
And rbee-hive spawns worker process "llm-worker-rbee" on port 8001 with cuda device 1

# 910-full-stack-integration.feature line 15
And mock-worker is healthy at 'http://localhost:8001'
```

### Evidence

**Product code says workers start at 8081:**
- `bin/rbee-hive/src/http/workers.rs` line 165: `let mut port = 8081u16;`

**Tests say workers start at 8001:**
- `160-end-to-end-flows.feature` line 64: "port 8001"
- `910-full-stack-integration.feature` line 15: "port 8001"

### The Problem

**Contradiction:** Product code and tests disagree on worker port allocation

**Possible Explanations:**
1. Product code was changed from 8001 to 8081, tests not updated
2. Tests use different port range for isolation
3. Configuration allows different port ranges

**Impact:** Medium - Could cause confusion about actual port allocation

---

## ⚠️ MINOR CONTRADICTION: Bind Address Notation

### The Contradiction

**Tests use THREE different notations for the same thing:**

| Notation | Meaning | Files Using It |
|----------|---------|----------------|
| `http://localhost:8080` | Loopback only | 22 files (76%) |
| `http://0.0.0.0:8080` | All interfaces | 4 files (14%) |
| `http://127.0.0.1:8080` | Loopback (explicit) | 3 files (10%) |

### Evidence

**Most tests use localhost:**
```gherkin
And queen-rbee is running at "http://localhost:8080"
```

**Security tests use 0.0.0.0:**
```gherkin
# 300-authentication.feature line 21
Given queen-rbee is running with auth enabled at "http://0.0.0.0:8080"

# 330-audit-logging.feature line 22
Given queen-rbee is running at "http://0.0.0.0:8080"
```

**Some tests use 127.0.0.1:**
```gherkin
# 300-authentication.feature line 67
Given queen-rbee is running at "http://127.0.0.1:8080"
```

### The Problem

**Inconsistency:** Same concept expressed three different ways

**Why It Matters:**
- `localhost` and `127.0.0.1` are functionally equivalent
- `0.0.0.0` means "bind to all interfaces" (different behavior)
- Security tests NEED `0.0.0.0` to test public binding
- Other tests should use `localhost` for consistency

**Impact:** Low - Semantically correct but stylistically inconsistent

---

## 📊 Summary of Contradictions

| Contradiction | Severity | Files Affected | Impact |
|---------------|----------|----------------|--------|
| **rbee-hive port (8081 vs 9200)** | 🚨 HIGH | 6 files, ~50 scenarios | Tests expect wrong port |
| **Worker port (8001 vs 8081)** | ⚠️ MEDIUM | 2 files, ~10 scenarios | Port allocation confusion |
| **Bind address notation** | ⚠️ LOW | 29 files | Style inconsistency only |

---

## 🔧 Recommended Fixes

### Fix 1: Correct rbee-hive Port in Tests (HIGH PRIORITY)

**Files to Update:**
1. `040-worker-provisioning.feature` line 20
2. `080-rbee-hive-preflight-validation.feature` line 23
3. `140-input-validation.feature` lines 189, 197, 204, 213, 221, 228, 251, 260, 267, 280
4. `230-resource-management.feature` line 19
5. `300-authentication.feature` lines 95, 220
6. `340-deadline-propagation.feature` line 19

**Change:**
```gherkin
# FROM:
And rbee-hive is running at "http://localhost:8081"

# TO:
And rbee-hive is running at "http://localhost:9200"
```

**Verification:**
Check product code confirms 9200:
- `bin/rbee-hive/src/http/metrics.rs` line 51
- `bin/rbee-hive/src/http/routes.rs` line 115

---

### Fix 2: Clarify Worker Port Allocation (MEDIUM PRIORITY)

**Options:**

**Option A:** Update product code to use 8001 (match tests)
```rust
// Change from:
let mut port = 8081u16;

// To:
let mut port = 8001u16;
```

**Option B:** Update tests to use 8081 (match product)
```gherkin
# Change from:
And rbee-hive spawns worker process "llm-worker-rbee" on port 8001

# To:
And rbee-hive spawns worker process "llm-worker-rbee" on port 8081
```

**Recommendation:** Option B (update tests) because:
- Product code is already implemented
- 8081 avoids conflict with common dev ports (8000, 8080)
- Tests should match product, not vice versa

---

### Fix 3: Standardize Bind Address Notation (LOW PRIORITY)

**Recommendation:**
- Use `http://localhost:8080` for general tests
- Use `http://0.0.0.0:8080` ONLY for security tests that need public binding
- Never use `http://127.0.0.1:8080` (redundant with localhost)

**Add comment when using 0.0.0.0:**
```gherkin
# Security test requires public bind to test auth properly
Given queen-rbee is running with auth enabled at "http://0.0.0.0:8080"
```

---

## 🎯 Root Cause Analysis

### Why These Contradictions Exist

**1. Port Confusion (8081 vs 9200):**
- **Root Cause:** Product code changed ports after tests were written
- **Evidence:** Comments in product code show TEAM-053 fixed port confusion
- **Quote:** "TEAM-053: Fixed port - rbee-hive uses 9200, not 8080"
- **Implication:** Tests were never updated after architectural change

**2. Worker Port Confusion (8001 vs 8081):**
- **Root Cause:** Different teams used different conventions
- **Evidence:** Integration tests use 8001, unit tests use 8081
- **Implication:** No single source of truth for port allocation

**3. Bind Address Inconsistency:**
- **Root Cause:** Different test authors, no style guide
- **Evidence:** Security tests intentionally use 0.0.0.0
- **Implication:** Inconsistency is partly intentional, partly accidental

---

## ✅ Verification Steps

### After Fixing Contradictions

1. **Update all 6 feature files** to use port 9200 for rbee-hive
2. **Run tests** to verify they still pass
3. **Check step implementations** to ensure they use correct ports
4. **Update documentation** to reflect correct port allocation
5. **Add port allocation comment** to Background section

**Example Background with Clarification:**
```gherkin
Background:
  # Port Allocation:
  #   queen-rbee: 8080 (orchestrator)
  #   rbee-hive: 9200 (pool manager)
  #   workers: 8081+ (dynamic allocation)
  Given the following topology:
    ...
```

---

## 📝 Conclusion

**Found 3 contradictions:**
1. 🚨 **HIGH:** rbee-hive port mismatch (6 files wrong)
2. ⚠️ **MEDIUM:** Worker port confusion (2 files inconsistent)
3. ⚠️ **LOW:** Bind address notation (style issue)

**Key Finding:** Tests were written before product code stabilized, then never updated when ports changed.

**Action Required:** Update 6 feature files to use correct rbee-hive port (9200)

---

**TEAM-112 Contradiction Analysis Complete**
