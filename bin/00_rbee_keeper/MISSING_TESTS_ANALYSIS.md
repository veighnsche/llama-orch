# rbee_keeper Missing Tests Analysis

**Date:** 2025-10-29  
**Analyzer:** TEAM-375  
**Status:** ANALYSIS COMPLETE

---

## Executive Summary

**Current Test Coverage:** ~15% (3 modules with tests)  
**Missing Test Coverage:** ~85% (15 modules without tests)  
**Total Test Files:** 1 integration test + 3 inline test modules + BDD tests  
**Recommendation:** Add unit tests for critical modules

---

## Current Test Inventory

### ✅ **Existing Tests**

#### 1. **Integration Tests** (tests/ directory)
- **tests/process_output_tests.rs** (106 LOC, 8 tests)
  - ✅ Process spawning with output streaming
  - ✅ stdout/stderr handling
  - ✅ Error exit codes
  - ✅ Nonexistent command handling
  - ✅ Long-running processes
  - ✅ Narration format output

#### 2. **Inline Unit Tests** (src/ modules)
- **src/ssh_resolver.rs** (3 tests)
  - ✅ `test_localhost_resolution()` - Localhost bypass
  - ✅ `test_parse_ssh_config()` - SSH config parsing
  - ✅ `test_missing_host()` - Error handling for missing hosts

- **src/platform/mod.rs** (3 tests)
  - ✅ `test_platform_paths_exist()` - Path resolution
  - ✅ `test_exe_extension()` - Platform-specific extensions
  - ✅ `test_ssh_support()` - SSH availability check
  - ⚠️ **NOTE:** Platform module scheduled for deletion (RULE ZERO)

- **src/tauri_commands.rs** (1 test)
  - ✅ `export_typescript_bindings()` - TypeScript type generation
  - ⚠️ Not a runtime test, build-time validation only

#### 3. **BDD Tests** (bdd/ directory)
- **bdd/tests/features/** (3 feature files)
  - Health check scenarios
  - SSE streaming scenarios
  - Integration scenarios
- ✅ End-to-end behavior testing
- ✅ Cucumber/Gherkin format

**Total Existing Tests:** ~15 tests (excluding BDD)

---

## Missing Tests by Module

### ❌ **CRITICAL - No Tests (High Priority)**

#### 1. **config.rs** (87 LOC) - Configuration Management
**Missing Tests:**
- [ ] Load config from valid TOML file
- [ ] Load config from missing file (creates default)
- [ ] Load config from invalid TOML (parse error)
- [ ] Save config to file
- [ ] Save config creates parent directories
- [ ] Validate config (valid cases)
- [ ] Validate config (invalid cases: bad URLs, ports)
- [ ] Config path resolution (~/.config/rbee/config.toml)
- [ ] Default config values

**Why Critical:**
- Configuration errors cause runtime failures
- User-facing file I/O (corruption risk)
- Validation logic needs verification

**Estimated Tests:** 10-12 tests

---

#### 2. **job_client.rs** (105 LOC) - HTTP Client
**Missing Tests:**
- [ ] Submit job and stream results (success)
- [ ] Submit job with timeout (30s enforcer)
- [ ] Submit job with connection failure
- [ ] Submit job with invalid operation
- [ ] Stream job results with [DONE] marker
- [ ] Stream job results with failure detection
- [ ] Stream job results with timeout
- [ ] submit_and_stream_job_to_hive (alias verification)
- [ ] Job failure tracking (job_failed flag)
- [ ] Narration emission during streaming

**Why Critical:**
- Core communication with queen-rbee
- Timeout handling is critical (hangs)
- Error detection affects user experience

**Estimated Tests:** 12-15 tests

**Note:** Uses shared `job-client` crate, but keeper-specific wrapper needs tests

---

#### 3. **tracing_init.rs** (315 LOC) - Tracing Setup
**Missing Tests:**
- [ ] init_cli_tracing() initializes stderr output
- [ ] init_gui_tracing() initializes dual layers
- [ ] StderrNarrationLayer formats messages correctly
- [ ] TauriNarrationLayer emits events to Tauri
- [ ] EventVisitor extracts "human" field
- [ ] EventVisitor extracts "action" field
- [ ] EventVisitor extracts "actor" field
- [ ] EventVisitor extracts "fn_name" field
- [ ] EventVisitor extracts "context" field
- [ ] EventVisitor extracts "target" field
- [ ] EventVisitor handles missing fields (fallback)
- [ ] EventVisitor removes debug quotes
- [ ] NarrationEvent serialization
- [ ] Narration mode switching (Human/Cute/Story)

**Why Critical:**
- Complex field extraction logic (150+ LOC)
- Bug fix history (TEAM-337) needs regression tests
- Dual-layer setup is fragile
- EventVisitor is the most complex code in keeper

**Estimated Tests:** 15-20 tests

**Complexity:** HIGH - EventVisitor has detailed field extraction logic

---

#### 4. **process_utils.rs** (89 LOC) - Process Streaming
**Status:** ✅ HAS INTEGRATION TESTS (tests/process_output_tests.rs)

**Additional Unit Tests Needed:**
- [ ] spawn_with_output_streaming() returns Child handle
- [ ] stream_child_output() handles already-spawned process
- [ ] stdout streaming task spawns correctly
- [ ] stderr streaming task spawns correctly
- [ ] Handles process with no stdout
- [ ] Handles process with no stderr

**Why Additional Tests:**
- Integration tests cover end-to-end, but not edge cases
- stream_child_output() has NO tests

**Estimated Tests:** 6-8 tests

---

### ⚠️ **MEDIUM PRIORITY - Handler Tests**

All handler modules are thin wrappers around shared crates, but still need tests:

#### 5. **handlers/queen.rs** (125 LOC)
**Missing Tests:**
- [ ] handle_queen(Start) calls start_daemon with correct config
- [ ] handle_queen(Stop) calls stop_daemon with correct config
- [ ] handle_queen(Status) calls check_daemon_health
- [ ] handle_queen(Rebuild) calls rebuild_daemon
- [ ] handle_queen(Install) calls install_daemon
- [ ] handle_queen(Uninstall) calls uninstall_daemon
- [ ] Port extraction from queen_url
- [ ] SshConfig::localhost() used for all operations
- [ ] Health URL includes /health path (TEAM-341 bug fix)

**Estimated Tests:** 10-12 tests

---

#### 6. **handlers/hive.rs** (157 LOC)
**Missing Tests:**
- [ ] handle_hive(Start) resolves SSH config
- [ ] handle_hive(Start) uses localhost for "localhost" alias
- [ ] handle_hive(Start) uses SSH config for remote alias
- [ ] handle_hive(Stop) resolves SSH config
- [ ] handle_hive(Status) resolves SSH config
- [ ] handle_hive(Install) resolves SSH config
- [ ] handle_hive(Uninstall) resolves SSH config
- [ ] handle_hive(Rebuild) resolves SSH config
- [ ] Port defaults to 7835
- [ ] Args include --hive-id flag

**Estimated Tests:** 12-15 tests

---

#### 7. **handlers/worker.rs** (87 LOC)
**Missing Tests:**
- [ ] handle_worker(Spawn) creates WorkerSpawnRequest
- [ ] Device parsing: "cuda:0" → worker="cuda", device=0
- [ ] Device parsing: "cpu" → worker="cpu", device=0
- [ ] Device parsing: "metal:1" → worker="metal", device=1
- [ ] handle_worker(Process(List)) creates WorkerProcessListRequest
- [ ] handle_worker(Process(Get)) creates WorkerProcessGetRequest
- [ ] handle_worker(Process(Delete)) creates WorkerProcessDeleteRequest
- [ ] Calls submit_and_stream_job with correct operation

**Estimated Tests:** 8-10 tests

---

#### 8. **handlers/model.rs** (38 LOC)
**Missing Tests:**
- [ ] handle_model(Download) creates ModelDownloadRequest
- [ ] handle_model(List) creates ModelListRequest
- [ ] handle_model(Get) creates ModelGetRequest
- [ ] handle_model(Delete) creates ModelDeleteRequest
- [ ] Calls submit_and_stream_job with correct operation

**Estimated Tests:** 5-6 tests

---

#### 9. **handlers/infer.rs** (41 LOC)
**Missing Tests:**
- [ ] handle_infer creates InferRequest with all fields
- [ ] handle_infer passes hive_id correctly
- [ ] handle_infer passes model correctly
- [ ] handle_infer passes prompt correctly
- [ ] handle_infer passes max_tokens correctly
- [ ] handle_infer passes temperature correctly
- [ ] handle_infer passes optional top_p
- [ ] handle_infer passes optional top_k
- [ ] handle_infer passes optional device
- [ ] handle_infer passes optional worker_id
- [ ] handle_infer passes stream flag

**Estimated Tests:** 8-10 tests

---

#### 10. **handlers/status.rs** (15 LOC)
**Missing Tests:**
- [ ] handle_status creates Operation::Status
- [ ] handle_status calls submit_and_stream_job

**Estimated Tests:** 2-3 tests

---

#### 11. **handlers/self_check.rs** (113 LOC)
**Missing Tests:**
- [ ] handle_self_check runs without errors
- [ ] handle_self_check emits narration events
- [ ] handle_self_check tests all 3 narration modes
- [ ] handle_self_check tests format specifiers
- [ ] handle_self_check tests sequential narrations
- [ ] handle_self_check loads config successfully
- [ ] handle_self_check handles config load failure

**Estimated Tests:** 8-10 tests

**Note:** This is a test command itself, but still needs tests to verify it works

---

### ⚠️ **MEDIUM PRIORITY - Tauri Commands**

#### 12. **tauri_commands.rs** (537 LOC)
**Current Tests:**
- ✅ export_typescript_bindings() - Build-time validation

**Missing Runtime Tests:**
- [ ] queen_status() returns DaemonStatus
- [ ] queen_start() calls handle_queen
- [ ] queen_stop() calls handle_queen
- [ ] queen_install() calls handle_queen
- [ ] queen_rebuild() calls handle_queen
- [ ] queen_uninstall() calls handle_queen
- [ ] hive_start() calls handle_hive
- [ ] hive_stop() calls handle_hive
- [ ] hive_status() calls handle_hive with narration
- [ ] hive_install() calls handle_hive
- [ ] hive_uninstall() calls handle_hive
- [ ] hive_rebuild() calls handle_hive
- [ ] ssh_open_config() opens editor
- [ ] ssh_list() parses SSH config
- [ ] ssh_list() deduplicates by hostname
- [ ] ssh_list() includes localhost
- [ ] get_installed_hives() checks all hives
- [ ] test_narration() emits events

**Estimated Tests:** 20-25 tests

**Complexity:** MEDIUM - Mostly thin wrappers, but many functions

---

### 🟢 **LOW PRIORITY - Already Tested or Simple**

#### 13. **cli/commands.rs** (104 LOC)
**Why Low Priority:**
- ✅ Clap definitions (validated by Clap at compile time)
- ✅ No business logic to test
- ✅ Structural validation happens automatically

**Optional Tests:**
- [ ] Cli parses with no command (launches GUI)
- [ ] Cli parses with subcommands
- [ ] Default values are correct

**Estimated Tests:** 3-5 tests (optional)

---

#### 14. **cli/mod.rs** (14 LOC)
**Why Low Priority:**
- ✅ Just re-exports, no logic
- ✅ Compilation validates correctness

**No tests needed**

---

#### 15. **handlers/mod.rs** (32 LOC)
**Why Low Priority:**
- ✅ Just re-exports, no logic
- ✅ Compilation validates correctness

**No tests needed**

---

#### 16. **lib.rs** (28 LOC)
**Why Low Priority:**
- ✅ Just re-exports, no logic
- ✅ Compilation validates correctness

**No tests needed**

---

#### 17. **main.rs** (190 LOC)
**Why Low Priority:**
- ✅ Entry point, tested via integration tests
- ✅ launch_gui() tested via BDD
- ✅ handle_command() tested via BDD

**Optional Tests:**
- [ ] main() with no args launches GUI
- [ ] main() with subcommand calls handle_command

**Estimated Tests:** 2-3 tests (optional, covered by BDD)

---

#### 18. **platform/** (450 LOC)
**Status:** ❌ SCHEDULED FOR DELETION (RULE ZERO)

**No tests needed** - Module is dead code

---

## Test Priority Matrix

### 🔴 **CRITICAL (Must Have) - 60-75 tests**

| Module | LOC | Tests Needed | Complexity | Priority |
|--------|-----|--------------|------------|----------|
| tracing_init.rs | 315 | 15-20 | HIGH | 🔴 CRITICAL |
| job_client.rs | 105 | 12-15 | MEDIUM | 🔴 CRITICAL |
| config.rs | 87 | 10-12 | MEDIUM | 🔴 CRITICAL |
| process_utils.rs | 89 | 6-8 | LOW | 🔴 CRITICAL |

**Rationale:**
- **tracing_init.rs:** Most complex code, bug fix history needs regression tests
- **job_client.rs:** Core communication, timeout handling critical
- **config.rs:** User-facing I/O, validation logic
- **process_utils.rs:** Already has integration tests, needs edge case coverage

---

### 🟡 **HIGH (Should Have) - 50-65 tests**

| Module | LOC | Tests Needed | Complexity | Priority |
|--------|-----|--------------|------------|----------|
| tauri_commands.rs | 537 | 20-25 | MEDIUM | 🟡 HIGH |
| handlers/queen.rs | 125 | 10-12 | LOW | 🟡 HIGH |
| handlers/hive.rs | 157 | 12-15 | LOW | 🟡 HIGH |
| handlers/worker.rs | 87 | 8-10 | LOW | 🟡 HIGH |

**Rationale:**
- **tauri_commands.rs:** Many functions, but thin wrappers
- **handlers/queen.rs:** Critical daemon lifecycle
- **handlers/hive.rs:** SSH config resolution needs verification
- **handlers/worker.rs:** Device parsing logic

---

### 🟢 **MEDIUM (Nice to Have) - 20-30 tests**

| Module | LOC | Tests Needed | Complexity | Priority |
|--------|-----|--------------|------------|----------|
| handlers/infer.rs | 41 | 8-10 | LOW | 🟢 MEDIUM |
| handlers/self_check.rs | 113 | 8-10 | LOW | 🟢 MEDIUM |
| handlers/model.rs | 38 | 5-6 | LOW | 🟢 MEDIUM |

**Rationale:**
- Thin wrappers around shared crates
- Less critical than lifecycle operations

---

### ⚪ **LOW (Optional) - 5-10 tests**

| Module | LOC | Tests Needed | Complexity | Priority |
|--------|-----|--------------|------------|----------|
| handlers/status.rs | 15 | 2-3 | LOW | ⚪ LOW |
| cli/commands.rs | 104 | 3-5 | LOW | ⚪ LOW |
| main.rs | 190 | 2-3 | LOW | ⚪ LOW |

**Rationale:**
- Already covered by integration/BDD tests
- Minimal logic to test

---

## Recommended Test Implementation Plan

### Phase 1: Critical Tests (2-3 days)
**Target:** 60-75 tests

1. **tracing_init.rs** (15-20 tests)
   - EventVisitor field extraction
   - Dual-layer setup
   - Narration mode switching
   - Regression tests for TEAM-337 bug fix

2. **job_client.rs** (12-15 tests)
   - Job submission and streaming
   - Timeout handling
   - Error detection
   - [DONE] marker detection

3. **config.rs** (10-12 tests)
   - Load/save operations
   - Validation logic
   - Default config generation
   - Error handling

4. **process_utils.rs** (6-8 tests)
   - Edge cases not covered by integration tests
   - stream_child_output() function

---

### Phase 2: High Priority Tests (2-3 days)
**Target:** 50-65 tests

1. **tauri_commands.rs** (20-25 tests)
   - All 16 Tauri command functions
   - SSH list deduplication
   - Installed hives detection

2. **handlers/queen.rs** (10-12 tests)
   - All 6 queen lifecycle operations
   - Port extraction
   - Health URL construction

3. **handlers/hive.rs** (12-15 tests)
   - All 6 hive lifecycle operations
   - SSH config resolution
   - Localhost vs remote detection

4. **handlers/worker.rs** (8-10 tests)
   - Device parsing logic
   - Operation construction

---

### Phase 3: Medium Priority Tests (1-2 days)
**Target:** 20-30 tests

1. **handlers/infer.rs** (8-10 tests)
2. **handlers/self_check.rs** (8-10 tests)
3. **handlers/model.rs** (5-6 tests)

---

### Phase 4: Optional Tests (0.5-1 day)
**Target:** 5-10 tests

1. **handlers/status.rs** (2-3 tests)
2. **cli/commands.rs** (3-5 tests)
3. **main.rs** (2-3 tests)

---

## Test File Structure Proposal

```
bin/00_rbee_keeper/
├── src/
│   ├── config.rs
│   ├── job_client.rs
│   ├── tracing_init.rs
│   ├── process_utils.rs
│   ├── ssh_resolver.rs (✅ has tests)
│   └── ...
│
├── tests/
│   ├── process_output_tests.rs (✅ exists)
│   ├── config_tests.rs           # NEW
│   ├── job_client_tests.rs       # NEW
│   ├── tracing_init_tests.rs     # NEW
│   ├── tauri_commands_tests.rs   # NEW
│   └── handlers/                 # NEW
│       ├── queen_tests.rs
│       ├── hive_tests.rs
│       ├── worker_tests.rs
│       ├── model_tests.rs
│       ├── infer_tests.rs
│       ├── status_tests.rs
│       └── self_check_tests.rs
│
└── bdd/ (✅ exists)
    └── tests/features/
```

---

## Testing Strategy Recommendations

### 1. **Use Mock HTTP Servers**
For testing job_client.rs and tauri_commands.rs:
```rust
use wiremock::{MockServer, Mock, ResponseTemplate};

#[tokio::test]
async fn test_submit_job_success() {
    let mock_server = MockServer::start().await;
    
    Mock::given(method("POST"))
        .and(path("/v1/jobs"))
        .respond_with(ResponseTemplate::new(200)
            .set_body_json(json!({"job_id": "test-123"})))
        .mount(&mock_server)
        .await;
    
    // Test job submission
}
```

### 2. **Use Temporary Files**
For testing config.rs:
```rust
use tempfile::TempDir;

#[test]
fn test_config_save() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("config.toml");
    
    // Test config save/load
}
```

### 3. **Use Test Fixtures**
For testing ssh_resolver.rs (already done):
```rust
use tempfile::NamedTempFile;

#[test]
fn test_parse_ssh_config() {
    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "Host workstation").unwrap();
    // ...
}
```

### 4. **Use Tracing Subscriber for Testing**
For testing tracing_init.rs:
```rust
use tracing_subscriber::layer::SubscriberExt;

#[test]
fn test_event_visitor_extracts_human_field() {
    let (subscriber, handle) = tracing_subscriber::fmt()
        .with_test_writer()
        .finish();
    
    tracing::subscriber::with_default(subscriber, || {
        // Test narration emission
    });
}
```

---

## Estimated Effort

| Phase | Tests | Estimated Time | Priority |
|-------|-------|----------------|----------|
| Phase 1: Critical | 60-75 | 2-3 days | 🔴 MUST |
| Phase 2: High | 50-65 | 2-3 days | 🟡 SHOULD |
| Phase 3: Medium | 20-30 | 1-2 days | 🟢 NICE |
| Phase 4: Optional | 5-10 | 0.5-1 day | ⚪ OPTIONAL |
| **TOTAL** | **135-180** | **6-9 days** | |

---

## Key Insights

### ✅ **What's Working**
1. **ssh_resolver.rs** has good unit tests (3 tests)
2. **process_utils.rs** has comprehensive integration tests (8 tests)
3. **BDD tests** cover end-to-end scenarios
4. **tauri_commands.rs** has TypeScript binding validation

### ❌ **Critical Gaps**
1. **tracing_init.rs** - Most complex code, ZERO tests
2. **job_client.rs** - Core communication, ZERO tests
3. **config.rs** - User-facing I/O, ZERO tests
4. **All handlers** - Thin wrappers, but ZERO tests

### 🎯 **Biggest ROI**
1. **tracing_init.rs** - Prevents regression of TEAM-337 bug fix
2. **job_client.rs** - Prevents timeout hangs and error detection issues
3. **config.rs** - Prevents user data corruption

---

## RULE ZERO Compliance

### ✅ **No Test Duplication**
- Integration tests in tests/
- Unit tests inline in src/ (ssh_resolver.rs, platform/mod.rs)
- BDD tests in bdd/
- **No overlap** - each test type serves different purpose

### ✅ **No Backwards Compatibility Tests**
- No tests for deprecated functions
- No tests for "v2" functions
- Clean test suite

### ⚠️ **Platform Module Tests**
- platform/mod.rs has 3 tests
- **BUT:** Module scheduled for deletion (RULE ZERO)
- **Action:** Delete tests when deleting module

---

## Summary

**Current Coverage:** ~15% (3 modules with inline tests + 1 integration test file)  
**Target Coverage:** ~70% (135-180 tests across all critical modules)  
**Estimated Effort:** 6-9 days  
**Biggest Gaps:** tracing_init.rs, job_client.rs, config.rs, all handlers

**Recommendation:**
1. **IMMEDIATE:** Implement Phase 1 (critical tests) - 2-3 days
2. **SHORT-TERM:** Implement Phase 2 (high priority) - 2-3 days
3. **MEDIUM-TERM:** Implement Phase 3 (medium priority) - 1-2 days
4. **OPTIONAL:** Implement Phase 4 (optional) - 0.5-1 day

**ROI:** Prevents regression of complex EventVisitor logic, timeout hangs, config corruption, and lifecycle operation failures.

---

**Document Version:** 1.0  
**Last Updated:** 2025-10-29  
**Next Review:** After Phase 1 implementation
