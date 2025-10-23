# rbee-keeper BEHAVIOR INVENTORY

**Team:** TEAM-216  
**Component:** `bin/00_rbee_keeper` (rbee-keeper CLI)  
**Date:** Oct 22, 2025  
**LOC:** ~940 lines (464 main.rs + 74 config.rs + 171 job_client.rs + 306 queen_lifecycle.rs - BDD excluded)

---

## 1. Public API Surface

### CLI Commands (main.rs:61-272)

**Top-Level Commands:**
- `status` - Show live status of all hives/workers (line 76)
- `queen <action>` - Manage queen-rbee daemon (line 78-82)
- `hive <action>` - Hive management (line 84-88)
- `worker <action>` - Worker management (line 90-97)
- `model <action>` - Model management (line 99-106)
- `infer` - Run inference (line 108-139)

**Queen Actions (main.rs:142-150):**
- `start` - Start queen-rbee daemon
- `stop` - Stop queen-rbee daemon
- `status` - Check queen-rbee health

**Hive Actions (main.rs:187-240):**
- `ssh-test -a <alias>` - Test SSH connection (line 188-193)
- `install -a <alias>` - Install hive binary (line 194-199)
- `uninstall -a <alias>` - Uninstall hive (line 200-205)
- `list` - List all configured hives (line 206-207)
- `start -a <alias>` - Start hive daemon (line 208-213, default: localhost)
- `stop -a <alias>` - Stop hive daemon (line 214-219, default: localhost)
- `get -a <alias>` - Get hive details (line 220-225, default: localhost)
- `status -a <alias>` - Check hive health (line 227-232, default: localhost)
- `refresh-capabilities -a <alias>` - Refresh device capabilities (line 234-239)

**Worker Actions (main.rs:242-263):**
- `spawn --model <model> --worker <type> --device <id>` - Spawn worker (line 244-255)
- `list` - List workers on hive (line 256)
- `get <id>` - Get worker details (line 257-259)
- `delete <id>` - Delete worker (line 260-262)

**Model Actions (main.rs:265-271):**
- `download <model>` - Download model (line 267)
- `list` - List models on hive (line 268)
- `get <id>` - Get model details (line 269)
- `delete <id>` - Delete model (line 270)

**Infer Parameters (main.rs:108-139):**
- `--hive-id` - Target hive (default: localhost)
- `--model` - Model identifier (required)
- `prompt` - Input prompt (positional, required)
- `--max-tokens` - Max tokens to generate (default: 20)
- `--temperature` - Sampling temperature (default: 0.7)
- `--top-p` - Nucleus sampling (optional)
- `--top-k` - Top-k sampling (optional)
- `--device` - Device filter: cpu/cuda/metal (optional)
- `--worker-id` - Specific worker ID (optional)
- `--stream` - Stream tokens (default: true)

### HTTP Client API (job_client.rs:36-170)

**Function:** `submit_and_stream_job(client, queen_url, operation)` (line 36-170)
- Serializes Operation to JSON (line 42)
- Ensures queen is running (line 48)
- POSTs to `/v1/jobs` with 10s timeout (line 51-56)
- Extracts `job_id` and `sse_url` from response (line 60-62)
- GETs SSE stream with 10s timeout (line 91-96)
- Streams narration events to stdout (line 114-158)
- Detects `[DONE]` marker for completion (line 134)
- Tracks job failures for proper status display (line 115, 129-131)
- Returns on timeout (30s) or completion (line 86-163)

### Queen Lifecycle API (queen_lifecycle.rs:94-305)

**Function:** `ensure_queen_running(base_url)` (line 94-101)
- 30-second timeout with visual countdown (line 96-100)
- Delegates to `ensure_queen_running_inner()` (line 103-205)

**Function:** `ensure_queen_running_inner(base_url)` (line 103-205)
- Checks if queen is healthy (line 107-110)
- Loads and validates RbeeConfig (line 113-164)
- Finds queen-rbee binary in target/ (line 170-177)
- Spawns queen process on port 8500 (line 180-188)
- Polls health until ready (line 191-193)
- Returns QueenHandle (line 204)

**Function:** `is_queen_healthy(base_url)` (line 216-238)
- GETs `/health` endpoint with 500ms timeout (line 219)
- Returns `Ok(true)` if 2xx status (line 224)
- Returns `Ok(false)` if connection refused (line 231)
- Returns `Err` for other errors (line 234)

**Function:** `poll_until_healthy(base_url, timeout)` (line 251-305)
- Exponential backoff: 100ms → 3200ms (line 253-254)
- Polls until healthy or timeout (line 257-304)
- Emits narration for each attempt (line 282-296)

### Configuration API (config.rs:26-73)

**Function:** `Config::load()` (line 29-44)
- Loads from `~/.config/rbee/config.toml` (line 30)
- Creates default config if missing (line 32-36)
- Parses TOML (line 39-41)

**Function:** `Config::save()` (line 47-60)
- Saves to `~/.config/rbee/config.toml` (line 48)
- Creates parent directory if needed (line 51-53)
- Serializes to TOML (line 55-57)

**Function:** `Config::queen_url()` (line 70-72)
- Returns `http://localhost:{queen_port}` (default: 8500)

---

## 2. State Machine Behaviors

### Queen Lifecycle States (queen_lifecycle.rs:22-73)

**QueenHandle States:**
1. **Already Running** - Queen was running before keeper started (line 40-42)
   - `started_by_us: false`
   - `pid: None`
   - Keeper will NOT shutdown queen
2. **Started By Us** - Keeper spawned queen (line 45-47)
   - `started_by_us: true`
   - `pid: Some(u32)`
   - Keeper WILL shutdown queen (via `std::mem::forget` pattern)

**Queen Startup Flow (queen_lifecycle.rs:103-205):**
```
Check Health → Already Running? → Return Handle (started_by_us=false)
              ↓ Not Running
          Load Config → Validate → Find Binary → Spawn Process
              ↓
          Poll Health (30s timeout) → Ready → Return Handle (started_by_us=true)
              ↓ Timeout
          Error
```

### Job Submission Flow (job_client.rs:36-170)

```
Serialize Operation → Ensure Queen Running → POST /v1/jobs
    ↓
Extract job_id + sse_url → GET SSE stream → Stream to stdout
    ↓
Detect [DONE] or timeout (30s) → Cleanup queen handle → Return
```

### CLI Command Flow (main.rs:279-462)

```
Parse CLI → Load Config → Match Command
    ↓
Queen Commands → Direct HTTP (no job submission)
    ↓
All Other Commands → submit_and_stream_job() → SSE streaming
```

---

## 3. Data Flows

### Inputs

**CLI Arguments (main.rs:61-272):**
- Command-line flags parsed by clap
- Defaults: `--hive-id localhost`, `--max-tokens 20`, `--temperature 0.7`, `--stream true`

**Configuration Files:**
- `~/.config/rbee/config.toml` - Queen port (default: 8500)
- `~/.config/rbee/hives.conf` - Hive aliases (loaded by queen-rbee, not keeper)
- `~/.config/rbee/capabilities.json` - Device capabilities (loaded by queen-rbee)

**Environment Variables:**
- None (config is file-based only)

### Outputs

**Stdout (job_client.rs:126, queen_lifecycle.rs:68-70):**
- Narration events from observability-narration-core
- SSE stream data (raw JSON from queen-rbee)
- Status messages (✅/❌/⚠️ emoji prefixes)

**Stderr:**
- Error messages from anyhow (automatic)

**HTTP Requests:**
- `POST http://localhost:8500/v1/jobs` - Job submission (job_client.rs:56)
- `GET http://localhost:8500/jobs/{job_id}/stream` - SSE streaming (job_client.rs:96)
- `GET http://localhost:8500/health` - Health checks (queen_lifecycle.rs:219, main.rs:304, 350)
- `POST http://localhost:8500/v1/shutdown` - Queen shutdown (main.rs:321)

**File Writes:**
- `~/.config/rbee/config.toml` - Config creation (config.rs:57)

---

## 4. Error Handling

### Error Types (All use anyhow::Result)

**Network Errors (job_client.rs:56, 96, queen_lifecycle.rs:219):**
- Connection refused → Queen not running (queen_lifecycle.rs:231)
- Timeout → Operation took too long (job_client.rs:86, queen_lifecycle.rs:261-268)
- HTTP errors → Propagated as anyhow::Error

**Configuration Errors (config.rs:39-41, queen_lifecycle.rs:115-136):**
- Missing config file → Auto-create default (config.rs:32-36)
- Invalid TOML → Parse error (config.rs:41)
- Validation failure → Detailed error list (queen_lifecycle.rs:127-136)

**Binary Resolution Errors (queen_lifecycle.rs:170-171):**
- queen-rbee not found in target/ → Error with context

**Timeout Errors:**
- Queen startup timeout (30s) → Error (queen_lifecycle.rs:261-268)
- Job submission timeout (10s) → Error (job_client.rs:51-53)
- SSE connection timeout (10s) → Error (job_client.rs:91-93)
- SSE streaming timeout (30s) → Error (job_client.rs:86-163)

**Shutdown Errors (main.rs:321-342):**
- Connection closed during shutdown → Expected, treated as success (main.rs:328-330)
- Other errors → Propagated (main.rs:332-340)

### Error Recovery

**Queen Auto-Start (queen_lifecycle.rs:94-205):**
- If queen not running → Auto-spawn and wait for health
- If spawn fails → Error (no retry)
- If health timeout → Error (no retry)

**Graceful Degradation:**
- Queen status check → Returns Ok(()) even if queen not running (main.rs:372-379)
- SSE stream errors → Printed to stdout, then error returned

**No Retry Logic:**
- All operations fail-fast (no automatic retries)
- User must re-run command

---

## 5. Integration Points

### Dependencies (Cargo.toml:14-44)

**External Crates:**
- `clap` - CLI parsing (derive API)
- `tokio` - Async runtime (full features)
- `reqwest` - HTTP client (json, stream features)
- `futures` - Stream utilities
- `serde`, `serde_json` - Serialization
- `toml` - Config parsing
- `dirs` - Config directory resolution
- `anyhow` - Error handling

**Internal Crates:**
- `observability-narration-core` - Narration system
- `daemon-lifecycle` - Binary resolution, process spawning
- `timeout-enforcer` - Timeout wrapper with visual countdown
- `rbee-operations` - Shared Operation enum
- `rbee-config` - Configuration validation

### Dependents

**None** - rbee-keeper is a leaf binary (no other components depend on it)

### Contracts

**queen-rbee HTTP API:**
- `POST /v1/jobs` - Accepts Operation JSON, returns `{job_id, sse_url}`
- `GET /jobs/{job_id}/stream` - SSE stream with narration events
- `GET /health` - Returns 2xx if healthy
- `POST /v1/shutdown` - Graceful shutdown (may close connection before responding)

**Operation Serialization (rbee-operations crate):**
- Tagged enum with `"operation"` field
- Serde JSON serialization/deserialization
- Shared between keeper and queen

---

## 6. Critical Invariants

### Queen Lifecycle Invariants

1. **Cleanup Safety:** Only shutdown queens that keeper started (queen_lifecycle.rs:50-52)
   - `QueenHandle.started_by_us` tracks ownership
   - `std::mem::forget()` prevents automatic cleanup (main.rs:299, job_client.rs:166)

2. **Health Check Before Use:** Always check queen health before operations (queen_lifecycle.rs:107-110)

3. **Timeout Enforcement:** All network operations have timeouts
   - Health check: 500ms (queen_lifecycle.rs:219)
   - Job submission: 10s (job_client.rs:51-53)
   - SSE connection: 10s (job_client.rs:91-93)
   - SSE streaming: 30s (job_client.rs:86)
   - Queen startup: 30s (queen_lifecycle.rs:96)

### Configuration Invariants

1. **Auto-Creation:** Missing config files are auto-created with defaults (config.rs:32-36)
2. **Validation Before Start:** Config is validated before spawning queen (queen_lifecycle.rs:125-136)

### SSE Streaming Invariants

1. **[DONE] Marker:** Stream completes when `[DONE]` marker received (job_client.rs:134)
2. **Failure Tracking:** Job failures detected by parsing narration events (job_client.rs:129-131)
3. **Proper Status Display:** ✅ for success, ❌ for failure (job_client.rs:136-148)

---

## 7. Existing Test Coverage

### BDD Tests (bdd/tests/features/)

**queen_health_check.feature (35 lines):**
- ✅ Queen not running (connection refused)
- ✅ Queen running and healthy
- ✅ Custom port health check
- ✅ Health check timeout

**sse_streaming.feature (60 lines):**
- ✅ Submit job and establish SSE connection
- ✅ POST /jobs returns job_id and sse_url
- ✅ GET /jobs/{job_id}/stream establishes SSE
- ✅ SSE stream handles missing job (404)
- ✅ Full dual-call pattern flow

**placeholder.feature:**
- ❌ Placeholder only (no real tests)

### Unit Tests

**None** - No unit tests in src/ files

### Coverage Gaps

**CLI Command Coverage:**
- ❌ No tests for hive commands (install, uninstall, start, stop, list, get, status, refresh-capabilities)
- ❌ No tests for worker commands (spawn, list, get, delete)
- ❌ No tests for model commands (download, list, get, delete)
- ❌ No tests for infer command with various parameters
- ❌ No tests for status command (TEAM-190)

**Error Path Coverage:**
- ❌ No tests for config validation failures
- ❌ No tests for binary resolution failures
- ❌ No tests for timeout scenarios (job submission, SSE connection, SSE streaming)
- ❌ No tests for malformed SSE responses
- ❌ No tests for queen shutdown during operation

**Edge Case Coverage:**
- ❌ No tests for concurrent keeper sessions
- ❌ No tests for queen crash during operation
- ❌ No tests for network interruption during SSE streaming
- ❌ No tests for invalid Operation JSON
- ❌ No tests for missing/invalid config files

**Integration Coverage:**
- ❌ No end-to-end tests with real queen-rbee
- ❌ No tests for full inference flow (keeper → queen → hive → worker)

---

## 8. Behavior Checklist

- [x] All public APIs documented (CLI commands, HTTP client, queen lifecycle, config)
- [x] All state transitions documented (QueenHandle states, job flow, command flow)
- [x] All error paths documented (network, config, timeout, shutdown)
- [x] All integration points documented (dependencies, contracts, queen API)
- [x] All edge cases identified (timeouts, failures, concurrent sessions)
- [x] Existing test coverage assessed (2 BDD features, no unit tests)
- [x] Coverage gaps identified (CLI commands, error paths, edge cases, integration)
- [x] Code signatures added (TEAM-216: Investigated)

---

**Critical Findings:**

1. **Thin Client Architecture:** rbee-keeper is correctly implemented as a thin HTTP client (~940 LOC)
2. **No SSH:** Correctly delegates all SSH operations to queen-rbee
3. **Queen Lifecycle:** Proper ownership tracking prevents shutting down pre-existing queens
4. **Timeout Enforcement:** All network operations have timeouts (prevents hanging)
5. **Test Coverage:** Only 2 BDD features (health check, SSE streaming) - needs comprehensive CLI command tests
6. **Error Handling:** Uses anyhow for error propagation, no retry logic (fail-fast)
7. **Configuration:** Auto-creates missing config files, validates before queen startup

**Recommended Test Priorities:**
1. CLI command tests (all hive/worker/model/infer commands)
2. Error path tests (timeouts, failures, validation errors)
3. Edge case tests (concurrent sessions, crashes, network interruption)
4. Integration tests (full keeper → queen → hive → worker flow)
