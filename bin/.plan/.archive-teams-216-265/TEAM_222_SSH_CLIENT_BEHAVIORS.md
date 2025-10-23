# SSH-CLIENT BEHAVIOR INVENTORY

**Team:** TEAM-222  
**Component:** `bin/15_queen_rbee_crates/ssh-client`  
**Date:** Oct 22, 2025  
**LOC:** 263 (lib.rs)

---

## 1. Public API Surface

### Core Function
```rust
// src/lib.rs:102-137
pub async fn test_ssh_connection(config: SshConfig) -> Result<SshTestResult>
```
**Purpose:** Test SSH connectivity to remote hosts  
**Behavior:** Async wrapper around blocking SSH operations using `tokio::spawn_blocking`

### Configuration Types
```rust
// src/lib.rs:33-49
pub struct SshConfig {
    pub host: String,        // SSH host address
    pub port: u16,           // SSH port (default: 22)
    pub user: String,        // SSH username
    pub timeout_secs: u64,   // Connection timeout in seconds
}

impl Default for SshConfig {
    fn default() -> Self {
        Self { host: String::new(), port: 22, user: String::new(), timeout_secs: 5 }
    }
}
```

### Result Types
```rust
// src/lib.rs:52-60
pub struct SshTestResult {
    pub success: bool,              // Whether the connection was successful
    pub error: Option<String>,      // Error message if connection failed
    pub test_output: Option<String>, // Test command output (if successful)
}
```

### Dependencies
- `ssh2 = "0.9"` - SSH protocol implementation
- `tokio` - Async runtime (for spawn_blocking)
- `anyhow` - Error handling
- `observability-narration-core` - Narration emission

---

## 2. State Machine Behaviors

### SSH Connection Lifecycle (5 Steps)

**Step 1: Pre-flight Check (lines 159-162)**
- Check if SSH agent is running via `SSH_AUTH_SOCK` environment variable
- **Success:** Proceed to TCP connection
- **Failure:** Return helpful error message with instructions to start SSH agent

**Step 2: TCP Connection (lines 165-177)**
- Establish TCP connection with timeout
- Parse host:port address
- **Success:** Proceed to SSH handshake
- **Failure:** Return `SshTestResult` with TCP connection error

**Step 3: SSH Handshake (lines 186-195)**
- Create SSH session
- Set TCP stream
- Perform SSH handshake
- **Success:** Proceed to authentication
- **Failure:** Return `SshTestResult` with handshake error

**Step 4: Authentication (lines 198-218)**
- Authenticate using SSH agent (`session.userauth_agent`)
- Verify authentication succeeded
- **Success:** Proceed to test command
- **Failure:** Return `SshTestResult` with authentication error + helpful message

**Step 5: Test Command Execution (lines 220-258)**
- Open SSH channel
- Execute `echo test` command
- Read stdout
- Check exit status (must be 0)
- **Success:** Return `SshTestResult` with success=true and output
- **Failure:** Return `SshTestResult` with command execution error

### State Transitions
```
[Start] → Pre-flight → TCP → Handshake → Auth → Test Command → [Success]
            ↓           ↓        ↓         ↓          ↓
         [Fail]      [Fail]   [Fail]    [Fail]    [Fail]
```

**Critical:** All failures return `Ok(SshTestResult)` with `success=false`, NOT `Err`. Only panics or critical errors return `Err`.

---

## 3. Data Flows

### Input Flow
```
SshConfig → test_ssh_connection() → tokio::spawn_blocking() → test_ssh_connection_blocking()
```

### Output Flow
```
test_ssh_connection_blocking() → SshTestResult → test_ssh_connection() → Result<SshTestResult>
```

### Narration Flow (3 emission points)
```rust
// 1. Start (line 105-107)
Narration::new(ACTOR_SSH_CLIENT, ACTION_TEST, &target)
    .human(format!("🔐 Testing SSH connection to {}", target))
    .emit();

// 2. Success (line 116-118)
Narration::new(ACTOR_SSH_CLIENT, ACTION_TEST, "success")
    .human(format!("✅ SSH connection to {} successful", target))
    .emit();

// 3. Failure (line 121-127)
Narration::new(ACTOR_SSH_CLIENT, ACTION_TEST, "failed")
    .human(format!("❌ SSH connection to {} failed: {}", target, error))
    .emit();
```

**Note:** Narration does NOT include `job_id` - this is a leaf crate, job_id propagation happens in hive-lifecycle wrapper.

---

## 4. Error Handling

### Error Categories

**1. Pre-flight Errors (line 143-152)**
- **Cause:** SSH agent not running (`SSH_AUTH_SOCK` not set)
- **Handling:** Return `SshTestResult` with helpful instructions
- **Recovery:** User must start SSH agent and add keys

**2. TCP Connection Errors (line 170-176)**
- **Causes:** Host unreachable, port closed, network timeout
- **Handling:** Return `SshTestResult` with TCP error message
- **Recovery:** Check network, firewall, host availability

**3. SSH Handshake Errors (line 189-195)**
- **Causes:** SSH protocol mismatch, server not responding
- **Handling:** Return `SshTestResult` with handshake error
- **Recovery:** Verify SSH server is running, check SSH version compatibility

**4. Authentication Errors (line 199-218)**
- **Causes:** No SSH keys loaded, wrong user, permission denied
- **Handling:** Return `SshTestResult` with auth error + helpful message
- **Recovery:** Load SSH keys, verify username, check server authorized_keys

**5. Command Execution Errors (line 223-257)**
- **Causes:** Channel creation failed, command exec failed, read timeout, non-zero exit
- **Handling:** Return `SshTestResult` with command error
- **Recovery:** Check SSH server health, verify command permissions

**6. Critical Errors (line 112)**
- **Cause:** Tokio task panic
- **Handling:** Return `Err` (propagates to caller)
- **Recovery:** None - indicates serious runtime issue

### Error Message Quality
- ✅ All errors include context (what failed)
- ✅ Pre-flight errors include actionable instructions
- ✅ Authentication errors suggest checking SSH agent
- ✅ No raw error types exposed (all wrapped in strings)

---

## 5. Integration Points

### Upstream Dependents
**hive-lifecycle crate** (`bin/15_queen_rbee_crates/hive-lifecycle/src/ssh_test.rs`)
- Wraps `test_ssh_connection()` in `execute_ssh_test()`
- Adds job_id propagation for SSE routing
- Converts `SshTestResult` to `SshTestResponse`
- Used by queen-rbee for remote hive SSH validation

### Downstream Dependencies
**ssh2 crate** (external)
- `Session::new()` - Create SSH session
- `session.set_tcp_stream()` - Attach TCP stream
- `session.handshake()` - Perform SSH handshake
- `session.userauth_agent()` - Authenticate using SSH agent
- `session.authenticated()` - Check authentication status
- `session.channel_session()` - Open SSH channel
- `channel.exec()` - Execute command
- `channel.read_to_string()` - Read stdout
- `channel.exit_status()` - Get command exit code

**tokio** (external)
- `tokio::spawn_blocking()` - Run blocking SSH operations in thread pool
- Required because ssh2 is synchronous

### Configuration Sources
- `SSH_AUTH_SOCK` environment variable (SSH agent socket)
- User's `~/.ssh/` directory (SSH keys, managed by SSH agent)
- TCP timeout from `SshConfig.timeout_secs` (default: 5s)

---

## 6. Critical Invariants

### 1. Blocking Operations Must Use spawn_blocking
**Invariant:** All ssh2 operations MUST run in `tokio::spawn_blocking`  
**Reason:** ssh2 is synchronous, will block tokio runtime  
**Location:** Line 110-112  
**Enforcement:** Wrapper function `test_ssh_connection_blocking()` is private

### 2. Timeouts Must Be Set on TCP Stream
**Invariant:** TCP stream MUST have read/write timeouts set  
**Reason:** Prevent indefinite hangs on slow/dead connections  
**Location:** Lines 180-183  
**Values:** Both set to `config.timeout_secs`

### 3. Authentication Must Be Verified
**Invariant:** After `userauth_agent()`, MUST check `session.authenticated()`  
**Reason:** `userauth_agent()` may succeed but authentication may still fail  
**Location:** Lines 210-218  
**Enforcement:** Explicit check with error return

### 4. Failures Return Ok(SshTestResult), Not Err
**Invariant:** Connection/auth failures return `Ok(SshTestResult { success: false })`  
**Reason:** These are expected failures, not errors. Only panics/critical issues return Err.  
**Location:** All error paths (lines 161, 171, 191, 201, 224, 234, 243, 253)  
**Enforcement:** Consistent pattern throughout

### 5. SSH Agent Must Be Running
**Invariant:** SSH agent MUST be running before attempting connection  
**Reason:** `userauth_agent()` requires SSH agent, will fail without helpful error  
**Location:** Lines 159-162 (pre-flight check)  
**Enforcement:** Explicit check with actionable error message

### 6. Test Command Must Be Simple
**Invariant:** Test command is hardcoded to `echo test`  
**Reason:** Simple, safe, no side effects, works on all systems  
**Location:** Line 232  
**Enforcement:** Hardcoded string (no user input)

---

## 7. Existing Test Coverage

### Unit Tests
**Status:** ❌ NO UNIT TESTS  
**Reason:** Requires SSH server for testing (integration test territory)

### Integration Tests
**Status:** ❌ NO INTEGRATION TESTS  
**Location:** BDD harness exists but not implemented

### BDD Tests
**Location:** `bdd/tests/features/placeholder.feature`  
**Status:** 🚧 STUB ONLY  
**Content:** Placeholder feature, no real tests

**BDD Infrastructure:**
- ✅ BDD runner (`bdd/src/main.rs`) - Complete
- ✅ World struct (`bdd/src/steps/world.rs`) - Basic scaffolding
- ❌ Step definitions - None implemented
- ❌ Feature files - Only placeholder

### Example Code
**Location:** `examples/test_connection.rs`  
**Status:** ✅ COMPLETE  
**Purpose:** Manual testing tool  
**Usage:** `cargo run --example test_connection -- <host> <user>`

### Test Gaps (IMPLEMENTED code with NO tests)

**Gap 1: SSH Agent Pre-flight Check**
- **Code:** Lines 140-153 (`check_ssh_agent()`)
- **Missing:** Test with SSH_AUTH_SOCK unset
- **Missing:** Test with SSH_AUTH_SOCK set but empty

**Gap 2: TCP Connection Timeout**
- **Code:** Lines 165-177
- **Missing:** Test with unreachable host (timeout behavior)
- **Missing:** Test with invalid host:port format

**Gap 3: SSH Handshake Failure**
- **Code:** Lines 186-195
- **Missing:** Test with non-SSH server (e.g., HTTP server on port)
- **Missing:** Test with SSH protocol version mismatch

**Gap 4: Authentication Failure**
- **Code:** Lines 198-218
- **Missing:** Test with no SSH keys loaded
- **Missing:** Test with wrong username
- **Missing:** Test with userauth_agent success but authenticated() = false

**Gap 5: Command Execution Failure**
- **Code:** Lines 220-258
- **Missing:** Test with channel creation failure
- **Missing:** Test with command exec failure
- **Missing:** Test with non-zero exit status

**Gap 6: Narration Emission**
- **Code:** Lines 105-134
- **Missing:** Verify narration emitted on success
- **Missing:** Verify narration emitted on failure
- **Missing:** Verify narration includes correct target/error

**Gap 7: Tokio spawn_blocking Panic**
- **Code:** Line 110-112
- **Missing:** Test task panic handling (if possible)

### Coverage Summary
- **Total Behaviors:** 7 (pre-flight, TCP, handshake, auth, command, narration, async wrapper)
- **Tested Behaviors:** 0
- **Test Coverage:** 0%
- **Critical Gaps:** All behaviors untested

---

## 8. Behavior Checklist

- [x] All public APIs documented
- [x] All state transitions documented
- [x] All error paths documented
- [x] All integration points documented
- [x] All edge cases documented
- [x] Existing test coverage assessed
- [x] Coverage gaps identified

---

## Additional Notes

### Security Considerations
**From README (lines 233-278):**
- ✅ Uses ssh2 crate (safe, no command injection)
- ✅ Respects user's SSH keys via SSH agent
- ✅ No password handling (agent-only authentication)
- ✅ Simple test command (`echo test`) - no side effects

**Potential Security Issues:**
- ⚠️ No host key verification documented (ssh2 may accept any host key)
- ⚠️ No known_hosts handling documented
- ⚠️ No SSH config file support (~/.ssh/config)

### Future Enhancements (from README)
- Remote command execution (beyond test command)
- File transfer (upload/download)
- Connection pooling
- Password authentication (currently agent-only)
- SSH config file support

### Naming Discussion (from README lines 33-53)
- Current name: `ssh-client`
- Proposed name: `remote` (more accurate for purpose)
- Reason: "ssh-client" sounds like it receives connections, but queen-rbee is SSH initiator
- Status: Not renamed yet

### LOC Analysis
- `src/lib.rs`: 263 LOC
- `examples/test_connection.rs`: 38 LOC
- `bdd/` infrastructure: ~60 LOC (scaffolding only)
- **Total:** ~361 LOC

### Code Quality
- ✅ Well-documented (rustdoc comments)
- ✅ Clear error messages
- ✅ Consistent narration pattern
- ✅ Proper async/blocking separation
- ✅ No TODOs in implementation code
- ⚠️ Zero test coverage

---

**TEAM-222: Investigated**
