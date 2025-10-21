# Queen Status Command

**Team:** TEAM-186  
**Date:** 2025-10-21  
**Status:** ✅ Complete

## Summary

Added `rbee queen status` command to check if queen-rbee daemon is running and healthy.

## Usage

```bash
# Check queen-rbee status
rbee queen status

# Output examples:
# ✅ Queen is running on http://localhost:8500
# Status: {"status":"healthy"}

# ❌ Queen is not running on http://localhost:8500
```

## Implementation

### 1. Added CLI Command

**File:** `src/main.rs`

Added `Status` variant to `QueenAction` enum:

```rust
#[derive(Subcommand)]
pub enum QueenAction {
    /// Start queen-rbee daemon
    Start,
    /// Stop queen-rbee daemon
    Stop,
    /// Check queen-rbee daemon status
    Status,  // TEAM-186: New command
}
```

### 2. Implemented Status Check

**File:** `src/main.rs`

The status command:
1. Makes HTTP GET request to `/health` endpoint
2. Uses 5-second timeout
3. Returns three possible states:
   - **Running** (200 OK) - Queen is healthy
   - **Unhealthy** (non-200) - Queen responded but with error status
   - **Not Running** (connection error) - Queen is not accessible

```rust
QueenAction::Status => {
    let client = reqwest::Client::builder()
        .timeout(tokio::time::Duration::from_secs(5))
        .build()?;
    
    match client.get(format!("{}/health", queen_url)).send().await {
        Ok(response) if response.status().is_success() => {
            // ✅ Running
        }
        Ok(response) => {
            // ⚠️ Unhealthy
        }
        Err(_) => {
            // ❌ Not running
        }
    }
}
```

### 3. Added Narration Constant

**File:** `src/operations.rs`

```rust
pub const ACTION_QUEEN_STATUS: &str = "queen_status";
```

### 4. Updated Documentation

**File:** `bin/API_REFERENCE.md`

```bash
rbee queen start                              # Start queen-rbee daemon
rbee queen stop                               # Stop queen-rbee daemon
rbee queen status                             # Check queen-rbee daemon status
```

## Status States

### 1. Running (Healthy)
```
✅ Queen is running on http://localhost:8500
Status: {"status":"healthy"}
```

**Narration:**
```
actor: 🧑‍🌾 rbee-keeper
action: queen_status
target: running
```

### 2. Unhealthy
```
⚠️  Queen responded with status: 500
```

**Narration:**
```
actor: 🧑‍🌾 rbee-keeper
action: queen_status
target: unhealthy
```

### 3. Not Running
```
❌ Queen is not running on http://localhost:8500
```

**Narration:**
```
actor: 🧑‍🌾 rbee-keeper
action: queen_status
target: not_running
```

## Technical Details

### HTTP Request
- **Method:** GET
- **Endpoint:** `/health`
- **Timeout:** 5 seconds
- **Expected Response:** 200 OK with JSON body

### Error Handling
- Connection refused → "Not running"
- Timeout → "Not running"
- Non-200 status → "Unhealthy"
- 200 status → "Running"

### Narration
All status checks emit narration events for observability:
- Actor: `ACTOR_RBEE_KEEPER` ("🧑‍🌾 rbee-keeper")
- Action: `ACTION_QUEEN_STATUS` ("queen_status")
- Target: "running" | "unhealthy" | "not_running"

## Use Cases

### 1. Health Check Scripts
```bash
#!/bin/bash
if rbee queen status | grep -q "✅"; then
    echo "Queen is healthy"
    exit 0
else
    echo "Queen is not healthy"
    exit 1
fi
```

### 2. Monitoring
```bash
# Check status every 30 seconds
watch -n 30 rbee queen status
```

### 3. Startup Verification
```bash
# Start queen and verify it's running
rbee queen start
sleep 2
rbee queen status
```

### 4. Debugging
```bash
# Check if queen is running before submitting jobs
rbee queen status
rbee hive list
```

## Testing

```bash
# Build
cargo build -p rbee-keeper

# Test help
./target/debug/rbee-keeper queen --help

# Test status (queen not running)
./target/debug/rbee-keeper queen status
# Expected: ❌ Queen is not running on http://localhost:8500

# Start queen and test status
./target/debug/rbee-keeper queen start
./target/debug/rbee-keeper queen status
# Expected: ✅ Queen is running on http://localhost:8500
```

## Files Modified

- `bin/00_rbee_keeper/src/main.rs` - Added Status command and implementation
- `bin/00_rbee_keeper/src/operations.rs` - Added ACTION_QUEEN_STATUS constant
- `bin/API_REFERENCE.md` - Documented new command

## Future Enhancements

Possible improvements:
1. **JSON output mode** - `rbee queen status --json` for scripting
2. **Detailed status** - Show uptime, version, active jobs
3. **Exit codes** - Return 0 for running, 1 for not running
4. **Verbose mode** - Show connection details, response time

## Related Commands

- `rbee queen start` - Start queen-rbee daemon
- `rbee queen stop` - Stop queen-rbee daemon
- `rbee hive list` - List hives (requires queen to be running)
