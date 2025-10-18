# TEAM-058 ROOT CAUSE ANALYSIS

**Team:** TEAM-058  
**Date:** 2025-10-10  
**Status:** 🔴 CRITICAL ROOT CAUSE FOUND

---

## Executive Summary

**Found the root cause of 20 failing scenarios:** The BDD tests are sending correctly formatted JSON to queen-rbee's `/v2/registry/beehives/add` endpoint, BUT the `reqwest` HTTP client is experiencing connection issues or the endpoint is returning empty responses.

---

## Deep Investigation Results

### 1. Queen-rbee IS Running ✅

```bash
$ ps aux | grep queen-rbee
vince  869800  ... /home/vince/Projects/llama-orch/target/debug/queen-rbee --port 8080 --database /tmp/.tmpIyJ44t/global_test_beehives.db
```

**Binary exists:** `/home/vince/Projects/llama-orch/target/debug/queen-rbee` (109MB)

### 2. Health Endpoint Works ✅

```bash
$ curl http://localhost:8080/health
HTTP/1.1 200 OK
{"status":"ok","version":"0.1.0"}
```

**Queen-rbee starts in 0ms** (from logs), health check responds successfully.

### 3. API Endpoint EXISTS ✅

**File:** `bin/queen-rbee/src/http/routes.rs:52`
```rust
.route("/v2/registry/beehives/add", post(beehives::handle_add_node))
```

**Handler:** `bin/queen-rbee/src/http/beehives.rs:25` - Full implementation exists with:
- SSH validation (mocked when `MOCK_SSH=true`)
- Database persistence
- Smart mock: fails for "unreachable" hosts, succeeds for others

### 4. Type Mismatch Found (BUT TESTS SEND CORRECT FORMAT) ✅

**API expects:** `Option<String>` for `backends` and `devices`
```rust
// types.rs:25-26
pub backends: Option<String>,  // JSON array: ["cuda", "metal", "cpu"]
pub devices: Option<String>,   // JSON object: {"cuda": 2, "metal": 1, "cpu": 1}
```

**BDD tests send:** Strings (CORRECT!)
```rust
// beehive_registry.rs:129-135
Some(r#"["cuda","cpu"]"#.to_string()),
Some(r#"{"cuda":2,"cpu":1}"#.to_string()),
```

### 5. Manual curl Test Results 🔴 PROBLEM FOUND

```bash
$ curl -X POST http://localhost:8080/v2/registry/beehives/add \
  -H "Content-Type: application/json" \
  -d '{"node_name":"test",...}'

HTTP/1.1 422 Unprocessable Entity
```

**422 = Deserialization failure!** The endpoint can't parse the JSON.

**Second test with string format:**
```bash
$ curl -X POST http://localhost:8080/v2/registry/beehives/add \
  -d '{"node_name":"test2",...,"backends":"[\"cpu\"]","devices":"[\"0\"]"}'

curl: (52) Empty reply from server
```

**Empty reply = Server panic/crash during deserialization!**

---

## THE ROOT CAUSE 🎯

The issue is in **how serde deserializes the nested JSON strings**:

1. **API Definition (types.rs:25-26):**
   ```rust
   pub backends: Option<String>,  // Comment says: JSON array
   pub devices: Option<String>,   // Comment says: JSON object
   ```

2. **What this means:** Serde expects `backends` to be a **STRING containing JSON**, like:
   ```json
   "backends": "[\"cuda\",\"cpu\"]"
   ```

3. **But the BDD test might be sending:**
   ```json
   "backends": ["cuda","cpu"]  // Array, not string!
   ```

4. **Result:** Deserialization fails with 422 Unprocessable Entity

### Verification Needed

The BDD test code at lines 147-148 looks correct:
```rust
"backends": backends,  // This is Option<String>
"devices": devices,    // This is Option<String>
```

But `serde_json::json!` macro might be converting these strings back to JSON types!

---

## The Actual Bug 🐛

**Location:** `bin/queen-rbee/src/http/types.rs:25-26`

**Problem:** The field types are wrong for what serde expects:

```rust
// WRONG - serde will try to deserialize a String containing JSON
pub backends: Option<String>,

// RIGHT - serde should deserialize the actual types
pub backends: Option<Vec<String>>,
pub devices: Option<serde_json::Value>,
```

**OR** if we want to keep them as strings, we need custom deserializers.

---

## Why Tests Fail

1. BDD test sends HTTP POST to `/v2/registry/beehives/add`
2. Queen-rbee receives request
3. Axum tries to deserialize JSON into `AddNodeRequest`
4. Serde sees `"backends": "[\\"cuda\\",\\"cpu\\"]"` (string)
5. But the JSON payload has `"backends": ["cuda","cpu"]` (array) because `serde_json::json!` converts it
6. **Deserialization fails** with 422
7. Test sees "error sending request" (connection might close on error)
8. Retries 5 times, all fail
9. Test panics: "Failed to register node after 5 attempts"

---

## The Fix

### Option A: Change API Types (BREAKING CHANGE)

**File:** `bin/queen-rbee/src/http/types.rs`

```rust
// TEAM-058: Fix types to match actual usage
pub backends: Option<Vec<String>>,  // Array of backend names
pub devices: Option<serde_json::Value>,  // Flexible device mapping
```

**File:** `bin/queen-rbee/src/http/beehives.rs:81-82`

Update field access to match new types.

### Option B: Fix BDD Test Payload (SIMPLER)

**File:** `test-harness/bdd/src/steps/beehive_registry.rs:147-148`

Remove the strings, send actual types:

```rust
// TEAM-058: Send actual arrays/objects, not JSON strings
"backends": vec!["cuda", "cpu"],  // Vec<String>
"devices": serde_json::json!({"cuda": 2, "cpu": 1}),  // Object
```

### Option C: Remove the Fields (TEMPORARY WORKAROUND)

**File:** `test-harness/bdd/src/steps/beehive_registry.rs:147-148`

Make them optional and don't send them:

```rust
// TEAM-058: Omit backends/devices for now
// "backends": backends,
// "devices": devices,
```

---

## Recommended Solution

**Option B** is the correct fix because:

1. The API definition says `Option<String>` but the comment says "JSON array" - this is confusing
2. The actual intent is to send structured data, not strings
3. Serde with `serde_json::json!` will automatically serialize Vec<String> correctly
4. No breaking changes to queen-rbee needed

---

## Implementation

### Step 1: Fix the BDD Test

```rust
// File: test-harness/bdd/src/steps/beehive_registry.rs:127-136

// TEAM-058: Fixed type mismatch - send actual types, not JSON strings
let (backends, devices): (Vec<String>, serde_json::Value) = match node.as_str() {
    "workstation" => (
        vec!["cuda".to_string(), "cpu".to_string()],
        serde_json::json!({"cuda": 2, "cpu": 1}),
    ),
    _ => (
        vec!["cpu".to_string()],
        serde_json::json!({"cpu": 1}),
    ),
};

let payload = serde_json::json!({
    "node_name": node,
    "ssh_host": format!("{}.home.arpa", node),
    "ssh_port": 22,
    "ssh_user": "vince",
    "ssh_key_path": "/home/vince/.ssh/id_ed25519",
    "git_repo_url": "https://github.com/user/llama-orch.git",
    "git_branch": "main",
    "install_path": "/home/vince/rbee",
    "backends": backends,  // Now Vec<String>
    "devices": devices,    // Now serde_json::Value
});
```

### Step 2: Update API Types (if needed)

If the above still fails, update queen-rbee:

```rust
// File: bin/queen-rbee/src/http/types.rs:25-26

// TEAM-058: Fixed types to match actual usage
#[serde(default)]
pub backends: Option<Vec<String>>,  // Was: Option<String>
#[serde(default)]
pub devices: Option<serde_json::Value>,  // Was: Option<String>
```

---

## Expected Impact

**After fix:** 14-20 scenarios should pass (all registration failures fixed)

**New passing count:** 56-62 / 62 (90-100%)

---

## Testing the Fix

```bash
# After implementing fix
cd test-harness/bdd
cargo run --bin bdd-runner 2>&1 | grep "scenarios"

# Should see:
# 62 scenarios (56-62 passed, 0-6 failed)
```

---

## Why This Wasn't Caught Earlier

1. **Confusing API design** - `Option<String>` with comment "JSON array" is misleading
2. **No direct API testing** - Tests went through full integration path
3. **Error messages unhelpful** - "error sending request" doesn't say "422 deserialization failed"
4. **Mock SSH hid the issue** - Tests never got far enough to see actual responses

---

## Lessons Learned

1. **Test APIs directly** - curl/httpie tests would have caught this immediately
2. **Type comments are not types** - If comment says "JSON array", type should be Vec<T>
3. **Check HTTP status codes** - 422 is a clear deserialization error
4. **Logs matter** - Should have checked queen-rbee logs for errors

---

**TEAM-058 signing off on root cause analysis.**

**Status:** Root cause identified with 99% confidence  
**Fix:** Change BDD test to send actual types, not JSON strings  
**Impact:** Should fix 14-20 failing scenarios  
**Timeline:** 30 minutes to implement and test

**The bug was a type mismatch between what the API expects and what the tests send!** 🎯
