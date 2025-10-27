# #[with_timeout] Attribute Macro Guide

**Version:** 0.2.0 (TEAM-330)  
**Purpose:** Ergonomic timeout enforcement for async functions

---

## 🎯 Philosophy

The `#[with_timeout]` macro is **syntactic sugar** over the core `TimeoutEnforcer` struct:

- **Core remains king**: `TimeoutEnforcer` is the source of truth
- **Macro is sugar**: Reduces call-site boilerplate
- **Policy enforcement**: Makes timeouts mandatory by default
- **Zero runtime cost**: Expands to the same code you'd write manually

---

## 🚀 Basic Usage

### Simple Timeout

```rust
use timeout_enforcer::with_timeout;
use anyhow::Result;

#[with_timeout(secs = 30)]
async fn fetch_data() -> Result<String> {
    // ... operation ...
    Ok("data".into())
}
```

**Expands to:**

```rust
async fn fetch_data() -> Result<String> {
    async fn __fetch_data_inner() -> Result<String> {
        // ... operation ...
        Ok("data".into())
    }
    
    timeout_enforcer::TimeoutEnforcer::new(Duration::from_secs(30))
        .enforce(__fetch_data_inner())
        .await
}
```

### With Label

```rust
#[with_timeout(secs = 45, label = "Starting hive")]
async fn start_hive() -> Result<()> {
    // ... operation ...
    Ok(())
}
```

### With Countdown

```rust
#[with_timeout(secs = 60, label = "Long operation", countdown = true)]
async fn long_operation() -> Result<()> {
    // ... operation ...
    Ok(())
}
```

---

## 📋 Macro Parameters

| Parameter | Required | Type | Default | Description |
|-----------|----------|------|---------|-------------|
| `secs` | ✅ Yes | `u64` | - | Timeout duration in seconds |
| `label` | ❌ No | `&str` | `"Operation"` | Human-readable label |
| `countdown` | ❌ No | `bool` | `false` | Show visual progress bar |

---

## 🎨 Usage Patterns

### Pattern 1: Simple Function

```rust
#[with_timeout(secs = 10)]
async fn quick_check() -> Result<bool> {
    // ... check ...
    Ok(true)
}
```

### Pattern 2: With Parameters

```rust
#[with_timeout(secs = 30, label = "Processing")]
async fn process_data(data: Vec<u8>, config: &Config) -> Result<String> {
    // ... process ...
    Ok("processed".into())
}
```

### Pattern 3: With Mutable Parameters

```rust
#[with_timeout(secs = 20)]
async fn update_state(mut state: State) -> Result<State> {
    state.update();
    Ok(state)
}
```

### Pattern 4: With References

```rust
#[with_timeout(secs = 15, label = "Validating")]
async fn validate(input: &str) -> Result<bool> {
    // ... validate ...
    Ok(true)
}
```

### Pattern 5: Generic Functions

```rust
#[with_timeout(secs = 25)]
async fn fetch<T: DeserializeOwned>(url: &str) -> Result<T> {
    // ... fetch and deserialize ...
}
```

---

## 🔄 Context Propagation

The macro works seamlessly with `NarrationContext`:

```rust
use observability_narration_core::{NarrationContext, with_narration_context};

#[with_timeout(secs = 30, label = "Starting daemon")]
async fn start_daemon() -> Result<()> {
    // ... start daemon ...
    Ok(())
}

async fn handle_job(job_id: String) -> Result<()> {
    let ctx = NarrationContext::new().with_job_id(&job_id);
    
    with_narration_context(ctx, async {
        // Timeout narration automatically includes job_id!
        start_daemon().await?;
        Ok(())
    }).await
}
```

---

## ⚖️ When to Use Macro vs Struct

### Use `#[with_timeout]` Macro When:

✅ Function always needs the same timeout  
✅ Timeout is part of the function's contract  
✅ You want to enforce policy at the function level  
✅ You want cleaner call sites  

```rust
// Policy: All SSH operations timeout after 10s
#[with_timeout(secs = 10, label = "SSH command")]
async fn ssh_exec(cmd: &str) -> Result<String> {
    // ... execute ...
}
```

### Use `TimeoutEnforcer` Struct When:

✅ Timeout varies per call site  
✅ Timeout is configurable  
✅ You need conditional timeout logic  
✅ You're wrapping external functions  

```rust
// Timeout varies based on operation type
async fn execute_operation(op: Operation, timeout: Duration) -> Result<()> {
    TimeoutEnforcer::new(timeout)
        .with_label(&op.name)
        .enforce(op.execute())
        .await
}
```

---

## 🔍 Macro Expansion Examples

### Example 1: No Parameters

```rust
#[with_timeout(secs = 30)]
async fn my_fn() -> Result<()> {
    Ok(())
}
```

**Expands to:**

```rust
async fn my_fn() -> Result<()> {
    async fn __my_fn_inner() -> Result<()> {
        Ok(())
    }
    
    {
        use std::time::Duration;
        timeout_enforcer::TimeoutEnforcer::new(Duration::from_secs(30))
            .enforce(__my_fn_inner())
            .await
    }
}
```

### Example 2: With Parameters

```rust
#[with_timeout(secs = 20, label = "Processing")]
async fn process(x: i32, y: i32) -> Result<i32> {
    Ok(x + y)
}
```

**Expands to:**

```rust
async fn process(x: i32, y: i32) -> Result<i32> {
    async fn __process_inner(x: i32, y: i32) -> Result<i32> {
        Ok(x + y)
    }
    
    {
        use std::time::Duration;
        timeout_enforcer::TimeoutEnforcer::new(Duration::from_secs(20))
            .with_label("Processing")
            .enforce(__process_inner(x, y))
            .await
    }
}
```

---

## 🎯 Real-World Examples

### Example 1: Remote Daemon Operations

```rust
use timeout_enforcer::with_timeout;
use anyhow::Result;

#[with_timeout(secs = 10, label = "Finding binary")]
async fn find_remote_binary(ssh: &SshConfig, name: &str) -> Result<String> {
    // SSH command to find binary
    ssh_exec(ssh, &format!("which {}", name)).await
}

#[with_timeout(secs = 10, label = "Starting daemon")]
async fn start_remote_daemon(ssh: &SshConfig, binary: &str) -> Result<u32> {
    // SSH command to start daemon
    let output = ssh_exec(ssh, &format!("nohup {} &", binary)).await?;
    Ok(output.trim().parse()?)
}

#[with_timeout(secs = 30, label = "Health polling")]
async fn poll_daemon_health(url: &str) -> Result<()> {
    // Poll health endpoint with retries
    for attempt in 1..=10 {
        if check_health(url).await? {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(200 * attempt)).await;
    }
    anyhow::bail!("Health check failed")
}
```

### Example 2: HTTP Operations

```rust
#[with_timeout(secs = 5, label = "HTTP GET")]
async fn http_get(url: &str) -> Result<String> {
    let response = reqwest::get(url).await?;
    Ok(response.text().await?)
}

#[with_timeout(secs = 10, label = "HTTP POST")]
async fn http_post(url: &str, body: &str) -> Result<()> {
    let client = reqwest::Client::new();
    client.post(url).body(body.to_string()).send().await?;
    Ok(())
}
```

### Example 3: Database Operations

```rust
#[with_timeout(secs = 5, label = "Database query")]
async fn query_user(db: &Database, user_id: i64) -> Result<User> {
    db.query_one("SELECT * FROM users WHERE id = $1", &[&user_id]).await
}

#[with_timeout(secs = 10, label = "Database transaction")]
async fn update_user(db: &Database, user: &User) -> Result<()> {
    let tx = db.transaction().await?;
    tx.execute("UPDATE users SET name = $1 WHERE id = $2", &[&user.name, &user.id]).await?;
    tx.commit().await?;
    Ok(())
}
```

---

## ⚠️ Requirements

1. **Function must be `async`**
   ```rust
   // ❌ ERROR: Not async
   #[with_timeout(secs = 10)]
   fn sync_function() -> Result<()> { Ok(()) }
   
   // ✅ OK: Async function
   #[with_timeout(secs = 10)]
   async fn async_function() -> Result<()> { Ok(()) }
   ```

2. **Function must return `Result<T>`**
   ```rust
   // ❌ ERROR: No Result
   #[with_timeout(secs = 10)]
   async fn no_result() -> String { "ok".into() }
   
   // ✅ OK: Returns Result
   #[with_timeout(secs = 10)]
   async fn with_result() -> Result<String> { Ok("ok".into()) }
   ```

3. **`timeout-enforcer` must be in scope**
   ```rust
   use timeout_enforcer::with_timeout;  // ← Required!
   
   #[with_timeout(secs = 10)]
   async fn my_fn() -> Result<()> { Ok(()) }
   ```

---

## 🐛 Debugging

### Enable Countdown for Debugging

```rust
// Show visual progress bar (useful for debugging slow operations)
#[with_timeout(secs = 60, label = "Debug operation", countdown = true)]
async fn debug_operation() -> Result<()> {
    // ... operation ...
    Ok(())
}
```

### Check Macro Expansion

```bash
# See what the macro expands to
cargo expand --test macro_tests
```

---

## 📚 See Also

- **Core Implementation**: `TimeoutEnforcer` struct
- **Universal Context**: `TEAM_330_UNIVERSAL_TIMEOUT.md`
- **Quick Start**: `QUICK_START.md`
- **Tests**: `tests/macro_tests.rs`

---

**TEAM-330: Ergonomic timeout enforcement with zero runtime cost!** ✅
