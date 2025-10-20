# timeout-enforcer

**Created by:** TEAM-163  
**Purpose:** Hard timeout enforcement with visual countdown feedback

## Mission

**ZERO TOLERANCE FOR HANGING OPERATIONS.**

Every operation that could hang MUST use this crate. No exceptions.

## Features

- ✅ **Hard timeout enforcement** - Operation WILL fail after timeout
- ✅ **Visual countdown** - Shows remaining time in terminal
- ✅ **Clear error messages** - Tells you exactly what timed out
- ✅ **Easy to use** - Simple API, hard to misuse

## Usage

```rust
use timeout_enforcer::TimeoutEnforcer;
use std::time::Duration;

async fn potentially_hanging_operation() -> anyhow::Result<String> {
    // Your operation here
    Ok("done".to_string())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let result = TimeoutEnforcer::new(Duration::from_secs(30))
        .with_label("Starting queen-rbee")
        .enforce(potentially_hanging_operation())
        .await?;
    
    println!("Result: {}", result);
    Ok(())
}
```

## Visual Countdown

When running with countdown enabled (default):

```
⏱️  Starting queen-rbee (timeout: 30s)
⏱️  Starting queen-rbee ... 25s remaining
⏱️  Starting queen-rbee ... 20s remaining
⏱️  Starting queen-rbee ... 15s remaining
```

If timeout occurs:

```
❌ Starting queen-rbee TIMED OUT after 30s
Error: Starting queen-rbee timed out after 30 seconds - operation was hanging
```

## Silent Mode

For operations where you don't want countdown spam:

```rust
TimeoutEnforcer::new(Duration::from_secs(30))
    .silent()
    .enforce(my_operation())
    .await?;
```

## Where to Use

**MANDATORY for:**
- HTTP requests (client-side)
- Process spawning
- Health checks
- Shutdown operations
- Database operations
- File I/O that could block
- Network operations
- Any external system interaction

**Example locations:**
- `rbee-keeper` - All HTTP calls to queen/hive
- `queen-lifecycle` - Queen startup, health checks
- `daemon-lifecycle` - Process spawning
- HTTP clients - All requests
- Shutdown handlers - Graceful shutdown attempts

## Testing

```bash
cargo test -p timeout-enforcer
```

## Design Principles

1. **Fail fast** - Better to fail with clear error than hang forever
2. **Visual feedback** - User should see progress and know timeout is active
3. **Clear errors** - Error message must say what timed out and how long
4. **Easy to use** - Should be trivial to add timeout to any operation
5. **Hard to bypass** - No way to accidentally disable timeout

## Anti-Patterns

❌ **DON'T:**
```rust
// No timeout - can hang forever
let response = client.get(url).send().await?;
```

✅ **DO:**
```rust
// Hard timeout enforced
let response = TimeoutEnforcer::new(Duration::from_secs(30))
    .with_label("HTTP GET request")
    .enforce(async { 
        client.get(url).send().await.map_err(Into::into)
    })
    .await?;
```

## Maintenance

- **Owner:** Infrastructure team
- **Review:** Any PR that adds network/IO operations must use this crate
- **Violations:** PRs without timeout enforcement will be rejected
