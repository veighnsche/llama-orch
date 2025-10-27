# Timeout Enforcer - Quick Start Guide

**Version:** 0.2.0 (TEAM-330)  
**Universal**: Works everywhere (client, server, WASM)

---

## 🚀 Basic Usage (Client-Side)

```rust
use timeout_enforcer::TimeoutEnforcer;
use std::time::Duration;

// Just works! No context needed
let result = TimeoutEnforcer::new(Duration::from_secs(30))
    .with_label("Fetching data")
    .enforce(fetch_data())
    .await?;
```

---

## 🎯 Server-Side (SSE Routing)

```rust
use timeout_enforcer::TimeoutEnforcer;
use observability_narration_core::{NarrationContext, with_narration_context};
use std::time::Duration;

async fn handle_job(job_id: String) -> Result<()> {
    // Set context once at job start
    let ctx = NarrationContext::new().with_job_id(&job_id);
    
    with_narration_context(ctx, async {
        // All timeouts automatically include job_id!
        TimeoutEnforcer::new(Duration::from_secs(45))
            .with_label("Starting hive")
            .enforce(start_hive())
            .await?;
        
        TimeoutEnforcer::new(Duration::from_secs(30))
            .with_label("Health check")
            .enforce(health_check())
            .await?;
        
        Ok(())
    }).await
}
```

---

## 🔄 Migration from 0.1.0

### ❌ Old Way (Deprecated)

```rust
// Manual job_id passing (deprecated)
TimeoutEnforcer::new(timeout)
    .with_job_id(&job_id)  // ← Deprecated!
    .enforce(future).await
```

### ✅ New Way (Recommended)

```rust
// Context propagation (automatic)
let ctx = NarrationContext::new().with_job_id(&job_id);
with_narration_context(ctx, async {
    TimeoutEnforcer::new(timeout)
        .enforce(future).await  // ← job_id included automatically!
}).await
```

---

## 📋 Common Patterns

### Pattern 1: Simple Timeout (No Context)

```rust
TimeoutEnforcer::new(Duration::from_secs(10))
    .with_label("Quick operation")
    .enforce(quick_op())
    .await?;
```

### Pattern 2: With Progress Bar

```rust
TimeoutEnforcer::new(Duration::from_secs(60))
    .with_label("Long operation")
    .with_countdown()  // Show progress bar
    .enforce(long_op())
    .await?;
```

### Pattern 3: Silent Mode (Default)

```rust
TimeoutEnforcer::new(Duration::from_secs(30))
    .with_label("Background task")
    .silent()  // No progress bar (default)
    .enforce(background_task())
    .await?;
```

### Pattern 4: Multiple Operations (Same Context)

```rust
let ctx = NarrationContext::new()
    .with_job_id(&job_id)
    .with_correlation_id(&correlation_id);

with_narration_context(ctx, async {
    // All operations share the same context!
    TimeoutEnforcer::new(Duration::from_secs(10))
        .enforce(step1()).await?;
    
    TimeoutEnforcer::new(Duration::from_secs(20))
        .enforce(step2()).await?;
    
    TimeoutEnforcer::new(Duration::from_secs(30))
        .enforce(step3()).await?;
    
    Ok(())
}).await
```

### Pattern 5: Nested Timeouts

```rust
// Outer timeout: 60s total
TimeoutEnforcer::new(Duration::from_secs(60))
    .with_label("Full operation")
    .enforce(async {
        // Inner timeout: 10s for step 1
        TimeoutEnforcer::new(Duration::from_secs(10))
            .with_label("Step 1")
            .enforce(step1()).await?;
        
        // Inner timeout: 20s for step 2
        TimeoutEnforcer::new(Duration::from_secs(20))
            .with_label("Step 2")
            .enforce(step2()).await?;
        
        Ok(())
    })
    .await?;
```

### Pattern 6: Optional Context

```rust
pub async fn operation(job_id: Option<String>) -> Result<()> {
    let op = async {
        TimeoutEnforcer::new(Duration::from_secs(30))
            .enforce(do_work()).await
    };
    
    // Wrap in context if job_id provided
    if let Some(job_id) = job_id {
        let ctx = NarrationContext::new().with_job_id(&job_id);
        with_narration_context(ctx, op).await
    } else {
        op.await
    }
}
```

---

## 🎯 Recommended Timeouts

| Operation Type | Timeout | Reason |
|----------------|---------|--------|
| SSH command | 10s | Network operations should be fast |
| HTTP health check | 2s | Quick health endpoint |
| Daemon startup | 30-45s | Binary loading + initialization |
| File transfer (SCP) | 60s | Network + disk I/O |
| Health polling | 30s | Multiple retries with backoff |
| Database query | 5s | Should be fast or needs optimization |
| API call | 10s | Network + processing |

---

## 🔍 Debugging

### Enable Countdown for Debugging

```rust
// Show visual progress bar (useful for debugging)
TimeoutEnforcer::new(Duration::from_secs(30))
    .with_label("Debug operation")
    .with_countdown()  // ← Shows progress bar
    .enforce(operation())
    .await?;
```

### Check Context Propagation

```rust
use observability_narration_core::n;

let ctx = NarrationContext::new().with_job_id("test-123");
with_narration_context(ctx, async {
    // This narration will include job_id automatically
    n!("test", "Testing context propagation");
    
    // Timeout narration will also include job_id
    TimeoutEnforcer::new(Duration::from_secs(10))
        .enforce(operation()).await?;
    
    Ok(())
}).await
```

---

## ⚠️ Common Mistakes

### ❌ Mistake 1: Using deprecated `.with_job_id()`

```rust
// DON'T DO THIS (deprecated)
TimeoutEnforcer::new(timeout)
    .with_job_id(&job_id)  // ← Prints warning, does nothing
    .enforce(future).await
```

### ❌ Mistake 2: Setting context inside timeout

```rust
// DON'T DO THIS (context not propagated)
TimeoutEnforcer::new(timeout)
    .enforce(async {
        let ctx = NarrationContext::new().with_job_id(&job_id);
        with_narration_context(ctx, async {
            // Context set too late!
            operation().await
        }).await
    })
    .await
```

### ✅ Correct: Set context before timeout

```rust
// DO THIS (context propagates correctly)
let ctx = NarrationContext::new().with_job_id(&job_id);
with_narration_context(ctx, async {
    TimeoutEnforcer::new(timeout)
        .enforce(operation())
        .await
}).await
```

---

## 📚 See Also

- **Full Documentation**: `TEAM_330_UNIVERSAL_TIMEOUT.md`
- **Architecture**: How context propagation works
- **Migration Guide**: Upgrading from 0.1.0
- **Examples**: Real-world usage patterns

---

**TEAM-330: Universal timeout enforcement that works everywhere!** ✅
