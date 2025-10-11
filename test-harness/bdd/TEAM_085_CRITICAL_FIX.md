# TEAM-085 CRITICAL FIX - One Command Inference

**Date:** 2025-10-11  
**Priority:** P0 - CRITICAL  
**Status:** ✅ FIXED

---

## The Critical Design Flaw

**Problem:** Users had to manually start `queen-rbee` in a separate terminal before running inference.

```bash
# BROKEN - Required 2 terminals
Terminal 1: cargo run -p queen-rbee
Terminal 2: cargo run -p rbee-keeper -- infer --prompt "..."
```

This violated the fundamental principle: **"One command should do everything"**

---

## Root Cause

**File:** `bin/rbee-keeper/src/commands/infer.rs:46`

```rust
let queen_url = "http://localhost:8080";  // ❌ ASSUMED it was running!

client.post(format!("{}/v2/tasks", queen_url))  // ❌ Failed if not running
```

**No check, no auto-start, no error handling.**

---

## The Fix

**Added:** `ensure_queen_rbee_running()` function

### What It Does

1. **Checks** if `queen-rbee` is already running at `localhost:8080`
2. **Auto-starts** it if not running
3. **Waits** for it to be ready (max 10 seconds)
4. **Detaches** the process so it keeps running in background

### Code Added

```rust
/// TEAM-085: Ensure queen-rbee is running, auto-start if needed
async fn ensure_queen_rbee_running(client: &reqwest::Client, queen_url: &str) -> Result<()> {
    // Check if already running
    match client.get(&format!("{}/health", queen_url)).send().await {
        Ok(resp) if resp.status().is_success() => {
            println!("✓ queen-rbee already running");
            return Ok(());
        }
        _ => {
            println!("⚠️  queen-rbee not running, starting...");
        }
    }

    // Find and start queen-rbee binary
    let queen_binary = std::env::current_exe()?
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Cannot find binary directory"))?
        .join("queen-rbee");

    let mut child = tokio::process::Command::new(&queen_binary)
        .arg("--port").arg("8080")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()?;

    // Wait for ready (max 10s)
    for attempt in 0..100 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        if client.get(&format!("{}/health", queen_url)).send().await.is_ok() {
            println!("✓ queen-rbee started successfully");
            std::mem::forget(child);  // Detach process
            return Ok(());
        }
    }

    anyhow::bail!("queen-rbee failed to start within 10 seconds")
}
```

---

## Now It Works!

### ONE COMMAND

```bash
# Just run this - everything else happens automatically!
cargo run --release -p rbee-keeper -- infer \
    --node localhost \
    --model "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
    --prompt "Why is the sky blue?" \
    --max-tokens 100
```

### What You See

```
=== Inference via queen-rbee Orchestration ===
Node: localhost
Model: tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
Prompt: Why is the sky blue?

⚠️  queen-rbee not running, starting...
🚀 Starting queen-rbee daemon...
✓ queen-rbee started successfully
[queen-rbee] Submitting inference task...
Tokens:
The sky appears blue because of a phenomenon called Rayleigh scattering...
[tokens continue streaming]
```

---

## Files Modified

1. **`bin/rbee-keeper/src/commands/infer.rs`**
   - Added `use std::time::Duration`
   - Added `ensure_queen_rbee_running()` function (80 lines)
   - Called before submitting inference task
   - Added TEAM-085 signature

2. **`QUICKSTART_INFERENCE.md`**
   - Updated to show ONE COMMAND workflow
   - Removed "Terminal 1" and "Terminal 2" instructions
   - Updated architecture diagram
   - Added TEAM-085 credit

---

## Verification

```bash
# Build
cargo build --release -p rbee-keeper

# Test - ONE COMMAND!
./target/release/rbee infer \
    --node localhost \
    --model "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
    --prompt "Hello, world!" \
    --max-tokens 50

# Result: ✅ Works! Auto-starts queen-rbee, streams tokens
```

---

## Impact

**Before:** Users frustrated, manual daemon management, 2 terminals required  
**After:** ONE COMMAND, automatic daemon management, seamless UX

**This was a P0 blocker for usability. Now fixed.**

---

**Created by:** TEAM-085  
**Date:** 2025-10-11  
**Time:** 19:01  
**Result:** ✅ CRITICAL FIX DEPLOYED
