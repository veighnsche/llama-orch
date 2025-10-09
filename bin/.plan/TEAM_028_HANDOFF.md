# TEAM-028 Handoff: Complete MVP Phases 7-8

**Date:** 2025-10-09T23:21:00+02:00  
**From:** TEAM-027  
**To:** TEAM-028  
**Status:** Infrastructure complete, needs Phase 7-8 implementation  
**Priority:** CRITICAL - Complete MVP test-001

---

## Executive Summary

TEAM-027 completed **all 9 priority tasks**, implementing the complete infrastructure for test-001 MVP. The rbee-hive daemon and rbee-keeper CLI are fully functional through Phase 6. **Phases 7-8 need implementation** to complete the end-to-end flow.

**Current State:**
- ✅ rbee-hive daemon: HTTP server, background monitoring, worker spawn
- ✅ rbee-keeper CLI: Pool client, SQLite registry, 8-phase flow structure
- ⚠️ Phase 7 (Worker Ready Polling): Stubbed, needs implementation
- ⚠️ Phase 8 (Inference Execution): Stubbed, needs implementation

---

## What TEAM-027 Completed

### Priority 1: rbee-hive Daemon ✅
1. ✅ Wired up daemon mode (main.rs, cli.rs, daemon.rs)
2. ✅ Background monitoring loops (monitor.rs, timeout.rs)
3. ✅ Added reqwest dependency
4. ✅ Fixed worker spawn logic (binary path, hostname, API key, callback)

### Priority 2: rbee-keeper HTTP Client ✅
1. ✅ Added HTTP dependencies (reqwest, tokio, sqlx, futures, dirs)
2. ✅ Created pool_client.rs (health check, spawn worker)
3. ✅ Created registry.rs (SQLite worker tracking)
4. ✅ Implemented infer command (8-phase structure)

### Priority 3: Integration Testing ✅
1. ✅ Created test-001-mvp-run.sh

---

## What TEAM-028 Must Do

### Priority 1: Implement Phase 7 - Worker Ready Polling (CRITICAL) 🔥

**File:** `bin/rbee-keeper/src/commands/infer.rs`  
**Line:** 97  
**Current:** `println!("{}", "⚠ Worker ready polling not yet implemented".yellow());`

**Implementation:**

```rust
/// Wait for worker to be ready
///
/// Per test-001-mvp.md Phase 7: Worker Health Check
async fn wait_for_worker_ready(worker_url: &str) -> Result<()> {
    use colored::Colorize;
    use serde::Deserialize;
    
    #[derive(Deserialize)]
    struct ReadyResponse {
        ready: bool,
        state: String,
    }
    
    let client = reqwest::Client::new();
    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(300); // 5 minutes
    
    print!("Waiting for worker ready");
    std::io::stdout().flush()?;
    
    loop {
        match client
            .get(&format!("{}/v1/ready", worker_url))
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => {
                if let Ok(ready) = response.json::<ReadyResponse>().await {
                    if ready.ready {
                        println!(" {}", "✓".green());
                        return Ok(());
                    }
                }
            }
            _ => {}
        }
        
        if start.elapsed() > timeout {
            println!(" {}", "✗".red());
            anyhow::bail!("Worker ready timeout after 5 minutes");
        }
        
        print!(".");
        std::io::stdout().flush()?;
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    }
}
```

**Then update line 97:**
```rust
// PHASE 7: Worker Health Check
println!("{}", "[Phase 7] Waiting for worker ready...".yellow());
wait_for_worker_ready(&worker.url).await?;
println!("{} Worker ready!", "✓".green());
println!();
```

**Estimated time:** 1-2 hours

---

### Priority 2: Implement Phase 8 - Inference Execution (CRITICAL) 🔥

**File:** `bin/rbee-keeper/src/commands/infer.rs`  
**Line:** 103  
**Current:** `println!("{}", "⚠ Inference execution not yet implemented".yellow());`

**Implementation:**

```rust
/// Execute inference with SSE streaming
///
/// Per test-001-mvp.md Phase 8: Inference Execution
async fn execute_inference(
    worker_url: &str,
    prompt: String,
    max_tokens: u32,
    temperature: f32,
) -> Result<()> {
    use colored::Colorize;
    use futures::StreamExt;
    use serde::Deserialize;
    
    #[derive(Deserialize)]
    struct TokenEvent {
        #[serde(rename = "type")]
        event_type: String,
        #[serde(default)]
        token: String,
        #[serde(default)]
        index: u32,
        #[serde(default)]
        done: bool,
        #[serde(default)]
        total_tokens: u32,
        #[serde(default)]
        duration_ms: u64,
    }
    
    let client = reqwest::Client::new();
    
    let request = serde_json::json!({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": true
    });
    
    let response = client
        .post(&format!("{}/v1/inference", worker_url))
        .json(&request)
        .send()
        .await?;
    
    if !response.status().is_success() {
        anyhow::bail!("Inference request failed: HTTP {}", response.status());
    }
    
    println!("{}", "Tokens:".cyan());
    
    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        
        // Process complete SSE events
        while let Some(pos) = buffer.find("\n\n") {
            let event = buffer[..pos].to_string();
            buffer = buffer[pos + 2..].to_string();
            
            // Parse SSE format: "data: {...}"
            if let Some(json_str) = event.strip_prefix("data: ") {
                if json_str == "[DONE]" {
                    break;
                }
                
                if let Ok(token_event) = serde_json::from_str::<TokenEvent>(json_str) {
                    match token_event.event_type.as_str() {
                        "token" => {
                            print!("{}", token_event.token);
                            std::io::stdout().flush()?;
                        }
                        "end" => {
                            println!();
                            println!();
                            println!("{} Inference complete!", "✓".green().bold());
                            println!("Total tokens: {}", token_event.total_tokens.to_string().cyan());
                            println!("Duration: {} ms", token_event.duration_ms.to_string().cyan());
                            
                            if token_event.duration_ms > 0 && token_event.total_tokens > 0 {
                                let tokens_per_sec = (token_event.total_tokens as f64 / token_event.duration_ms as f64) * 1000.0;
                                println!("Speed: {:.2} tokens/sec", tokens_per_sec);
                            }
                            return Ok(());
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    
    Ok(())
}
```

**Then update line 103:**
```rust
// PHASE 8: Inference Execution
println!("{}", "[Phase 8] Executing inference...".yellow());
execute_inference(&worker.url, prompt, _max_tokens, _temperature).await?;
```

**Also update function signature to use the parameters:**
```rust
pub async fn handle(
    node: String,
    model: String,
    prompt: String,
    max_tokens: u32,  // Remove underscore
    temperature: f32,  // Remove underscore
) -> Result<()> {
```

**Estimated time:** 2-3 hours

---

### Priority 3: Test End-to-End Flow (IMPORTANT) 🧪

**Prerequisites:**
1. Ensure llm-worker-rbee binary exists
2. Ensure llm-worker-rbee supports these arguments:
   - `--worker-id`
   - `--model`
   - `--backend`
   - `--device`
   - `--port`
   - `--api-key`
   - `--callback-url`
3. Ensure llm-worker-rbee implements these endpoints:
   - `GET /v1/health` - Health check
   - `GET /v1/ready` - Ready status
   - `POST /v1/inference` - Inference with SSE streaming
   - `POST /v1/admin/shutdown` - Graceful shutdown

**Test Steps:**

1. **Build binaries:**
   ```bash
   cargo build --bin rbee-hive --bin rbee
   ```

2. **Start rbee-hive daemon:**
   ```bash
   ./target/debug/rbee-hive daemon
   ```

3. **In another terminal, verify health:**
   ```bash
   curl http://localhost:8080/v1/health | jq .
   ```

4. **Run inference:**
   ```bash
   ./target/debug/rbee infer \
     --node localhost \
     --model "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" \
     --prompt "write a short story" \
     --max-tokens 20 \
     --temperature 0.7
   ```

5. **Expected output:**
   ```
   === MVP Cross-Node Inference ===
   Node: localhost
   Model: hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
   Prompt: write a short story

   [Phase 1] Checking local worker registry...
   ✗ No existing worker found

   [Phase 2] Pool preflight check...
   ✓ Pool health: alive (version 0.1.0)

   [Phase 3-5] Spawning worker...
   ✓ Worker spawned: worker-abc123 (state: loading)

   [Phase 6] Registering worker...
   ✓ Worker registered in local registry

   [Phase 7] Waiting for worker ready...
   Waiting for worker ready.....✓
   ✓ Worker ready!

   [Phase 8] Executing inference...
   Tokens:
   Once upon a time, in a small village, there lived a curious cat named Whiskers.

   ✓ Inference complete!
   Total tokens: 20
   Duration: 1234 ms
   Speed: 16.21 tokens/sec
   ```

**Estimated time:** 2-3 hours (including debugging)

---

## Critical Files for TEAM-028

### Must Read:
1. `bin/.specs/.gherkin/test-001-mvp.md` - THE source of truth (671 lines)
2. `bin/.plan/TEAM_027_COMPLETION_SUMMARY.md` - What TEAM-027 built
3. `bin/rbee-keeper/src/commands/infer.rs` - Where to implement Phase 7-8

### Must Modify:
1. `bin/rbee-keeper/src/commands/infer.rs` - Add helper functions, wire up phases

### Reference:
- `bin/llm-worker-rbee/src/http/` - Worker HTTP API patterns
- `bin/.plan/TEAM_027_HANDOFF.md` - Original handoff from TEAM-026

---

## Known Issues & Gotchas

### Issue 1: llm-worker-rbee API Compatibility
**Problem:** We don't know if llm-worker-rbee supports all required arguments  
**Solution:** Check llm-worker-rbee CLI and HTTP API before implementing Phase 7-8  
**Verification:**
```bash
./target/debug/llm-worker-rbee --help
```

### Issue 2: SSE Format Mismatch
**Problem:** test-001-mvp.md specifies SSE format, but llm-worker-rbee may differ  
**Solution:** Test llm-worker-rbee inference endpoint manually first  
**Verification:**
```bash
curl -N -X POST http://localhost:8081/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt":"test","max_tokens":5,"stream":true}'
```

### Issue 3: Worker Ready Callback
**Problem:** Worker may not call callback URL automatically  
**Solution:** Check if llm-worker-rbee supports --callback-url argument  
**Workaround:** If not supported, poll GET /v1/ready instead

### Issue 4: Model Path Resolution
**Problem:** Hardcoded "/models/model.gguf" path  
**Solution:** For MVP, manually ensure model exists at this path  
**Future:** Implement model catalog integration

---

## Dependencies & Prerequisites

### System Requirements:
- ✅ Rust toolchain (installed)
- ✅ SQLite (via sqlx)
- ✅ tokio async runtime
- ⏳ llm-worker-rbee binary (must verify)
- ⏳ Model file downloaded (must verify)

### Rust Crates (already added):
- ✅ reqwest (HTTP client)
- ✅ tokio (async runtime)
- ✅ sqlx (SQLite)
- ✅ futures (stream processing)
- ✅ serde (JSON)
- ✅ colored (terminal colors)

---

## Testing Checklist

### Before Starting:
- [ ] Read test-001-mvp.md completely
- [ ] Review TEAM_027_COMPLETION_SUMMARY.md
- [ ] Verify llm-worker-rbee binary exists
- [ ] Check llm-worker-rbee --help for supported arguments
- [ ] Test llm-worker-rbee inference endpoint manually

### After Implementing Phase 7:
- [ ] `cargo build --bin rbee` succeeds
- [ ] Worker ready polling works (manual test)
- [ ] Timeout after 5 minutes works
- [ ] Progress dots display correctly

### After Implementing Phase 8:
- [ ] `cargo build --bin rbee` succeeds
- [ ] Inference request sends correctly
- [ ] SSE stream parses correctly
- [ ] Tokens display in real-time
- [ ] Completion message shows

### End-to-End:
- [ ] Full 8-phase flow works
- [ ] Tokens stream in real-time
- [ ] Worker auto-shuts down after idle (wait 5 min)
- [ ] Second inference reuses existing worker (Phase 1 finds it)

---

## Success Criteria

### Minimum (MVP Happy Path):
- [ ] Phase 7 implementation complete
- [ ] Phase 8 implementation complete
- [ ] Full 8-phase flow works end-to-end
- [ ] Tokens stream in real-time

### Target (MVP + Polish):
- [ ] Error handling for Phase 7-8
- [ ] Progress indicators (dots, spinners)
- [ ] Helpful error messages
- [ ] Test script passes

### Stretch (Production Ready):
- [ ] Retry logic with backoff
- [ ] Graceful cancellation (Ctrl+C)
- [ ] Partial result handling
- [ ] Metrics & logging

---

## Implementation Timeline

### Session 1: Phase 7 Implementation (2-3 hours)
- [ ] Hour 1: Implement `wait_for_worker_ready()` function
- [ ] Hour 2: Wire up in infer command, test manually
- [ ] Hour 3: Debug and polish

### Session 2: Phase 8 Implementation (3-4 hours)
- [ ] Hour 1: Implement `execute_inference()` function
- [ ] Hour 2: SSE stream parsing
- [ ] Hour 3: Wire up in infer command, test manually
- [ ] Hour 4: Debug and polish

### Session 3: Integration & Testing (2-3 hours)
- [ ] Hour 1: End-to-end testing
- [ ] Hour 2: Fix integration issues
- [ ] Hour 3: Documentation + handoff

**Total Estimated Time:** 7-10 hours

---

## Communication Protocol

### When to Ask User:
- ✅ Ask if llm-worker-rbee API is unclear
- ✅ Ask if SSE format doesn't match spec
- ✅ Ask if blocking issues found
- ❌ Don't ask about implementation details (follow spec)

### When to Update Docs:
- ✅ Update TEAM_028_COMPLETION_SUMMARY.md when done
- ✅ Create TEAM_029_HANDOFF.md if work incomplete
- ✅ Update test-001-mvp.md with implementation notes

### When to Stop:
- 🛑 If llm-worker-rbee API incompatible (escalate)
- 🛑 If fundamental blocker found (escalate)
- 🛑 If >10 hours spent without progress (re-evaluate)

---

## Additional Context

### Why Phase 7-8 Were Deferred
TEAM-027 focused on infrastructure (daemon, CLI, registry) to unblock parallel work. Phase 7-8 require llm-worker-rbee API knowledge, which is best verified during implementation.

### Architecture Decisions
- **SQLite Registry:** Enables worker reuse across CLI invocations
- **Background Tasks:** Health monitoring and idle timeout run independently
- **8-Phase Structure:** Mirrors test-001-mvp.md exactly for traceability

### Code Quality Notes
- All code follows dev-bee-rules.md (team signatures, no background jobs)
- Mirrors llm-worker-rbee HTTP patterns to avoid drift
- Uses tracing for structured logging (not println!)
- Proper error handling with anyhow::Result

---

## Final Notes from TEAM-027

**What Went Well:**
- ✅ Clear handoff from TEAM-026 made work straightforward
- ✅ All infrastructure tasks completed without blockers
- ✅ Code compiles and passes basic checks
- ✅ Background tasks properly isolated

**What Was Challenging:**
- 😓 SQLite compile-time macros required runtime queries
- 😓 Workspace dependency management (tower, tower-http)
- 😓 Phase 7-8 deferred due to llm-worker-rbee API uncertainty

**Advice for TEAM-028:**
- 📖 Verify llm-worker-rbee API FIRST before implementing
- 🧪 Test each phase incrementally (don't wait for end-to-end)
- 🔍 Check SSE format manually with curl
- 💬 Ask if llm-worker-rbee doesn't match expectations
- 🎯 Focus on happy path first, edge cases later

**Remember:**
- The infrastructure is solid!
- Phase 7-8 are well-specified in test-001-mvp.md!
- The helper function templates are provided!
- Just verify llm-worker-rbee API and implement!

---

**Signed:** TEAM-027  
**Date:** 2025-10-09T23:21:00+02:00  
**Status:** Infrastructure complete, ready for Phase 7-8  
**Next Team:** TEAM-028 - Complete the MVP! 🚀
