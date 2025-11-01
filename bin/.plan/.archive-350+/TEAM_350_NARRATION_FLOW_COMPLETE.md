# TEAM-350: Complete Narration Flow - Test Button to rbee-keeper UI

**Status:** 🔍 IN PROGRESS - Narration arriving as plain text, not JSON

## Current Status

✅ **Test button works** - Backend executes, returns success
❌ **Narration not reaching UI** - Coming as plain text, not JSON
🎯 **Goal:** Narration events should appear in rbee-keeper UI

## The Complete Flow

### 1. User Presses "Test" Button

**File:** `bin/10_queen_rbee/ui/app/src/components/RhaiIDE.tsx`

```typescript
<Button onClick={() => testScript(content)}>
  Test
</Button>
```

### 2. React Hook: `useRhaiScripts.testScript()`

**File:** `bin/10_queen_rbee/ui/packages/queen-rbee-react/src/hooks/useRhaiScripts.ts`

```typescript
const testScript = async (content: string) => {
  const client = new sdk.QueenClient(baseUrl)
  
  const operation = {
    operation: 'rhai_script_test',
    content
  }
  
  await client.submitAndStream(operation, (line: string) => {
    console.log('[RHAI Test] SSE line:', line)
    narrationHandler(line)  // ← Send to narration bridge
    
    if (line.includes('[DONE]')) {
      setTestResult({ success: true })
    }
  })
}
```

**Current logs:**
```
[RHAI Test] Starting test...
[RHAI Test] Client created, baseUrl: "http://localhost:7833"
[RHAI Test] Operation: {operation: "rhai_script_test", content: ""}
[RHAI Test] Submitting and streaming...
```

### 3. WASM SDK: `QueenClient.submitAndStream()`

**File:** `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/src/client.rs`

```rust
pub async fn submit_and_stream(
    &self,
    operation: JsValue,
    on_line: js_sys::Function,
) -> Result<String, JsValue> {
    let op: Operation = js_to_operation(operation)?;
    
    let job_id = self.inner
        .submit_and_stream(op, move |line| {
            // Call JavaScript callback with each SSE line
            let line_js = JsValue::from_str(line);
            let _ = callback.call1(&this, &line_js);
            Ok(())
        })
        .await
        .map_err(|e| JsValue::from_str(&format!("Submit failed: {}", e)))?;
    
    Ok(job_id)
}
```

### 4. Job Client: HTTP POST + SSE Stream

**File:** `bin/99_shared_crates/job-client/src/lib.rs`

```rust
pub async fn submit_and_stream<F>(
    &self,
    operation: Operation,
    mut on_line: F,
) -> Result<String>
where
    F: FnMut(&str) -> Result<()>,
{
    // POST to /v1/jobs
    let response = self.client
        .post(&format!("{}/v1/jobs", self.base_url))
        .json(&operation)
        .send()
        .await?;
    
    let job_id = response.json::<JobResponse>()?.job_id;
    
    // Connect to SSE stream: /v1/jobs/{job_id}/stream
    let stream_url = format!("{}/v1/jobs/{}/stream", self.base_url, job_id);
    let mut event_source = EventSource::new(&stream_url)?;
    
    while let Some(event) = event_source.next().await {
        let line = event.data;
        on_line(&line)?;  // ← Calls back to WASM SDK
        
        if line.contains("[DONE]") {
            break;
        }
    }
    
    Ok(job_id)
}
```

### 5. Backend: Queen Job Router

**File:** `bin/10_queen_rbee/src/job_router.rs`

```rust
pub async fn route_job(job_id: String, operation: Operation) -> Result<()> {
    match operation {
        Operation::RhaiScriptTest { content } => {
            let config = crate::rhai::RhaiTestConfig {
                job_id: Some(job_id.clone()),
                content,
            };
            crate::rhai::execute_rhai_script_test(config).await?;
        }
        // ...
    }
    Ok(())
}
```

### 6. RHAI Test Function (with #[with_job_id] macro)

**File:** `bin/10_queen_rbee/src/rhai/test.rs`

```rust
#[with_job_id(config_param = "test_config")]
pub async fn execute_rhai_script_test(test_config: RhaiTestConfig) -> Result<()> {
    n!("rhai_test_start", "🧪 Testing RHAI script");
    n!("rhai_test_content", "Script length: {} bytes", test_config.content.len());
    
    if test_config.content.trim().is_empty() {
        n!("rhai_test_error", "❌ Script is empty");
        anyhow::bail!("Script content cannot be empty");
    }
    
    n!("rhai_test_success", "✅ Script executed successfully");
    Ok(())
}
```

**What the macro does:**
```rust
// Expands to:
pub async fn execute_rhai_script_test(test_config: RhaiTestConfig) -> Result<()> {
    async fn __execute_rhai_script_test_inner(test_config: RhaiTestConfig) -> Result<()> {
        n!("rhai_test_start", "🧪 Testing RHAI script");
        // ... original body
    }
    
    if let Some(job_id) = test_config.job_id.as_ref() {
        let ctx = NarrationContext::new().with_job_id(job_id);
        with_narration_context(ctx, __execute_rhai_script_test_inner(test_config)).await
    } else {
        __execute_rhai_script_test_inner(test_config).await
    }
}
```

### 7. Narration Macro: `n!()`

**File:** `bin/99_shared_crates/narration-core/src/macros.rs`

```rust
#[macro_export]
macro_rules! n {
    ($action:expr, $($arg:tt)*) => {{
        let ctx = $crate::context::get_context();
        let job_id = ctx.as_ref().and_then(|c| c.job_id.as_ref());
        
        $crate::narrate(
            env!("CARGO_CRATE_NAME"),  // actor
            $action,
            &format!($($arg)*),        // message
            job_id,
        );
    }};
}
```

### 8. Narration Core: `narrate()`

**File:** `bin/99_shared_crates/narration-core/src/api/emit.rs`

```rust
pub fn narrate(actor: &str, action: &str, message: &str, job_id: Option<&String>) {
    let fields = NarrationFields {
        actor: actor.to_string(),
        action: action.to_string(),
        message: message.to_string(),
        timestamp: Utc::now(),
        job_id: job_id.cloned(),
        correlation_id: None,
    };
    
    // Emit to all sinks
    emit_to_sinks(&fields);
}
```

### 9. SSE Sink: Routes to Job Stream

**File:** `bin/99_shared_crates/narration-core/src/sinks/sse.rs`

```rust
pub fn emit_sse(fields: &NarrationFields) {
    if let Some(job_id) = &fields.job_id {
        // Get the SSE channel for this job_id
        if let Some(tx) = SSE_CHANNELS.get(job_id) {
            // Send narration as JSON
            let json = serde_json::to_string(&fields).unwrap();
            let _ = tx.send(format!("data: {}\n\n", json));
        }
    }
}
```

**🚨 PROBLEM: This is where it goes wrong!**

### 10. Stdout Sink: Also Emits (for debugging)

**File:** `bin/99_shared_crates/narration-core/src/sinks/stdout.rs`

```rust
pub fn emit_stdout(fields: &NarrationFields) {
    // Emits colored text to stdout
    println!("{} {} {}", 
        fields.actor.bold(),
        fields.action.dimmed(),
        fields.message
    );
}
```

**Current output (what we're seeing):**
```
queen_rbee::rhai::test::execute_rhai_script_test::{{closure}}::__execute_rhai_script_test_inner rhai_test_start     
🧪 Testing RHAI script
```

This is **stdout format**, not **JSON format**!

## The Problem

**SSE stream is sending stdout-formatted text, not JSON!**

**What we're receiving:**
```
"\u001b[1mqueen_rbee::rhai::test::...\u001b[0m \u001b[2mrhai_test_start\u001b[0m"
"🧪 Testing RHAI script"
```

**What we should receive:**
```json
{"actor":"queen_rbee","action":"rhai_test_start","message":"🧪 Testing RHAI script","timestamp":"...","job_id":"..."}
```

## Why This Happens

The SSE endpoint `/v1/jobs/{job_id}/stream` is likely:
1. Reading from stdout instead of SSE sink, OR
2. SSE sink is not being used, OR
3. SSE sink is emitting wrong format

## Next Steps to Debug

### 1. Check SSE Endpoint Implementation

**File:** `bin/10_queen_rbee/src/routes/jobs.rs` (or similar)

Look for the `/v1/jobs/{job_id}/stream` endpoint and see what it's sending.

### 2. Check if SSE Sink is Registered

**File:** `bin/10_queen_rbee/src/main.rs`

Look for sink registration:
```rust
narration_core::register_sink(SinkType::Sse);
```

### 3. Add More Logging

We need to add logs in:
- `narrationBridge.ts` - Is it receiving events?
- `useRhaiScripts.ts` - What format are the lines?
- SSE endpoint - What is it sending?

## Narration Bridge (Step 11)

**File:** `bin/10_queen_rbee/ui/packages/queen-rbee-react/src/utils/narrationBridge.ts`

```typescript
export function createNarrationStreamHandler(
  onEvent?: (event: NarrationEvent) => void
): (line: string) => void {
  return (line: string) => {
    // Try to parse as JSON
    try {
      const event = JSON.parse(line) as NarrationEvent
      
      // Send to parent window (rbee-keeper)
      if (window.parent !== window) {
        window.parent.postMessage({
          type: 'NARRATION_EVENT',
          payload: event
        }, 'http://localhost:7834')
      }
      
      onEvent?.(event)
    } catch (e) {
      console.warn('[Queen] Failed to parse narration line:', line, e)
    }
  }
}
```

**Current behavior:** Failing to parse because lines are plain text!

## rbee-keeper (Step 12)

**File:** `bin/00_rbee_keeper/ui/app/src/App.tsx` (or similar)

```typescript
useEffect(() => {
  const handleMessage = (event: MessageEvent) => {
    if (event.data.type === 'NARRATION_EVENT') {
      const narration = event.data.payload
      // Add to narration store
      narrationStore.add(narration)
    }
  }
  
  window.addEventListener('message', handleMessage)
  return () => window.removeEventListener('message', handleMessage)
}, [])
```

## Summary

**Flow is correct, but SSE format is wrong!**

✅ Button → Hook → SDK → Job Client → Backend → RHAI function → n!() macro
❌ SSE endpoint sending stdout format instead of JSON
❌ Narration bridge can't parse plain text
❌ Never reaches rbee-keeper

**Next:** Fix SSE endpoint to send JSON format!

---

**TEAM-350 Signature:** Documented complete narration flow from button to UI
