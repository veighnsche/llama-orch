# Real-Time Streaming Implementation Plan

**Team:** TEAM-148  
**Date:** 2025-10-20  
**Status:** PLAN - NOT IMPLEMENTED  
**Reference:** candle-vllm architecture

---

## Goal

Implement TRUE real-time token streaming where tokens are sent to the client as they're generated, not after all generation completes.

---

## Current Architecture (BROKEN)

```
HTTP Request → Lock Backend → execute_stream() → execute() (blocks 30s) → Return stream → SSE sends
                     ↑______________ LOCK HELD ENTIRE TIME _______________↑
```

**Problem:** Backend lock is held during entire generation, blocking the async runtime.

---

## Target Architecture (candle-vllm pattern)

```
HTTP Request → Add to Queue → Return Stream Immediately → SSE sends as tokens arrive
                                        ↓
                            Generation Loop (separate thread)
                                        ↓
                            Tokens → Channel → Stream
```

**Key:** Generation happens in `spawn_blocking`, HTTP handler returns immediately.

---

## Implementation Steps

### Step 1: Create Generation Request Queue

**File:** `src/backend/request_queue.rs` (NEW)

```rust
use tokio::sync::mpsc;
use std::sync::Arc;

pub struct GenerationRequest {
    pub request_id: String,
    pub prompt: String,
    pub config: SamplingConfig,
    pub response_tx: mpsc::UnboundedSender<TokenResponse>,
}

pub enum TokenResponse {
    Token(String),
    Error(String),
    Done,
}

pub struct RequestQueue {
    tx: mpsc::UnboundedSender<GenerationRequest>,
}

impl RequestQueue {
    pub fn new() -> (Self, mpsc::UnboundedReceiver<GenerationRequest>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self { tx }, rx)
    }
    
    pub fn add_request(&self, request: GenerationRequest) -> Result<(), String> {
        self.tx.send(request).map_err(|e| format!("Queue send failed: {}", e))
    }
}
```

---

### Step 2: Create Generation Engine Loop

**File:** `src/backend/generation_engine.rs` (NEW)

```rust
use tokio::sync::mpsc;
use std::sync::{Arc, Mutex};

pub struct GenerationEngine {
    backend: Arc<Mutex<CandleInferenceBackend>>,
    request_rx: mpsc::UnboundedReceiver<GenerationRequest>,
}

impl GenerationEngine {
    pub fn new(
        backend: Arc<Mutex<CandleInferenceBackend>>,
        request_rx: mpsc::UnboundedReceiver<GenerationRequest>,
    ) -> Self {
        Self { backend, request_rx }
    }
    
    pub fn start(mut self) {
        // Spawn blocking task for generation loop
        tokio::task::spawn_blocking(move || {
            // Get tokio runtime handle for async operations
            let rt = tokio::runtime::Handle::current();
            
            loop {
                // Wait for next request (blocking is OK here)
                let request = match rt.block_on(self.request_rx.recv()) {
                    Some(req) => req,
                    None => break, // Channel closed
                };
                
                tracing::info!(
                    request_id = %request.request_id,
                    "Processing generation request"
                );
                
                // Lock backend for this request only
                let mut backend = self.backend.lock().unwrap();
                
                // Generate tokens and send through channel
                if let Err(e) = Self::generate_streaming(
                    &mut backend,
                    &request.prompt,
                    &request.config,
                    request.response_tx,
                ) {
                    tracing::error!(
                        request_id = %request.request_id,
                        error = %e,
                        "Generation failed"
                    );
                }
                
                // Lock is released here, next request can proceed
            }
            
            tracing::info!("Generation engine stopped");
        });
    }
    
    fn generate_streaming(
        backend: &mut CandleInferenceBackend,
        prompt: &str,
        config: &SamplingConfig,
        response_tx: mpsc::UnboundedSender<TokenResponse>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Tokenize
        let encoding = backend.tokenizer.encode(prompt, true)?;
        let mut tokens = encoding.get_ids().to_vec();
        
        // Reset cache
        backend.model.reset_cache()?;
        
        // Create sampling components
        let mut logits_processor = sampling::create_logits_processor(config);
        let mut token_stream = TokenOutputStream::new(backend.tokenizer.clone());
        
        // Generate tokens one by one
        for pos in 0..config.max_tokens {
            let pos_usize = pos as usize;
            
            // Prepare input tensor
            let input_ids = if pos == 0 {
                Tensor::new(&tokens[..], &backend.device)?
                    .unsqueeze(0)?
            } else {
                Tensor::new(&[tokens[tokens.len() - 1]], &backend.device)?
                    .unsqueeze(0)?
            };
            
            // Forward pass
            let logits = backend.model.forward(&input_ids, pos_usize)?;
            
            // Get logits for last position
            let logits = logits.squeeze(0)?;
            let logits = if logits.dims().len() > 1 {
                logits.get(logits.dims()[0] - 1)?
            } else {
                logits
            };
            
            // Sample next token
            let next_token = logits_processor.sample(&logits)?;
            
            // Check for EOS
            let tokenizer_eos_id = backend.tokenizer.token_to_id("</s>");
            let is_eos = tokenizer_eos_id.map_or_else(
                || next_token == backend.model.eos_token_id(),
                |eos_id| next_token == eos_id,
            );
            
            if is_eos {
                break;
            }
            
            // Decode token and send IMMEDIATELY through channel
            if let Some(token_str) = token_stream.next_token(next_token)? {
                // CRITICAL: Send token as soon as it's generated!
                if response_tx.send(TokenResponse::Token(token_str)).is_err() {
                    // Client disconnected, stop generation
                    tracing::debug!("Client disconnected, stopping generation");
                    return Ok(());
                }
            }
            
            tokens.push(next_token);
        }
        
        // Send any remaining decoded bytes
        if let Some(rest) = token_stream.decode_rest()? {
            let _ = response_tx.send(TokenResponse::Token(rest));
        }
        
        // Send done signal
        let _ = response_tx.send(TokenResponse::Done);
        
        Ok(())
    }
}
```

---

### Step 3: Refactor Main to Start Generation Engine

**File:** `src/main.rs`

```rust
// After loading model (around line 186):

// TEAM-148: Create request queue and generation engine
let (request_queue, request_rx) = RequestQueue::new();
let request_queue = Arc::new(request_queue);

// TEAM-148: Start generation engine in background
let generation_engine = GenerationEngine::new(
    Arc::clone(&backend),
    request_rx,
);
generation_engine.start();

// TEAM-148: Create router with request queue (not backend directly)
let router = create_router(request_queue, expected_token);
```

---

### Step 4: Refactor HTTP Handler to Use Queue

**File:** `src/http/execute.rs`

```rust
use crate::backend::request_queue::{RequestQueue, TokenResponse};

pub async fn handle_execute(
    State(queue): State<Arc<RequestQueue>>,
    Json(req): Json<ExecuteRequest>,
) -> Result<Sse<EventStream>, ValidationErrorResponse> {
    // Validate request (same as before)
    if let Err(validation_errors) = req.validate_all() {
        return Err(validation_errors);
    }
    
    // Convert to sampling config (same as before)
    let config = SamplingConfig { /* ... */ };
    
    // TEAM-148: Create channel for this request
    let (response_tx, mut response_rx) = tokio::sync::mpsc::unbounded_channel();
    
    // TEAM-148: Add request to queue
    let generation_request = GenerationRequest {
        request_id: req.job_id.clone(),
        prompt: req.prompt.clone(),
        config,
        response_tx,
    };
    
    if let Err(e) = queue.add_request(generation_request) {
        warn!(job_id = %req.job_id, error = %e, "Failed to queue request");
        // Return error response
        let events = vec![InferenceEvent::Error {
            code: "QUEUE_FAILED".to_string(),
            message: e,
        }];
        let stream: EventStream = Box::new(
            stream::iter(events)
                .map(|event| Ok(Event::default().json_data(&event).unwrap()))
                .chain(stream::once(future::ready(Ok(Event::default().data("[DONE]"))))),
        );
        return Ok(Sse::new(stream));
    }
    
    // TEAM-148: Return stream immediately!
    // Generation happens in background, tokens flow through response_rx
    
    let started_event = InferenceEvent::Started {
        job_id: req.job_id.clone(),
        model: "model".to_string(),
        started_at: chrono::Utc::now().timestamp().to_string(),
    };
    
    // Convert token responses to SSE events
    let token_stream = async_stream::stream! {
        while let Some(token_response) = response_rx.recv().await {
            match token_response {
                TokenResponse::Token(token) => {
                    yield Ok(Event::default().json_data(&InferenceEvent::Token {
                        t: token,
                        i: 0, // TODO: track count
                    }).unwrap())
                }
                TokenResponse::Error(e) => {
                    yield Ok(Event::default().json_data(&InferenceEvent::Error {
                        code: "GENERATION_ERROR".to_string(),
                        message: e,
                    }).unwrap())
                }
                TokenResponse::Done => {
                    break;
                }
            }
        }
    };
    
    // Build complete stream
    let stream_with_done: EventStream = Box::new(
        stream::once(future::ready(Ok(Event::default().json_data(&started_event).unwrap())))
            .chain(token_stream)
            .chain(stream::once(future::ready(Ok(Event::default().data("[DONE]"))))),
    );
    
    info!(job_id = %req.job_id, "Streaming inference started");
    
    // CRITICAL: We return immediately here!
    // Generation is happening in spawn_blocking
    // Tokens will flow through the stream as they're generated
    Ok(Sse::new(stream_with_done))
}
```

---

### Step 5: Update Router Creation

**File:** `src/http/routes.rs`

```rust
pub fn create_router(
    queue: Arc<RequestQueue>,  // Changed from backend
    expected_token: String,
) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/inference", post(execute::handle_execute))
        .layer(Extension(queue))  // Changed
        .layer(/* auth middleware */)
}
```

---

### Step 6: Remove Old Backend Trait Method

**File:** `src/http/backend.rs`

```rust
// REMOVE execute_stream() from trait
// Keep only execute() for non-streaming use cases

#[async_trait]
pub trait InferenceBackend: Send + Sync {
    async fn execute(&mut self, ...) -> Result<InferenceResult, ...>;
    async fn cancel(&self, job_id: &str) -> Result<(), ...>;
    fn vram_usage(&self) -> u64;
    fn is_healthy(&self) -> bool;
}
```

---

### Step 7: Add Dependencies

**File:** `Cargo.toml`

```toml
[dependencies]
async-stream = "0.3"  # For stream! macro
```

---

## Testing Plan

### Unit Tests

1. **Test RequestQueue**
   - Add request succeeds
   - Receive request succeeds
   - Channel closed handling

2. **Test GenerationEngine**
   - Single request generates tokens
   - Multiple requests processed sequentially
   - Error handling

### Integration Tests

1. **Test HTTP endpoint**
   - Request returns immediately
   - Tokens arrive in real-time
   - [DONE] marker sent
   - Client disconnect stops generation

2. **Test with xtask worker:test**
   - Tokens stream as generated
   - Test completes without hanging
   - Heartbeats continue during generation

---

## Success Criteria

- ✅ HTTP handler returns in <100ms
- ✅ First token arrives within 1-2 seconds
- ✅ Tokens stream continuously (not batched)
- ✅ Test completes successfully
- ✅ No deadlocks or hangs
- ✅ Client disconnect stops generation
- ✅ Multiple concurrent requests work

---

## Migration Path

### Phase 1: Implement Core (2-3 hours)
1. Create `request_queue.rs`
2. Create `generation_engine.rs`
3. Update `main.rs` to start engine

### Phase 2: Refactor HTTP Layer (1-2 hours)
4. Update `execute.rs` to use queue
5. Update `routes.rs` 
6. Remove old `execute_stream()` from trait

### Phase 3: Test & Debug (1-2 hours)
7. Run `cargo xtask worker:test`
8. Fix any issues
9. Verify real-time streaming

### Phase 4: Cleanup (30 min)
10. Remove dead code
11. Update documentation
12. Add TEAM-148 signatures

**Total Estimated Time:** 5-8 hours

---

## Key Differences from Current Code

| Current | New |
|---------|-----|
| Backend in Arc<Mutex<>> passed to handlers | RequestQueue passed to handlers |
| Handler locks backend, calls execute_stream | Handler adds to queue, returns immediately |
| execute_stream blocks until done | Generation happens in spawn_blocking |
| Tokens buffered, then streamed | Tokens sent as generated |
| Single-threaded, blocks runtime | Multi-threaded, doesn't block |

---

## Why This Works

1. **`spawn_blocking`** moves CPU-intensive work off async runtime
2. **Request queue** decouples HTTP from generation
3. **Channels** enable true streaming (producer/consumer)
4. **Lock per request** (not per token) minimizes contention
5. **Immediate return** keeps HTTP layer responsive

---

## References

- `reference/candle-vllm/src/openai/openai_server.rs` (lines 213-265)
- `reference/candle-vllm/src/openai/pipelines/llm_engine.rs` (lines 138-211, 620-668)
- `reference/candle-vllm/src/openai/streaming.rs` (complete file)
- `reference/mistral.rs/mistralrs/examples/simple_stream/main.rs` (lines 41-62)

---

**Status:** READY TO IMPLEMENT  
**Blocker:** None  
**Next Action:** Start Phase 1 - Create request_queue.rs

---

**TEAM-148**  
**Date:** 2025-10-20 00:15
