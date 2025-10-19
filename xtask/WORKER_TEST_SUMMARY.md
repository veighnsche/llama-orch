# Worker Test - Correct Architecture

## What the test SHOULD do (from a_human_wrote_this.md):

### Worker Startup Flow:
1. **Bee hive starts worker** with model path, device, port, hive-url
2. **Worker starts HTTP server** on assigned port
3. **Worker automatically sends heartbeats** to bee hive at `/heartbeat/workers`
4. **Bee hive receives heartbeat** and knows worker is ready
5. **Bee hive includes worker heartbeat** in its own heartbeat to queen

### Worker Endpoints (from routes.rs):
- `GET /health` - Health check (public, no auth)
- `GET /v1/ready` - Readiness check (protected)
- `POST /v1/inference` - Inference request, returns SSE link (protected)
- `GET /v1/loading/progress` - Model loading progress SSE (protected)

### Inference Flow (from document):
1. Queen sends `POST /v1/inference` with prompt to worker
2. Worker responds with GET link for SSE connection
3. Queen connects to SSE link
4. Worker streams tokens via SSE
5. Worker sends `[DONE]` signal when complete

## Test Implementation:

### Phase 1: Heartbeat Test (CURRENT)
```
1. Start mock hive server (Rust TCP server)
2. Mock hive listens for POST /heartbeat/workers
3. Start worker with --hive-url pointing to mock hive
4. Wait 35 seconds (heartbeat interval is 30s)
5. Verify mock hive received at least 1 heartbeat
6. Check heartbeat payload has: worker_id, state, model_loaded
```

### Phase 2: Inference Test (FUTURE)
```
1. Start mock hive (receives heartbeats)
2. Start worker
3. Wait for heartbeat (worker ready)
4. Send POST /v1/inference with prompt
5. Parse response to get SSE link
6. Connect to SSE link
7. Stream tokens
8. Verify [DONE] signal
```

## Current Issues:
- ❌ Removed Python server (now Rust)
- ❌ Removed incorrect `/v1/ready` endpoint test
- ❌ Removed incorrect health endpoint polling
- ✅ Focus on heartbeat mechanism ONLY
- ✅ Rust mock hive server
- ✅ Proper heartbeat payload parsing

## Next Steps:
1. Get heartbeat test working
2. Add inference SSE test
3. Verify token streaming
4. Test [DONE] signal
