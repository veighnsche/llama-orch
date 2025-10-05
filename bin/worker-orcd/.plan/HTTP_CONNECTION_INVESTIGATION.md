# HTTP Connection Investigation

**Date**: 2025-10-05  
**Issue**: `/execute` endpoint connection fails, but `/health` works  
**Status**: 🔍 INVESTIGATING  

## Symptoms

1. ✅ Worker starts successfully
2. ✅ Model loads in 1 second (optimized!)
3. ✅ HTTP server listening on port
4. ✅ Health endpoint responds: `GET /health` works
5. ❌ Execute endpoint fails: `POST /execute` connection refused
6. ✅ Worker process is still running when request fails

## Evidence

### Health Endpoint Works
```
🔍 Testing health endpoint again...
✅ Health check passed
```

### Execute Endpoint Fails
```
✅ Worker process is still running
🔍 Sending POST to http://localhost:37475/execute
❌ Request failed: error sending request for url (http://localhost:37475/execute)
```

### Error Type
`reqwest` error: "error sending request for url"

This typically means:
- Connection refused (server not listening on that endpoint)
- Connection reset (server closed connection)
- Network/DNS issue (unlikely for localhost)

## What We Know

### Router Configuration
From `worker-http/src/routes.rs`:
```rust
pub fn create_router<B: InferenceBackend + 'static>(backend: Arc<B>) -> Router {
    Router::new()
        .route("/health", get(health::handle_health::<B>))
        .route("/execute", post(execute::handle_execute::<B>))
        .with_state(backend)
}
```

Both endpoints are defined in the same router.

### Server Startup
From logs:
```
{"message":"HTTP server initialized","addr":"0.0.0.0:39585"}
{"message":"HTTP server listening","addr":"0.0.0.0:39585"}
```

Server is listening on all interfaces.

### Request Details
- Health: `GET http://localhost:PORT/health` ✅ Works
- Execute: `POST http://localhost:PORT/execute` ❌ Fails

## Hypotheses

### 1. Middleware Issue ❓
Maybe there's middleware that blocks POST requests?
- **Unlikely**: Router shows no middleware between routes

### 2. State/Backend Issue ❓
Maybe the backend Arc is causing issues?
- **Unlikely**: Health endpoint uses same state

### 3. Request Body Issue ❓
Maybe the JSON body is malformed?
- **Possible**: But error is "sending request", not "parsing response"

### 4. Connection Timing ❓
Maybe the server closes connections between health and execute?
- **Possible**: But we wait 500ms and verify process is alive

### 5. Port Binding Issue ❓
Maybe the server is only partially listening?
- **Unlikely**: Health works on same port

### 6. Axum/Tower Issue ❓
Maybe there's a bug in the HTTP stack?
- **Possible**: But this would be very unusual

## Next Steps

1. **Test with curl** - Try direct HTTP request to isolate reqwest
2. **Check server logs** - See if request reaches the server
3. **Add debug logging** - Log all incoming requests
4. **Test health POST** - Try POST to /health to isolate method
5. **Check axum version** - Maybe there's a known bug

## Debug Commands

### Test with curl
```bash
# Start worker
cargo run -p worker-orcd --features cuda -- \
  --worker-id test \
  --model .test-models/qwen/qwen2.5-0.5b-instruct-fp16.gguf \
  --port 9999 \
  --gpu-device 0 \
  --callback-url http://localhost:9999/callback

# In another terminal
curl -v -X POST http://localhost:9999/execute \
  -H "Content-Type: application/json" \
  -d '{"job_id":"test","prompt":"hello","max_tokens":10}'
```

### Check if server is listening
```bash
netstat -tlnp | grep 9999
# or
lsof -i :9999
```

## Timeline

1. **22:14** - Optimized loading complete (1s load time!)
2. **22:15** - Discovered HTTP connection issue
3. **22:16** - Confirmed worker is running
4. **22:17** - Confirmed health endpoint works
5. **22:18** - Investigating execute endpoint failure

---

**Current Status**: Worker runs, health works, execute fails with connection error. Need to isolate whether this is a server-side or client-side issue.
