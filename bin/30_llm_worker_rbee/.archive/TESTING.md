# Worker Testing Guide

## Quick Start

### Run Isolation Test
```bash
cargo xtask worker:test
```

This replaces the old `test_worker_isolation.sh` script with a better Rust implementation.

---

## What Gets Tested

1. **Worker Startup** - Model loading, HTTP server initialization
2. **Heartbeat Mechanism** - Periodic heartbeats to mock hive (30s interval)
3. **HTTP Endpoints** - Health, ready, and inference endpoints
4. **Process Management** - Automatic cleanup on exit

---

## Custom Configuration

### Test CUDA Backend
```bash
cargo xtask worker:test --backend cuda --device 0
```

### Test with Different Model
```bash
cargo xtask worker:test --model /path/to/model.gguf
```

### Custom Ports
```bash
cargo xtask worker:test --port 8080 --hive-port 9000
```

### Longer Timeout
```bash
cargo xtask worker:test --timeout 60
```

---

## Test Output

### Success
```
🧪 Worker Isolation Test
==================================

📦 Checking model file...
✓ Model found

📡 Starting mock hive server...
✓ Mock hive server started

🚀 Starting worker...
✓ Worker started

⏳ Waiting for worker to be ready...
✅ Worker is ready!

⏳ Waiting for heartbeats (10 seconds)...

🔍 Testing health endpoint...
✅ Health endpoint responding

🔍 Testing ready endpoint...
✅ Ready endpoint responding

🤔 Testing inference...
✅ Inference endpoint responding

📋 Hive server logs:
   [10:54:23] ✅ Heartbeat #1 from worker
   [10:54:53] ✅ Heartbeat #2 from worker

📊 Test Summary
==================================
✅ Health test passed
✅ Ready test passed
✅ Inference test passed

✅ Isolation test complete - ALL TESTS PASSED!
```

---

## Troubleshooting

### Model Not Found
```
⚠️  Model not found: ../../.test-models/tinyllama/...
   Skipping test - no model available
   To run this test, provide a model path:
   cargo xtask worker:test --model /path/to/model.gguf
```

**Solution:** Provide path to an existing GGUF model file

### Worker Timeout
```
❌ Worker did not become ready within 30s
```

**Solutions:**
- Increase timeout: `--timeout 60`
- Check model path is correct
- Check sufficient RAM/VRAM available
- Review worker logs in output

### Port Already in Use
```
❌ Failed to start mock hive server
```

**Solutions:**
- Use different port: `--hive-port 19201`
- Kill existing process on that port
- Check firewall settings

---

## Comparison: Old vs New

### Old Bash Script
```bash
cd bin/30_llm_worker_rbee
./test_worker_isolation.sh
```

**Issues:**
- ❌ Manual process management
- ❌ No cleanup on error
- ❌ Hardcoded paths
- ❌ Platform-specific
- ❌ Used deprecated callback mechanism

### New xtask
```bash
cargo xtask worker:test
```

**Benefits:**
- ✅ Automatic cleanup (RAII)
- ✅ Cleanup on any exit path
- ✅ Configurable via CLI
- ✅ Cross-platform
- ✅ Uses current heartbeat mechanism
- ✅ Type-safe
- ✅ Better error messages

---

## Architecture

### Mock Hive Server (Python)
- Accepts heartbeat POSTs at `/heartbeat/workers`
- Logs all received heartbeats
- Runs on configurable port (default: 19200)
- Auto-cleanup via Drop trait

### Worker Process
- Spawned with `--hive-url` pointing to mock server
- Sends heartbeats every 30 seconds
- Exposes HTTP endpoints on configurable port (default: 18081)
- Auto-cleanup via Drop trait

### Test Flow
```
1. Check model exists
2. Start mock hive server
3. Start worker process
4. Wait for worker ready (health check)
5. Wait 10s for heartbeats
6. Test all HTTP endpoints
7. Show logs
8. Cleanup (automatic)
```

---

## See Also

- **Implementation:** `xtask/src/tasks/worker.rs`
- **Documentation:** `xtask/WORKER_TEST_XTASK.md`
- **Old Script:** `test_worker_isolation.sh` (deprecated)

---

**Status:** ✅ Ready to use  
**Last Updated:** 2025-10-19
