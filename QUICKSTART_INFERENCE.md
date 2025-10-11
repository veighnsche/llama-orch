# 🚀 QUICKSTART: Get LLM Response on Your Shell

**TEAM-035: End-to-End Inference Working!**  
**TEAM-085: ONE COMMAND INFERENCE - Auto-starts queen-rbee!**

## What Works Now

✅ **Phase 1**: Download Progress SSE (TEAM-034)  
✅ **Phase 2**: Loading Progress SSE (TEAM-035)  
✅ **Phase 3**: Inference Streaming with [DONE] marker (TEAM-035)  
✅ **ONE COMMAND**: Auto-starts queen-rbee if needed (TEAM-085)  
✅ **Full Flow**: rbee-keeper → queen-rbee → rbee-hive → llm-worker-rbee → **TOKENS ON YOUR SHELL!**

## Quick Test (ONE COMMAND!)

### Run Inference - That's It!

```bash
# ONE COMMAND - No need to start anything manually!
cargo run --release -p rbee-keeper -- infer \
    --node localhost \
    --model "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" \
    --prompt "Why is the sky blue?" \
    --max-tokens 100 \
    --temperature 0.7
```

### What Happens Automatically ✨

You'll see:
1. ✓ **queen-rbee auto-starts** (if not already running)
2. ✓ Pool health check
3. ✓ Worker spawning
4. ✓ Worker ready
5. **Tokens streaming to your shell in real-time!**

**No more manual daemon management! Just run one command!**

## Full Test Script

```bash
./test_inference.sh
```

This script:
- Starts rbee-hive
- Runs inference
- Cleans up automatically

## What Happens Under the Hood

Per `test-001-mvp.md`:

1. **Phase 1**: Worker Registry Check (skipped - ephemeral mode)
2. **Phase 2**: Pool Preflight (`GET /v1/health`)
3. **Phase 3-5**: Spawn Worker (`POST /v1/workers/spawn`)
4. **Phase 6**: Worker Registration (in-memory)
5. **Phase 7**: Worker Health Check (`GET /v1/ready`)
6. **Phase 8**: Inference Execution (`POST /v1/inference`) → **SSE STREAM!**

## SSE Event Format (OpenAI Compatible)

```
data: {"type":"started","job_id":"abc","model":"model","started_at":"0"}

data: {"type":"token","t":"Once","i":0}

data: {"type":"token","t":" upon","i":1}

data: {"type":"token","t":" a","i":2}

data: {"type":"end","tokens_out":20,"decode_time_ms":1234,"stop_reason":"MAX_TOKENS"}

data: [DONE]
```

## Cross-Node Inference (Multi-Machine)

### On Remote Node (e.g., mac.home.arpa)

```bash
# Start rbee-hive on the remote machine
RBEE_WORKER_HOST=mac.home.arpa cargo run -p rbee-hive -- daemon --addr 0.0.0.0:8080
```

### From Control Node (blep)

```bash
# Run inference on remote node
cargo run --release -p rbee-keeper -- infer \
    --node mac \
    --model "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" \
    --prompt "write a short story" \
    --max-tokens 50 \
    --temperature 0.7
```

## Endpoints Implemented

### Pool Manager (rbee-hive)
- `GET /v1/health` - Health check
- `POST /v1/workers/spawn` - Spawn worker
- `GET /v1/workers/list` - List workers
- `POST /v1/workers/ready` - Worker ready callback
- `GET /v1/models/download/progress?id=<id>` - Download progress SSE

### Worker (llm-worker-rbee)
- `GET /health` - Health check
- `GET /v1/ready` - Readiness check
- `GET /v1/loading/progress` - Loading progress SSE
- `POST /v1/inference` - Inference with SSE streaming
- `POST /v1/admin/shutdown` - Graceful shutdown

## Next Steps

- [ ] Implement actual model downloading (Phase 3)
- [ ] Implement actual model loading with progress (Phase 7)
- [ ] Add backend detection (CPU/CUDA/Metal)
- [ ] Add worker auto-shutdown after idle timeout
- [ ] Add cancellation support (Ctrl+C)

## Troubleshooting

### "Connection refused"
- Make sure rbee-hive is running: `curl http://localhost:8080/v1/health`

### "Worker failed to start"
- Check rbee-hive logs
- Verify worker binary is built: `cargo build -p llm-worker-rbee`

### "Model not found"
- Model downloading not yet implemented
- For now, worker will fail if model doesn't exist locally

## Architecture

```
┌─────────────┐
│rbee-keeper  │  (CLI - ONE COMMAND!)
│   (client)  │
└──────┬──────┘
       │ auto-starts if needed
       ↓
┌─────────────┐
│ queen-rbee  │  (Orchestrator daemon)
│  (daemon)   │
└──────┬──────┘
       │ HTTP
       ↓
┌─────────────┐
│  rbee-hive  │  (Pool manager)
│   (daemon)  │
└──────┬──────┘
       │ spawn
       ↓
┌─────────────┐
│llm-worker-  │  (Worker process)
│    rbee     │
└─────────────┘
       │ SSE
       ↓
   YOUR SHELL! 🎉
```

---

**Built by:** TEAM-034 (Download SSE), TEAM-035 (Loading SSE + Inference SSE), TEAM-085 (Auto-start fix)  
**Status:** ✅ All SSE phases complete! ✅ ONE COMMAND INFERENCE!  
**Date:** 2025-10-11
