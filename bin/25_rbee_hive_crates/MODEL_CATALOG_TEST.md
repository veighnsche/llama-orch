# Model Catalog - Manual Test Plan

## Quick Test: Verify Full Stack

### 1. Add Test Model to Catalog

```bash
# Create test model directory
mkdir -p ~/.cache/rbee/models/test-model-1

# Create metadata.json
cat > ~/.cache/rbee/models/test-model-1/metadata.json << 'EOF'
{
  "id": "test-model-1",
  "name": "Test Model 1",
  "path": "/home/vince/.cache/rbee/models/test-model-1/model.gguf",
  "size": 1073741824,
  "status": "Available",
  "added_at": "2025-10-31T20:30:00Z"
}
EOF

# Create dummy model file (1GB)
dd if=/dev/zero of=~/.cache/rbee/models/test-model-1/model.gguf bs=1M count=1024
```

### 2. Start Hive Backend

```bash
cd /home/vince/Projects/llama-orch
cargo run --bin rbee-hive
```

Expected output:
```
🐝 [rbee-hive] Starting on http://localhost:7835
```

### 3. Test ModelList via CLI (if available)

```bash
# Using curl
curl -X POST http://localhost:7835/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"operation":"model_list","hive_id":"localhost"}'
```

Expected response:
```json
{
  "job_id": "job-abc123",
  "sse_url": "/v1/jobs/job-abc123/stream"
}
```

Then connect to SSE:
```bash
curl -N http://localhost:7835/v1/jobs/job-abc123/stream
```

Expected SSE stream:
```
data: 📋 Listing models on hive 'localhost'
data: Found 1 model(s)
data: [{"id":"test-model-1","name":"Test Model 1","path":"...","size":1073741824,"status":"Available","added_at":"2025-10-31T20:30:00Z"}]
data: [DONE]
```

### 4. Test Frontend

```bash
# Terminal 1: Start Hive backend
cd /home/vince/Projects/llama-orch
cargo run --bin rbee-hive

# Terminal 2: Build WASM SDK
cd bin/20_rbee_hive/ui/packages/rbee-hive-sdk
pnpm build

# Terminal 3: Start frontend dev server
cd bin/20_rbee_hive/ui/app
pnpm dev
```

Open browser: http://localhost:7836

**Expected UI:**
- Model Management card shows "1 Models" badge
- Model list shows:
  - Name: "Test Model 1"
  - Size: "1.00 GB"
  - Status: "available" badge

### 5. Verify Network Requests (Browser DevTools)

**Network tab should show:**

1. POST http://localhost:7835/v1/jobs
   - Payload: `{"operation":"model_list","hive_id":"localhost"}`
   - Response: `{"job_id":"...","sse_url":"..."}`

2. GET http://localhost:7835/v1/jobs/{job_id}/stream
   - Type: EventStream
   - Data: SSE lines with model JSON

**Console should show:**
```
🎉 [Hive SDK] WASM module initialized successfully!
```

## Troubleshooting

### Issue: "No models found"

**Check:**
```bash
ls -la ~/.cache/rbee/models/
cat ~/.cache/rbee/models/test-model-1/metadata.json
```

**Verify JSON is valid:**
```bash
cat ~/.cache/rbee/models/test-model-1/metadata.json | jq .
```

### Issue: "Cannot connect to backend"

**Check Hive is running:**
```bash
curl http://localhost:7835/health
```

Expected: `{"status":"healthy"}`

**Check ports:**
```bash
lsof -i :7835  # Hive backend
lsof -i :7836  # Hive UI
```

### Issue: "WASM not loading"

**Check WASM build:**
```bash
cd bin/20_rbee_hive/ui/packages/rbee-hive-sdk
ls -la pkg/
```

Should contain:
- `rbee_hive_sdk_bg.wasm`
- `rbee_hive_sdk.js`
- `rbee_hive_sdk.d.ts`

**Rebuild if missing:**
```bash
pnpm build
```

### Issue: "JSON parsing error"

**Check backend output format:**
```bash
# Look for the JSON line in SSE stream
curl -N http://localhost:7835/v1/jobs/{job_id}/stream | grep '^\['
```

Should output valid JSON array starting with `[`

## Expected Results Summary

✅ **Backend:**
- Hive starts on port 7835
- ModelList operation returns JSON array
- SSE stream includes narration + JSON

✅ **Frontend:**
- UI loads on port 7836
- WASM SDK initializes
- useModels() hook fetches data
- ModelManagement component renders
- Model appears in list with correct data

✅ **Integration:**
- POST /v1/jobs creates job
- SSE stream delivers data
- React Query caches result
- UI updates automatically

## Cleanup

```bash
# Remove test model
rm -rf ~/.cache/rbee/models/test-model-1
```
