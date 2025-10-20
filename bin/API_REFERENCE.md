# rbee API Reference

## rbee CLI

```bash
rbee queen start
rbee queen stop
rbee hive start
rbee hive stop
rbee hive list
rbee hive get {id}
rbee hive create ...
rbee hive update {id} ...
rbee hive delete {id}
rbee worker spawn ...
rbee worker list
rbee worker get {id}
rbee worker delete {id}
rbee model download ...
rbee model list
rbee model get {id}
rbee model delete {id}
rbee infer --model M "prompt"
rbee job stream {id}
```

---

## queen-rbee (Port 8500)

```
GET  /health                     # Health check
POST /v1/shutdown                # Graceful shutdown
POST /v1/hive/start              # Start hive (spawn, fire & forget)
POST /v1/heartbeat               # Receive hive heartbeat (callback)
POST /v1/jobs                    # Create job
GET  /v1/jobs/{id}/stream        # Stream results (SSE)
GET  /v1/hives                   # List hives
GET  /v1/hives/{id}              # Get hive
POST /v1/hives                   # Create hive
PUT  /v1/hives/{id}              # Update hive
DELETE /v1/hives/{id}            # Delete hive
ROUTE /v1/hive/{id/localhost}/*  # Route Hive API  
```

---

## rbee-hive (Port 8600)

```
GET  /health                  # Health check
POST /v1/shutdown             # Graceful shutdown
POST /v1/heartbeat            # Receive worker heartbeat
GET  /v1/devices              # Device detection
POST /v1/workers/spawn        # Spawn worker
GET  /v1/workers              # List workers
GET  /v1/workers/{id}         # Get worker
DELETE /v1/workers/{id}       # Delete worker
POST /v1/models/download      # Download model
GET  /v1/models/download/{id}/stream  # Progress
GET  /v1/models               # List models
GET  /v1/models/{id}          # Get model
DELETE /v1/models/{id}        # Delete model
GET  /v1/capacity/{id}        # VRAM check
POST /v1/inference            # Create job
GET  /v1/inference/{id}/stream     # Stream (SSE)
GET  /metrics                 # Prometheus
```

---

## llm-worker-rbee (Dynamic Port)

```
GET  /health                  # Health check
POST /execute                 # Execute inference
```

---

## Flow

```
User → rbee CLI → queen-rbee → rbee-hive → llm-worker-rbee
```

## Heartbeat

```
worker → hive (30s) → queen (15s, aggregated)
```

## Callback

```
queen: POST /hive/start (spawn, return)
hive: starts up
hive: POST /heartbeat → queen (I'm ready!)
queen: GET /v1/devices → hive (detect)
queen: updates catalog (hive Online)
```
