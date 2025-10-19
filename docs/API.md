# API Documentation

**Created by:** TEAM-116  
**Date:** 2025-10-19  
**For:** v0.1.0 Production Deployment

---

## 📋 Overview

llama-orch exposes RESTful HTTP APIs for orchestration and pool management.

### Base URLs

- **queen-rbee:** `http://localhost:8080`
- **rbee-hive:** `http://localhost:8081`

### Authentication

All endpoints require Bearer token (except `/health` and `/metrics`):

```bash
Authorization: Bearer <api-token>
```

---

## 👑 queen-rbee API

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

### GET /api/v1/nodes

List registered nodes.

**Auth:** Required

**Response:**
```json
{
  "nodes": [
    {
      "node_name": "hive1",
      "ssh_host": "hive1.example.com",
      "status": "active"
    }
  ]
}
```

### POST /api/v1/nodes

Register new node.

**Auth:** Required

**Request:**
```json
{
  "node_name": "hive1",
  "ssh_host": "hive1.example.com",
  "ssh_port": 22,
  "ssh_user": "llama-orch"
}
```

### GET /metrics

Prometheus metrics (no auth required).

---

## 🐝 rbee-hive API

### GET /health

Health check with worker stats.

**Response:**
```json
{
  "status": "healthy",
  "workers": {
    "total": 3,
    "idle": 2,
    "busy": 1
  }
}
```

### GET /api/v1/models

List downloaded models.

**Auth:** Required

### POST /api/v1/models/download

Download model from Hugging Face.

**Auth:** Required

**Request:**
```json
{
  "model_ref": "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
  "filename": "tinyllama.gguf"
}
```

### POST /api/v1/workers

Spawn new worker.

**Auth:** Required

**Request:**
```json
{
  "model_ref": "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
  "backend": "cuda",
  "device": 0,
  "slots": 4
}
```

### GET /api/v1/workers

List all workers.

**Auth:** Required

### DELETE /api/v1/workers/{id}

Stop worker.

**Auth:** Required

### GET /metrics

Prometheus metrics (no auth required).

---

## ❌ Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 401 | Unauthorized |
| 404 | Not Found |
| 500 | Internal Error |

---

## 📝 Examples

### Spawn Worker

```bash
TOKEN=$(cat /etc/llama-orch/secrets/api-token)

curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model_ref": "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    "backend": "cpu",
    "device": 0,
    "slots": 4
  }' \
  http://localhost:8081/api/v1/workers
```

### Check Health

```bash
curl http://localhost:8081/health
```

### View Metrics

```bash
curl http://localhost:8081/metrics
```

---

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete setup guide.
