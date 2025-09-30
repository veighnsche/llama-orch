# Manual Model Staging for Cloud Profile

**Version**: 1.0  
**Date**: 2025-10-01  
**Audience**: Operations, DevOps Engineers

---

## Overview

In **CLOUD_PROFILE** deployments, each GPU node maintains its own local model catalog. Models must be manually staged to each node before they can be used for task execution.

This document describes how to stage models across multiple nodes and verify availability.

---

## Architecture

### Model Storage

- **Location**: Each node has a local filesystem catalog at `~/.cache/llama-orch/models/` (default)
- **Catalog**: Managed by `catalog-core` (FsCatalog)
- **No Sync**: Models are NOT automatically synchronized between nodes
- **Operator Responsibility**: You must ensure required models are present on nodes

### Placement Behavior

When a task requires a specific model:
1. **orchestratord** queries all online nodes for pool status
2. Pools report `models_available` in heartbeat
3. **Placement filters** to only pools that have the required model
4. If no pools have the model → task is rejected

---

## Staging Workflow

### Step 1: Identify Required Models

Check which models are needed for your workload:

```bash
# List models in orchestratord catalog
curl http://orchestratord:8080/v2/catalog/models \
  -H "Authorization: Bearer $LLORCH_API_TOKEN"
```

### Step 2: Check Current Distribution

Query catalog availability across all nodes:

```bash
curl http://orchestratord:8080/v2/catalog/availability \
  -H "Authorization: Bearer $LLORCH_API_TOKEN" | jq .
```

Example response:
```json
{
  "nodes": {
    "gpu-node-1": {
      "node_id": "gpu-node-1",
      "address": "http://192.168.1.100:9200",
      "online": true,
      "models": ["llama-3.1-8b-instruct"],
      "pools": [
        {
          "pool_id": "pool-0",
          "ready": true,
          "models": ["llama-3.1-8b-instruct"]
        }
      ]
    },
    "gpu-node-2": {
      "node_id": "gpu-node-2",
      "address": "http://192.168.1.101:9200",
      "online": true,
      "models": [],
      "pools": []
    }
  },
  "total_models": 1,
  "replicated_models": [],
  "single_node_models": ["llama-3.1-8b-instruct"]
}
```

**Key Observations**:
- `single_node_models`: Models on only one node (single point of failure)
- `replicated_models`: Models on all nodes (fully redundant)
- Empty `models` array: Node has no models staged

### Step 3: Stage Models to Nodes

#### Option A: Copy from Shared Storage (Recommended)

If you have models in S3, NFS, or other shared storage:

```bash
# On each GPU node
ssh gpu-node-2 '
  mkdir -p ~/.cache/llama-orch/models/
  aws s3 cp s3://my-models/llama-3.1-8b-instruct.gguf \
    ~/.cache/llama-orch/models/llama-3.1-8b-instruct.gguf
'
```

#### Option B: Copy from Another Node

```bash
# Copy from gpu-node-1 to gpu-node-2
scp gpu-node-1:~/.cache/llama-orch/models/llama-3.1-8b-instruct.gguf \
    gpu-node-2:~/.cache/llama-orch/models/
```

#### Option C: Download Directly on Node

```bash
ssh gpu-node-2 '
  cd ~/.cache/llama-orch/models/
  wget https://huggingface.co/models/llama-3.1-8b-instruct.gguf
'
```

### Step 4: Register Model in Catalog

After copying the file, register it in the local catalog:

```bash
ssh gpu-node-2 '
  # Register model (creates catalog entry)
  curl http://localhost:9200/v2/catalog/models \
    -X POST \
    -H "Content-Type: application/json" \
    -d "{
      \"id\": \"llama-3.1-8b-instruct\",
      \"digest\": \"sha256:abc123...\",
      \"source_url\": \"s3://my-models/llama-3.1-8b-instruct.gguf\"
    }"
'
```

**Note**: If pool-managerd doesn't expose catalog endpoints, use orchestratord's catalog API (models are per-node).

### Step 5: Trigger Pool Preload

Force pool-managerd to load the model:

```bash
ssh gpu-node-2 '
  curl http://localhost:9200/pools/pool-0/preload \
    -X POST \
    -H "Authorization: Bearer $LLORCH_API_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{
      \"model_id\": \"llama-3.1-8b-instruct\",
      \"engine\": \"llamacpp\"
    }"
'
```

This will:
1. Provision the engine with the model
2. Write handoff file when ready
3. pool-managerd detects handoff and marks pool ready
4. Next heartbeat reports `models_available: ["llama-3.1-8b-instruct"]`

### Step 6: Verify Availability

Check that orchestratord sees the model:

```bash
curl http://orchestratord:8080/v2/catalog/availability \
  -H "Authorization: Bearer $LLORCH_API_TOKEN" | jq '.nodes["gpu-node-2"].models'
```

Should now show:
```json
["llama-3.1-8b-instruct"]
```

---

## Best Practices

### 1. Replicate Critical Models

For production workloads, replicate models across **all nodes**:

```bash
# Stage to all nodes
for node in gpu-node-1 gpu-node-2 gpu-node-3; do
  echo "Staging to $node..."
  scp model.gguf $node:~/.cache/llama-orch/models/
  ssh $node "curl http://localhost:9200/pools/pool-0/preload -X POST ..."
done
```

**Benefits**:
- No single point of failure
- Load balancing across all nodes
- Faster failover

### 2. Use Checksums

Always verify file integrity after copying:

```bash
ssh gpu-node-2 '
  cd ~/.cache/llama-orch/models/
  sha256sum llama-3.1-8b-instruct.gguf
'
# Compare with expected: abc123...
```

### 3. Monitor Disk Space

Models can be large (5-70GB). Monitor disk usage:

```bash
ssh gpu-node-2 'df -h ~/.cache/llama-orch/models/'
```

### 4. Automate with Ansible/Terraform

Example Ansible playbook:

```yaml
- name: Stage models to GPU nodes
  hosts: gpu_nodes
  tasks:
    - name: Ensure model directory exists
      file:
        path: ~/.cache/llama-orch/models
        state: directory

    - name: Copy model from S3
      aws_s3:
        bucket: my-models
        object: llama-3.1-8b-instruct.gguf
        dest: ~/.cache/llama-orch/models/llama-3.1-8b-instruct.gguf
        mode: get

    - name: Register model in catalog
      uri:
        url: http://localhost:9200/v2/catalog/models
        method: POST
        body_format: json
        body:
          id: llama-3.1-8b-instruct
          digest: "sha256:abc123..."
```

### 5. Pre-warm on Deployment

Stage models **before** registering nodes with orchestratord:

1. Deploy pool-managerd
2. Stage all required models
3. Preload models into pools
4. Register node with orchestratord

This prevents tasks from being routed to nodes without models.

---

## Troubleshooting

### Model Not Showing in Availability

**Symptom**: Model staged but not in `/v2/catalog/availability`

**Diagnosis**:
```bash
# Check if model file exists
ssh gpu-node-2 'ls -lh ~/.cache/llama-orch/models/'

# Check if pool is ready
curl http://orchestratord:8080/v2/nodes | jq '.nodes[] | select(.node_id == "gpu-node-2") | .pools'

# Check pool-managerd logs
ssh gpu-node-2 'journalctl -u pool-managerd -n 100 --no-pager | grep model'
```

**Resolution**:
1. Verify model file exists and is readable
2. Trigger preload manually
3. Check for errors in pool-managerd logs
4. Restart pool-managerd if needed

### Task Rejected: No Pools Available

**Symptom**: Task submission fails even though nodes are online

**Diagnosis**:
```bash
# Check which models are available
curl http://orchestratord:8080/v2/catalog/availability \
  -H "Authorization: Bearer $LLORCH_API_TOKEN" | jq '.nodes[].models'

# Check task model requirement
# (model_id in task body)
```

**Resolution**:
- Stage the required model to at least one node
- Or submit task without model_id (uses any available pool)

### Disk Space Exhausted

**Symptom**: Cannot copy model, disk full

**Diagnosis**:
```bash
ssh gpu-node-2 'df -h'
```

**Resolution**:
1. Delete unused models:
   ```bash
   ssh gpu-node-2 'rm ~/.cache/llama-orch/models/old-model.gguf'
   ```
2. Increase disk size
3. Use external storage (NFS mount)

---

## Future Enhancements (v2.0)

Planned features for automated model distribution:

- **Catalog Sync Protocol**: Automatic model replication
- **Peer-to-Peer Transfer**: Nodes copy from each other
- **Lazy Loading**: Download models on first use
- **Cache Eviction**: LRU eviction when disk full
- **Model Registry**: Central registry with metadata

---

## References

- [Cloud Profile Specification](../.specs/01_cloud_profile.md)
- [Catalog Core Library](../../libs/catalog-core/README.md)
- [Phase 7 Completion Summary](../.docs/PHASE7_CATALOG_COMPLETE.md)
- [Incident Runbook](./runbooks/CLOUD_PROFILE_INCIDENTS.md)

---

**Last Updated**: 2025-10-01  
**Maintainer**: Operations Team  
**Review Cadence**: Quarterly
