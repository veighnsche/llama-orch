# Configuration Reference

**Version**: 1.0  
**Date**: 2025-10-01  
**Audience**: Operators, DevOps Engineers

---

## Overview

This document provides a complete reference for configuring llama-orch in both HOME_PROFILE and CLOUD_PROFILE deployments.

---

## Deployment Profiles

llama-orch supports two deployment profiles controlled via environment variables.

### Profile Selection

```bash
# HOME_PROFILE (default)
# No configuration needed - this is the default behavior

# CLOUD_PROFILE
ORCHESTRATORD_CLOUD_PROFILE=true
```

---

## queen-rbee Configuration

### Core Settings

#### ORCHESTRATORD_CLOUD_PROFILE
- **Type**: Boolean
- **Default**: `false`
- **Valid Values**: `true`, `false`, `1`, `0`
- **Description**: Enable cloud profile mode for distributed deployment
- **Example**:
  ```bash
  ORCHESTRATORD_CLOUD_PROFILE=true
  ```

#### ORCHESTRATORD_BIND_ADDR
- **Type**: String (host:port)
- **Default**: `127.0.0.1:8080` (HOME_PROFILE), `0.0.0.0:8080` (CLOUD_PROFILE)
- **Description**: Address to bind HTTP server
- **Notes**: 
  - Use `127.0.0.1` for local-only access
  - Use `0.0.0.0` to accept external connections
- **Example**:
  ```bash
  ORCHESTRATORD_BIND_ADDR=0.0.0.0:8080
  ```

#### ORCHD_ADDR
- **Type**: String (host:port)
- **Default**: `0.0.0.0:8080`
- **Description**: Legacy alias for `ORCHESTRATORD_BIND_ADDR`
- **Example**:
  ```bash
  ORCHD_ADDR=0.0.0.0:8080
  ```

### Authentication (CLOUD_PROFILE Only)

#### LLORCH_API_TOKEN
- **Type**: String
- **Default**: None
- **Required**: Yes (for CLOUD_PROFILE)
- **Description**: Bearer token for inter-service authentication
- **Security**: Use a cryptographically secure random token
- **Example**:
  ```bash
  # Generate secure token
  LLORCH_API_TOKEN=$(openssl rand -hex 32)
  ```

### Node Management (CLOUD_PROFILE Only)

#### ORCHESTRATORD_NODE_TIMEOUT_MS
- **Type**: Integer (milliseconds)
- **Default**: `30000` (30 seconds)
- **Description**: Node heartbeat timeout - nodes not sending heartbeats within this interval are marked offline
- **Range**: `5000` to `300000` (5 seconds to 5 minutes)
- **Example**:
  ```bash
  ORCHESTRATORD_NODE_TIMEOUT_MS=30000
  ```

#### ORCHESTRATORD_STALE_CHECK_INTERVAL_SECS
- **Type**: Integer (seconds)
- **Default**: `10`
- **Description**: Interval for checking stale nodes
- **Range**: `1` to `60`
- **Example**:
  ```bash
  ORCHESTRATORD_STALE_CHECK_INTERVAL_SECS=10
  ```

### Handoff Watcher (HOME_PROFILE Only)

#### ORCHD_RUNTIME_DIR
- **Type**: String (path)
- **Default**: `.runtime/engines`
- **Description**: Directory where engine handoff files are written
- **Notes**: Must be accessible to both queen-rbee and engine-provisioner
- **Example**:
  ```bash
  ORCHD_RUNTIME_DIR=/var/lib/llama-orch/engines
  ```

#### ORCHD_HANDOFF_WATCH_INTERVAL_MS
- **Type**: Integer (milliseconds)
- **Default**: `1000` (1 second)
- **Description**: Polling interval for handoff file detection
- **Range**: `100` to `10000`
- **Example**:
  ```bash
  ORCHD_HANDOFF_WATCH_INTERVAL_MS=1000
  ```

### Placement Configuration

#### ORCHESTRATORD_PLACEMENT_STRATEGY
- **Type**: String (enum)
- **Default**: `round-robin`
- **Valid Values**: `round-robin`, `least-loaded`, `random`
- **Description**: Strategy for selecting pools across nodes
- **Example**:
  ```bash
  ORCHESTRATORD_PLACEMENT_STRATEGY=least-loaded
  ```

### Admission Queue

#### ORCHD_ADMISSION_CAPACITY
- **Type**: Integer
- **Default**: `100`
- **Description**: Maximum tasks in admission queue
- **Range**: `1` to `10000`
- **Example**:
  ```bash
  ORCHD_ADMISSION_CAPACITY=500
  ```

#### ORCHD_ADMISSION_POLICY
- **Type**: String (enum)
- **Default**: `drop-lru`
- **Valid Values**: `drop-lru`, `reject-new`
- **Description**: Backpressure policy when queue is full
- **Example**:
  ```bash
  ORCHD_ADMISSION_POLICY=drop-lru
  ```

### Observability

#### OTEL_EXPORTER_OTLP_ENDPOINT
- **Type**: String (URL)
- **Default**: None
- **Description**: OpenTelemetry OTLP exporter endpoint
- **Example**:
  ```bash
  OTEL_EXPORTER_OTLP_ENDPOINT=http://tempo:4317
  ```

#### PROMETHEUS_METRICS_PORT
- **Type**: Integer (port)
- **Default**: `9090`
- **Description**: Port for Prometheus metrics scraping
- **Example**:
  ```bash
  PROMETHEUS_METRICS_PORT=9090
  ```

#### RUST_LOG
- **Type**: String (log level)
- **Default**: `info`
- **Valid Values**: `trace`, `debug`, `info`, `warn`, `error`
- **Description**: Logging verbosity
- **Example**:
  ```bash
  RUST_LOG=info,queen-rbee=debug
  ```

---

## pool-managerd Configuration

### Core Settings

#### POOL_MANAGERD_BIND_ADDR
- **Type**: String (host:port)
- **Default**: `0.0.0.0:9200`
- **Description**: Address to bind pool-managerd HTTP server
- **Example**:
  ```bash
  POOL_MANAGERD_BIND_ADDR=0.0.0.0:9200
  ```

### Node Registration (CLOUD_PROFILE Only)

#### POOL_MANAGERD_NODE_ID
- **Type**: String
- **Default**: None
- **Required**: Yes (for CLOUD_PROFILE)
- **Description**: Unique identifier for this GPU node
- **Notes**: Must be unique across all nodes in the cluster
- **Example**:
  ```bash
  POOL_MANAGERD_NODE_ID=gpu-node-1
  ```

#### ORCHESTRATORD_URL
- **Type**: String (URL)
- **Default**: None
- **Required**: Yes (for CLOUD_PROFILE)
- **Description**: URL of queen-rbee control plane
- **Example**:
  ```bash
  ORCHESTRATORD_URL=http://queen-rbee:8080
  ```

#### LLORCH_API_TOKEN
- **Type**: String
- **Default**: None
- **Required**: Yes (for CLOUD_PROFILE)
- **Description**: Same Bearer token as queen-rbee
- **Example**:
  ```bash
  LLORCH_API_TOKEN=<same-token-as-queen-rbee>
  ```

#### POOL_MANAGERD_HEARTBEAT_INTERVAL_SECS
- **Type**: Integer (seconds)
- **Default**: `10`
- **Description**: Interval for sending heartbeats to queen-rbee
- **Range**: `1` to `60`
- **Notes**: Should be less than `ORCHESTRATORD_NODE_TIMEOUT_MS / 1000`
- **Example**:
  ```bash
  POOL_MANAGERD_HEARTBEAT_INTERVAL_SECS=10
  ```

### Handoff Watcher

#### POOL_MANAGERD_RUNTIME_DIR
- **Type**: String (path)
- **Default**: `.runtime/engines`
- **Description**: Directory where engine handoff files are written
- **Notes**: Must be accessible to both pool-managerd and engine-provisioner on the same node
- **Example**:
  ```bash
  POOL_MANAGERD_RUNTIME_DIR=/var/lib/llama-orch/engines
  ```

#### POOL_MANAGERD_WATCH_INTERVAL_MS
- **Type**: Integer (milliseconds)
- **Default**: `1000`
- **Description**: Polling interval for handoff file detection
- **Range**: `100` to `10000`
- **Example**:
  ```bash
  POOL_MANAGERD_WATCH_INTERVAL_MS=1000
  ```

#### POOL_MANAGERD_AUTO_DELETE_HANDOFF
- **Type**: Boolean
- **Default**: `true`
- **Description**: Automatically delete handoff files after processing
- **Example**:
  ```bash
  POOL_MANAGERD_AUTO_DELETE_HANDOFF=true
  ```

### GPU Discovery

#### POOL_MANAGERD_GPU_DISCOVERY_INTERVAL_MS
- **Type**: Integer (milliseconds)
- **Default**: `10000` (10 seconds)
- **Description**: Interval for GPU discovery scans
- **Range**: `1000` to `60000`
- **Example**:
  ```bash
  POOL_MANAGERD_GPU_DISCOVERY_INTERVAL_MS=10000
  ```

### Observability

#### OTEL_EXPORTER_OTLP_ENDPOINT
- **Type**: String (URL)
- **Default**: None
- **Description**: OpenTelemetry OTLP exporter endpoint
- **Example**:
  ```bash
  OTEL_EXPORTER_OTLP_ENDPOINT=http://tempo:4317
  ```

#### PROMETHEUS_METRICS_PORT
- **Type**: Integer (port)
- **Default**: `9091`
- **Description**: Port for Prometheus metrics scraping
- **Example**:
  ```bash
  PROMETHEUS_METRICS_PORT=9091
  ```

#### RUST_LOG
- **Type**: String (log level)
- **Default**: `info`
- **Valid Values**: `trace`, `debug`, `info`, `warn`, `error`
- **Description**: Logging verbosity
- **Example**:
  ```bash
  RUST_LOG=info,pool_managerd=debug
  ```

---

## engine-provisioner Configuration

### Core Settings

#### ENGINE_PROVISIONER_HANDOFF_DIR
- **Type**: String (path)
- **Default**: `.runtime/engines`
- **Description**: Directory where handoff files are written
- **Notes**: Must match `POOL_MANAGERD_RUNTIME_DIR`
- **Example**:
  ```bash
  ENGINE_PROVISIONER_HANDOFF_DIR=/var/lib/llama-orch/engines
  ```

#### ENGINE_PROVISIONER_CACHE_DIR
- **Type**: String (path)
- **Default**: `~/.cache/llama-orch`
- **Description**: Directory for caching downloaded engines
- **Example**:
  ```bash
  ENGINE_PROVISIONER_CACHE_DIR=/var/cache/llama-orch
  ```

### GPU Configuration

#### ENGINE_PROVISIONER_GPU_MASK
- **Type**: String (comma-separated GPU indices)
- **Default**: `0`
- **Description**: Which GPUs to use for engine provisioning
- **Example**:
  ```bash
  ENGINE_PROVISIONER_GPU_MASK=0,1  # Use GPUs 0 and 1
  ```

### Observability

#### OTEL_EXPORTER_OTLP_ENDPOINT
- **Type**: String (URL)
- **Default**: None
- **Description**: OpenTelemetry OTLP exporter endpoint
- **Example**:
  ```bash
  OTEL_EXPORTER_OTLP_ENDPOINT=http://tempo:4317
  ```

---

## Configuration Examples

### HOME_PROFILE (Single Machine)

```bash
# queen-rbee
ORCHD_ADDR=127.0.0.1:8080
ORCHD_RUNTIME_DIR=.runtime/engines
ORCHD_ADMISSION_CAPACITY=100
RUST_LOG=info

# pool-managerd (same machine)
POOL_MANAGERD_BIND_ADDR=127.0.0.1:9200
POOL_MANAGERD_RUNTIME_DIR=.runtime/engines
RUST_LOG=info

# engine-provisioner (same machine)
ENGINE_PROVISIONER_HANDOFF_DIR=.runtime/engines
```

### CLOUD_PROFILE (Multi-Machine)

**Control Plane Node** (no GPU):
```bash
# queen-rbee
ORCHESTRATORD_CLOUD_PROFILE=true
ORCHESTRATORD_BIND_ADDR=0.0.0.0:8080
LLORCH_API_TOKEN=abc123def456789...
ORCHESTRATORD_NODE_TIMEOUT_MS=30000
ORCHESTRATORD_PLACEMENT_STRATEGY=least-loaded
ORCHD_ADMISSION_CAPACITY=500
OTEL_EXPORTER_OTLP_ENDPOINT=http://tempo:4317
PROMETHEUS_METRICS_PORT=9090
RUST_LOG=info,queen-rbee=debug
```

**GPU Worker Node 1**:
```bash
# pool-managerd
POOL_MANAGERD_NODE_ID=gpu-node-1
ORCHESTRATORD_URL=http://control-plane:8080
LLORCH_API_TOKEN=abc123def456789...
POOL_MANAGERD_BIND_ADDR=0.0.0.0:9200
POOL_MANAGERD_RUNTIME_DIR=/var/lib/llama-orch/engines
POOL_MANAGERD_HEARTBEAT_INTERVAL_SECS=10
OTEL_EXPORTER_OTLP_ENDPOINT=http://tempo:4317
PROMETHEUS_METRICS_PORT=9091
RUST_LOG=info,pool_managerd=debug

# engine-provisioner
ENGINE_PROVISIONER_HANDOFF_DIR=/var/lib/llama-orch/engines
ENGINE_PROVISIONER_CACHE_DIR=/var/cache/llama-orch
ENGINE_PROVISIONER_GPU_MASK=0
OTEL_EXPORTER_OTLP_ENDPOINT=http://tempo:4317
```

**GPU Worker Node 2**:
```bash
# Same as Node 1, but with different NODE_ID and GPU_MASK
POOL_MANAGERD_NODE_ID=gpu-node-2
ENGINE_PROVISIONER_GPU_MASK=1
# ... rest same as Node 1
```

---

## Configuration Validation

### Startup Validation

queen-rbee and pool-managerd validate configuration on startup and will refuse to start if:

- Required environment variables are missing (CLOUD_PROFILE mode)
- Invalid values for type or range
- Conflicting settings (e.g., cloud profile enabled but no API token)

### Runtime Validation

Configuration is immutable after startup. Changes require service restart.

---

## Security Best Practices

### API Tokens

1. **Generation**: Use cryptographically secure random tokens
   ```bash
   openssl rand -hex 32
   ```

2. **Storage**: 
   - Use Kubernetes Secrets or Docker Secrets
   - Never commit tokens to version control
   - Rotate tokens periodically

3. **Distribution**: Same token must be configured on queen-rbee and all pool-managerd instances

### Network Binding

1. **HOME_PROFILE**: Use `127.0.0.1` to restrict access to localhost
2. **CLOUD_PROFILE**: Use `0.0.0.0` but deploy behind firewall/network policies

### Filesystem Permissions

1. Handoff directories (`POOL_MANAGERD_RUNTIME_DIR`) should be:
   - Readable/writable by pool-managerd and engine-provisioner
   - Not accessible to other users

2. Model cache directories should have restrictive permissions

---

## Troubleshooting

### Configuration Not Taking Effect

**Problem**: Changed environment variable but behavior unchanged

**Solution**: Restart the service - configuration is read only on startup

### Authentication Failures

**Problem**: `401 Unauthorized` on node registration/heartbeat

**Diagnosis**:
```bash
# Check token matches on both sides
echo $LLORCH_API_TOKEN  # on control plane
echo $LLORCH_API_TOKEN  # on worker node

# Check queen-rbee logs for token fingerprint
journalctl -u queen-rbee | grep token_fp6
```

**Solution**: Ensure same token is configured on all nodes

### Nodes Not Appearing

**Problem**: GPU node not showing up in `/v2/nodes`

**Diagnosis**:
```bash
# Check if node is attempting registration
journalctl -u pool-managerd | grep registration

# Check queen-rbee received registration
journalctl -u queen-rbee | grep node_registration
```

**Solution**: 
- Verify `ORCHESTRATORD_URL` is correct
- Check network connectivity: `curl http://queen-rbee:8080/v2/meta/capabilities`
- Verify API token matches

---

## References

- [Cloud Profile Specification](../.specs/01_cloud_profile.md)
- [Deployment Guides](./deployments/)
- [Troubleshooting Guide](./TROUBLESHOOTING.md)
- [Security Policy](../SECURITY.md)

---

**Last Updated**: 2025-10-01  
**Maintainer**: Engineering Team  
**Review Cadence**: Quarterly
