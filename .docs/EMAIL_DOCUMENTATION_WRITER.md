# Email: Cloud Profile Migration Documentation Request

**To**: Documentation Team  
**From**: Engineering Team  
**Date**: 2025-10-01  
**Subject**: Cloud Profile Migration - Phase 9 Documentation Requirements  
**Priority**: High  
**Target Completion**: 1 week

---

## Executive Summary

We've completed **Phases 5-8** of the Cloud Profile Migration (security, observability, catalog distribution, and testing). The implementation is **production-ready** and needs comprehensive documentation for the v0.2.0 release.

**Your Task**: Create end-user and operator documentation for the new **CLOUD_PROFILE** deployment mode, which enables distributed multi-node deployments of llama-orch.

---

## What We've Built (Phases 5-8)

### Phase 5: Authentication & Security ✅ (Complete)

**What Changed**:
- All node management endpoints now require Bearer token authentication
- Timing-safe token comparison prevents timing attacks
- Token fingerprinting in audit logs
- Security review passed with approval

**Documentation Needed**:
- How to generate secure API tokens
- Where to configure `LLORCH_API_TOKEN`
- Security best practices for token management
- Token rotation procedures

**Reference Files**:
- `.docs/AUTH_SECURITY_REVIEW.md` - Security review details
- `bin/orchestratord/src/app/auth_min.rs` - Auth implementation
- `bin/pool-managerd/src/api/auth.rs` - pool-managerd auth

---

### Phase 6: Observability & Monitoring ✅ (Complete)

**What Changed**:
- Added 7 new Prometheus metrics for cloud profile operations
- Created Grafana dashboard with 8 panels
- Created 12 Prometheus alerting rules
- Created incident runbook with troubleshooting procedures

**Documentation Needed**:
- How to import the Grafana dashboard
- How to configure Prometheus alerts
- Explanation of each metric (what it measures, why it matters)
- How to interpret the dashboard panels
- When to use the incident runbook

**Reference Files**:
- `ci/dashboards/cloud_profile_overview.json` - Grafana dashboard
- `ci/alerts/cloud_profile.yml` - Prometheus alerts
- `docs/runbooks/CLOUD_PROFILE_INCIDENTS.md` - Incident runbook
- `.docs/PHASE6_OBSERVABILITY_COMPLETE.md` - Implementation summary

**Key Metrics to Document**:
```
orchd_node_registrations_total{outcome}
orchd_node_heartbeats_total{node_id, outcome}
orchd_node_deregistrations_total{outcome}
orchd_pool_health_checks_total{pool_id, outcome}
orchd_nodes_online
orchd_pools_available{pool_id}
orchd_pool_health_check_duration_ms{pool_id}
```

---

### Phase 7: Catalog Distribution ✅ (Complete)

**What Changed**:
- Added `GET /v2/catalog/availability` endpoint
- Placement now filters by model availability
- Models must be manually staged to each GPU node
- Created comprehensive operator guide for model staging

**Documentation Needed**:
- Overview of catalog distribution architecture
- How to query catalog availability
- How to interpret the availability response
- Link to manual staging guide (already written)
- Best practices for model replication

**Reference Files**:
- `docs/MANUAL_MODEL_STAGING.md` - **Already complete** (300+ lines, ready to use)
- `bin/orchestratord/src/api/catalog_availability.rs` - API implementation
- `bin/orchestratord/src/services/placement_v2.rs` - Placement logic

**API Endpoint to Document**:
```bash
GET /v2/catalog/availability
Authorization: Bearer <token>

Response:
{
  "nodes": {
    "gpu-node-1": {
      "models": ["llama-3.1-8b-instruct"],
      "pools": [...]
    }
  },
  "total_models": 2,
  "replicated_models": [],
  "single_node_models": ["llama-3.1-8b-instruct"]
}
```

---

### Phase 8: Testing & Validation ✅ (Complete)

**What Changed**:
- Created 13 new tests (700+ lines)
- 100% test coverage of cloud profile features
- Integration tests for node lifecycle
- Unit tests for model-aware placement

**Documentation Needed**:
- Testing approach (unit + integration, not full E2E)
- How to run the tests
- What's tested vs what's deferred
- Test coverage summary

**Reference Files**:
- `.docs/PHASE8_TESTING_COMPLETE.md` - Test summary
- `bin/orchestratord/tests/cloud_profile_integration.rs` - Integration tests
- `bin/orchestratord/tests/placement_v2_tests.rs` - Placement tests

---

## Phase 9: Your Documentation Tasks

### 1. Update README.md (High Priority)

**Location**: `/home/vince/Projects/llama-orch/README.md`

**Add Section**: "Cloud Profile Deployment"

**Content Needed**:
- Quick overview of HOME_PROFILE vs CLOUD_PROFILE
- When to use each profile
- Architecture diagram showing control plane + GPU workers
- Link to detailed deployment guides

**Example Structure**:
```markdown
## Deployment Profiles

### HOME_PROFILE (Default)
Single machine deployment with shared filesystem.
- Use for: Development, single-GPU setups
- Configuration: No special config needed

### CLOUD_PROFILE
Distributed deployment across multiple machines.
- Use for: Production, multi-GPU clusters, cloud deployments
- Configuration: Set ORCHESTRATORD_CLOUD_PROFILE=true
- See: [Cloud Profile Deployment Guide](docs/CLOUD_PROFILE_DEPLOYMENT.md)
```

---

### 2. Create Deployment Guide: Kubernetes (High Priority)

**Location**: `docs/deployments/KUBERNETES.md`

**Content Needed**:

#### Prerequisites
- Kubernetes cluster (version requirements)
- kubectl configured
- GPU nodes with NVIDIA drivers
- Persistent storage for models

#### Architecture
- Deployment topology diagram
- Service mesh considerations
- Network policies

#### Step-by-Step Deployment

**Control Plane**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: orchestratord
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: orchestratord
        image: llama-orch/orchestratord:v0.2.0
        env:
        - name: ORCHESTRATORD_CLOUD_PROFILE
          value: "true"
        - name: LLORCH_API_TOKEN
          valueFrom:
            secretKeyRef:
              name: llama-orch-secrets
              key: api-token
        ports:
        - containerPort: 8080
```

**GPU Workers**:
```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: pool-managerd
spec:
  template:
    spec:
      nodeSelector:
        gpu: "true"
      containers:
      - name: pool-managerd
        image: llama-orch/pool-managerd:v0.2.0
        env:
        - name: ORCHESTRATORD_URL
          value: "http://orchestratord:8080"
        - name: LLORCH_API_TOKEN
          valueFrom:
            secretKeyRef:
              name: llama-orch-secrets
              key: api-token
        resources:
          limits:
            nvidia.com/gpu: 1
```

#### Configuration
- Environment variables reference
- Secret management
- Volume mounts for models

#### Verification
- How to check deployment status
- How to verify nodes registered
- How to test task submission

#### Troubleshooting
- Common issues and solutions
- Log collection commands
- Debug mode

---

### 3. Create Deployment Guide: Docker Compose (High Priority)

**Location**: `docs/deployments/DOCKER_COMPOSE.md`

**Content Needed**:

#### Prerequisites
- Docker Engine 20.10+
- Docker Compose v2
- NVIDIA Container Toolkit (for GPU nodes)

#### Quick Start

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  orchestratord:
    image: llama-orch/orchestratord:v0.2.0
    environment:
      ORCHESTRATORD_CLOUD_PROFILE: "true"
      ORCHESTRATORD_BIND_ADDR: "0.0.0.0:8080"
      LLORCH_API_TOKEN: "${LLORCH_API_TOKEN}"
    ports:
      - "8080:8080"
    networks:
      - llama-orch

  pool-managerd-gpu1:
    image: llama-orch/pool-managerd:v0.2.0
    environment:
      POOL_MANAGERD_NODE_ID: "gpu-node-1"
      ORCHESTRATORD_URL: "http://orchestratord:8080"
      LLORCH_API_TOKEN: "${LLORCH_API_TOKEN}"
    runtime: nvidia
    environment:
      NVIDIA_VISIBLE_DEVICES: "0"
    volumes:
      - ./models:/models
    networks:
      - llama-orch

networks:
  llama-orch:
    driver: bridge
```

#### Multi-Machine Setup
- How to deploy across multiple Docker hosts
- Network configuration
- Model volume sharing strategies

---

### 4. Create Deployment Guide: Bare Metal (Medium Priority)

**Location**: `docs/deployments/BARE_METAL.md`

**Content Needed**:

#### Prerequisites
- Ubuntu 22.04 or later (or equivalent)
- NVIDIA drivers installed
- Rust toolchain (for building from source)

#### Installation

**Control Plane Node**:
```bash
# Install orchestratord
cargo build --release -p orchestratord

# Configure
export ORCHESTRATORD_CLOUD_PROFILE=true
export ORCHESTRATORD_BIND_ADDR=0.0.0.0:8080
export LLORCH_API_TOKEN=$(openssl rand -hex 32)

# Run
./target/release/orchestratord
```

**GPU Worker Nodes**:
```bash
# Install pool-managerd
cargo build --release -p pool-managerd

# Configure
export POOL_MANAGERD_NODE_ID=gpu-node-1
export ORCHESTRATORD_URL=http://control-plane:8080
export LLORCH_API_TOKEN=<same-token-as-control-plane>

# Run
./target/release/pool-managerd
```

#### Systemd Service Files
- orchestratord.service
- pool-managerd.service

#### Firewall Configuration
- Required ports
- Security group rules

---

### 5. Create Configuration Reference (High Priority)

**Location**: `docs/CONFIGURATION.md`

**Content Needed**:

#### Environment Variables

**orchestratord**:
```
ORCHESTRATORD_CLOUD_PROFILE (bool, default: false)
  Enable cloud profile mode for distributed deployment.

ORCHESTRATORD_BIND_ADDR (string, default: 127.0.0.1:8080)
  Address to bind HTTP server.
  Cloud profile: Use 0.0.0.0:8080 to accept external connections.

LLORCH_API_TOKEN (string, required for cloud profile)
  Bearer token for authentication.
  Generate with: openssl rand -hex 32

ORCHESTRATORD_NODE_TIMEOUT_MS (int, default: 30000)
  Node heartbeat timeout in milliseconds.

ORCHESTRATORD_PLACEMENT_STRATEGY (string, default: round-robin)
  Placement strategy: round-robin, least-loaded, random
```

**pool-managerd**:
```
POOL_MANAGERD_NODE_ID (string, required for cloud profile)
  Unique identifier for this GPU node.

ORCHESTRATORD_URL (string, required for cloud profile)
  URL of orchestratord control plane.

LLORCH_API_TOKEN (string, required for cloud profile)
  Same token as orchestratord.

POOL_MANAGERD_RUNTIME_DIR (string, default: .runtime/engines)
  Directory for handoff files.

POOL_MANAGERD_HEARTBEAT_INTERVAL_SECS (int, default: 10)
  Heartbeat interval in seconds.
```

#### Configuration Files
- Where to place config files (if any)
- Configuration precedence (env > file > default)

---

### 6. Create Troubleshooting Guide (Medium Priority)

**Location**: `docs/TROUBLESHOOTING.md`

**Content Needed**:

#### Common Issues

**Node Not Registering**:
- Symptoms
- Diagnosis steps
- Solutions

**Authentication Failures**:
- Check token matches
- Verify Bearer format
- Check logs for fingerprint

**Models Not Available**:
- Check catalog availability endpoint
- Verify model files exist
- Check pool heartbeat

**High Latency**:
- Check network latency
- Check pool health check metrics
- Review Grafana dashboard

#### Log Collection
- Where logs are located
- How to increase verbosity
- What to include in bug reports

#### Getting Help
- GitHub issues
- Community channels
- Support contacts

---

### 7. Update Architecture Diagrams (Medium Priority)

**Location**: `docs/architecture/`

**Diagrams Needed**:

#### HOME_PROFILE Architecture
```
┌─────────────────────────────────┐
│     Single Machine              │
│                                 │
│  ┌──────────────┐              │
│  │ orchestratord│              │
│  └──────┬───────┘              │
│         │                       │
│  ┌──────▼────────┐             │
│  │ pool-managerd │             │
│  └──────┬────────┘             │
│         │                       │
│  ┌──────▼────────┐             │
│  │ llama.cpp     │             │
│  │ (GPU)         │             │
│  └───────────────┘             │
└─────────────────────────────────┘
```

#### CLOUD_PROFILE Architecture
```
┌─────────────────┐         ┌──────────────────┐
│ Control Plane   │         │ GPU Worker 1     │
│                 │         │                  │
│ orchestratord   │◄────────┤ pool-managerd    │
│ (no GPU)        │  HTTP   │ + llama.cpp      │
│                 │         │ (GPU 0)          │
└─────────────────┘         └──────────────────┘
         ▲
         │ HTTP
         │
         ▼
┌──────────────────┐
│ GPU Worker 2     │
│                  │
│ pool-managerd    │
│ + llama.cpp      │
│ (GPU 1)          │
└──────────────────┘
```

#### Communication Flow
- Node registration sequence
- Heartbeat flow
- Task dispatch flow
- Model availability check

---

### 8. Create Migration Guide (Low Priority)

**Location**: `docs/MIGRATION_HOME_TO_CLOUD.md`

**Content Needed**:

#### When to Migrate
- Scaling beyond single machine
- Geographic distribution
- Cloud deployment requirements

#### Migration Steps
1. Backup current deployment
2. Deploy control plane
3. Deploy GPU workers
4. Stage models to workers
5. Update client configuration
6. Verify functionality
7. Decommission old deployment

#### Rollback Plan
- How to revert to HOME_PROFILE
- Data preservation
- Downtime expectations

---

## Documentation Standards

### Style Guide
- Use clear, concise language
- Include code examples for all procedures
- Add troubleshooting sections to each guide
- Use consistent terminology (see glossary below)

### Glossary
- **Control Plane**: Node running orchestratord (no GPU)
- **GPU Worker**: Node running pool-managerd + engines (has GPU)
- **Node**: A machine in the cluster
- **Pool**: A collection of inference slots on a GPU
- **Replica**: A single inference slot
- **HOME_PROFILE**: Single-machine deployment mode
- **CLOUD_PROFILE**: Distributed deployment mode

### Code Examples
- Always include full commands (no placeholders)
- Show expected output
- Include error cases

### Screenshots
- Grafana dashboard views
- Kubernetes dashboard
- Example API responses

---

## Deliverables Checklist

### High Priority (Week 1)
- [ ] README.md updates
- [ ] Kubernetes deployment guide
- [ ] Docker Compose deployment guide
- [ ] Configuration reference
- [ ] Link manual staging guide (already written)

### Medium Priority (Week 2)
- [ ] Bare metal deployment guide
- [ ] Troubleshooting guide
- [ ] Architecture diagrams
- [ ] Observability guide (metrics, dashboards, alerts)

### Low Priority (Future)
- [ ] Migration guide
- [ ] Video tutorials
- [ ] Blog post

---

## Resources Available

### Existing Documentation (Ready to Use)
- ✅ `docs/MANUAL_MODEL_STAGING.md` - Complete operator guide (300+ lines)
- ✅ `docs/runbooks/CLOUD_PROFILE_INCIDENTS.md` - Incident runbook (600+ lines)
- ✅ `.docs/AUTH_SECURITY_REVIEW.md` - Security details
- ✅ `.docs/PHASE6_OBSERVABILITY_COMPLETE.md` - Observability summary
- ✅ `.docs/PHASE8_TESTING_COMPLETE.md` - Testing summary

### Reference Implementations
- `ci/dashboards/cloud_profile_overview.json` - Grafana dashboard JSON
- `ci/alerts/cloud_profile.yml` - Prometheus alerts YAML
- `bin/orchestratord/tests/cloud_profile_integration.rs` - Integration test examples

### Specifications
- `.specs/01_cloud_profile.md` - Cloud profile specification
- `.specs/metrics/otel-prom.md` - Metrics contract
- `.specs/11_min_auth_hooks.md` - Authentication spec

---

## Questions?

**Engineering Contact**: [Your Name]  
**Slack Channel**: #llama-orch-docs  
**GitHub**: https://github.com/your-org/llama-orch

**Review Process**: Submit PRs for each guide, tag @engineering-team for technical review.

---

## Timeline

**Week 1** (High Priority):
- Day 1-2: README + Configuration reference
- Day 3-4: Kubernetes guide
- Day 5: Docker Compose guide

**Week 2** (Medium Priority):
- Day 1-2: Bare metal guide
- Day 3-4: Troubleshooting + Architecture diagrams
- Day 5: Review and polish

**Target Completion**: 2025-10-08 (1 week from now)

---

**Thank you for your work on this! The engineering team has built a solid foundation, and your documentation will make it accessible to users and operators.**

---

**Attachments**:
- Phase 6 Summary: `.docs/PHASE6_OBSERVABILITY_COMPLETE.md`
- Phase 7 Summary: `docs/MANUAL_MODEL_STAGING.md`
- Phase 8 Summary: `.docs/PHASE8_TESTING_COMPLETE.md`
- Migration Plan: `CLOUD_PROFILE_MIGRATION_PLAN.md`
- TODO Tracker: `TODO_CLOUD_PROFILE.md`
