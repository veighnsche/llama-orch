# TEAM-019 Handoff: Observability for Homelab Build & Test Scripts

**Date:** 2025-10-09  
**From:** TEAM-018  
**To:** TEAM-019 (Observability)  
**Subject:** Observability hooks for homelab automation scripts  
**Status:** 🚧 Observability infrastructure needed

---

## Executive Summary

TEAM-018 has created 4 homelab automation scripts for building and testing llm-worker-rbee on remote machines (CUDA workstation, Metal Mac). These scripts currently provide human-readable terminal output but lack structured logging, metrics collection, and observability infrastructure.

**Your mission:** Add observability to these scripts so we can:
1. Track build/test success rates over time
2. Monitor build duration and performance trends
3. Correlate failures with git commits
4. Expose metrics via Prometheus
5. Stream real-time build logs via SSE (optional)

---

## Current State

### Scripts Created by TEAM-018

Located in `/scripts/homelab/`:

| Script | Target | Backend | Purpose |
|--------|--------|---------|---------|
| `workstation-build.sh` | workstation.home.arpa | CUDA | Build CUDA binary |
| `workstation-test.sh` | workstation.home.arpa | CUDA | Run CUDA tests |
| `mac-build.sh` | mac.home.arpa | Metal | Build Metal binary |
| `mac-test.sh` | mac.home.arpa | Metal | Run Metal tests |

### Current Output Format

**Human-readable only:**
- ASCII box drawing for visual separation
- Emoji indicators (🖥️, 🍎, 🧪, ✅)
- Timestamps in ISO 8601 format
- Full cargo output (no suppression)

**Example:**
```
════════════════════════════════════════════════════════════════
🖥️  WORKSTATION BUILD (CUDA Backend)
════════════════════════════════════════════════════════════════
Target: workstation.home.arpa
Backend: CUDA (NVIDIA GPU)
Started: 2025-10-09T09:27:33+02:00
════════════════════════════════════════════════════════════════
...
✅ Workstation build completed successfully
Finished: 2025-10-09T09:28:45+02:00
════════════════════════════════════════════════════════════════
```

---

## Observability Gaps

### 1. No Structured Logging

**Current:** Plain text output to stdout  
**Needed:** JSON-structured logs with:
- Timestamp (ISO 8601)
- Log level (INFO, WARN, ERROR)
- Context (script name, target host, backend)
- Event type (build_start, build_complete, test_start, etc.)
- Metadata (git commit, duration, exit code)

**Example desired output:**
```json
{
  "timestamp": "2025-10-09T09:27:33+02:00",
  "level": "INFO",
  "event": "build_start",
  "script": "workstation-build.sh",
  "target": "workstation.home.arpa",
  "backend": "cuda",
  "git_commit": "a1b2c3d",
  "git_branch": "main"
}
```

### 2. No Metrics Collection

**Current:** No metrics exported  
**Needed:** Prometheus metrics:
- `homelab_build_duration_seconds{host,backend}` - Build duration
- `homelab_build_success_total{host,backend}` - Successful builds counter
- `homelab_build_failure_total{host,backend}` - Failed builds counter
- `homelab_test_duration_seconds{host,backend}` - Test duration
- `homelab_test_success_total{host,backend}` - Successful test runs
- `homelab_test_failure_total{host,backend}` - Failed test runs
- `homelab_binary_size_bytes{host,backend}` - Built binary size

### 3. No Build Correlation

**Current:** Git commit shown in output but not captured  
**Needed:** Correlation between:
- Git commit hash → Build success/failure
- Git commit hash → Test results
- Git commit hash → Build duration trends

### 4. No Alerting

**Current:** Failures only visible in terminal  
**Needed:** 
- Slack/Discord webhook on build failure
- Email notification on test failure
- Threshold alerts (e.g., build duration > 5 minutes)

### 5. No Historical Data

**Current:** No persistence  
**Needed:**
- Time-series database (Prometheus, InfluxDB)
- Grafana dashboards
- Build history retention (30 days minimum)

---

## Proposed Architecture

### Phase 1: Structured Logging

**Add to each script:**
```bash
# At top of script
LOG_FILE="/var/log/llama-orch/homelab-$(basename "$0" .sh)-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

log_event() {
  local level="$1"
  local event="$2"
  shift 2
  local metadata="$*"
  
  jq -n \
    --arg ts "$(date -Iseconds)" \
    --arg level "$level" \
    --arg event "$event" \
    --arg script "$(basename "$0")" \
    --arg metadata "$metadata" \
    '{timestamp: $ts, level: $level, event: $event, script: $script, metadata: $metadata}' \
    | tee -a "$LOG_FILE"
}

# Usage:
log_event INFO build_start "host=workstation.home.arpa backend=cuda"
```

### Phase 2: Metrics Collection

**Option A: Pushgateway (recommended for batch jobs)**
```bash
# After build completes
BUILD_DURATION=$((END_TIME - START_TIME))
cat <<EOF | curl --data-binary @- http://pushgateway.home.arpa:9091/metrics/job/homelab_build/instance/workstation
# TYPE homelab_build_duration_seconds gauge
homelab_build_duration_seconds{host="workstation",backend="cuda"} $BUILD_DURATION
# TYPE homelab_build_success_total counter
homelab_build_success_total{host="workstation",backend="cuda"} 1
EOF
```

**Option B: Node exporter textfile collector**
```bash
# Write metrics to file
cat > /var/lib/node_exporter/textfile_collector/homelab_build.prom <<EOF
homelab_build_duration_seconds{host="workstation",backend="cuda"} $BUILD_DURATION
homelab_build_success_total{host="workstation",backend="cuda"} 1
EOF
```

### Phase 3: Real-Time Streaming (Optional)

**SSE endpoint for live build logs:**
```bash
# Stream to SSE server
exec > >(tee >(curl -X POST -H "Content-Type: text/event-stream" \
  http://sse-server.home.arpa:8080/stream/build/workstation))
```

### Phase 4: Alerting

**Webhook on failure:**
```bash
trap 'send_failure_alert' ERR

send_failure_alert() {
  curl -X POST http://alertmanager.home.arpa:9093/api/v1/alerts \
    -H "Content-Type: application/json" \
    -d '{
      "labels": {
        "alertname": "HomelabBuildFailure",
        "severity": "warning",
        "host": "workstation.home.arpa",
        "backend": "cuda"
      },
      "annotations": {
        "summary": "Build failed on workstation",
        "description": "CUDA build failed at $(date)"
      }
    }'
}
```

---

## Observability Hook Points

### In Each Script

TEAM-018 has marked observability hook points with comments:

```bash
# TEAM-019: Hook pre-build telemetry here (start timestamp, git commit hash)
# TEAM-019: Capture git metadata for build correlation
# TEAM-019: Hook repo sync telemetry here (bytes pulled, duration)
# TEAM-019: Hook build telemetry here (duration, binary size, warnings count)
# TEAM-019: Capture build artifacts metadata
# TEAM-019: Hook post-build telemetry here (end timestamp, success/failure)
```

**Search for:** `# TEAM-019:` in all 4 scripts

### Specific Metrics to Capture

#### Build Scripts

1. **Pre-build:**
   - Start timestamp
   - Git commit hash
   - Git branch
   - Host info (hostname, OS version)

2. **During build:**
   - Repo sync duration
   - Bytes pulled from git
   - Cargo build duration
   - Compiler warnings count
   - Clippy warnings count

3. **Post-build:**
   - End timestamp
   - Total duration
   - Binary size (bytes)
   - Exit code (0 = success)

#### Test Scripts

1. **Pre-test:**
   - Start timestamp
   - Environment info (GPU model, CUDA version, macOS version)

2. **During test:**
   - Test execution duration
   - Tests run count
   - Tests passed count
   - Tests failed count
   - Tests ignored count

3. **Post-test:**
   - End timestamp
   - Total duration
   - Exit code (0 = success)

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Homelab Scripts (workstation, mac)                          │
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│ │ Build Script│  │ Test Script │  │ Cron Job    │          │
│ └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
│        │                 │                 │                 │
│        └─────────────────┴─────────────────┘                │
│                          │                                   │
└──────────────────────────┼───────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │ Observability Layer (TEAM-019)      │
         │ ┌─────────────┐  ┌─────────────┐   │
         │ │ Structured  │  │ Metrics     │   │
         │ │ Logs (JSON) │  │ (Prometheus)│   │
         │ └──────┬──────┘  └──────┬──────┘   │
         │        │                 │          │
         │        ▼                 ▼          │
         │ ┌─────────────┐  ┌─────────────┐   │
         │ │ Log Storage │  │ Pushgateway │   │
         │ │ (Files/Loki)│  │ or Node Exp │   │
         │ └─────────────┘  └──────┬──────┘   │
         └────────────────────────┼────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────┐
                  │ Visualization & Alerting  │
                  │ ┌─────────┐  ┌─────────┐ │
                  │ │ Grafana │  │ Alerts  │ │
                  │ │ Dashbrd │  │ Manager │ │
                  │ └─────────┘  └─────────┘ │
                  └───────────────────────────┘
```

---

## Recommended Tech Stack

### Logging
- **Structured logs:** JSON format via `jq`
- **Storage:** Local files + Loki (optional)
- **Retention:** 30 days

### Metrics
- **Collection:** Prometheus Pushgateway (recommended for batch jobs)
- **Storage:** Prometheus
- **Retention:** 90 days

### Visualization
- **Dashboards:** Grafana
- **Suggested panels:**
  - Build success rate (last 7 days)
  - Build duration trend (by backend)
  - Test pass rate (by backend)
  - Binary size over time

### Alerting
- **Alertmanager** for Prometheus alerts
- **Webhooks** for Slack/Discord
- **Suggested alerts:**
  - Build failure (immediate)
  - Test failure (immediate)
  - Build duration > 5 minutes (warning)
  - Test duration > 2 minutes (warning)

---

## Implementation Plan

### Week 1: Foundation
- [ ] Add structured logging to all 4 scripts
- [ ] Set up log file rotation
- [ ] Create JSON schema for log events

### Week 2: Metrics
- [ ] Deploy Prometheus Pushgateway (or configure node exporter)
- [ ] Add metrics push to all scripts
- [ ] Verify metrics in Prometheus UI

### Week 3: Visualization
- [ ] Create Grafana dashboards
- [ ] Add panels for build/test metrics
- [ ] Set up dashboard auto-refresh

### Week 4: Alerting
- [ ] Configure Alertmanager
- [ ] Set up Slack/Discord webhooks
- [ ] Test alert delivery

---

## Testing Strategy

### Validation Checklist

- [ ] Run each script manually, verify JSON logs are written
- [ ] Verify metrics appear in Prometheus
- [ ] Trigger intentional build failure, verify alert fires
- [ ] Check Grafana dashboard shows recent builds
- [ ] Verify log retention policy works (30 days)

### Test Scenarios

1. **Successful build:** Metrics show success, no alerts
2. **Failed build:** Metrics show failure, alert fires
3. **Long build:** Duration metric captured correctly
4. **Network failure:** Script fails gracefully, logs error

---

## Security Considerations

### Secrets Management

**Do not hardcode:**
- Pushgateway URLs
- Webhook URLs
- API tokens

**Use environment variables:**
```bash
PUSHGATEWAY_URL="${PUSHGATEWAY_URL:-http://pushgateway.home.arpa:9091}"
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
```

### Log Sanitization

**Ensure logs do not contain:**
- SSH keys
- API tokens
- Passwords
- Personal data

---

## Future Enhancements

### Phase 5: Advanced Features (Post-MVP)

1. **Build caching metrics:**
   - Cache hit rate
   - Incremental build time

2. **Resource utilization:**
   - CPU usage during build
   - Memory usage during test
   - Disk I/O

3. **Distributed tracing:**
   - OpenTelemetry spans
   - End-to-end build trace

4. **ML-based anomaly detection:**
   - Detect unusual build durations
   - Predict build failures

---

## References

### Internal Documentation
- `/scripts/homelab/` - Scripts created by TEAM-018
- `bin/llm-worker-rbee/docs/metal.md` - Metal backend guide
- `.specs/TEAM_018_HANDOFF.md` - Metal migration handoff

### External Resources
- [Prometheus Pushgateway](https://github.com/prometheus/pushgateway)
- [Grafana Dashboards](https://grafana.com/docs/grafana/latest/dashboards/)
- [Alertmanager Configuration](https://prometheus.io/docs/alerting/latest/configuration/)
- [Loki for Log Aggregation](https://grafana.com/docs/loki/latest/)

---

## Questions for TEAM-019

Before you start, please clarify:

1. **Infrastructure:** Do we have Prometheus/Grafana already deployed in homelab?
2. **Storage:** Where should logs be stored? (Local files, Loki, S3?)
3. **Alerting:** What notification channels? (Slack, Discord, Email?)
4. **Retention:** Confirm retention policies (logs: 30d, metrics: 90d?)
5. **Access:** Do you need SSH access to homelab machines?

---

## Acceptance Criteria for TEAM-019

Your work is complete when:

- [x] All 4 scripts emit structured JSON logs
- [x] Metrics are pushed to Prometheus (or equivalent)
- [x] Grafana dashboard shows build/test history
- [x] Alerts fire on build/test failures
- [x] Documentation updated with observability setup

---

**Handoff completed:** 2025-10-09  
**From:** TEAM-018  
**To:** TEAM-019 (Observability)  
**Status:** ✅ Scripts ready, observability hooks marked  
**Next action:** TEAM-019 to implement structured logging and metrics collection

---

**Signed:**  
TEAM-018  
2025-10-09T09:27:33+02:00
