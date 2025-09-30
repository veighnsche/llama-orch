# Cloud Profile Incident Runbook

**Version**: 1.0  
**Date**: 2025-10-01  
**Audience**: Operations, SRE, On-call Engineers

---

## Overview

This runbook provides troubleshooting procedures for common cloud profile incidents. Cloud profile enables distributed deployment with orchestratord (control plane) and multiple pool-managerd instances (GPU workers).

**Prerequisites**:
- Access to Grafana dashboard: `Cloud Profile Overview`
- Access to Prometheus metrics endpoint
- SSH access to control plane and GPU nodes
- Knowledge of deployment topology

---

## Quick Reference

| Alert | Severity | MTTR Target | First Action |
|-------|----------|-------------|--------------|
| NoNodesOnline | Critical | 5 min | Check node connectivity |
| NoPoolsAvailable | Critical | 5 min | Check pool-managerd processes |
| HeartbeatStalled | Critical | 10 min | Restart pool-managerd |
| HighHealthCheckLatency | Warning | 30 min | Check network latency |
| HandoffProcessingStalled | Warning | 15 min | Check handoff watcher |

---

## Incident Response Workflow

1. **Acknowledge alert** in alerting system
2. **Check Grafana dashboard** for context
3. **Follow runbook procedure** for specific alert
4. **Document actions** in incident log
5. **Resolve alert** when metrics return to normal
6. **Post-mortem** for critical incidents

---

## Incidents

### No Nodes Online

**Alert**: `NoNodesOnline`  
**Severity**: Critical  
**Impact**: No tasks can be processed

#### Symptoms
- `sum(orchd_nodes_online) == 0`
- All task submissions fail with placement errors
- orchestratord logs show no registered nodes

#### Diagnosis

1. **Check orchestratord logs**:
   ```bash
   journalctl -u orchestratord -n 100 --no-pager | grep -i "node"
   ```

2. **Check service registry state**:
   ```bash
   curl http://orchestratord:8080/v2/nodes
   ```

3. **Check pool-managerd processes on GPU nodes**:
   ```bash
   ssh gpu-node-1 'systemctl status pool-managerd'
   ssh gpu-node-2 'systemctl status pool-managerd'
   ```

4. **Check network connectivity**:
   ```bash
   ssh gpu-node-1 'curl -v http://orchestratord:8080/health'
   ```

#### Resolution

**If pool-managerd is down**:
```bash
ssh gpu-node-1 'systemctl restart pool-managerd'
```

**If network is down**:
- Check firewall rules (port 8080 must be open)
- Check DNS resolution
- Check routing tables

**If authentication is failing**:
- Verify `LLORCH_API_TOKEN` matches on both sides
- Check orchestratord logs for `auth_failed` events
- Regenerate token if compromised

#### Verification
```bash
# Should show online nodes
curl http://orchestratord:8080/v2/nodes | jq '.count'

# Metric should be > 0
curl http://orchestratord:8080/metrics | grep orchd_nodes_online
```

---

### Low Node Availability

**Alert**: `LowNodeAvailability`  
**Severity**: Warning  
**Impact**: Reduced capacity, no redundancy

#### Symptoms
- `sum(orchd_nodes_online) < 2`
- Single point of failure
- Increased load on remaining nodes

#### Diagnosis

1. **Identify missing nodes**:
   ```bash
   curl http://orchestratord:8080/v2/nodes | jq '.nodes[] | select(.online == false)'
   ```

2. **Check why nodes went offline**:
   - Heartbeat timeout (30s default)
   - Graceful shutdown
   - Crash/restart

3. **Check node logs**:
   ```bash
   ssh gpu-node-X 'journalctl -u pool-managerd -n 200 --no-pager'
   ```

#### Resolution

**If node crashed**:
```bash
ssh gpu-node-X 'systemctl restart pool-managerd'
```

**If node is draining**:
- Wait for drain to complete
- Or cancel drain if urgent

**If node is under maintenance**:
- Document in incident log
- Suppress alert if planned

---

### No Pools Available

**Alert**: `NoPoolsAvailable`  
**Severity**: Critical  
**Impact**: Tasks cannot be dispatched

#### Symptoms
- `sum(orchd_pools_available) == 0`
- Tasks enqueued but not started
- Placement failures in logs

#### Diagnosis

1. **Check pool status**:
   ```bash
   curl http://orchestratord:8080/v2/nodes | jq '.nodes[].pools[] | select(.ready == false)'
   ```

2. **Check pool-managerd registry**:
   ```bash
   ssh gpu-node-1 'curl http://localhost:9200/pools/pool-0/status'
   ```

3. **Check engine processes**:
   ```bash
   ssh gpu-node-1 'ps aux | grep llama'
   ```

#### Resolution

**If engines are not running**:
```bash
# Trigger preload
curl -X POST http://gpu-node-1:9200/pools/pool-0/preload \
  -H "Authorization: Bearer $LLORCH_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model_id": "llama-3.1-8b-instruct", "engine": "llamacpp"}'
```

**If handoff files are stuck**:
```bash
# Check handoff directory
ssh gpu-node-1 'ls -la /var/lib/llama-orch/engines/*.json'

# Manually process handoff
ssh gpu-node-1 'systemctl restart pool-managerd'
```

**If pools are draining**:
- Cancel drain or wait for completion

---

### High Health Check Latency

**Alert**: `HighHealthCheckLatency`  
**Severity**: Warning  
**Impact**: Slow placement decisions

#### Symptoms
- P95 health check latency > 100ms
- Increased admission latency
- Slow task dispatch

#### Diagnosis

1. **Check network latency**:
   ```bash
   ping -c 10 gpu-node-1
   ```

2. **Check pool-managerd load**:
   ```bash
   ssh gpu-node-1 'top -b -n 1 | head -20'
   ```

3. **Check for network congestion**:
   ```bash
   ssh gpu-node-1 'iftop -t -s 10'
   ```

#### Resolution

**If network latency is high**:
- Check for network congestion
- Check for misconfigured routes
- Consider moving nodes to same subnet

**If pool-managerd is overloaded**:
- Reduce heartbeat frequency
- Add more CPU cores
- Optimize handoff processing

**If database is slow** (future):
- Check query performance
- Add indexes
- Scale database

---

### Heartbeat Failures

**Alert**: `HeartbeatFailureRate` or `NodeHeartbeatStalled`  
**Severity**: Warning/Critical  
**Impact**: Node may be marked offline

#### Symptoms
- Heartbeat failure rate > 10%
- No heartbeats for 2+ minutes
- Node shows as offline despite being up

#### Diagnosis

1. **Check pool-managerd logs**:
   ```bash
   ssh gpu-node-1 'journalctl -u pool-managerd -n 100 --no-pager | grep heartbeat'
   ```

2. **Check authentication**:
   ```bash
   # Should return 200
   curl -H "Authorization: Bearer $LLORCH_API_TOKEN" \
     http://orchestratord:8080/v2/nodes/gpu-node-1/heartbeat \
     -X POST -d '{"timestamp":"2025-10-01T00:00:00Z","pools":[]}'
   ```

3. **Check network connectivity**:
   ```bash
   ssh gpu-node-1 'curl -v http://orchestratord:8080/health'
   ```

#### Resolution

**If authentication is failing**:
```bash
# Verify token
ssh gpu-node-1 'echo $LLORCH_API_TOKEN'

# Update token
ssh gpu-node-1 'export LLORCH_API_TOKEN=new-token && systemctl restart pool-managerd'
```

**If network is intermittent**:
- Check for packet loss
- Check firewall rules
- Check DNS resolution

**If pool-managerd is stuck**:
```bash
ssh gpu-node-1 'systemctl restart pool-managerd'
```

---

### Handoff Processing Stalled

**Alert**: `HandoffProcessingStalled`  
**Severity**: Warning  
**Impact**: New engines not detected

#### Symptoms
- No handoff files processed for 5+ minutes
- Engines running but pools not ready
- Handoff files accumulating in directory

#### Diagnosis

1. **Check handoff directory**:
   ```bash
   ssh gpu-node-1 'ls -la /var/lib/llama-orch/engines/*.json'
   ```

2. **Check handoff watcher logs**:
   ```bash
   ssh gpu-node-1 'journalctl -u pool-managerd -n 100 --no-pager | grep handoff'
   ```

3. **Check file permissions**:
   ```bash
   ssh gpu-node-1 'ls -la /var/lib/llama-orch/engines/'
   ```

#### Resolution

**If watcher is stuck**:
```bash
ssh gpu-node-1 'systemctl restart pool-managerd'
```

**If files are malformed**:
```bash
# Validate JSON
ssh gpu-node-1 'cat /var/lib/llama-orch/engines/pool-0-r0.json | jq .'

# Remove invalid files
ssh gpu-node-1 'rm /var/lib/llama-orch/engines/invalid.json'
```

**If permissions are wrong**:
```bash
ssh gpu-node-1 'chown pool-managerd:pool-managerd /var/lib/llama-orch/engines/*.json'
```

---

### Registration Failures

**Alert**: `HighRegistrationFailureRate`  
**Severity**: Warning  
**Impact**: Nodes cannot join cluster

#### Symptoms
- Registration failure rate > 20%
- New nodes fail to register
- orchestratord logs show auth failures

#### Diagnosis

1. **Check orchestratord logs**:
   ```bash
   journalctl -u orchestratord -n 100 --no-pager | grep register
   ```

2. **Check authentication**:
   ```bash
   # Should return 200
   curl -H "Authorization: Bearer $LLORCH_API_TOKEN" \
     http://orchestratord:8080/v2/nodes/register \
     -X POST -d '{"node_id":"test","machine_id":"test","address":"http://test:9200","pools":[],"capabilities":{},"version":"0.1.0"}'
   ```

3. **Check cloud profile enabled**:
   ```bash
   echo $ORCHESTRATORD_CLOUD_PROFILE
   ```

#### Resolution

**If cloud profile is disabled**:
```bash
export ORCHESTRATORD_CLOUD_PROFILE=true
systemctl restart orchestratord
```

**If authentication is failing**:
- Verify tokens match
- Check for token expiry (if implemented)
- Regenerate tokens

**If service registry is full** (future):
- Increase capacity
- Clean up stale nodes

---

## Escalation

### When to Escalate

- **Critical alerts** not resolved within MTTR target
- **Multiple simultaneous incidents**
- **Unknown root cause** after 30 minutes
- **Data loss** or corruption suspected

### Escalation Path

1. **L1 → L2**: On-call engineer → Senior SRE
2. **L2 → L3**: Senior SRE → Engineering team lead
3. **L3 → L4**: Team lead → CTO

### Contact Information

- **On-call rotation**: PagerDuty
- **Slack channel**: `#llama-orch-incidents`
- **Email**: `sre@example.com`

---

## Post-Incident

### Required Actions

1. **Document incident** in post-mortem template
2. **Update runbook** with new findings
3. **Create follow-up tickets** for preventive measures
4. **Share learnings** in team meeting

### Post-Mortem Template

```markdown
# Incident Post-Mortem: [Title]

**Date**: YYYY-MM-DD  
**Duration**: X hours  
**Severity**: Critical/Warning  
**Impact**: [Description]

## Timeline
- HH:MM - Alert triggered
- HH:MM - Engineer acknowledged
- HH:MM - Root cause identified
- HH:MM - Mitigation applied
- HH:MM - Incident resolved

## Root Cause
[Detailed analysis]

## Resolution
[What fixed it]

## Prevention
- [ ] Action item 1
- [ ] Action item 2

## Lessons Learned
[Key takeaways]
```

---

## Useful Commands

### Check System Health
```bash
# orchestratord health
curl http://orchestratord:8080/health

# pool-managerd health
curl http://gpu-node-1:9200/health

# Metrics snapshot
curl http://orchestratord:8080/metrics | grep orchd_
```

### Check Logs
```bash
# orchestratord logs (last hour)
journalctl -u orchestratord --since "1 hour ago" --no-pager

# pool-managerd logs (last hour)
ssh gpu-node-1 'journalctl -u pool-managerd --since "1 hour ago" --no-pager'

# Filter for errors
journalctl -u orchestratord -p err --no-pager
```

### Manual Operations
```bash
# Force node deregistration
curl -X DELETE http://orchestratord:8080/v2/nodes/gpu-node-1 \
  -H "Authorization: Bearer $LLORCH_API_TOKEN"

# Trigger pool preload
curl -X POST http://gpu-node-1:9200/pools/pool-0/preload \
  -H "Authorization: Bearer $LLORCH_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model_id":"llama-3.1-8b-instruct","engine":"llamacpp"}'

# Check pool status
curl http://gpu-node-1:9200/pools/pool-0/status
```

---

## References

- [Cloud Profile Specification](../../.specs/01_cloud_profile.md)
- [Cloud Profile Migration Plan](../../CLOUD_PROFILE_MIGRATION_PLAN.md)
- [Authentication Security Review](../../.docs/AUTH_SECURITY_REVIEW.md)
- [Grafana Dashboard](../../ci/dashboards/cloud_profile_overview.json)
- [Prometheus Alerts](../../ci/alerts/cloud_profile.yml)

---

**Last Updated**: 2025-10-01  
**Maintainer**: SRE Team  
**Review Cadence**: Quarterly
