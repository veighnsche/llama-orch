# Chaos Testing Suite

**Created by:** TEAM-107 | 2025-10-18  
**Purpose:** Validate system resilience under failure conditions

---

## Overview

This chaos testing suite validates that the rbee system can survive and recover from:
- Random network failures
- Worker crashes
- Resource exhaustion
- Disk full scenarios
- OOM conditions

**Tools Used:**
- **Toxiproxy** - Network failure injection
- **Docker** - Container orchestration and process control
- **Python** - Chaos scenario orchestration

---

## Quick Start

```bash
# Run all chaos tests
./run-chaos-tests.sh

# Results will be saved to ./results/
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Toxiproxy Layer                      │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐     │
│  │ Queen    │      │ Hive     │      │ Worker   │     │
│  │ Proxy    │─────▶│ Proxy    │─────▶│ Proxy    │     │
│  │ :8080    │      │ :9200    │      │ :xxxx    │     │
│  └──────────┘      └──────────┘      └──────────┘     │
└─────────────────────────────────────────────────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Queen-rbee   │   │ Rbee-hive    │   │ Mock Worker  │
│ :8081        │   │ :9201        │   │              │
└──────────────┘   └──────────────┘   └──────────────┘
```

**Toxiproxy** sits between services and injects failures:
- Latency
- Packet loss
- Connection resets
- Bandwidth limits
- Complete network partitions

---

## Chaos Scenarios

### Network Failures (`scenarios/network-failures.json`)

| ID | Scenario | Toxic | Duration | Expected Behavior |
|----|----------|-------|----------|-------------------|
| NF-001 | Complete partition | timeout | 30s | Retry with backoff |
| NF-002 | High latency | latency | 60s | Degraded performance |
| NF-003 | Packet loss | bandwidth | 45s | TCP retransmit |
| NF-004 | Slow network | bandwidth | 60s | Throttled responses |
| NF-005 | Connection reset | reset_peer | 30s | Client retry |

### Worker Crashes (`scenarios/worker-crashes.json`)

| ID | Scenario | Signal | Timing | Expected Behavior |
|----|----------|--------|--------|-------------------|
| WC-001 | Crash during inference | SIGKILL | During request | Retry on another worker |
| WC-002 | Crash during registration | SIGKILL | During handshake | Timeout and mark failed |
| WC-003 | OOM kill | SIGKILL | Random | Hive removes worker |
| WC-004 | Graceful shutdown timeout | SIGTERM | 5s timeout | Force-kill after timeout |
| WC-005 | Multiple crashes | SIGKILL | 50% of workers | Continue with remaining |

### Resource Exhaustion (`scenarios/resource-exhaustion.json`)

| ID | Scenario | Resource | Target | Expected Behavior |
|----|----------|----------|--------|-------------------|
| RE-001 | Disk full | Disk | Hive | 507 Insufficient Storage |
| RE-002 | Memory exhaustion | Memory | Worker | OOM kill, remove worker |
| RE-003 | CPU saturation | CPU | Queen | Queue requests |
| RE-004 | FD exhaustion | File descriptors | Hive | Reject connections |
| RE-005 | VRAM exhaustion | VRAM | Worker | 507 error |

---

## Running Tests

### Prerequisites

```bash
# Ensure Docker and Docker Compose are installed
docker --version
docker-compose --version

# Ensure integration services are built
cd test-harness/bdd
docker-compose -f docker-compose.integration.yml build
```

### Run All Chaos Tests

```bash
cd test-harness/chaos
./run-chaos-tests.sh
```

This will:
1. Start Docker Compose infrastructure
2. Wait for services to be ready
3. Configure toxiproxy proxies
4. Run all chaos scenarios
5. Collect results
6. Stop infrastructure

### Run Individual Scenario Types

```bash
# Start infrastructure
docker-compose -f docker-compose.chaos.yml up -d

# Run only network failure scenarios
docker-compose exec chaos-controller python3 /scripts/chaos_controller.py --scenarios network

# Run only worker crash scenarios
docker-compose exec chaos-controller python3 /scripts/chaos_controller.py --scenarios crashes

# Stop infrastructure
docker-compose -f docker-compose.chaos.yml down
```

---

## Results

Results are saved to `./results/`:

```
results/
├── chaos_test_20251018_140530.log          # Full test log
├── chaos_results_20251018_140530.json      # Structured results
└── chaos_summary_20251018_140530.md        # Human-readable summary
```

### Result Format

```json
{
  "test_run": "20251018_140530",
  "total_scenarios": 15,
  "completed": 14,
  "failed": 1,
  "scenarios": [
    {
      "scenario_id": "NF-001",
      "name": "Complete network partition",
      "type": "network_failure",
      "status": "completed",
      "start_time": "2025-10-18T14:05:30Z",
      "end_time": "2025-10-18T14:06:00Z",
      "error": null
    }
  ]
}
```

---

## Acceptance Criteria

From TEAM-107 plan:

- ✅ System survives chaos scenarios
- ✅ Random worker crashes handled
- ✅ Random network failures recovered
- ✅ Random disk full handled
- ✅ Random OOM handled
- ✅ System recovers automatically

**Success Threshold:** 90%+ of scenarios should complete successfully

---

## Troubleshooting

### Services Not Starting

```bash
# Check Docker logs
docker-compose -f docker-compose.chaos.yml logs

# Check specific service
docker-compose -f docker-compose.chaos.yml logs queen-rbee
```

### Toxiproxy Not Responding

```bash
# Check toxiproxy health
curl http://localhost:8474/version

# List configured proxies
curl http://localhost:8474/proxies

# Reset all toxics
curl -X POST http://localhost:8474/reset
```

### Tests Timing Out

Increase timeouts in `docker-compose.chaos.yml`:
```yaml
environment:
  - CHAOS_TIMEOUT=120  # Increase from default 60s
```

---

## Adding New Scenarios

1. Add scenario to appropriate JSON file in `scenarios/`
2. Update `chaos_controller.py` if new scenario type
3. Document expected behavior
4. Update this README

Example:
```json
{
  "id": "NF-006",
  "name": "Intermittent connectivity",
  "description": "Randomly drop connections",
  "toxic": "timeout",
  "attributes": {
    "timeout": 1000
  },
  "duration_seconds": 60,
  "expected_behavior": "Clients should retry and eventually succeed"
}
```

---

## Integration with CI/CD

```yaml
# .github/workflows/chaos-tests.yml
name: Chaos Tests
on: [push]
jobs:
  chaos:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run chaos tests
        run: |
          cd test-harness/chaos
          ./run-chaos-tests.sh
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: chaos-results
          path: test-harness/chaos/results/
```

---

## References

- [Toxiproxy Documentation](https://github.com/Shopify/toxiproxy)
- [Chaos Engineering Principles](https://principlesofchaos.org/)
- [TEAM-106 Integration Testing](../../.docs/components/PLAN/TEAM_106_INTEGRATION_TESTING.md)
- [TEAM-107 Plan](../../.docs/components/PLAN/TEAM_107_CHAOS_LOAD_TESTING.md)

---

**Created by:** TEAM-107 | 2025-10-18  
**Status:** ✅ Complete and ready to use
