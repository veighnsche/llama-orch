# Load Testing Suite

**Created by:** TEAM-107 | 2025-10-18  
**Purpose:** Validate system performance under load

---

## Overview

This load testing suite validates that the rbee system can handle:
- 1000+ concurrent requests
- Sustained load for extended periods
- Sudden traffic spikes
- Gradual load increases to breaking point

**Tools Used:**
- **k6** - Modern load testing tool
- **Grafana k6 Cloud** (optional) - Results visualization

---

## Quick Start

```bash
# Install k6 (if not already installed)
# See: https://k6.io/docs/getting-started/installation/

# Run all load tests
./run-load-tests.sh

# Results will be saved to ./results/
```

---

## Test Scenarios

### 1. Inference Load Test (`inference-load.js`)

**Goal:** Validate system handles 1000+ concurrent users with acceptable latency

**Configuration:**
- Ramp up: 0 → 1000 users over 5 minutes
- Sustained: 1000 users for 10 minutes
- Ramp down: 1000 → 0 over 1 minute
- **Total duration:** ~16 minutes

**Thresholds:**
- ✅ Error rate < 1%
- ✅ p95 latency < 500ms
- ✅ p99 latency < 1000ms
- ✅ Success rate > 99%

**Load Pattern:**
```
Users
1000 │         ┌──────────────────┐
     │        ╱                    ╲
 500 │      ╱                        ╲
     │    ╱                            ╲
   0 └──┴────────────────────────────────┴──▶ Time
      0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 min
```

### 2. Stress Test (`stress-test.js`)

**Goal:** Find the system's breaking point

**Configuration:**
- Gradual ramp: 100 → 5000 users over 12 minutes
- Hold at peak: 5000 users for 5 minutes
- Ramp down: 5000 → 0 over 2 minutes
- **Total duration:** ~19 minutes

**Thresholds:**
- ⚠️ Error rate < 5% (relaxed for stress testing)
- ⚠️ p95 latency < 2000ms (relaxed)

**Load Pattern:**
```
Users
5000 │                   ┌────────┐
     │                 ╱            ╲
3000 │              ╱                 ╲
     │           ╱                      ╲
1000 │        ╱                           ╲
     │     ╱                                ╲
   0 └──┴────────────────────────────────────┴──▶ Time
      0  2  4  6  8 10 12 14 16 18 20 min
```

### 3. Spike Test (`spike-test.js`)

**Goal:** Test recovery from sudden traffic spikes

**Configuration:**
- Normal load: 100 users
- Spike 1: 100 → 2000 users in 10 seconds
- Hold: 2000 users for 1 minute
- Drop: 2000 → 100 users in 10 seconds
- Spike 2: 100 → 3000 users in 10 seconds
- Hold: 3000 users for 1 minute
- **Total duration:** ~7 minutes

**Thresholds:**
- ✅ Error rate < 2%
- ✅ p95 latency < 1000ms

**Load Pattern:**
```
Users
3000 │                           ┌─┐
     │                           │ │
2000 │         ┌─┐               │ │
     │         │ │               │ │
 100 │────┬────┘ └────┬──────────┘ └────┬──▶ Time
   0 └────────────────────────────────────
      0   1   2   3   4   5   6   7  min
```

---

## Running Tests

### Prerequisites

```bash
# Install k6
# Ubuntu/Debian:
sudo gpg -k
sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg \
  --keyserver hkp://keyserver.ubuntu.com:80 \
  --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | \
  sudo tee /etc/apt/sources.list.d/k6.list
sudo apt-get update
sudo apt-get install k6

# macOS:
brew install k6

# Verify installation
k6 version
```

### Run All Tests

```bash
cd test-harness/load
./run-load-tests.sh
```

### Run Individual Tests

```bash
# Inference load test
k6 run -e QUEEN_URL=http://localhost:8080 inference-load.js

# Stress test
k6 run -e QUEEN_URL=http://localhost:8080 stress-test.js

# Spike test
k6 run -e QUEEN_URL=http://localhost:8080 spike-test.js
```

### Custom Configuration

```bash
# Override target URL
export QUEEN_URL=http://production.example.com:8080
./run-load-tests.sh

# Run with k6 Cloud (requires account)
k6 cloud inference-load.js

# Run with custom VUs
k6 run --vus 500 --duration 5m inference-load.js
```

---

## Results

Results are saved to `./results/`:

```
results/
├── load_test_20251018_150000.log           # Full test log
├── load_test_summary_20251018_150000.md    # Summary report
├── inference_load_20251018_150000.json     # Raw k6 data
├── inference_summary_20251018_150000.json  # Metrics summary
├── stress_test_20251018_150000.json        # Stress test data
├── stress_summary_20251018_150000.json     # Stress metrics
├── spike_test_20251018_150000.json         # Spike test data
└── spike_summary_20251018_150000.json      # Spike metrics
```

### Interpreting Results

**Summary JSON format:**
```json
{
  "metrics": {
    "http_req_duration": {
      "avg": 245.3,
      "min": 12.1,
      "med": 198.7,
      "max": 1234.5,
      "p(90)": 387.2,
      "p(95)": 456.8,
      "p(99)": 892.1
    },
    "http_req_failed": {
      "rate": 0.0023,
      "passes": 49885,
      "fails": 115
    },
    "iterations": {
      "count": 50000,
      "rate": 55.2
    }
  },
  "root_group": {
    "checks": {
      "status is 200": {
        "passes": 49885,
        "fails": 115
      }
    }
  }
}
```

**Key Metrics:**
- `http_req_duration.p(95)` - 95th percentile latency (target: < 500ms)
- `http_req_failed.rate` - Error rate (target: < 0.01 = 1%)
- `iterations.rate` - Throughput (requests/second)
- `checks.passes` - Successful assertions

---

## Acceptance Criteria

From TEAM-107 plan:

- ✅ 1000+ concurrent requests handled
- ✅ Sustained load for 1 hour (10 min in quick test)
- ✅ p95 latency < 500ms
- ✅ p99 latency < 1000ms
- ✅ Error rate < 1%
- ✅ System recovers from spikes

**Success Threshold:** All thresholds must pass

---

## Performance Baselines

Expected performance on reference hardware:

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| p50 latency | < 200ms | < 150ms | < 100ms |
| p95 latency | < 500ms | < 300ms | < 200ms |
| p99 latency | < 1000ms | < 600ms | < 400ms |
| Error rate | < 1% | < 0.1% | < 0.01% |
| Throughput | > 50 req/s | > 100 req/s | > 200 req/s |

---

## Troubleshooting

### High Error Rates

```bash
# Check service logs
docker-compose logs queen-rbee
docker-compose logs rbee-hive

# Check resource usage
docker stats

# Reduce load
k6 run --vus 100 --duration 1m inference-load.js
```

### High Latency

Possible causes:
- CPU saturation (check `docker stats`)
- Memory pressure (check available RAM)
- Network bottleneck (check network I/O)
- Database contention (check query performance)

### k6 Errors

```bash
# Increase timeout
k6 run --http-debug inference-load.js

# Enable verbose logging
k6 run --verbose inference-load.js

# Check k6 version
k6 version
```

---

## Advanced Usage

### Custom Scenarios

Create custom load patterns:

```javascript
export const options = {
  scenarios: {
    constant_load: {
      executor: 'constant-vus',
      vus: 500,
      duration: '10m',
    },
    ramping_load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 1000 },
        { duration: '5m', target: 1000 },
        { duration: '2m', target: 0 },
      ],
    },
  },
};
```

### Distributed Testing

Run k6 on multiple machines:

```bash
# Machine 1
k6 run --out json=results1.json inference-load.js

# Machine 2
k6 run --out json=results2.json inference-load.js

# Merge results
jq -s 'add' results1.json results2.json > combined.json
```

### Real-time Monitoring

Stream metrics to InfluxDB + Grafana:

```bash
k6 run --out influxdb=http://localhost:8086/k6 inference-load.js
```

---

## Integration with CI/CD

```yaml
# .github/workflows/load-tests.yml
name: Load Tests
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
jobs:
  load:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install k6
        run: |
          sudo gpg -k
          sudo gpg --no-default-keyring \
            --keyring /usr/share/keyrings/k6-archive-keyring.gpg \
            --keyserver hkp://keyserver.ubuntu.com:80 \
            --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
          echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | \
            sudo tee /etc/apt/sources.list.d/k6.list
          sudo apt-get update
          sudo apt-get install k6
      - name: Run load tests
        run: |
          cd test-harness/load
          ./run-load-tests.sh
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: load-results
          path: test-harness/load/results/
```

---

## References

- [k6 Documentation](https://k6.io/docs/)
- [k6 Test Types](https://k6.io/docs/test-types/introduction/)
- [k6 Metrics](https://k6.io/docs/using-k6/metrics/)
- [TEAM-107 Plan](../../.docs/components/PLAN/TEAM_107_CHAOS_LOAD_TESTING.md)

---

**Created by:** TEAM-107 | 2025-10-18  
**Status:** ✅ Complete and ready to use
