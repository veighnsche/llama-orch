# metrics-contract

**Contract tests for Prometheus metrics compliance**

`test-harness/metrics-contract` — Tests that verify metrics conform to naming conventions and labels.

---

## What This Test Suite Does

metrics-contract provides **metrics compliance testing** for llama-orch:

- **Naming conventions** — Verify metric names follow Prometheus standards
- **Label validation** — Required labels present and valid
- **Type checking** — Counter, gauge, histogram types correct
- **Help text** — Descriptive help text present
- **Cardinality** — Label cardinality within bounds

**Purpose**: Ensure metrics are consistent, discoverable, and Prometheus-compliant

---

## Metrics Contract

### Naming Conventions

- **Prefix** — All metrics start with `llorch_`
- **Subsystem** — Second component (e.g., `llorch_queue_`, `llorch_pool_`)
- **Suffix** — Type suffix (`_total`, `_seconds`, `_bytes`)
- **Snake case** — Lowercase with underscores

Examples:
- ✅ `llorch_queue_depth`
- ✅ `llorch_pool_replicas_total`
- ✅ `llorch_request_duration_seconds`
- ❌ `queue_depth` (missing prefix)
- ❌ `llorch_queueDepth` (camelCase)

### Required Labels

All metrics must include:
- **service** — Service name (orchestratord, pool-managerd)
- **version** — Service version

Additional labels by subsystem:
- **Queue metrics**: `pool_id`
- **Pool metrics**: `pool_id`, `replica_id`
- **Request metrics**: `pool_id`, `status`

### Metric Types

- **Counter** — Monotonically increasing (suffix `_total`)
- **Gauge** — Can go up or down (no suffix)
- **Histogram** — Distribution (suffix `_seconds`, `_bytes`)

---

## Test Scenarios

### Naming Convention

```rust
#[test]
fn test_metric_naming_conventions() {
    let metrics = collect_metrics().await?;
    
    for metric in metrics {
        // Must start with llorch_
        assert!(metric.name.starts_with("llorch_"), 
            "Metric {} missing llorch_ prefix", metric.name);
        
        // Must be snake_case
        assert!(is_snake_case(&metric.name),
            "Metric {} not snake_case", metric.name);
        
        // Counter must end with _total
        if metric.type_ == MetricType::Counter {
            assert!(metric.name.ends_with("_total"),
                "Counter {} missing _total suffix", metric.name);
        }
    }
}
```

### Required Labels

```rust
#[test]
fn test_required_labels() {
    let metrics = collect_metrics().await?;
    
    for metric in metrics {
        // All metrics must have service and version
        assert!(metric.labels.contains_key("service"),
            "Metric {} missing service label", metric.name);
        assert!(metric.labels.contains_key("version"),
            "Metric {} missing version label", metric.name);
        
        // Queue metrics must have pool_id
        if metric.name.starts_with("llorch_queue_") {
            assert!(metric.labels.contains_key("pool_id"),
                "Queue metric {} missing pool_id label", metric.name);
        }
    }
}
```

### Help Text

```rust
#[test]
fn test_help_text() {
    let metrics = collect_metrics().await?;
    
    for metric in metrics {
        // Must have help text
        assert!(!metric.help.is_empty(),
            "Metric {} missing help text", metric.name);
        
        // Help text must be descriptive (>10 chars)
        assert!(metric.help.len() > 10,
            "Metric {} help text too short: {}", metric.name, metric.help);
    }
}
```

---

## Running Tests

### All Contract Tests

```bash
# Run all tests
cargo test -p test-harness-metrics-contract -- --nocapture
```

### Specific Test

```bash
# Naming conventions
cargo test -p test-harness-metrics-contract -- test_metric_naming_conventions --nocapture

# Required labels
cargo test -p test-harness-metrics-contract -- test_required_labels --nocapture

# Help text
cargo test -p test-harness-metrics-contract -- test_help_text --nocapture
```

---

## Metrics Linter

The test suite includes a linter that checks:

```bash
# Run linter
cargo run -p test-harness-metrics-contract --bin metrics-lint

# Output
✅ llorch_queue_depth: OK
✅ llorch_pool_replicas_total: OK
❌ queue_depth: Missing llorch_ prefix
❌ llorch_queueDepth: Not snake_case
```

---

## Example Metrics

### Queue Depth (Gauge)

```
# HELP llorch_queue_depth Number of jobs in queue
# TYPE llorch_queue_depth gauge
llorch_queue_depth{service="orchestratord",version="0.0.0",pool_id="default"} 5
```

### Requests Total (Counter)

```
# HELP llorch_requests_total Total number of requests
# TYPE llorch_requests_total counter
llorch_requests_total{service="orchestratord",version="0.0.0",pool_id="default",status="success"} 42
```

### Request Duration (Histogram)

```
# HELP llorch_request_duration_seconds Request duration in seconds
# TYPE llorch_request_duration_seconds histogram
llorch_request_duration_seconds_bucket{service="orchestratord",version="0.0.0",pool_id="default",le="0.1"} 10
llorch_request_duration_seconds_bucket{service="orchestratord",version="0.0.0",pool_id="default",le="1.0"} 42
llorch_request_duration_seconds_sum{service="orchestratord",version="0.0.0",pool_id="default"} 123.45
llorch_request_duration_seconds_count{service="orchestratord",version="0.0.0",pool_id="default"} 42
```

---

## Testing

### Unit Tests

```bash
# Run all tests
cargo test -p test-harness-metrics-contract -- --nocapture
```

---

## Dependencies

### Internal

- None (contract testing)

### External

- `prometheus` — Metrics parsing
- `regex` — Pattern matching

---

## Specifications

Implements requirements from:
- ORCH-3050 (Metrics contract)
- ORCH-3051 (Prometheus compliance)

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha
- **Maintainers**: @llama-orch-maintainers
