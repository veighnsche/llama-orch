# Monitoring Setup Guide

**Created by:** TEAM-116  
**Date:** 2025-10-19  
**For:** v0.1.0 Production Deployment

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prometheus Setup](#prometheus-setup)
3. [Grafana Setup](#grafana-setup)
4. [Alerting Rules](#alerting-rules)
5. [Log Aggregation](#log-aggregation)
6. [Audit Log Retention](#audit-log-retention)
7. [Key Metrics](#key-metrics)

---

## 📊 Overview

llama-orch exposes Prometheus metrics on `/metrics` endpoints for comprehensive monitoring.

### Monitoring Stack

- **Metrics**: Prometheus + Grafana
- **Logs**: journald or ELK stack
- **Audit**: File-based NDJSON logs
- **Alerts**: Alertmanager

---

## 🔧 Prometheus Setup

### 1. Install Prometheus

```bash
# Ubuntu/Debian
sudo apt-get install prometheus

# Or download from prometheus.io
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
tar -xzf prometheus-2.45.0.linux-amd64.tar.gz
sudo mv prometheus-2.45.0.linux-amd64 /opt/prometheus
```

### 2. Configure Prometheus

**`/etc/prometheus/prometheus.yml`:**
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'llama-orch-prod'
    environment: 'production'

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - localhost:9093

# Load alerting rules
rule_files:
  - '/etc/prometheus/rules/*.yml'

# Scrape configurations
scrape_configs:
  # queen-rbee (Orchestrator)
  - job_name: 'queen-rbee'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 15s
    scrape_timeout: 10s

  # rbee-hive (Pool Managers)
  - job_name: 'rbee-hive'
    static_configs:
      - targets:
          - 'hive1.example.com:8081'
          - 'hive2.example.com:8081'
          - 'hive3.example.com:8081'
    metrics_path: '/metrics'
    scrape_interval: 15s
    scrape_timeout: 10s

  # Prometheus self-monitoring
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

### 3. Start Prometheus

```bash
# systemd service
sudo systemctl enable prometheus
sudo systemctl start prometheus

# Verify
curl http://localhost:9090/-/healthy
```

---

## 📈 Grafana Setup

### 1. Install Grafana

```bash
# Ubuntu/Debian
sudo apt-get install -y software-properties-common
sudo add-apt-repository "deb https://packages.grafana.com/oss/deb stable main"
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
sudo apt-get update
sudo apt-get install grafana
```

### 2. Configure Data Source

```bash
# Start Grafana
sudo systemctl enable grafana-server
sudo systemctl start grafana-server

# Access Grafana at http://localhost:3000
# Default credentials: admin/admin
```

**Add Prometheus Data Source:**
1. Navigate to Configuration → Data Sources
2. Add Prometheus
3. URL: `http://localhost:9090`
4. Save & Test

### 3. Import Dashboard

**Dashboard JSON** (`/etc/grafana/dashboards/llama-orch.json`):
```json
{
  "dashboard": {
    "title": "llama-orch Production Dashboard",
    "panels": [
      {
        "title": "Worker Count by State",
        "targets": [
          {
            "expr": "sum by (state) (rbee_hive_workers_total)"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Memory Usage",
        "targets": [
          {
            "expr": "rbee_hive_memory_available_bytes / rbee_hive_memory_total_bytes * 100"
          }
        ],
        "type": "gauge"
      },
      {
        "title": "Shutdown Duration",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(rbee_hive_shutdown_duration_seconds_bucket[5m]))"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Force-Killed Workers",
        "targets": [
          {
            "expr": "rate(rbee_hive_workers_force_killed_total[5m])"
          }
        ],
        "type": "graph"
      }
    ]
  }
}
```

---

## 🚨 Alerting Rules

### 1. Create Alert Rules

**`/etc/prometheus/rules/llama-orch.yml`:**
```yaml
groups:
  - name: llama-orch-alerts
    interval: 30s
    rules:
      # Worker health alerts
      - alert: HighFailedHealthChecks
        expr: rbee_hive_workers_failed_health_checks > 3
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High number of failed health checks"
          description: "{{ $value }} workers have failed health checks"

      - alert: NoIdleWorkers
        expr: sum(rbee_hive_workers_total{state="idle"}) == 0
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "No idle workers available"
          description: "All workers are busy or loading"

      # Resource alerts
      - alert: LowMemory
        expr: rbee_hive_memory_available_bytes / rbee_hive_memory_total_bytes < 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Low memory available"
          description: "Only {{ $value | humanizePercentage }} memory available"

      - alert: LowDiskSpace
        expr: rbee_hive_disk_available_bytes / rbee_hive_disk_total_bytes < 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Low disk space"
          description: "Only {{ $value | humanizePercentage }} disk space available"

      # Shutdown alerts
      - alert: HighForceKillRate
        expr: rate(rbee_hive_workers_force_killed_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High rate of force-killed workers"
          description: "{{ $value }} workers/sec being force-killed"

      - alert: SlowShutdown
        expr: histogram_quantile(0.95, rate(rbee_hive_shutdown_duration_seconds_bucket[5m])) > 25
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow graceful shutdown"
          description: "95th percentile shutdown time is {{ $value }}s"

      # Service health
      - alert: ServiceDown
        expr: up{job=~"queen-rbee|rbee-hive"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.job }} is down"
          description: "{{ $labels.instance }} has been down for 1 minute"

      # Restart alerts
      - alert: HighRestartRate
        expr: rate(rbee_hive_worker_restart_failures_total[5m]) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High worker restart failure rate"
          description: "{{ $value }} restart failures/sec"

      - alert: CircuitBreakerActive
        expr: increase(rbee_hive_circuit_breaker_activations_total[5m]) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Circuit breaker activated"
          description: "Worker restart circuit breaker has been activated"
```

### 2. Configure Alertmanager

**`/etc/alertmanager/alertmanager.yml`:**
```yaml
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname', 'cluster']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'team-notifications'

receivers:
  - name: 'team-notifications'
    email_configs:
      - to: 'ops-team@example.com'
        from: 'alertmanager@example.com'
        smarthost: 'smtp.example.com:587'
        auth_username: 'alertmanager@example.com'
        auth_password: 'password'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
        channel: '#llama-orch-alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'cluster']
```

---

## 📝 Log Aggregation

### journald (Simple Setup)

```bash
# View logs
sudo journalctl -u queen-rbee -f
sudo journalctl -u rbee-hive -f

# Filter by time
sudo journalctl -u queen-rbee --since "1 hour ago"

# Export logs
sudo journalctl -u queen-rbee -o json > queen-rbee.json
```

### ELK Stack (Advanced Setup)

**Filebeat configuration** (`/etc/filebeat/filebeat.yml`):
```yaml
filebeat.inputs:
  - type: journald
    id: queen-rbee
    include_matches:
      - _SYSTEMD_UNIT=queen-rbee.service

  - type: journald
    id: rbee-hive
    include_matches:
      - _SYSTEMD_UNIT=rbee-hive.service

  - type: log
    enabled: true
    paths:
      - /var/log/llama-orch/audit/*.ndjson
    json.keys_under_root: true
    json.add_error_key: true

output.elasticsearch:
  hosts: ["localhost:9200"]
  index: "llama-orch-%{+yyyy.MM.dd}"

setup.kibana:
  host: "localhost:5601"
```

---

## 🗄️ Audit Log Retention

### Retention Policy

```bash
# Default retention: 90 days
# Configured in service config:
# retention_policy = { days = 90 }
```

### Manual Cleanup

```bash
# Remove audit logs older than 90 days
find /var/log/llama-orch/audit -name "*.ndjson" -mtime +90 -delete

# Compress old logs
find /var/log/llama-orch/audit -name "*.ndjson" -mtime +30 -exec gzip {} \;
```

### Automated Cleanup

```bash
#!/bin/bash
# /usr/local/bin/cleanup-audit-logs.sh

AUDIT_DIR="/var/log/llama-orch/audit"
RETENTION_DAYS=90
COMPRESS_DAYS=30

# Compress logs older than 30 days
find "$AUDIT_DIR" -name "*.ndjson" -mtime +$COMPRESS_DAYS -exec gzip {} \;

# Delete logs older than 90 days
find "$AUDIT_DIR" -name "*.ndjson.gz" -mtime +$RETENTION_DAYS -delete

echo "Audit log cleanup completed"
```

**Add to crontab:**
```bash
# Run daily at 3 AM
0 3 * * * /usr/local/bin/cleanup-audit-logs.sh
```

---

## 📊 Key Metrics

### Worker Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `rbee_hive_workers_total{state,backend,device}` | Gauge | Total workers by state |
| `rbee_hive_workers_failed_health_checks` | Gauge | Workers with failed health checks |
| `rbee_hive_workers_restart_count` | Gauge | Total restart count |
| `rbee_hive_worker_restart_failures_total` | Counter | Total restart failures |
| `rbee_hive_circuit_breaker_activations_total` | Counter | Circuit breaker activations |

### Resource Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `rbee_hive_memory_total_bytes` | Gauge | Total system memory |
| `rbee_hive_memory_available_bytes` | Gauge | Available memory |
| `rbee_hive_disk_total_bytes` | Gauge | Total disk space |
| `rbee_hive_disk_available_bytes` | Gauge | Available disk space |
| `rbee_hive_worker_spawn_resource_failures_total` | Counter | Resource limit failures |

### Shutdown Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `rbee_hive_shutdown_duration_seconds` | Histogram | Shutdown duration |
| `rbee_hive_workers_graceful_shutdown_total` | Counter | Graceful shutdowns |
| `rbee_hive_workers_force_killed_total` | Counter | Force-killed workers |

### Model Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `rbee_hive_models_downloaded_total` | Counter | Total models downloaded |
| `rbee_hive_download_active` | Gauge | Active downloads |

---

## 🔍 Useful Queries

### PromQL Examples

```promql
# Worker utilization
sum(rbee_hive_workers_total{state="busy"}) / sum(rbee_hive_workers_total) * 100

# Memory usage percentage
(1 - rbee_hive_memory_available_bytes / rbee_hive_memory_total_bytes) * 100

# Disk usage percentage
(1 - rbee_hive_disk_available_bytes / rbee_hive_disk_total_bytes) * 100

# Average shutdown duration (95th percentile)
histogram_quantile(0.95, rate(rbee_hive_shutdown_duration_seconds_bucket[5m]))

# Force-kill rate
rate(rbee_hive_workers_force_killed_total[5m])

# Worker restart failure rate
rate(rbee_hive_worker_restart_failures_total[5m])
```

---

## 🔗 Related Documentation

- [Deployment Guide](DEPLOYMENT.md)
- [Configuration Guide](CONFIGURATION.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
- [API Documentation](API.md)
