# Production Deployment Guide

**Created by:** TEAM-116  
**Date:** 2025-10-19  
**For:** v0.1.0 Production Deployment

---

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Secret Management](#secret-management)
5. [Monitoring Setup](#monitoring-setup)
6. [Backup Procedures](#backup-procedures)
7. [Upgrade Procedures](#upgrade-procedures)
8. [Health Checks](#health-checks)

---

## 🖥️ System Requirements

### Minimum Requirements

**queen-rbee (Orchestrator):**
- CPU: 2 cores
- RAM: 2GB
- Disk: 10GB
- OS: Linux (Ubuntu 20.04+ or RHEL 8+)
- Network: Static IP or DNS name

**rbee-hive (Pool Manager):**
- CPU: 4 cores
- RAM: 8GB (16GB recommended)
- Disk: 100GB+ (for model storage)
- OS: Linux (Ubuntu 20.04+ or RHEL 8+)
- GPU: Optional (CUDA 11.8+ or Metal for macOS)

**llm-worker-rbee (Inference Worker):**
- CPU: 4+ cores
- RAM: 8GB minimum (depends on model size)
- VRAM: 4GB+ (for GPU inference)
- Disk: 20GB

### Recommended Production Setup

- **queen-rbee**: 1 instance (HA setup with 2+ instances recommended)
- **rbee-hive**: 1+ per GPU node
- **llm-worker-rbee**: Spawned dynamically by rbee-hive

---

## 📦 Installation

### 1. Download Binaries

```bash
# Download latest release
wget https://github.com/your-org/llama-orch/releases/download/v0.1.0/llama-orch-v0.1.0-linux-x86_64.tar.gz

# Extract
tar -xzf llama-orch-v0.1.0-linux-x86_64.tar.gz
cd llama-orch-v0.1.0

# Install binaries
sudo cp bin/* /usr/local/bin/
sudo chmod +x /usr/local/bin/{queen-rbee,rbee-hive,llm-worker-rbee}
```

### 2. Create System User

```bash
# Create dedicated user
sudo useradd -r -s /bin/false -d /var/lib/llama-orch llama-orch

# Create directories
sudo mkdir -p /var/lib/llama-orch/{models,data}
sudo mkdir -p /var/log/llama-orch/audit
sudo mkdir -p /etc/llama-orch

# Set permissions
sudo chown -R llama-orch:llama-orch /var/lib/llama-orch
sudo chown -R llama-orch:llama-orch /var/log/llama-orch
sudo chown -R llama-orch:llama-orch /etc/llama-orch
```

### 3. Install systemd Services

**queen-rbee.service:**
```ini
[Unit]
Description=rbee Orchestrator Daemon
After=network.target

[Service]
Type=simple
User=llama-orch
Group=llama-orch
ExecStart=/usr/local/bin/queen-rbee --port 8080 --database /var/lib/llama-orch/data/beehive.db
Restart=on-failure
RestartSec=10s

# Environment
Environment="LLORCH_API_TOKEN_FILE=/etc/llama-orch/secrets/api-token"
Environment="LLORCH_AUDIT_MODE=local"
Environment="LLORCH_AUDIT_DIR=/var/log/llama-orch/audit"

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/llama-orch /var/log/llama-orch

[Install]
WantedBy=multi-user.target
```

**rbee-hive.service:**
```ini
[Unit]
Description=rbee Pool Manager Daemon
After=network.target

[Service]
Type=simple
User=llama-orch
Group=llama-orch
ExecStart=/usr/local/bin/rbee-hive daemon 0.0.0.0:8081
Restart=on-failure
RestartSec=10s

# Environment
Environment="RBEE_MODEL_BASE_DIR=/var/lib/llama-orch/models"
Environment="LLORCH_API_TOKEN_FILE=/etc/llama-orch/secrets/api-token"
Environment="LLORCH_AUDIT_MODE=local"
Environment="LLORCH_AUDIT_DIR=/var/log/llama-orch/audit"

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/llama-orch /var/log/llama-orch

[Install]
WantedBy=multi-user.target
```

### 4. Enable and Start Services

```bash
# Copy service files
sudo cp systemd/*.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable services
sudo systemctl enable queen-rbee
sudo systemctl enable rbee-hive

# Start services
sudo systemctl start queen-rbee
sudo systemctl start rbee-hive

# Check status
sudo systemctl status queen-rbee
sudo systemctl status rbee-hive
```

---

## ⚙️ Configuration

See [CONFIGURATION.md](CONFIGURATION.md) for detailed configuration options.

### Quick Start Configuration

**Minimal queen-rbee config** (`/etc/llama-orch/queen-rbee.toml`):
```toml
[server]
port = 8080
bind_address = "0.0.0.0"

[database]
path = "/var/lib/llama-orch/data/beehive.db"

[audit]
mode = "local"
base_dir = "/var/log/llama-orch/audit"
```

**Minimal rbee-hive config** (`/etc/llama-orch/rbee-hive.toml`):
```toml
[server]
port = 8081
bind_address = "0.0.0.0"

[models]
base_dir = "/var/lib/llama-orch/models"
catalog_db = "/var/lib/llama-orch/data/models.db"

[resources]
max_worker_memory_gb = 8
min_free_memory_gb = 2
min_free_disk_gb = 10

[audit]
mode = "local"
base_dir = "/var/log/llama-orch/audit"
```

---

## 🔐 Secret Management

See [SECRETS.md](SECRETS.md) for detailed secret management.

### Quick Setup

```bash
# Generate API token
openssl rand -hex 32 > /etc/llama-orch/secrets/api-token

# Set permissions (CRITICAL!)
sudo chmod 600 /etc/llama-orch/secrets/api-token
sudo chown llama-orch:llama-orch /etc/llama-orch/secrets/api-token

# Verify permissions
ls -la /etc/llama-orch/secrets/api-token
# Should show: -rw------- 1 llama-orch llama-orch
```

---

## 📊 Monitoring Setup

See [MONITORING.md](MONITORING.md) for detailed monitoring setup.

### Quick Prometheus Setup

```yaml
# /etc/prometheus/prometheus.yml
scrape_configs:
  - job_name: 'queen-rbee'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 15s

  - job_name: 'rbee-hive'
    static_configs:
      - targets: ['localhost:8081']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Key Metrics to Monitor

- `rbee_hive_workers_total{state}` - Worker count by state
- `rbee_hive_memory_available_bytes` - Available memory
- `rbee_hive_shutdown_duration_seconds` - Shutdown duration
- `rbee_hive_workers_force_killed_total` - Force-killed workers

---

## 💾 Backup Procedures

### What to Backup

1. **SQLite Databases:**
   - `/var/lib/llama-orch/data/beehive.db` (queen-rbee registry)
   - `/var/lib/llama-orch/data/models.db` (model catalog)

2. **Configuration:**
   - `/etc/llama-orch/*.toml`
   - `/etc/llama-orch/secrets/*`

3. **Audit Logs:**
   - `/var/log/llama-orch/audit/*`

### Backup Script

```bash
#!/bin/bash
# /usr/local/bin/backup-llama-orch.sh

BACKUP_DIR="/var/backups/llama-orch"
DATE=$(date +%Y%m%d-%H%M%S)

mkdir -p "$BACKUP_DIR"

# Backup databases
sqlite3 /var/lib/llama-orch/data/beehive.db ".backup $BACKUP_DIR/beehive-$DATE.db"
sqlite3 /var/lib/llama-orch/data/models.db ".backup $BACKUP_DIR/models-$DATE.db"

# Backup config
tar -czf "$BACKUP_DIR/config-$DATE.tar.gz" /etc/llama-orch

# Backup audit logs (last 7 days)
find /var/log/llama-orch/audit -type f -mtime -7 | \
  tar -czf "$BACKUP_DIR/audit-$DATE.tar.gz" -T -

# Cleanup old backups (keep 30 days)
find "$BACKUP_DIR" -type f -mtime +30 -delete

echo "Backup completed: $DATE"
```

### Automated Backups

```bash
# Add to crontab
sudo crontab -e -u llama-orch

# Run daily at 2 AM
0 2 * * * /usr/local/bin/backup-llama-orch.sh >> /var/log/llama-orch/backup.log 2>&1
```

---

## 🔄 Upgrade Procedures

### Pre-Upgrade Checklist

- [ ] Backup all data (see Backup Procedures)
- [ ] Review release notes for breaking changes
- [ ] Schedule maintenance window
- [ ] Notify users of downtime

### Upgrade Steps

```bash
# 1. Stop services
sudo systemctl stop rbee-hive
sudo systemctl stop queen-rbee

# 2. Backup current binaries
sudo cp /usr/local/bin/queen-rbee /usr/local/bin/queen-rbee.backup
sudo cp /usr/local/bin/rbee-hive /usr/local/bin/rbee-hive.backup

# 3. Download new version
wget https://github.com/your-org/llama-orch/releases/download/v0.2.0/llama-orch-v0.2.0-linux-x86_64.tar.gz
tar -xzf llama-orch-v0.2.0-linux-x86_64.tar.gz

# 4. Install new binaries
sudo cp llama-orch-v0.2.0/bin/* /usr/local/bin/
sudo chmod +x /usr/local/bin/{queen-rbee,rbee-hive,llm-worker-rbee}

# 5. Run database migrations (if any)
# Check release notes for migration scripts

# 6. Start services
sudo systemctl start queen-rbee
sudo systemctl start rbee-hive

# 7. Verify services
sudo systemctl status queen-rbee
sudo systemctl status rbee-hive

# 8. Check logs
sudo journalctl -u queen-rbee -f
sudo journalctl -u rbee-hive -f
```

### Rollback Procedure

```bash
# If upgrade fails, rollback to previous version
sudo systemctl stop rbee-hive
sudo systemctl stop queen-rbee

# Restore old binaries
sudo cp /usr/local/bin/queen-rbee.backup /usr/local/bin/queen-rbee
sudo cp /usr/local/bin/rbee-hive.backup /usr/local/bin/rbee-hive

# Restore database from backup (if needed)
sudo cp /var/backups/llama-orch/beehive-YYYYMMDD-HHMMSS.db /var/lib/llama-orch/data/beehive.db
sudo cp /var/backups/llama-orch/models-YYYYMMDD-HHMMSS.db /var/lib/llama-orch/data/models.db

# Start services
sudo systemctl start queen-rbee
sudo systemctl start rbee-hive
```

---

## 🏥 Health Checks

### Service Health

```bash
# Check if services are running
systemctl is-active queen-rbee
systemctl is-active rbee-hive

# Check HTTP endpoints
curl http://localhost:8080/health
curl http://localhost:8081/health
```

### Expected Health Response

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "uptime_seconds": 3600
}
```

### Automated Health Monitoring

```bash
#!/bin/bash
# /usr/local/bin/health-check-llama-orch.sh

# Check queen-rbee
if ! curl -f -s http://localhost:8080/health > /dev/null; then
    echo "ERROR: queen-rbee health check failed"
    exit 1
fi

# Check rbee-hive
if ! curl -f -s http://localhost:8081/health > /dev/null; then
    echo "ERROR: rbee-hive health check failed"
    exit 1
fi

echo "OK: All services healthy"
exit 0
```

---

## 🚨 Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions.

### Quick Diagnostics

```bash
# Check service status
sudo systemctl status queen-rbee rbee-hive

# View recent logs
sudo journalctl -u queen-rbee -n 100
sudo journalctl -u rbee-hive -n 100

# Check disk space
df -h /var/lib/llama-orch

# Check memory usage
free -h

# Check open files
sudo lsof -u llama-orch

# Check network connectivity
ss -tlnp | grep -E '8080|8081'
```

---

## 📞 Support

- **Documentation:** https://docs.llama-orch.dev
- **Issues:** https://github.com/your-org/llama-orch/issues
- **Community:** https://discord.gg/llama-orch

---

**Next Steps:**
- [Configuration Guide](CONFIGURATION.md)
- [Secret Management](SECRETS.md)
- [Monitoring Setup](MONITORING.md)
- [Troubleshooting](TROUBLESHOOTING.md)
- [API Documentation](API.md)
