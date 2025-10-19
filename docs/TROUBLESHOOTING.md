# Troubleshooting Guide

**Created by:** TEAM-116  
**Date:** 2025-10-19  
**For:** v0.1.0 Production Deployment

---

## 📋 Table of Contents

1. [Common Issues](#common-issues)
2. [Debug Logging](#debug-logging)
3. [Health Check Procedures](#health-check-procedures)
4. [Performance Tuning](#performance-tuning)
5. [Recovery Procedures](#recovery-procedures)
6. [Error Messages](#error-messages)

---

## 🔧 Common Issues

### Issue: Service Won't Start

**Symptoms:**
- `systemctl start queen-rbee` fails
- Service immediately exits

**Diagnosis:**
```bash
# Check service status
sudo systemctl status queen-rbee

# View logs
sudo journalctl -u queen-rbee -n 100

# Check for port conflicts
sudo ss -tlnp | grep 8080
```

**Solutions:**

1. **Port already in use:**
```bash
# Find process using port
sudo lsof -i :8080

# Kill process or change port in config
```

2. **Missing API token file:**
```bash
# Verify token file exists
ls -la /etc/llama-orch/secrets/api-token

# Create if missing (see SECRETS.md)
openssl rand -hex 32 | sudo tee /etc/llama-orch/secrets/api-token
sudo chmod 600 /etc/llama-orch/secrets/api-token
sudo chown llama-orch:llama-orch /etc/llama-orch/secrets/api-token
```

3. **Database locked:**
```bash
# Check for stale lock
sudo lsof /var/lib/llama-orch/data/beehive.db

# Remove if no process is using it
sudo rm /var/lib/llama-orch/data/beehive.db-wal
sudo rm /var/lib/llama-orch/data/beehive.db-shm
```

---

### Issue: Worker Spawn Failures

**Symptoms:**
- Workers fail to spawn
- "Insufficient memory" errors
- "Insufficient VRAM" errors

**Diagnosis:**
```bash
# Check memory
free -h

# Check disk space
df -h /var/lib/llama-orch

# Check GPU VRAM (if applicable)
nvidia-smi

# Check worker logs
sudo journalctl -u rbee-hive -n 100 | grep "spawn"
```

**Solutions:**

1. **Insufficient memory:**
```bash
# Check current limits
cat /etc/llama-orch/rbee-hive.toml | grep memory

# Adjust limits
sudo nano /etc/llama-orch/rbee-hive.toml
# Set: max_worker_memory_gb = 4  # Lower limit

# Restart service
sudo systemctl restart rbee-hive
```

2. **Insufficient disk space:**
```bash
# Clean up old models
sudo rbee-hive models list
sudo rbee-hive models remove <model-ref>

# Or expand disk
```

3. **VRAM exhausted:**
```bash
# Check GPU usage
nvidia-smi

# Kill hung workers
sudo pkill -f llm-worker-rbee

# Restart rbee-hive
sudo systemctl restart rbee-hive
```

---

### Issue: Workers Not Responding

**Symptoms:**
- Workers stuck in "loading" state
- Health checks failing
- Inference requests timeout

**Diagnosis:**
```bash
# Check worker processes
ps aux | grep llm-worker-rbee

# Check worker health
curl http://localhost:8082/health

# Check worker logs
sudo journalctl -u rbee-hive | grep "worker-"
```

**Solutions:**

1. **Worker hung during model load:**
```bash
# Force-kill worker
sudo pkill -9 -f "llm-worker-rbee.*worker-id"

# rbee-hive will detect and restart
```

2. **Model file corrupted:**
```bash
# Remove model
sudo rbee-hive models remove <model-ref>

# Re-download
sudo rbee-hive models download <model-ref>
```

3. **Network issues:**
```bash
# Check connectivity
ping -c 3 localhost

# Check firewall
sudo iptables -L -n | grep 8082
```

---

### Issue: High Memory Usage

**Symptoms:**
- System running out of memory
- OOM killer activating
- Workers being killed

**Diagnosis:**
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head -n 20

# Check OOM killer logs
sudo dmesg | grep -i "out of memory"

# Check worker memory limits
curl http://localhost:8081/metrics | grep memory
```

**Solutions:**

1. **Reduce worker memory limits:**
```bash
# Edit config
sudo nano /etc/llama-orch/rbee-hive.toml

[resources]
max_worker_memory_gb = 4  # Reduce from 8
min_free_memory_gb = 4    # Increase from 2

# Restart
sudo systemctl restart rbee-hive
```

2. **Reduce concurrent workers:**
```bash
# Kill excess workers
sudo rbee-hive workers list
sudo rbee-hive workers stop <worker-id>
```

3. **Add swap space:**
```bash
# Create 8GB swap file
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

### Issue: Slow Shutdown

**Symptoms:**
- Shutdown takes > 30 seconds
- Workers being force-killed
- High `workers_force_killed_total` metric

**Diagnosis:**
```bash
# Check shutdown metrics
curl http://localhost:8081/metrics | grep shutdown

# Check worker processes during shutdown
sudo systemctl stop rbee-hive &
sleep 5
ps aux | grep llm-worker-rbee
```

**Solutions:**

1. **Reduce graceful timeout:**
```bash
# Edit shutdown config (future feature)
# For now, workers should shutdown within 30s
```

2. **Fix hung workers:**
```bash
# Identify hung workers
ps aux | grep llm-worker-rbee

# Check what they're doing
sudo strace -p <pid>

# If stuck on I/O, check disk health
sudo smartctl -a /dev/sda
```

---

### Issue: Authentication Failures

**Symptoms:**
- "Unauthorized" errors
- "Invalid token" errors
- 401 HTTP responses

**Diagnosis:**
```bash
# Check token file exists
ls -la /etc/llama-orch/secrets/api-token

# Check token permissions
stat /etc/llama-orch/secrets/api-token

# Check audit logs
sudo grep "authentication_failed" /var/log/llama-orch/audit/*.ndjson
```

**Solutions:**

1. **Token file missing:**
```bash
# Generate new token
openssl rand -hex 32 | sudo tee /etc/llama-orch/secrets/api-token
sudo chmod 600 /etc/llama-orch/secrets/api-token
sudo chown llama-orch:llama-orch /etc/llama-orch/secrets/api-token

# Restart services
sudo systemctl restart queen-rbee rbee-hive
```

2. **Token mismatch:**
```bash
# Ensure all services use same token
sudo md5sum /etc/llama-orch/secrets/api-token

# Copy to other nodes if needed
```

3. **Wrong permissions:**
```bash
# Fix permissions
sudo chmod 600 /etc/llama-orch/secrets/api-token
sudo chown llama-orch:llama-orch /etc/llama-orch/secrets/api-token
```

---

## 🐛 Debug Logging

### Enable Debug Logging

**queen-rbee:**
```bash
# Set environment variable
sudo systemctl edit queen-rbee

[Service]
Environment="RUST_LOG=debug"

# Restart
sudo systemctl restart queen-rbee
```

**rbee-hive:**
```bash
# Set environment variable
sudo systemctl edit rbee-hive

[Service]
Environment="RUST_LOG=rbee_hive=debug,tower_http=debug"

# Restart
sudo systemctl restart rbee-hive
```

### View Debug Logs

```bash
# Follow logs in real-time
sudo journalctl -u queen-rbee -f

# Filter by level
sudo journalctl -u rbee-hive -p debug -f

# Export to file
sudo journalctl -u queen-rbee --since "1 hour ago" > debug.log
```

### Disable Debug Logging

```bash
# Remove environment override
sudo systemctl revert queen-rbee
sudo systemctl revert rbee-hive

# Restart
sudo systemctl restart queen-rbee rbee-hive
```

---

## 🏥 Health Check Procedures

### Manual Health Checks

```bash
#!/bin/bash
# /usr/local/bin/health-check.sh

set -e

echo "=== llama-orch Health Check ==="

# 1. Check services running
echo "Checking services..."
systemctl is-active queen-rbee || echo "ERROR: queen-rbee not running"
systemctl is-active rbee-hive || echo "ERROR: rbee-hive not running"

# 2. Check HTTP endpoints
echo "Checking HTTP endpoints..."
curl -f -s http://localhost:8080/health > /dev/null || echo "ERROR: queen-rbee health check failed"
curl -f -s http://localhost:8081/health > /dev/null || echo "ERROR: rbee-hive health check failed"

# 3. Check disk space
echo "Checking disk space..."
DISK_USAGE=$(df -h /var/lib/llama-orch | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 90 ]; then
    echo "WARNING: Disk usage is ${DISK_USAGE}%"
fi

# 4. Check memory
echo "Checking memory..."
MEM_AVAILABLE=$(free -m | grep Mem | awk '{print $7}')
if [ "$MEM_AVAILABLE" -lt 1024 ]; then
    echo "WARNING: Only ${MEM_AVAILABLE}MB memory available"
fi

# 5. Check workers
echo "Checking workers..."
WORKER_COUNT=$(curl -s http://localhost:8081/metrics | grep 'rbee_hive_workers_total{state="idle"}' | awk '{print $2}')
if [ "$WORKER_COUNT" = "0" ]; then
    echo "WARNING: No idle workers available"
fi

echo "=== Health Check Complete ==="
```

### Automated Health Monitoring

```bash
# Add to crontab
*/5 * * * * /usr/local/bin/health-check.sh >> /var/log/llama-orch/health-check.log 2>&1
```

---

## ⚡ Performance Tuning

### Worker Performance

**Optimize model loading:**
```toml
# /etc/llama-orch/rbee-hive.toml
[workers]
preload_models = ["hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"]
model_cache_size_gb = 20
```

**Optimize inference:**
```toml
[inference]
batch_size = 4
context_length = 2048
threads = 4  # CPU threads per worker
```

### System Performance

**Increase file descriptors:**
```bash
# /etc/security/limits.conf
llama-orch soft nofile 65536
llama-orch hard nofile 65536

# Verify
sudo -u llama-orch bash -c 'ulimit -n'
```

**Optimize disk I/O:**
```bash
# Use faster disk for models
sudo mkdir -p /mnt/nvme/llama-orch/models
sudo ln -s /mnt/nvme/llama-orch/models /var/lib/llama-orch/models
```

**Optimize network:**
```bash
# Increase TCP buffer sizes
sudo sysctl -w net.core.rmem_max=16777216
sudo sysctl -w net.core.wmem_max=16777216
```

---

## 🔄 Recovery Procedures

### Recover from Crash

```bash
# 1. Check what crashed
sudo journalctl -u queen-rbee -n 200 | grep -i "panic\|error\|fatal"

# 2. Check core dumps
sudo coredumpctl list
sudo coredumpctl info <pid>

# 3. Restore from backup
sudo systemctl stop queen-rbee
sudo cp /var/backups/llama-orch/beehive-latest.db /var/lib/llama-orch/data/beehive.db
sudo systemctl start queen-rbee
```

### Recover from Corruption

```bash
# 1. Stop services
sudo systemctl stop rbee-hive queen-rbee

# 2. Check database integrity
sqlite3 /var/lib/llama-orch/data/beehive.db "PRAGMA integrity_check;"

# 3. If corrupted, restore from backup
sudo cp /var/backups/llama-orch/beehive-latest.db /var/lib/llama-orch/data/beehive.db

# 4. Start services
sudo systemctl start queen-rbee rbee-hive
```

### Emergency Restart

```bash
# Nuclear option: restart everything
sudo systemctl restart queen-rbee rbee-hive

# Kill all workers
sudo pkill -9 -f llm-worker-rbee

# Clear worker registry
sudo rm -rf /var/lib/llama-orch/data/workers/*

# Restart
sudo systemctl restart rbee-hive
```

---

## ❌ Error Messages

### "Insufficient memory"

**Cause:** Not enough RAM available for worker spawn

**Solution:**
```bash
# Reduce worker memory limit
sudo nano /etc/llama-orch/rbee-hive.toml
# Set: max_worker_memory_gb = 4

# Or free up memory
sudo systemctl restart rbee-hive
```

### "Insufficient VRAM"

**Cause:** Not enough GPU memory for model

**Solution:**
```bash
# Use smaller model or CPU backend
sudo rbee-hive workers spawn --model <smaller-model> --backend cpu

# Or use quantized model
sudo rbee-hive workers spawn --model <model>-Q4_K_M.gguf
```

### "Model not found"

**Cause:** Model not downloaded or wrong reference

**Solution:**
```bash
# List available models
sudo rbee-hive models list

# Download model
sudo rbee-hive models download <model-ref>
```

### "Worker timeout"

**Cause:** Worker took too long to respond

**Solution:**
```bash
# Check worker health
curl http://localhost:8082/health

# Force-kill if hung
sudo pkill -9 -f "llm-worker-rbee.*<worker-id>"
```

### "Database locked"

**Cause:** SQLite database locked by another process

**Solution:**
```bash
# Find process holding lock
sudo lsof /var/lib/llama-orch/data/beehive.db

# Kill process or wait for it to finish
```

---

## 📞 Getting Help

### Collect Diagnostic Information

```bash
#!/bin/bash
# /usr/local/bin/collect-diagnostics.sh

DIAG_DIR="/tmp/llama-orch-diagnostics-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$DIAG_DIR"

# System info
uname -a > "$DIAG_DIR/system-info.txt"
free -h > "$DIAG_DIR/memory.txt"
df -h > "$DIAG_DIR/disk.txt"

# Service status
systemctl status queen-rbee > "$DIAG_DIR/queen-rbee-status.txt"
systemctl status rbee-hive > "$DIAG_DIR/rbee-hive-status.txt"

# Logs
journalctl -u queen-rbee -n 500 > "$DIAG_DIR/queen-rbee.log"
journalctl -u rbee-hive -n 500 > "$DIAG_DIR/rbee-hive.log"

# Config
cp /etc/llama-orch/*.toml "$DIAG_DIR/" 2>/dev/null || true

# Metrics
curl -s http://localhost:8080/metrics > "$DIAG_DIR/queen-rbee-metrics.txt"
curl -s http://localhost:8081/metrics > "$DIAG_DIR/rbee-hive-metrics.txt"

# Create archive
tar -czf "$DIAG_DIR.tar.gz" -C /tmp "$(basename $DIAG_DIR)"
echo "Diagnostics collected: $DIAG_DIR.tar.gz"
```

### Submit Issue

1. Run diagnostic collection script
2. Attach `llama-orch-diagnostics-*.tar.gz` to GitHub issue
3. Include steps to reproduce
4. Include expected vs actual behavior

---

## 🔗 Related Documentation

- [Deployment Guide](DEPLOYMENT.md)
- [Configuration Guide](CONFIGURATION.md)
- [Monitoring Guide](MONITORING.md)
- [API Documentation](API.md)
