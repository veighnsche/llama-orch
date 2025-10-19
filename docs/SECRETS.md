# Secret Management Guide

**Created by:** TEAM-116  
**Date:** 2025-10-19  
**For:** v0.1.0 Production Deployment

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [API Token Management](#api-token-management)
3. [File Permissions](#file-permissions)
4. [systemd Integration](#systemd-integration)
5. [Token Rotation](#token-rotation)
6. [Emergency Revocation](#emergency-revocation)
7. [Best Practices](#best-practices)

---

## 🔐 Overview

llama-orch uses **file-based secret management** for production deployments. Secrets are loaded from files with strict permissions (0600) to prevent unauthorized access.

### Why File-Based?

- ✅ **Security**: Secrets not visible in process listings
- ✅ **Audit Trail**: File access can be monitored
- ✅ **systemd Integration**: Works with systemd credentials
- ✅ **Simple**: No external dependencies

### Supported Secrets

- **API Tokens**: Authentication between components
- **SSH Keys**: For remote pool management (future)
- **TLS Certificates**: For HTTPS (future)

---

## 🔑 API Token Management

### Generate API Token

```bash
# Generate secure random token (64 hex characters)
openssl rand -hex 32 > /tmp/api-token

# Or use uuidgen
uuidgen | tr -d '-' > /tmp/api-token

# Verify token length
wc -c /tmp/api-token
# Should be 64 bytes (32 hex pairs)
```

### Install API Token

```bash
# Create secrets directory
sudo mkdir -p /etc/llama-orch/secrets

# Move token to secure location
sudo mv /tmp/api-token /etc/llama-orch/secrets/api-token

# Set ownership
sudo chown llama-orch:llama-orch /etc/llama-orch/secrets/api-token

# Set permissions (CRITICAL!)
sudo chmod 600 /etc/llama-orch/secrets/api-token

# Verify permissions
ls -la /etc/llama-orch/secrets/api-token
# Expected: -rw------- 1 llama-orch llama-orch 65 Oct 19 12:00 api-token
```

### Configure Services

**queen-rbee:**
```bash
# Set environment variable
export LLORCH_API_TOKEN_FILE=/etc/llama-orch/secrets/api-token

# Or in systemd service file
Environment="LLORCH_API_TOKEN_FILE=/etc/llama-orch/secrets/api-token"
```

**rbee-hive:**
```bash
# Set environment variable
export LLORCH_API_TOKEN_FILE=/etc/llama-orch/secrets/api-token

# Or in systemd service file
Environment="LLORCH_API_TOKEN_FILE=/etc/llama-orch/secrets/api-token"
```

---

## 🔒 File Permissions

### Critical Permission Requirements

| File | Owner | Group | Permissions | Octal |
|------|-------|-------|-------------|-------|
| `/etc/llama-orch/secrets/` | llama-orch | llama-orch | `drwx------` | 700 |
| `/etc/llama-orch/secrets/api-token` | llama-orch | llama-orch | `-rw-------` | 600 |

### Verify Permissions

```bash
#!/bin/bash
# /usr/local/bin/verify-secrets-permissions.sh

SECRETS_DIR="/etc/llama-orch/secrets"
API_TOKEN="$SECRETS_DIR/api-token"

# Check directory permissions
DIR_PERMS=$(stat -c "%a" "$SECRETS_DIR")
if [ "$DIR_PERMS" != "700" ]; then
    echo "ERROR: $SECRETS_DIR has incorrect permissions: $DIR_PERMS (expected 700)"
    exit 1
fi

# Check file permissions
FILE_PERMS=$(stat -c "%a" "$API_TOKEN")
if [ "$FILE_PERMS" != "600" ]; then
    echo "ERROR: $API_TOKEN has incorrect permissions: $FILE_PERMS (expected 600)"
    exit 1
fi

# Check ownership
OWNER=$(stat -c "%U:%G" "$API_TOKEN")
if [ "$OWNER" != "llama-orch:llama-orch" ]; then
    echo "ERROR: $API_TOKEN has incorrect ownership: $OWNER (expected llama-orch:llama-orch)"
    exit 1
fi

echo "OK: All secret permissions correct"
exit 0
```

### Fix Permissions

```bash
# Fix directory permissions
sudo chmod 700 /etc/llama-orch/secrets

# Fix file permissions
sudo chmod 600 /etc/llama-orch/secrets/api-token

# Fix ownership
sudo chown -R llama-orch:llama-orch /etc/llama-orch/secrets
```

---

## 🔄 systemd Integration

### Using systemd Credentials

systemd 250+ supports `LoadCredential` for secure secret loading:

```ini
[Service]
LoadCredential=api-token:/etc/llama-orch/secrets/api-token
Environment="LLORCH_API_TOKEN_FILE=%d/api-token"
```

### Benefits

- ✅ Secrets loaded into memory-backed tmpfs
- ✅ Automatic cleanup on service stop
- ✅ SELinux/AppArmor compatible
- ✅ Audit trail via systemd

### Example Service File

```ini
[Unit]
Description=rbee Orchestrator Daemon
After=network.target

[Service]
Type=simple
User=llama-orch
Group=llama-orch
ExecStart=/usr/local/bin/queen-rbee --port 8080

# Load credentials securely
LoadCredential=api-token:/etc/llama-orch/secrets/api-token
Environment="LLORCH_API_TOKEN_FILE=%d/api-token"

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true

[Install]
WantedBy=multi-user.target
```

---

## 🔄 Token Rotation

### Rotation Procedure

Token rotation should be performed regularly (recommended: every 90 days).

#### 1. Generate New Token

```bash
# Generate new token
openssl rand -hex 32 > /tmp/api-token-new

# Set permissions
chmod 600 /tmp/api-token-new
```

#### 2. Update Services (Rolling Update)

```bash
# Update queen-rbee first
sudo cp /tmp/api-token-new /etc/llama-orch/secrets/api-token-new
sudo chown llama-orch:llama-orch /etc/llama-orch/secrets/api-token-new
sudo chmod 600 /etc/llama-orch/secrets/api-token-new

# Update queen-rbee config to accept both tokens temporarily
# (Requires dual-token support - future feature)

# Restart queen-rbee
sudo systemctl restart queen-rbee

# Update rbee-hive instances
for host in hive1 hive2 hive3; do
    ssh $host "sudo cp /tmp/api-token-new /etc/llama-orch/secrets/api-token"
    ssh $host "sudo systemctl restart rbee-hive"
done

# Remove old token
sudo rm /etc/llama-orch/secrets/api-token-old
```

#### 3. Verify Rotation

```bash
# Check all services are healthy
curl http://localhost:8080/health
curl http://localhost:8081/health

# Check audit logs for authentication events
sudo grep "authentication" /var/log/llama-orch/audit/*.ndjson
```

### Automated Rotation

```bash
#!/bin/bash
# /usr/local/bin/rotate-api-token.sh

set -e

OLD_TOKEN="/etc/llama-orch/secrets/api-token"
NEW_TOKEN="/etc/llama-orch/secrets/api-token.new"
BACKUP_TOKEN="/etc/llama-orch/secrets/api-token.backup"

# Generate new token
openssl rand -hex 32 > "$NEW_TOKEN"
chmod 600 "$NEW_TOKEN"
chown llama-orch:llama-orch "$NEW_TOKEN"

# Backup old token
cp "$OLD_TOKEN" "$BACKUP_TOKEN"

# Replace token
mv "$NEW_TOKEN" "$OLD_TOKEN"

# Restart services
systemctl restart queen-rbee
systemctl restart rbee-hive

# Verify services are healthy
sleep 5
if ! curl -f -s http://localhost:8080/health > /dev/null; then
    echo "ERROR: queen-rbee health check failed after rotation"
    # Rollback
    mv "$BACKUP_TOKEN" "$OLD_TOKEN"
    systemctl restart queen-rbee
    exit 1
fi

echo "Token rotation successful"
rm "$BACKUP_TOKEN"
```

---

## 🚨 Emergency Revocation

### Immediate Token Revocation

If a token is compromised:

```bash
# 1. Generate new token immediately
openssl rand -hex 32 | sudo tee /etc/llama-orch/secrets/api-token > /dev/null

# 2. Fix permissions
sudo chmod 600 /etc/llama-orch/secrets/api-token
sudo chown llama-orch:llama-orch /etc/llama-orch/secrets/api-token

# 3. Restart all services
sudo systemctl restart queen-rbee
sudo systemctl restart rbee-hive

# 4. Verify services
sudo systemctl status queen-rbee
sudo systemctl status rbee-hive

# 5. Check audit logs for unauthorized access
sudo grep "authentication_failed" /var/log/llama-orch/audit/*.ndjson
```

### Incident Response Checklist

- [ ] Generate new token
- [ ] Update all services
- [ ] Restart all services
- [ ] Verify services healthy
- [ ] Review audit logs for unauthorized access
- [ ] Document incident
- [ ] Update security procedures

---

## ✅ Best Practices

### DO

✅ **Use file-based secrets** with 0600 permissions  
✅ **Rotate tokens regularly** (every 90 days)  
✅ **Use systemd credentials** when available  
✅ **Monitor file access** via audit logs  
✅ **Backup tokens securely** (encrypted)  
✅ **Document rotation procedures**  
✅ **Test rotation in staging first**  
✅ **Use strong random tokens** (32+ bytes)

### DON'T

❌ **Never use environment variables** in production  
❌ **Never commit secrets to git**  
❌ **Never use weak tokens** (< 16 bytes)  
❌ **Never share tokens between environments**  
❌ **Never log tokens** (even in debug mode)  
❌ **Never store tokens in world-readable files**  
❌ **Never reuse tokens across services**

### Security Checklist

```bash
# Run this checklist before production deployment

# 1. Verify file permissions
sudo stat -c "%a %U:%G" /etc/llama-orch/secrets/api-token
# Expected: 600 llama-orch:llama-orch

# 2. Verify token length
wc -c /etc/llama-orch/secrets/api-token
# Expected: 64 bytes minimum

# 3. Verify token randomness
hexdump -C /etc/llama-orch/secrets/api-token | head -n 5
# Should look random, not sequential

# 4. Verify no environment variables
env | grep LLORCH_API_TOKEN
# Should be empty

# 5. Verify services use file-based loading
sudo journalctl -u queen-rbee | grep "API token loaded"
# Should show "loaded from file"

# 6. Verify audit logging enabled
ls -la /var/log/llama-orch/audit/
# Should contain recent .ndjson files
```

---

## 📊 Audit Trail

### Monitor Secret Access

```bash
# Monitor file access with auditd
sudo auditctl -w /etc/llama-orch/secrets/api-token -p r -k secret-access

# View audit events
sudo ausearch -k secret-access
```

### Review Authentication Events

```bash
# Check successful authentications
sudo grep "authentication_success" /var/log/llama-orch/audit/*.ndjson

# Check failed authentications
sudo grep "authentication_failed" /var/log/llama-orch/audit/*.ndjson

# Count authentication events
sudo grep "authentication" /var/log/llama-orch/audit/*.ndjson | wc -l
```

---

## 🔗 Related Documentation

- [Deployment Guide](DEPLOYMENT.md)
- [Configuration Guide](CONFIGURATION.md)
- [Monitoring Guide](MONITORING.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)

---

## 📞 Security Contact

For security issues, please email: security@llama-orch.dev

**Do not** open public GitHub issues for security vulnerabilities.
