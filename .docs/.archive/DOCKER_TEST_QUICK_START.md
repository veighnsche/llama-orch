# Docker Test Quick Start Guide

**Date:** Oct 24, 2025  
**Purpose:** Get Docker tests running in 15 minutes  
**For:** Developers who want to start testing immediately

---

## Prerequisites

- Docker and docker-compose installed
- Rust toolchain installed
- Project cloned and at workspace root

---

## 5-Minute Setup

### Step 1: Build Binaries (2 min)
```bash
cargo build --bin queen-rbee --bin rbee-hive
```

### Step 2: Create Test Infrastructure (1 min)
```bash
# Create directories
mkdir -p tests/docker/{keys,configs,scripts}

# Generate SSH keys
cd tests/docker/keys
ssh-keygen -t ed25519 -f test_id_rsa -N "" -C "rbee-docker-tests"
cd ../../..
```

### Step 3: Copy Minimal Configs (1 min)

**File:** `tests/docker/configs/hives.conf`
```toml
[[hives]]
alias = "hive-localhost"
hostname = "172.20.0.20"
hive_port = 9000
ssh_user = "rbee"
ssh_port = 22
```

**File:** `tests/docker/configs/supervisord.conf`
```ini
[supervisord]
nodaemon=true
user=root

[program:sshd]
command=/usr/sbin/sshd -D
autostart=true
autorestart=true

[program:rbee-hive]
command=/home/rbee/.local/bin/rbee-hive --port 9000
user=rbee
autostart=true
autorestart=true
environment=HOME="/home/rbee",USER="rbee"
```

### Step 4: Create Dockerfiles (1 min)

**File:** `tests/docker/Dockerfile.base`
```dockerfile
FROM rust:1.75-slim
RUN apt-get update && apt-get install -y openssh-server git build-essential pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /var/run/sshd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN useradd -m -s /bin/bash rbee && echo 'rbee:rbee' | chpasswd
RUN mkdir -p /home/rbee/.ssh
COPY tests/docker/keys/test_id_rsa.pub /home/rbee/.ssh/authorized_keys
RUN chown -R rbee:rbee /home/rbee/.ssh && chmod 700 /home/rbee/.ssh && chmod 600 /home/rbee/.ssh/authorized_keys
WORKDIR /home/rbee
USER rbee
EXPOSE 22
```

**File:** `tests/docker/Dockerfile.queen`
```dockerfile
FROM rbee-base:latest
USER root
RUN mkdir -p /home/rbee/.local/bin /home/rbee/.config/rbee
USER rbee
COPY --chown=rbee:rbee target/debug/queen-rbee /home/rbee/.local/bin/queen-rbee
RUN chmod +x /home/rbee/.local/bin/queen-rbee
COPY --chown=rbee:rbee tests/docker/configs/hives.conf /home/rbee/.config/rbee/hives.conf
EXPOSE 8500
CMD ["/home/rbee/.local/bin/queen-rbee", "--port", "8500"]
```

**File:** `tests/docker/Dockerfile.hive`
```dockerfile
FROM rbee-base:latest
USER root
RUN apt-get update && apt-get install -y supervisor && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /home/rbee/.local/bin /home/rbee/.config/rbee
USER rbee
COPY --chown=rbee:rbee target/debug/rbee-hive /home/rbee/.local/bin/rbee-hive
RUN chmod +x /home/rbee/.local/bin/rbee-hive
USER root
COPY tests/docker/configs/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
EXPOSE 22 9000
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
```

**File:** `tests/docker/docker-compose.localhost.yml`
```yaml
version: '3.8'
services:
  queen:
    build:
      context: ../..
      dockerfile: tests/docker/Dockerfile.queen
    container_name: rbee-queen-localhost
    ports:
      - "8500:8500"
    networks:
      rbee-test:
        ipv4_address: 172.20.0.10
    environment:
      - RUST_LOG=debug

  hive-localhost:
    build:
      context: ../..
      dockerfile: tests/docker/Dockerfile.hive
    container_name: rbee-hive-localhost
    ports:
      - "9000:9000"
      - "2222:22"
    networks:
      rbee-test:
        ipv4_address: 172.20.0.20
    environment:
      - RUST_LOG=debug

networks:
  rbee-test:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

---

## First Test Run

### Build Images
```bash
docker build -f tests/docker/Dockerfile.base -t rbee-base:latest .
docker build -f tests/docker/Dockerfile.queen -t rbee-queen:latest .
docker build -f tests/docker/Dockerfile.hive -t rbee-hive:latest .
```

### Start Environment
```bash
docker-compose -f tests/docker/docker-compose.localhost.yml up -d
```

### Verify Services
```bash
# Check containers are running
docker ps

# Check queen health
curl http://localhost:8500/health

# Check hive health
curl http://localhost:9000/health

# Check hive capabilities
curl http://localhost:9000/capabilities
```

### Manual Tests
```bash
# SSH into hive
ssh -i tests/docker/keys/test_id_rsa -p 2222 rbee@localhost

# Check hive binary
docker exec rbee-hive-localhost ls -la /home/rbee/.local/bin/rbee-hive

# Check logs
docker logs rbee-queen-localhost
docker logs rbee-hive-localhost
```

### Cleanup
```bash
docker-compose -f tests/docker/docker-compose.localhost.yml down -v
```

---

## First Automated Test

### Create Test File
**File:** `xtask/tests/docker/smoke_test.rs`
```rust
use std::time::Duration;

#[tokio::test]
#[ignore]
async fn test_docker_smoke() {
    // Assumes containers are already running
    
    // Test queen health
    let response = ureq::get("http://localhost:8500/health")
        .timeout(Duration::from_secs(5))
        .call()
        .expect("Queen health check failed");
    
    assert_eq!(response.status(), 200);
    assert_eq!(response.into_string().unwrap(), "ok");
    
    // Test hive health
    let response = ureq::get("http://localhost:9000/health")
        .timeout(Duration::from_secs(5))
        .call()
        .expect("Hive health check failed");
    
    assert_eq!(response.status(), 200);
    assert_eq!(response.into_string().unwrap(), "ok");
    
    // Test hive capabilities
    let response = ureq::get("http://localhost:9000/capabilities")
        .timeout(Duration::from_secs(5))
        .call()
        .expect("Hive capabilities check failed");
    
    assert_eq!(response.status(), 200);
    
    let json: serde_json::Value = response.into_json().unwrap();
    assert!(json["devices"].is_array());
    
    println!("✅ Smoke test passed!");
}
```

### Run Test
```bash
# Start containers
docker-compose -f tests/docker/docker-compose.localhost.yml up -d

# Wait for services
sleep 10

# Run test
cargo test --package xtask --test smoke_test --ignored -- --nocapture

# Cleanup
docker-compose -f tests/docker/docker-compose.localhost.yml down -v
```

---

## Common Issues

### "Cannot connect to Docker daemon"
```bash
sudo systemctl start docker
sudo usermod -aG docker $USER
# Log out and back in
```

### "Port already in use"
```bash
# Find process using port
sudo lsof -i :8500
sudo lsof -i :9000

# Kill process or change ports in docker-compose.yml
```

### "SSH connection refused"
```bash
# Check SSH is running
docker exec rbee-hive-localhost ps aux | grep sshd

# Restart container
docker restart rbee-hive-localhost
```

### "Binary not found in container"
```bash
# Rebuild binary
cargo build --bin rbee-hive

# Rebuild image
docker build -f tests/docker/Dockerfile.hive -t rbee-hive:latest . --no-cache
```

---

## Next Steps

1. ✅ You now have a working Docker test environment
2. Read `DOCKER_NETWORK_TESTING_PLAN.md` for full test strategy
3. Read `DOCKER_TEST_IMPLEMENTATION_GUIDE.md` for detailed examples
4. Implement test categories one by one
5. Add to CI/CD pipeline

---

## Helper Scripts

### Build Everything
```bash
#!/bin/bash
# tests/docker/scripts/build-all.sh
set -e
cargo build --bin queen-rbee --bin rbee-hive
docker build -f tests/docker/Dockerfile.base -t rbee-base:latest .
docker build -f tests/docker/Dockerfile.queen -t rbee-queen:latest .
docker build -f tests/docker/Dockerfile.hive -t rbee-hive:latest .
echo "✅ Build complete"
```

### Start Environment
```bash
#!/bin/bash
# tests/docker/scripts/start.sh
set -e
docker-compose -f tests/docker/docker-compose.localhost.yml up -d
echo "⏳ Waiting for services..."
sleep 10
curl -f http://localhost:8500/health && echo "✅ Queen ready"
curl -f http://localhost:9000/health && echo "✅ Hive ready"
```

### Stop Environment
```bash
#!/bin/bash
# tests/docker/scripts/stop.sh
docker-compose -f tests/docker/docker-compose.localhost.yml down -v
echo "✅ Environment stopped"
```

### Run All Tests
```bash
#!/bin/bash
# tests/docker/scripts/test-all.sh
set -e
./tests/docker/scripts/start.sh
cargo test --package xtask --test smoke_test --ignored -- --nocapture
./tests/docker/scripts/stop.sh
echo "✅ All tests passed"
```

Make scripts executable:
```bash
chmod +x tests/docker/scripts/*.sh
```

---

## Success Checklist

- [ ] Binaries built
- [ ] SSH keys generated
- [ ] Dockerfiles created
- [ ] docker-compose.yml created
- [ ] Images built successfully
- [ ] Containers start without errors
- [ ] Queen health check returns "ok"
- [ ] Hive health check returns "ok"
- [ ] Hive capabilities returns JSON
- [ ] SSH connection works
- [ ] First automated test passes

**If all checked: You're ready to build comprehensive tests!** 🎉
