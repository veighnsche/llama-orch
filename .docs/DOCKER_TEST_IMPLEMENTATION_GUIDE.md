# Docker Test Implementation Guide

**Date:** Oct 24, 2025  
**Purpose:** Step-by-step implementation guide with concrete examples  
**Companion to:** DOCKER_NETWORK_TESTING_PLAN.md

---

## Phase 1: Foundation Setup

### Step 1: Create Directory Structure

```bash
mkdir -p tests/docker/{keys,configs,scripts}
mkdir -p xtask/tests/docker
mkdir -p xtask/src/integration
```

### Step 2: Generate SSH Keys

```bash
#!/bin/bash
# tests/docker/scripts/generate-keys.sh

cd tests/docker/keys
ssh-keygen -t ed25519 -f test_id_rsa -N "" -C "rbee-docker-tests"
chmod 600 test_id_rsa
chmod 644 test_id_rsa.pub
echo "✅ SSH keys generated"
```

### Step 3: Create Base Dockerfile

```dockerfile
# tests/docker/Dockerfile.base

FROM rust:1.75-slim

# Install SSH server, git, and build tools
RUN apt-get update && apt-get install -y \
    openssh-server \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Setup SSH
RUN mkdir -p /var/run/sshd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# Create test user
RUN useradd -m -s /bin/bash rbee
RUN echo 'rbee:rbee' | chpasswd

# Setup SSH keys
RUN mkdir -p /home/rbee/.ssh
COPY tests/docker/keys/test_id_rsa.pub /home/rbee/.ssh/authorized_keys
RUN chown -R rbee:rbee /home/rbee/.ssh
RUN chmod 700 /home/rbee/.ssh
RUN chmod 600 /home/rbee/.ssh/authorized_keys

WORKDIR /home/rbee
USER rbee

EXPOSE 22
```

### Step 4: Create Queen Dockerfile

```dockerfile
# tests/docker/Dockerfile.queen

FROM rbee-base:latest

USER root
RUN mkdir -p /home/rbee/.local/bin /home/rbee/.config/rbee
USER rbee

# Copy pre-built queen-rbee binary
COPY --chown=rbee:rbee target/debug/queen-rbee /home/rbee/.local/bin/queen-rbee
RUN chmod +x /home/rbee/.local/bin/queen-rbee

# Copy default config
COPY --chown=rbee:rbee tests/docker/configs/hives.conf /home/rbee/.config/rbee/hives.conf

EXPOSE 8500

CMD ["/home/rbee/.local/bin/queen-rbee", "--port", "8500"]
```

### Step 5: Create Hive Dockerfile

```dockerfile
# tests/docker/Dockerfile.hive

FROM rbee-base:latest

USER root

# Install supervisor to run SSH + hive
RUN apt-get update && apt-get install -y supervisor && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/rbee/.local/bin /home/rbee/.config/rbee
USER rbee

# Copy pre-built rbee-hive binary
COPY --chown=rbee:rbee target/debug/rbee-hive /home/rbee/.local/bin/rbee-hive
RUN chmod +x /home/rbee/.local/bin/rbee-hive

USER root

# Setup supervisor config
COPY tests/docker/configs/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 22 9000

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
```

### Step 6: Create Supervisor Config

```ini
# tests/docker/configs/supervisord.conf

[supervisord]
nodaemon=true
user=root

[program:sshd]
command=/usr/sbin/sshd -D
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0

[program:rbee-hive]
command=/home/rbee/.local/bin/rbee-hive --port 9000
user=rbee
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
environment=HOME="/home/rbee",USER="rbee"
```

### Step 7: Create Hives Config

```toml
# tests/docker/configs/hives.conf

[[hives]]
alias = "hive-1"
hostname = "172.20.0.20"
hive_port = 9000
ssh_user = "rbee"
ssh_port = 22

[[hives]]
alias = "hive-2"
hostname = "172.20.0.21"
hive_port = 9000
ssh_user = "rbee"
ssh_port = 22
```

---

## Phase 2: Docker Compose Files

### Localhost Topology

```yaml
# tests/docker/docker-compose.localhost.yml

version: '3.8'

services:
  queen:
    build:
      context: ../..
      dockerfile: tests/docker/Dockerfile.queen
    container_name: rbee-queen-localhost
    hostname: queen
    ports:
      - "8500:8500"
    networks:
      rbee-test:
        ipv4_address: 172.20.0.10
    environment:
      - RBEE_CONFIG_DIR=/home/rbee/.config/rbee
      - RBEE_DATA_DIR=/home/rbee/.local/share/rbee
      - RUST_LOG=debug
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8500/health"]
      interval: 5s
      timeout: 3s
      retries: 3

  hive-localhost:
    build:
      context: ../..
      dockerfile: tests/docker/Dockerfile.hive
    container_name: rbee-hive-localhost
    hostname: hive-localhost
    ports:
      - "9000:9000"
      - "2222:22"
    networks:
      rbee-test:
        ipv4_address: 172.20.0.20
    environment:
      - RBEE_CONFIG_DIR=/home/rbee/.config/rbee
      - RBEE_DATA_DIR=/home/rbee/.local/share/rbee
      - RUST_LOG=debug
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/health"]
      interval: 5s
      timeout: 3s
      retries: 3

networks:
  rbee-test:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

---

## Phase 3: DockerTestHarness Implementation

```rust
// xtask/src/integration/docker_harness.rs

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;
use tokio::time::sleep;

#[derive(Debug, Clone)]
pub enum Topology {
    Localhost,
    MultiHive,
    FullStack,
}

pub struct DockerTestHarness {
    compose_file: PathBuf,
    test_id: String,
}

impl DockerTestHarness {
    /// Create new Docker test environment
    pub async fn new(topology: Topology) -> Result<Self> {
        let compose_file = match topology {
            Topology::Localhost => "tests/docker/docker-compose.localhost.yml",
            Topology::MultiHive => "tests/docker/docker-compose.multi-hive.yml",
            Topology::FullStack => "tests/docker/docker-compose.full-stack.yml",
        };

        let test_id = uuid::Uuid::new_v4().to_string();

        println!("🐳 Starting Docker environment: {:?}", topology);
        println!("📋 Test ID: {}", test_id);

        // Start containers
        Self::docker_compose_up(compose_file).await?;

        // Wait for services to be healthy
        Self::wait_for_services(compose_file).await?;

        Ok(Self {
            compose_file: compose_file.into(),
            test_id,
        })
    }

    /// Start containers via docker-compose
    async fn docker_compose_up(compose_file: &str) -> Result<()> {
        let output = Command::new("docker-compose")
            .args(&["-f", compose_file, "up", "-d"])
            .output()
            .context("Failed to start docker-compose")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("docker-compose up failed: {}", stderr);
        }

        println!("✅ Containers started");
        Ok(())
    }

    /// Wait for all services to be healthy
    async fn wait_for_services(compose_file: &str) -> Result<()> {
        println!("⏳ Waiting for services to be healthy...");

        let start = std::time::Instant::now();
        let timeout = Duration::from_secs(60);

        loop {
            let output = Command::new("docker-compose")
                .args(&["-f", compose_file, "ps", "--format", "json"])
                .output()
                .context("Failed to check service status")?;

            let stdout = String::from_utf8_lossy(&output.stdout);

            // Check if all services are healthy
            if stdout.contains("\"Health\":\"healthy\"") {
                println!("✅ All services healthy");
                return Ok(());
            }

            if start.elapsed() > timeout {
                anyhow::bail!("Timeout waiting for services to be healthy");
            }

            sleep(Duration::from_secs(2)).await;
        }
    }

    /// Execute command in container
    pub async fn exec(&self, container: &str, cmd: &[&str]) -> Result<String> {
        let output = Command::new("docker")
            .arg("exec")
            .arg(container)
            .args(cmd)
            .output()
            .context(format!("Failed to exec in container {}", container))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Command failed: {}", stderr);
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    /// Get container logs
    pub async fn logs(&self, container: &str) -> Result<String> {
        let output = Command::new("docker")
            .args(&["logs", container])
            .output()
            .context(format!("Failed to get logs for {}", container))?;

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    /// Restart container
    pub async fn restart(&self, container: &str) -> Result<()> {
        Command::new("docker")
            .args(&["restart", container])
            .output()
            .context(format!("Failed to restart {}", container))?;

        println!("🔄 Restarted container: {}", container);
        Ok(())
    }

    /// Kill container (simulate crash)
    pub async fn kill(&self, container: &str) -> Result<()> {
        Command::new("docker")
            .args(&["kill", container])
            .output()
            .context(format!("Failed to kill {}", container))?;

        println!("💀 Killed container: {}", container);
        Ok(())
    }

    /// Block network between containers
    pub async fn block_network(&self, from: &str, to_ip: &str) -> Result<()> {
        self.exec(from, &["iptables", "-A", "OUTPUT", "-d", to_ip, "-j", "DROP"]).await?;
        println!("🚫 Blocked network: {} → {}", from, to_ip);
        Ok(())
    }

    /// Restore network between containers
    pub async fn restore_network(&self, from: &str, to_ip: &str) -> Result<()> {
        self.exec(from, &["iptables", "-D", "OUTPUT", "-d", to_ip, "-j", "DROP"]).await?;
        println!("✅ Restored network: {} → {}", from, to_ip);
        Ok(())
    }

    /// Wait for HTTP endpoint to be healthy
    pub async fn wait_for_http(&self, url: &str, timeout: Duration) -> Result<()> {
        let start = std::time::Instant::now();

        loop {
            match ureq::get(url).timeout(Duration::from_secs(2)).call() {
                Ok(response) if response.status() == 200 => {
                    println!("✅ HTTP endpoint healthy: {}", url);
                    return Ok(());
                }
                _ => {}
            }

            if start.elapsed() > timeout {
                anyhow::bail!("Timeout waiting for HTTP endpoint: {}", url);
            }

            sleep(Duration::from_millis(500)).await;
        }
    }
}

impl Drop for DockerTestHarness {
    fn drop(&mut self) {
        println!("🧹 Cleaning up Docker environment...");

        let _ = Command::new("docker-compose")
            .args(&["-f", self.compose_file.to_str().unwrap(), "down", "-v"])
            .output();

        println!("✅ Cleanup complete");
    }
}
```

---

## Phase 4: Example Tests

### HTTP Communication Test

```rust
// xtask/tests/docker/http_communication_tests.rs

use xtask::integration::docker_harness::{DockerTestHarness, Topology};
use std::time::Duration;

#[tokio::test]
#[ignore]
async fn test_queen_to_hive_health_check() {
    // Setup
    let harness = DockerTestHarness::new(Topology::Localhost).await.unwrap();

    // Wait for services
    harness.wait_for_http("http://localhost:8500/health", Duration::from_secs(30)).await.unwrap();
    harness.wait_for_http("http://localhost:9000/health", Duration::from_secs(30)).await.unwrap();

    // Test: Queen calls hive health endpoint
    let response = ureq::get("http://localhost:9000/health")
        .timeout(Duration::from_secs(5))
        .call()
        .unwrap();

    assert_eq!(response.status(), 200);
    assert_eq!(response.into_string().unwrap(), "ok");
}

#[tokio::test]
#[ignore]
async fn test_queen_to_hive_capabilities() {
    let harness = DockerTestHarness::new(Topology::Localhost).await.unwrap();

    harness.wait_for_http("http://localhost:9000/health", Duration::from_secs(30)).await.unwrap();

    // Test: Queen fetches capabilities
    let response = ureq::get("http://localhost:9000/capabilities")
        .timeout(Duration::from_secs(5))
        .call()
        .unwrap();

    assert_eq!(response.status(), 200);

    let json: serde_json::Value = response.into_json().unwrap();
    assert!(json["devices"].is_array());
    assert!(json["devices"].as_array().unwrap().len() > 0);
}
```

### SSH Communication Test

```rust
// xtask/tests/docker/ssh_communication_tests.rs

use xtask::integration::docker_harness::{DockerTestHarness, Topology};
use std::time::Duration;

#[tokio::test]
#[ignore]
async fn test_ssh_connection_to_hive() {
    let harness = DockerTestHarness::new(Topology::Localhost).await.unwrap();

    // Wait for SSH to be ready
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Test: Execute command via SSH
    let output = harness.exec("rbee-hive-localhost", &["echo", "test"]).await.unwrap();

    assert_eq!(output.trim(), "test");
}

#[tokio::test]
#[ignore]
async fn test_ssh_binary_check() {
    let harness = DockerTestHarness::new(Topology::Localhost).await.unwrap();

    tokio::time::sleep(Duration::from_secs(5)).await;

    // Test: Check if rbee-hive binary exists
    let output = harness.exec("rbee-hive-localhost", &["ls", "-la", "/home/rbee/.local/bin/rbee-hive"]).await.unwrap();

    assert!(output.contains("rbee-hive"));
    assert!(output.contains("-rwx")); // Executable
}
```

### Failure Scenario Test

```rust
// xtask/tests/docker/failure_tests.rs

use xtask::integration::docker_harness::{DockerTestHarness, Topology};
use std::time::Duration;

#[tokio::test]
#[ignore]
async fn test_hive_crash_during_operation() {
    let harness = DockerTestHarness::new(Topology::Localhost).await.unwrap();

    harness.wait_for_http("http://localhost:9000/health", Duration::from_secs(30)).await.unwrap();

    // Kill hive mid-operation
    harness.kill("rbee-hive-localhost").await.unwrap();

    // Verify connection fails
    let result = ureq::get("http://localhost:9000/health")
        .timeout(Duration::from_secs(2))
        .call();

    assert!(result.is_err());
}

#[tokio::test]
#[ignore]
async fn test_network_partition() {
    let harness = DockerTestHarness::new(Topology::MultiHive).await.unwrap();

    harness.wait_for_http("http://localhost:8500/health", Duration::from_secs(30)).await.unwrap();

    // Block network between queen and hive-1
    harness.block_network("rbee-queen", "172.20.0.20").await.unwrap();

    // Verify queen cannot reach hive-1
    let output = harness.exec("rbee-queen", &["curl", "-m", "2", "http://172.20.0.20:9000/health"]).await;

    assert!(output.is_err() || output.unwrap().contains("timeout"));

    // Restore network
    harness.restore_network("rbee-queen", "172.20.0.20").await.unwrap();

    // Verify connection restored
    tokio::time::sleep(Duration::from_secs(2)).await;
    let output = harness.exec("rbee-queen", &["curl", "http://172.20.0.20:9000/health"]).await.unwrap();

    assert_eq!(output.trim(), "ok");
}
```

---

## Running the Tests

### Build and Setup
```bash
# Build binaries
cargo build --bin queen-rbee --bin rbee-hive --bin llm-worker-rbee

# Generate SSH keys
./tests/docker/scripts/generate-keys.sh

# Build Docker images
docker build -f tests/docker/Dockerfile.base -t rbee-base:latest .
docker build -f tests/docker/Dockerfile.queen -t rbee-queen:latest .
docker build -f tests/docker/Dockerfile.hive -t rbee-hive:latest .
```

### Run Tests
```bash
# Run all Docker tests
cargo test --package xtask --test docker_http_communication_tests --ignored
cargo test --package xtask --test docker_ssh_communication_tests --ignored
cargo test --package xtask --test docker_failure_tests --ignored
```

---

## Troubleshooting

### Containers won't start
```bash
# Check logs
docker-compose -f tests/docker/docker-compose.localhost.yml logs

# Rebuild images
docker-compose -f tests/docker/docker-compose.localhost.yml build --no-cache
```

### SSH connection fails
```bash
# Check SSH is running in container
docker exec rbee-hive-localhost ps aux | grep sshd

# Test SSH manually
ssh -i tests/docker/keys/test_id_rsa -p 2222 rbee@localhost
```

### Network issues
```bash
# Check network exists
docker network ls | grep rbee-test

# Inspect network
docker network inspect rbee-test

# Check container IPs
docker inspect rbee-queen-localhost | grep IPAddress
```

---

## Next Steps

1. Implement Phase 1 (Foundation)
2. Test basic Docker setup
3. Implement Phase 2 (HTTP tests)
4. Implement Phase 3 (SSH tests)
5. Implement Phase 4 (Failure tests)
6. Add CI/CD integration
