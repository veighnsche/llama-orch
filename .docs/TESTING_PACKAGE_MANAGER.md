# Testing Package Manager - Options & Recommendations

**Date:** Oct 24, 2025  
**Status:** Testing Strategy Document

---

## Overview

The package manager (daemon-sync) performs SSH-based installation of hives and workers on remote hosts. Testing this requires simulating remote environments while maintaining reproducibility.

---

## Testing Challenges

1. **SSH Dependency** - Requires SSH access to remote hosts
2. **Git Clone** - Needs git repository access
3. **Cargo Build** - Requires Rust toolchain on remote hosts
4. **State Management** - Installation creates files/directories
5. **Concurrency** - Multiple hives/workers install in parallel
6. **Network I/O** - SSH commands have latency
7. **Build Time** - Cargo builds take 2-5 minutes

---

## Testing Levels

### 1. Unit Tests (Fast, Isolated)
### 2. Integration Tests (Medium, Docker)
### 3. E2E Tests (Slow, Real SSH)
### 4. CI Tests (Automated, Reproducible)

---

## Option 1: Mock SSH Client (Unit Tests)

### Approach
Create a mock SSH client that simulates command execution without real SSH.

### Implementation

```rust
// bin/99_shared_crates/daemon-sync/src/test_helpers.rs

pub struct MockSSHClient {
    commands: Vec<String>,
    responses: HashMap<String, (String, String, i32)>, // stdout, stderr, exit_code
}

impl MockSSHClient {
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
            responses: HashMap::new(),
        }
    }
    
    pub fn expect_command(&mut self, cmd: &str, stdout: &str, stderr: &str, exit_code: i32) {
        self.responses.insert(cmd.to_string(), (stdout.to_string(), stderr.to_string(), exit_code));
    }
    
    pub async fn exec(&mut self, cmd: &str) -> Result<(String, String, i32)> {
        self.commands.push(cmd.to_string());
        
        if let Some(response) = self.responses.get(cmd) {
            Ok(response.clone())
        } else {
            Ok(("".to_string(), "".to_string(), 0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_git_clone_success() {
        let mut client = MockSSHClient::new();
        
        // Mock git clone
        client.expect_command(
            "git clone --depth 1 --branch main git@github.com:veighnsche/llama-orch ~/.local/share/rbee/build",
            "",
            "",
            0
        );
        
        // Mock cargo build
        client.expect_command(
            "cd ~/.local/share/rbee/build && cargo build --release --bin rbee-hive",
            "",
            "",
            0
        );
        
        // Test installation
        let result = install_hive_from_git(&mut client, "git@github.com:veighnsche/llama-orch", "main", "job-123", "test-hive").await;
        
        assert!(result.is_ok());
        assert_eq!(client.commands.len(), 2);
    }
    
    #[tokio::test]
    async fn test_git_clone_failure() {
        let mut client = MockSSHClient::new();
        
        // Mock failed git clone
        client.expect_command(
            "git clone --depth 1 --branch main git@github.com:veighnsche/llama-orch ~/.local/share/rbee/build",
            "",
            "fatal: repository not found",
            128
        );
        
        let result = install_hive_from_git(&mut client, "git@github.com:veighnsche/llama-orch", "main", "job-123", "test-hive").await;
        
        assert!(result.is_err());
    }
}
```

### Pros
- ✅ Fast (milliseconds)
- ✅ No external dependencies
- ✅ Easy to test error cases
- ✅ Runs in CI without setup

### Cons
- ❌ Doesn't test real SSH
- ❌ Doesn't test real git/cargo
- ❌ Mock might diverge from reality

### Recommendation
**Use for:** Command generation, error handling, state transitions

---

## Option 2: Docker Containers (Integration Tests)

### Approach
Spin up Docker containers with SSH, git, and Rust toolchain. Test against real SSH.

### Implementation

```dockerfile
# tests/docker/Dockerfile.test-host

FROM rust:1.75

# Install SSH server
RUN apt-get update && apt-get install -y openssh-server git

# Setup SSH
RUN mkdir /var/run/sshd
RUN echo 'root:testpassword' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# Create test user
RUN useradd -m -s /bin/bash testuser
RUN echo 'testuser:testpassword' | chpasswd

# Setup SSH keys
RUN mkdir -p /home/testuser/.ssh
COPY test_id_rsa.pub /home/testuser/.ssh/authorized_keys
RUN chown -R testuser:testuser /home/testuser/.ssh
RUN chmod 700 /home/testuser/.ssh
RUN chmod 600 /home/testuser/.ssh/authorized_keys

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
```

```yaml
# tests/docker/docker-compose.yml

version: '3.8'

services:
  test-host-1:
    build:
      context: .
      dockerfile: Dockerfile.test-host
    container_name: rbee-test-host-1
    ports:
      - "2222:22"
    networks:
      - rbee-test

  test-host-2:
    build:
      context: .
      dockerfile: Dockerfile.test-host
    container_name: rbee-test-host-2
    ports:
      - "2223:22"
    networks:
      - rbee-test

networks:
  rbee-test:
    driver: bridge
```

```rust
// tests/integration/docker_tests.rs

#[tokio::test]
#[ignore] // Run with: cargo test --ignored
async fn test_install_hive_docker() {
    // Start docker containers
    let output = Command::new("docker-compose")
        .args(&["-f", "tests/docker/docker-compose.yml", "up", "-d"])
        .output()
        .expect("Failed to start docker");
    
    assert!(output.status.success());
    
    // Wait for SSH to be ready
    tokio::time::sleep(Duration::from_secs(5)).await;
    
    // Create test config
    let config = HivesConfig {
        hives: vec![
            HiveConfig {
                alias: "test-host-1".to_string(),
                hostname: "localhost".to_string(),
                ssh_user: "testuser".to_string(),
                ssh_port: 2222,
                install_method: InstallMethod::Git {
                    repo: "git@github.com:veighnsche/llama-orch".to_string(),
                    branch: "main".to_string(),
                },
                workers: vec![],
                ..Default::default()
            }
        ]
    };
    
    // Run sync
    let result = sync_all_hives(config, SyncOptions::default(), "test-job").await;
    
    assert!(result.is_ok());
    
    // Verify installation
    let mut client = RbeeSSHClient::connect("localhost", 2222, "testuser").await.unwrap();
    let (stdout, _, exit_code) = client.exec("~/.local/bin/rbee-hive --version").await.unwrap();
    assert_eq!(exit_code, 0);
    assert!(stdout.contains("rbee-hive"));
    
    // Cleanup
    Command::new("docker-compose")
        .args(&["-f", "tests/docker/docker-compose.yml", "down"])
        .output()
        .expect("Failed to stop docker");
}
```

### Pros
- ✅ Tests real SSH
- ✅ Tests real git/cargo
- ✅ Reproducible environment
- ✅ Can test multiple hosts
- ✅ Isolated from host system

### Cons
- ❌ Slower (minutes for cargo build)
- ❌ Requires Docker
- ❌ More complex setup
- ❌ Larger CI resource usage

### Recommendation
**Use for:** Integration tests, CI pipeline, multi-host scenarios

---

## Option 3: Local SSH Loopback (Quick Integration)

### Approach
SSH to localhost (127.0.0.1) for quick integration testing without Docker.

### Implementation

```bash
# Setup SSH loopback (one-time)
ssh-keygen -t rsa -f ~/.ssh/rbee_test_key -N ""
cat ~/.ssh/rbee_test_key.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

# Test SSH works
ssh -i ~/.ssh/rbee_test_key localhost "echo 'SSH works'"
```

```toml
# tests/fixtures/localhost.conf

[[hive]]
alias = "localhost-test"
hostname = "127.0.0.1"
ssh_user = "vince"  # Your username
ssh_port = 22
install_method = { git = { repo = "git@github.com:veighnsche/llama-orch", branch = "main" } }
workers = [
    { type = "test", version = "latest", features = ["cpu"] }
]
```

```rust
// tests/integration/localhost_tests.rs

#[tokio::test]
#[ignore] // Run with: cargo test --ignored localhost
async fn test_install_localhost() {
    let config = HivesConfig::load_from("tests/fixtures/localhost.conf").unwrap();
    
    // Run sync
    let result = sync_all_hives(config, SyncOptions {
        dry_run: false,
        remove_extra: false,
        force: true,
    }, "test-job").await;
    
    assert!(result.is_ok());
    
    // Verify installation
    let hive_path = shellexpand::tilde("~/.local/bin/rbee-hive");
    assert!(std::path::Path::new(hive_path.as_ref()).exists());
}
```

### Pros
- ✅ Fast setup (no Docker)
- ✅ Tests real SSH/git/cargo
- ✅ Easy to debug
- ✅ Good for local development

### Cons
- ❌ Modifies local system
- ❌ Not isolated
- ❌ Cleanup required
- ❌ Can't test multiple hosts easily

### Recommendation
**Use for:** Local development, quick manual testing

---

## Option 4: GitHub Actions with Docker (CI)

### Approach
Use GitHub Actions with Docker containers for automated testing.

### Implementation

```yaml
# .github/workflows/test-package-manager.yml

name: Test Package Manager

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test-package-manager:
    runs-on: ubuntu-latest
    
    services:
      test-host:
        image: rust:1.75
        ports:
          - 2222:22
        options: >-
          --health-cmd "sshd -t"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          override: true
      
      - name: Setup SSH in container
        run: |
          docker exec test-host apt-get update
          docker exec test-host apt-get install -y openssh-server git
          docker exec test-host mkdir /var/run/sshd
          docker exec test-host useradd -m testuser
          docker exec test-host bash -c "echo 'testuser:testpass' | chpasswd"
          docker exec test-host service ssh start
      
      - name: Run integration tests
        run: cargo test --test integration_tests --ignored
        env:
          TEST_SSH_HOST: localhost
          TEST_SSH_PORT: 2222
          TEST_SSH_USER: testuser
          TEST_SSH_PASS: testpass
      
      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: target/test-results/
```

### Pros
- ✅ Automated on every push
- ✅ Reproducible
- ✅ Tests real SSH/git/cargo
- ✅ No manual intervention

### Cons
- ❌ Slow (5-10 minutes)
- ❌ Uses CI minutes
- ❌ Complex debugging

### Recommendation
**Use for:** CI/CD pipeline, PR validation

---

## Option 5: Vagrant VMs (Full E2E)

### Approach
Use Vagrant to spin up full VMs for end-to-end testing.

### Implementation

```ruby
# tests/vagrant/Vagrantfile

Vagrant.configure("2") do |config|
  # GPU host
  config.vm.define "gpu-host" do |gpu|
    gpu.vm.box = "ubuntu/jammy64"
    gpu.vm.hostname = "rbee-gpu-host"
    gpu.vm.network "private_network", ip: "192.168.56.10"
    
    gpu.vm.provider "virtualbox" do |vb|
      vb.memory = "4096"
      vb.cpus = 2
    end
    
    gpu.vm.provision "shell", inline: <<-SHELL
      apt-get update
      apt-get install -y git build-essential
      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    SHELL
  end
  
  # CPU host
  config.vm.define "cpu-host" do |cpu|
    cpu.vm.box = "ubuntu/jammy64"
    cpu.vm.hostname = "rbee-cpu-host"
    cpu.vm.network "private_network", ip: "192.168.56.11"
    
    cpu.vm.provider "virtualbox" do |vb|
      vb.memory = "2048"
      vb.cpus = 1
    end
    
    cpu.vm.provision "shell", inline: <<-SHELL
      apt-get update
      apt-get install -y git build-essential
      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    SHELL
  end
end
```

```toml
# tests/fixtures/vagrant.conf

[[hive]]
alias = "gpu-host"
hostname = "192.168.56.10"
ssh_user = "vagrant"
workers = [
    { type = "vllm", version = "latest", features = ["cuda"] }
]

[[hive]]
alias = "cpu-host"
hostname = "192.168.56.11"
ssh_user = "vagrant"
workers = [
    { type = "llama-cpp", version = "latest", features = ["cpu"] }
]
```

### Pros
- ✅ Full VMs (most realistic)
- ✅ Can test GPU passthrough
- ✅ Multiple hosts
- ✅ Persistent for debugging

### Cons
- ❌ Very slow (10+ minutes)
- ❌ Large resource usage
- ❌ Requires VirtualBox/VMware
- ❌ Complex setup

### Recommendation
**Use for:** Manual E2E testing, demo environments

---

## Recommended Testing Strategy

### Phase 1: Development (Fast Feedback)

**Unit Tests (Mock SSH)**
```bash
cargo test
```
- Test command generation
- Test error handling
- Test state transitions
- Run on every save

### Phase 2: Pre-Commit (Medium Confidence)

**Integration Tests (Docker)**
```bash
docker-compose -f tests/docker/docker-compose.yml up -d
cargo test --ignored
docker-compose -f tests/docker/docker-compose.yml down
```
- Test real SSH
- Test git clone + build
- Test concurrent installation
- Run before committing

### Phase 3: CI/CD (High Confidence)

**GitHub Actions**
- Run on every push
- Run on every PR
- Block merge if tests fail
- Cache cargo builds

### Phase 4: Manual E2E (Full Validation)

**Localhost or Vagrant**
```bash
# Create test config
cat > ~/.config/rbee/test.conf << EOF
[[hive]]
alias = "test"
hostname = "127.0.0.1"
ssh_user = "$(whoami)"
workers = [{ type = "test", version = "latest" }]
EOF

# Run sync
rbee sync --dry-run
rbee sync

# Verify
rbee package-status
```

---

## Test Scenarios to Cover

### 1. Happy Path
- ✅ Install hive successfully
- ✅ Install worker successfully
- ✅ Multiple hives concurrently
- ✅ Multiple workers concurrently

### 2. Error Handling
- ❌ Git clone fails (repo not found)
- ❌ Cargo build fails (compilation error)
- ❌ SSH connection fails (host unreachable)
- ❌ Permission denied (can't write to directory)
- ❌ Binary verification fails (--version fails)

### 3. Edge Cases
- 🔄 Re-install (force flag)
- 🔄 Upgrade (different branch/tag)
- 🔄 Partial failure (some hives succeed, some fail)
- 🔄 Network interruption during clone
- 🔄 Disk full during build

### 4. Feature Flags
- 🎯 Build with cuda features
- 🎯 Build with metal features
- 🎯 Build with cpu features
- 🎯 Build with multiple features

### 5. Install Methods
- 📦 Git clone + build
- 📦 GitHub release download
- 📦 Local binary path

---

## Test Utilities to Create

### 1. Test Harness

```rust
// tests/common/harness.rs

pub struct TestHarness {
    docker_compose: Option<Child>,
    temp_dir: TempDir,
    config: HivesConfig,
}

impl TestHarness {
    pub async fn new() -> Self {
        // Start docker containers
        // Create temp directory
        // Generate test config
    }
    
    pub async fn sync(&self) -> Result<SyncReport> {
        sync_all_hives(self.config.clone(), SyncOptions::default(), "test").await
    }
    
    pub async fn verify_hive_installed(&self, alias: &str) -> bool {
        // SSH and check binary exists
    }
}

impl Drop for TestHarness {
    fn drop(&mut self) {
        // Stop docker containers
        // Cleanup temp directory
    }
}
```

### 2. Assertion Helpers

```rust
// tests/common/assertions.rs

pub async fn assert_hive_installed(host: &str, port: u16, user: &str) {
    let mut client = RbeeSSHClient::connect(host, port, user).await.unwrap();
    let (stdout, _, exit_code) = client.exec("~/.local/bin/rbee-hive --version").await.unwrap();
    assert_eq!(exit_code, 0);
    assert!(stdout.contains("rbee-hive"));
}

pub async fn assert_worker_installed(host: &str, port: u16, user: &str, worker_type: &str) {
    let mut client = RbeeSSHClient::connect(host, port, user).await.unwrap();
    let binary = format!("~/.local/share/rbee/workers/rbee-worker-{}", worker_type);
    let (stdout, _, exit_code) = client.exec(&format!("{} --version", binary)).await.unwrap();
    assert_eq!(exit_code, 0);
}
```

### 3. Fixture Generator

```rust
// tests/common/fixtures.rs

pub fn create_test_config(hives: usize, workers_per_hive: usize) -> HivesConfig {
    let mut config = HivesConfig { hives: vec![] };
    
    for i in 0..hives {
        let mut hive = HiveConfig {
            alias: format!("test-hive-{}", i),
            hostname: "localhost".to_string(),
            ssh_port: 2222 + i as u16,
            ssh_user: "testuser".to_string(),
            workers: vec![],
            ..Default::default()
        };
        
        for j in 0..workers_per_hive {
            hive.workers.push(WorkerConfig {
                worker_type: format!("test-worker-{}", j),
                version: "latest".to_string(),
                features: vec!["cpu".to_string()],
                ..Default::default()
            });
        }
        
        config.hives.push(hive);
    }
    
    config
}
```

---

## Metrics to Track

### Performance
- Git clone time
- Cargo build time
- Total installation time
- Concurrent vs sequential speedup

### Reliability
- Success rate
- Failure modes
- Retry behavior
- Error recovery

### Coverage
- Code coverage %
- Scenario coverage
- Error path coverage

---

## Quick Start Commands

```bash
# Unit tests (fast)
cargo test

# Integration tests (Docker)
cd tests/docker
docker-compose up -d
cd ../..
cargo test --ignored
cd tests/docker
docker-compose down

# Localhost test (manual)
rbee sync --dry-run
rbee sync
rbee package-status

# CI test (GitHub Actions)
git push  # Triggers workflow
```

---

## Recommendations Summary

| Test Type | Tool | When | Speed | Confidence |
|-----------|------|------|-------|------------|
| Unit | Mock SSH | Every save | ⚡ Fast | 🟡 Low |
| Integration | Docker | Pre-commit | 🐢 Medium | 🟢 High |
| E2E | Localhost | Manual | 🐌 Slow | 🟢 High |
| CI | GitHub Actions | Every push | 🐢 Medium | 🟢 High |
| Demo | Vagrant | As needed | 🦥 Very Slow | 🟢 Very High |

**Best Practice:**
1. Write unit tests for all logic
2. Run Docker integration tests before committing
3. Use CI for automated validation
4. Manual E2E testing for major changes

---

## Next Steps

1. **Create test harness** - `tests/common/harness.rs`
2. **Add Docker setup** - `tests/docker/`
3. **Write integration tests** - `tests/integration/`
4. **Setup GitHub Actions** - `.github/workflows/`
5. **Document test commands** - Update README

---

**Testing is critical for reliable package management!**
