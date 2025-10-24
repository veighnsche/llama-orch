# Docker Network Testing Infrastructure

**Purpose:** Comprehensive Docker-based testing for queen-rbee → rbee-hive communication

---

## Quick Start

### 1. Build Everything (First Time)
```bash
./tests/docker/scripts/build-all.sh
```

This will:
- Build `queen-rbee` and `rbee-hive` binaries
- Generate SSH keys
- Build Docker images

### 2. Start Environment
```bash
./tests/docker/scripts/start.sh
```

Services will be available at:
- **Queen:** http://localhost:8500
- **Hive:** http://localhost:9000
- **SSH:** `ssh -i tests/docker/keys/test_id_rsa -p 2222 rbee@localhost`

### 3. Run Tests
```bash
# Run all tests
./tests/docker/scripts/test-all.sh

# Or run specific test categories
cargo test --package xtask --test docker_smoke_test --ignored
cargo test --package xtask --test http_communication_tests --ignored
cargo test --package xtask --test ssh_communication_tests --ignored
cargo test --package xtask --test failure_tests --ignored
```

### 4. Stop Environment
```bash
./tests/docker/scripts/stop.sh
```

---

## Test Categories

### Smoke Tests (`docker_smoke_test.rs`)
Basic connectivity and health checks:
- Queen health check
- Hive health check
- Hive capabilities
- SSH connection
- Binary existence

### HTTP Communication Tests (`http_communication_tests.rs`)
HTTP communication patterns:
- Health checks
- Capabilities discovery
- Connection timeouts
- Connection refused scenarios
- Concurrent requests
- Large responses

### SSH Communication Tests (`ssh_communication_tests.rs`)
SSH operations:
- Connection establishment
- Command execution
- Binary checks
- File operations
- Concurrent connections
- Environment variables

### Failure Tests (`failure_tests.rs`)
Failure scenarios and recovery:
- Hive crash during operation
- Hive restart recovery
- Queen restart recovery
- Concurrent operations
- Rapid restart cycles
- Service logs after failure

---

## Architecture

### Network Topology
```
Docker Network: rbee-test-net (172.20.0.0/16)
├─ queen-rbee (172.20.0.10:8500)
└─ rbee-hive (172.20.0.20:9000, :22)
```

### Container Details

**Base Image (`rbee-base:latest`)**
- Rust 1.75 slim
- SSH server
- Git + build tools
- Test user: `rbee:rbee`
- SSH keys configured

**Queen Image (`rbee-queen:latest`)**
- Based on rbee-base
- Pre-built queen-rbee binary
- Default hives.conf
- Exposed port: 8500

**Hive Image (`rbee-hive:latest`)**
- Based on rbee-base
- Pre-built rbee-hive binary
- Supervisor (runs SSH + hive)
- Exposed ports: 22, 9000

---

## Directory Structure

```
tests/docker/
├── Dockerfile.base          # Base image with Rust + SSH
├── Dockerfile.queen         # Queen image
├── Dockerfile.hive          # Hive image
├── docker-compose.localhost.yml
├── docker-compose.multi-hive.yml
├── keys/
│   ├── test_id_rsa         # SSH private key
│   └── test_id_rsa.pub     # SSH public key
├── configs/
│   ├── hives.conf          # Hive configuration
│   └── supervisord.conf    # Supervisor config for hive
└── scripts/
    ├── generate-keys.sh    # Generate SSH keys
    ├── build-all.sh        # Build everything
    ├── start.sh            # Start environment
    ├── stop.sh             # Stop environment
    ├── test-all.sh         # Run all tests
    └── cleanup.sh          # Clean up Docker resources
```

---

## Manual Testing

### Check Service Health
```bash
# Queen
curl http://localhost:8500/health

# Hive
curl http://localhost:9000/health

# Hive capabilities
curl http://localhost:9000/capabilities | jq
```

### SSH into Hive
```bash
ssh -i tests/docker/keys/test_id_rsa -p 2222 rbee@localhost
```

### View Logs
```bash
# Queen logs
docker logs rbee-queen-localhost

# Hive logs
docker logs rbee-hive-localhost

# Follow logs
docker logs -f rbee-hive-localhost
```

### Execute Commands in Containers
```bash
# Execute in hive
docker exec rbee-hive-localhost ls -la /home/rbee/.local/bin

# Execute in queen
docker exec rbee-queen-localhost ps aux
```

---

## Troubleshooting

### Containers won't start
```bash
# Check logs
docker-compose -f tests/docker/docker-compose.localhost.yml logs

# Rebuild images
./tests/docker/scripts/cleanup.sh
./tests/docker/scripts/build-all.sh
```

### Port conflicts
```bash
# Check what's using ports
sudo lsof -i :8500
sudo lsof -i :9000

# Kill processes or change ports in docker-compose.yml
```

### SSH connection fails
```bash
# Check SSH is running
docker exec rbee-hive-localhost ps aux | grep sshd

# Restart hive
docker restart rbee-hive-localhost
```

### Tests fail
```bash
# Ensure containers are running
docker ps

# Check service health
curl http://localhost:8500/health
curl http://localhost:9000/health

# View test output with details
cargo test --package xtask --test docker_smoke_test --ignored -- --nocapture
```

---

## Development Workflow

### After Code Changes
```bash
# 1. Rebuild binaries
cargo build --bin queen-rbee --bin rbee-hive

# 2. Rebuild Docker images
docker build -f tests/docker/Dockerfile.queen -t rbee-queen:latest .
docker build -f tests/docker/Dockerfile.hive -t rbee-hive:latest .

# 3. Restart environment
./tests/docker/scripts/stop.sh
./tests/docker/scripts/start.sh

# 4. Run tests
cargo test --package xtask --test docker_smoke_test --ignored
```

### Adding New Tests
1. Create test file in `xtask/tests/docker/`
2. Use `DockerTestHarness` for setup/teardown
3. Mark tests with `#[ignore]` (Docker tests are opt-in)
4. Run with `cargo test --package xtask --test <test_name> --ignored`

---

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Docker Tests
on: [push, pull_request]
jobs:
  docker-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
      - run: ./tests/docker/scripts/build-all.sh
      - run: ./tests/docker/scripts/test-all.sh
```

---

## Test Coverage

| Category | Tests | Status |
|----------|-------|--------|
| Smoke Tests | 6 | ✅ Implemented |
| HTTP Communication | 6 | ✅ Implemented |
| SSH Communication | 6 | ✅ Implemented |
| Failure Scenarios | 6 | ✅ Implemented |
| **Total** | **24** | **✅ Ready** |

---

## Next Steps

1. ✅ Basic infrastructure complete
2. ✅ Smoke tests implemented
3. ✅ HTTP communication tests implemented
4. ✅ SSH communication tests implemented
5. ✅ Failure scenario tests implemented
6. 🔄 Add heartbeat tests
7. 🔄 Add lifecycle tests (hive start/stop)
8. 🔄 Add worker lifecycle tests
9. 🔄 Add E2E workflow tests
10. 🔄 Add multi-hive topology tests

---

## Resources

- **Planning Docs:**
  - `.docs/DOCKER_NETWORK_TESTING_PLAN.md` - Comprehensive test strategy
  - `.docs/DOCKER_TEST_IMPLEMENTATION_GUIDE.md` - Detailed implementation guide
  - `.docs/DOCKER_TEST_QUICK_START.md` - 15-minute quick start

- **Related Tests:**
  - `xtask/src/integration/` - Integration test harness
  - `xtask/src/chaos/` - Chaos testing
  - `bin/15_queen_rbee_crates/ssh-client/tests/` - SSH client tests
