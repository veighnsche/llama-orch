# Docker SSH Tests

**Date:** Oct 24, 2025  
**Status:** ✅ COMPLETE

---

## SSH Tests (9 tests)

**File:** `xtask/tests/docker/ssh_tests.rs`

Tests SSH communication between queen-rbee and rbee-hive using `RbeeSSHClient` (russh library).

### Tests
1. `test_ssh_connection_to_hive` - SSH connection
2. `test_ssh_authentication` - SSH authentication
3. `test_ssh_command_execution` - Execute commands via SSH
4. `test_ssh_binary_check` - Check rbee-hive binary via SSH
5. `test_ssh_file_operations` - File operations via SSH
6. `test_ssh_concurrent_connections` - 5 concurrent SSH connections
7. `test_ssh_connection_timeout` - SSH timeout when server down
8. `test_ssh_environment_variables` - Environment variables via SSH

---

## How to Run

```bash
# Build everything
./tests/docker/scripts/build-all.sh

# Start environment
./tests/docker/scripts/start.sh

# Run SSH tests
cargo test --package xtask --test docker_ssh_tests --ignored -- --nocapture

# Stop environment
./tests/docker/scripts/stop.sh
```

---

## What These Test

- SSH connections to Docker containers
- SSH authentication with keys
- SSH command execution
- SSH file operations
- Concurrent SSH connections
- SSH failure scenarios

---

## Architecture

```
Test Process → SSH (localhost:2222) → Docker Container (port 22)
```

- Tests connect via SSH to `localhost:2222`
- Docker maps port `2222` → container's port `22`
- Uses `RbeeSSHClient` (russh library)
- Tests actual SSH protocol communication
