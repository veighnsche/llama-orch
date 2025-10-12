# Dev Server Lifecycle Management

**Created by:** TEAM-FE-011 (aka TEAM-DX-000)

## Overview

The DX tool must properly manage dev server lifecycle to ensure reliable testing without manual intervention.

## Problem Statement

Frontend engineers need to:
- Start dev server automatically
- Wait for server to be ready
- Run verification commands
- Clean up server process
- Handle server crashes/hangs

**Without proper lifecycle management, tests are flaky and unreliable.**

## Core Features

### 1. Server Detection

Check if server is already running before starting a new one.

```bash
dx server --check http://localhost:3000
```

**Output:**
```
✓ Server is running at http://localhost:3000
  Response time: 45ms
  Status: 200 OK
```

### 2. Server Start

Start dev server and wait for readiness.

```bash
dx server --start --wait --cwd frontend/bin/commercial
```

**Implementation:**
```rust
pub async fn start_server(cwd: &Path, port: u16) -> Result<ServerHandle> {
    // 1. Check if already running
    if is_server_running(port).await? {
        return Ok(ServerHandle::existing(port));
    }
    
    // 2. Start process
    let mut child = Command::new("pnpm")
        .arg("dev")
        .current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    
    // 3. Wait for readiness
    wait_for_server(port, Duration::from_secs(30)).await?;
    
    Ok(ServerHandle::new(child, port))
}

pub async fn wait_for_server(port: u16, timeout: Duration) -> Result<()> {
    let start = Instant::now();
    let url = format!("http://localhost:{}", port);
    
    loop {
        if start.elapsed() > timeout {
            return Err(Error::ServerTimeout);
        }
        
        match reqwest::get(&url).await {
            Ok(response) if response.status().is_success() => {
                return Ok(());
            }
            _ => {
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }
    }
}
```

### 3. Server Stop

Gracefully stop server process.

```bash
dx server --stop
```

**Implementation:**
```rust
pub struct ServerHandle {
    child: Child,
    port: u16,
}

impl ServerHandle {
    pub async fn stop(mut self) -> Result<()> {
        // Send SIGTERM
        self.child.kill()?;
        
        // Wait for process to exit
        tokio::time::timeout(
            Duration::from_secs(5),
            self.child.wait()
        ).await??;
        
        Ok(())
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        // Ensure cleanup on panic/error
        let _ = self.child.kill();
    }
}
```

### 4. Integrated Testing

Run commands with automatic server lifecycle.

```bash
dx test --start-server --cwd frontend/bin/commercial \
  css --class-exists "cursor-pointer" http://localhost:3000
```

**Flow:**
1. Check if server running
2. If not, start server and wait
3. Run verification command
4. Keep server running (for subsequent commands)
5. Stop server on explicit `--stop` or process exit

### 5. Health Checks

Verify server is responding correctly.

```bash
dx server --health http://localhost:3000
```

**Output:**
```
✓ Server health check
  URL: http://localhost:3000
  Status: 200 OK
  Response time: 42ms
  Content-Type: text/html
  
  Checks:
    ✓ Server responding
    ✓ HTML content returned
    ✓ No error pages (404, 500)
    ✓ Response time < 1s
```

## Advanced Features

### 6. Port Management

Find available port if default is taken.

```rust
pub async fn find_available_port(start: u16) -> Result<u16> {
    for port in start..start + 100 {
        if !is_port_in_use(port).await? {
            return Ok(port);
        }
    }
    Err(Error::NoAvailablePort)
}
```

### 7. Log Capture

Capture server logs for debugging.

```bash
dx server --start --log-file server.log
```

**Implementation:**
```rust
pub async fn start_with_logging(cwd: &Path, log_path: &Path) -> Result<ServerHandle> {
    let log_file = File::create(log_path)?;
    
    let mut child = Command::new("pnpm")
        .arg("dev")
        .current_dir(cwd)
        .stdout(Stdio::from(log_file.try_clone()?))
        .stderr(Stdio::from(log_file))
        .spawn()?;
    
    // ... wait for readiness
}
```

### 8. Crash Detection

Detect if server crashes during testing.

```rust
pub async fn monitor_server(handle: &mut ServerHandle) -> Result<()> {
    match handle.child.try_wait()? {
        Some(status) => {
            Err(Error::ServerCrashed(status))
        }
        None => Ok(())
    }
}
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Frontend Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install dependencies
        run: pnpm install
      
      - name: Build DX tool
        run: cargo build --release --manifest-path frontend/.dx-tool/Cargo.toml
      
      - name: Run frontend tests
        run: |
          cd frontend/.dx-tool
          cargo run --release -- test \
            --start-server \
            --cwd ../bin/commercial \
            --timeout 60 \
            css --class-exists "cursor-pointer" http://localhost:3000
```

### pnpm Scripts

```json
{
  "scripts": {
    "dx:test": "dx test --start-server --cwd . css --class-exists cursor-pointer http://localhost:3000",
    "dx:verify": "dx test --start-server --cwd . snapshot --compare --name homepage http://localhost:3000",
    "dx:server": "dx server --start --wait",
    "dx:stop": "dx server --stop"
  }
}
```

## Configuration

`.dxrc.json`:
```json
{
  "server": {
    "command": "pnpm dev",
    "cwd": "frontend/bin/commercial",
    "port": 3000,
    "startupTimeout": 30,
    "healthCheckInterval": 500,
    "logFile": ".dx-server.log"
  }
}
```

## Error Handling

### Server Won't Start

```
✗ Error: Server failed to start
  Command: pnpm dev
  Working directory: frontend/bin/commercial
  
  Possible causes:
    - Port 3000 already in use
    - Dependencies not installed (run: pnpm install)
    - Build errors in code
  
  Check logs: .dx-server.log
```

### Server Timeout

```
✗ Error: Server startup timeout (30s)
  Server process is running but not responding
  
  Suggestions:
    - Increase timeout: --timeout 60
    - Check server logs: .dx-server.log
    - Verify port is correct: --port 3000
```

### Server Crashed

```
✗ Error: Server crashed during test
  Exit code: 1
  
  Last 10 lines of log:
    [error] Failed to compile
    [error] Module not found: 'cursor-pointer'
    ...
  
  Full logs: .dx-server.log
```

## Testing Strategy

```rust
#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn test_server_start_stop() {
        let handle = start_server(Path::new("test-app"), 3000).await.unwrap();
        assert!(is_server_running(3000).await.unwrap());
        
        handle.stop().await.unwrap();
        assert!(!is_server_running(3000).await.unwrap());
    }
    
    #[tokio::test]
    async fn test_wait_for_readiness() {
        let handle = start_server(Path::new("test-app"), 3000).await.unwrap();
        
        // Server should be ready immediately after start
        let result = reqwest::get("http://localhost:3000").await;
        assert!(result.is_ok());
        
        handle.stop().await.unwrap();
    }
}
```

## Implementation Priority

**Phase 2 (Week 2):**
- [ ] Server detection (`is_server_running`)
- [ ] Basic start/stop
- [ ] Wait for readiness
- [ ] Graceful cleanup

**Phase 3 (Week 3):**
- [ ] Health checks
- [ ] Log capture
- [ ] Crash detection
- [ ] Port management

**Phase 4 (Week 4):**
- [ ] Configuration file support
- [ ] CI/CD integration examples
- [ ] Comprehensive error messages

---

**This is critical for reliable automated testing. Implement carefully.**
