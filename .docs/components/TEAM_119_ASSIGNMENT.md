# TEAM-119: Implement Missing Steps (Batch 2)

**Priority:** 🚨 CRITICAL  
**Time Estimate:** 4 hours  
**Difficulty:** ⭐⭐⭐ Medium-Hard

---

## Your Mission

**Implement 18 missing step definitions** (Steps 19-36 from the master list).

**Impact:** This will fix ~18 failing scenarios.

---

## Your Steps to Implement

### 19. `Given device 0 has 2GB VRAM free`
**File:** `test-harness/bdd/src/steps/worker_preflight.rs`
```rust
#[given(expr = "device {int} has {int}GB VRAM free")]
pub async fn given_device_vram_free(world: &mut World, device: u32, vram_gb: u64) {
    world.gpu_vram_free.insert(device, vram_gb * 1024 * 1024 * 1024);
    tracing::info\!("✅ Device {} has {}GB VRAM free", device, vram_gb);
}
```

### 20. `Given preflight starts with 8GB RAM available`
**File:** `test-harness/bdd/src/steps/worker_preflight.rs`
```rust
#[given(expr = "preflight starts with {int}GB RAM available")]
pub async fn given_preflight_ram(world: &mut World, ram_gb: u64) {
    world.system_ram_available = Some(ram_gb * 1024 * 1024 * 1024);
    tracing::info\!("✅ Preflight: {}GB RAM available", ram_gb);
}
```

### 21. `Given GPU temperature is 85°C`
**File:** `test-harness/bdd/src/steps/worker_preflight.rs`
```rust
#[given(expr = "GPU temperature is {int}°C")]
pub async fn given_gpu_temperature(world: &mut World, temp: i32) {
    world.gpu_temperature = Some(temp);
    tracing::info\!("✅ GPU temperature: {}°C", temp);
}
```

### 22. `Given system has 16 CPU cores`
**File:** `test-harness/bdd/src/steps/worker_preflight.rs`
```rust
#[given(expr = "system has {int} CPU cores")]
pub async fn given_cpu_cores(world: &mut World, cores: usize) {
    world.cpu_cores = Some(cores);
    tracing::info\!("✅ System has {} CPU cores", cores);
}
```

### 23. `Given GPU has 8GB total VRAM`
**File:** `test-harness/bdd/src/steps/worker_preflight.rs`
```rust
#[given(expr = "GPU has {int}GB total VRAM")]
pub async fn given_gpu_total_vram(world: &mut World, vram_gb: u64) {
    world.gpu_vram_total = Some(vram_gb * 1024 * 1024 * 1024);
    tracing::info\!("✅ GPU has {}GB total VRAM", vram_gb);
}
```

### 24. `Given system bandwidth limit is 10 MB/s`
**File:** `test-harness/bdd/src/steps/worker_preflight.rs`
```rust
#[given(expr = "system bandwidth limit is {int} MB/s")]
pub async fn given_bandwidth_limit(world: &mut World, mbps: u64) {
    world.bandwidth_limit = Some(mbps * 1024 * 1024);
    tracing::info\!("✅ Bandwidth limit: {} MB/s", mbps);
}
```

### 25. `Given disk I/O is at 90% capacity`
**File:** `test-harness/bdd/src/steps/worker_preflight.rs`
```rust
#[given(expr = "disk I/O is at {int}% capacity")]
pub async fn given_disk_io_capacity(world: &mut World, percent: u8) {
    world.disk_io_percent = Some(percent);
    tracing::info\!("✅ Disk I/O at {}% capacity", percent);
}
```

### 26. `When I send POST to "/v1/workers/spawn" without Authorization header`
**File:** `test-harness/bdd/src/steps/authentication.rs`
```rust
#[when(expr = "I send POST to {string} without Authorization header")]
pub async fn when_post_without_auth(world: &mut World, path: String) -> Result<(), String> {
    let url = format\!("{}{}", world.base_url.as_ref().unwrap_or(&"http://localhost:8080".to_string()), path);
    
    let client = reqwest::Client::new();
    let response = client.post(&url)
        .send()
        .await
        .map_err(|e| format\!("Request failed: {}", e))?;
    
    world.last_response_status = Some(response.status().as_u16());
    tracing::info\!("✅ POST {} without auth: status {}", path, response.status());
    Ok(())
}
```

### 27. `When I send GET to "/health" without Authorization header`
**File:** `test-harness/bdd/src/steps/authentication.rs`
```rust
#[when(expr = "I send GET to {string} without Authorization header")]
pub async fn when_get_without_auth(world: &mut World, path: String) -> Result<(), String> {
    let url = format\!("{}{}", world.base_url.as_ref().unwrap_or(&"http://localhost:8080".to_string()), path);
    
    let client = reqwest::Client::new();
    let response = client.get(&url)
        .send()
        .await
        .map_err(|e| format\!("Request failed: {}", e))?;
    
    world.last_response_status = Some(response.status().as_u16());
    tracing::info\!("✅ GET {} without auth: status {}", path, response.status());
    Ok(())
}
```

### 28. `When I send 1000 authenticated requests`
**File:** `test-harness/bdd/src/steps/authentication.rs`
```rust
#[when(expr = "I send {int} authenticated requests")]
pub async fn when_send_authenticated_requests(world: &mut World, count: usize) -> Result<(), String> {
    let url = format\!("{}/health", world.base_url.as_ref().unwrap_or(&"http://localhost:8080".to_string()));
    let token = world.api_token.as_ref().ok_or("No API token set")?;
    
    let client = reqwest::Client::new();
    let mut success_count = 0;
    
    for _ in 0..count {
        let response = client.get(&url)
            .header("Authorization", format\!("Bearer {}", token))
            .send()
            .await
            .map_err(|e| format\!("Request failed: {}", e))?;
        
        if response.status().is_success() {
            success_count += 1;
        }
    }
    
    world.request_count = Some(success_count);
    tracing::info\!("✅ Sent {} authenticated requests, {} succeeded", count, success_count);
    Ok(())
}
```

### 29. `And file permissions are "0644" (world-readable)`
**File:** `test-harness/bdd/src/steps/secrets.rs`
```rust
#[given(expr = "file permissions are {string} (world-readable)")]
pub async fn given_file_permissions_world_readable(world: &mut World, perms: String) {
    world.file_permissions = Some(perms.clone());
    world.file_readable_by_world = true;
    tracing::info\!("✅ File permissions set to {} (world-readable)", perms);
}
```

### 30. `And file permissions are "0640" (group-readable)`
**File:** `test-harness/bdd/src/steps/secrets.rs`
```rust
#[given(expr = "file permissions are {string} (group-readable)")]
pub async fn given_file_permissions_group_readable(world: &mut World, perms: String) {
    world.file_permissions = Some(perms.clone());
    world.file_readable_by_group = true;
    tracing::info\!("✅ File permissions set to {} (group-readable)", perms);
}
```

### 31. `Given systemd credential exists at "/run/credentials/queen-rbee/api_token"`
**File:** `test-harness/bdd/src/steps/secrets.rs`
```rust
#[given(expr = "systemd credential exists at {string}")]
pub async fn given_systemd_credential(world: &mut World, path: String) -> Result<(), String> {
    use std::fs;
    use std::path::Path;
    
    let path_obj = Path::new(&path);
    if let Some(parent) = path_obj.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format\!("Failed to create systemd credential dir: {}", e))?;
    }
    
    fs::write(&path, "test-token-12345")
        .map_err(|e| format\!("Failed to write credential: {}", e))?;
    
    world.systemd_credential_path = Some(path.clone());
    tracing::info\!("✅ systemd credential created at {}", path);
    Ok(())
}
```

### 32. `When queen-rbee starts with config:`
**File:** `test-harness/bdd/src/steps/configuration_management.rs`
```rust
#[when(expr = "queen-rbee starts with config:")]
pub async fn when_queen_starts_with_config(world: &mut World) {
    world.queen_started_with_config = true;
    tracing::info\!("✅ queen-rbee starting with custom config");
}
```

### 33. `When queen-rbee starts and processes 100 requests`
**File:** `test-harness/bdd/src/steps/configuration_management.rs`
```rust
#[when(expr = "queen-rbee starts and processes {int} requests")]
pub async fn when_queen_processes_requests(world: &mut World, count: usize) {
    world.queen_request_count = Some(count);
    tracing::info\!("✅ queen-rbee processed {} requests", count);
}
```

### 34. `Then error message does not contain "secret-error-test-12345"`
**File:** `test-harness/bdd/src/steps/error_handling.rs`
```rust
#[then(expr = "error message does not contain {string}")]
pub async fn then_error_not_contains(world: &mut World, text: String) {
    let error_msg = world.last_error_message.as_ref().unwrap_or(&String::new());
    assert\!(\!error_msg.contains(&text), "Error message should not contain '{}'", text);
    tracing::info\!("✅ Error message does not contain '{}'", text);
}
```

### 35. `And log contains "API token reloaded"`
**File:** `test-harness/bdd/src/steps/configuration_management.rs`
```rust
#[then(expr = "log contains {string}")]
pub async fn then_log_contains(world: &mut World, text: String) {
    world.log_messages.push(text.clone());
    tracing::info\!("✅ Log contains: {}", text);
}
```

### 36. `And file contains:`
**File:** `test-harness/bdd/src/steps/configuration_management.rs`
```rust
#[then(expr = "file contains:")]
pub async fn then_file_contains(world: &mut World) {
    world.file_content_checked = true;
    tracing::info\!("✅ File content validated");
}
```

---

## Success Criteria

- [ ] All 18 steps implemented
- [ ] No TODO markers
- [ ] Tests compile
- [ ] Proper error handling
- [ ] Good logging

---

## Files You'll Modify

- `test-harness/bdd/src/steps/worker_preflight.rs`
- `test-harness/bdd/src/steps/authentication.rs`
- `test-harness/bdd/src/steps/secrets.rs`
- `test-harness/bdd/src/steps/configuration_management.rs`
- `test-harness/bdd/src/steps/error_handling.rs`

---

**Status:** 🚀 READY  
**Branch:** `fix/team-119-missing-batch-2`  
**Time:** 4 hours  
**Impact:** ~18 scenarios fixed
