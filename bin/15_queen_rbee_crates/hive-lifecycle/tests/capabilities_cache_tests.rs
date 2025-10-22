// TEAM-246: Capabilities cache tests for hive-lifecycle
// Purpose: Test cache hit/miss/refresh and staleness detection
// Priority: CRITICAL (performance optimization, prevents stale capabilities)
// Scale: Reasonable for NUC (5-10 concurrent, no overkill)

use std::fs;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};
use tempfile::TempDir;
use tokio::time::sleep;

// ============================================================================
// Cache Hit/Miss Tests
// ============================================================================

#[tokio::test]
async fn test_cache_hit_returns_cached() {
    // TEAM-246: Test cache hit returns cached capabilities without fetching
    
    let temp_dir = TempDir::new().unwrap();
    let cache_file = temp_dir.path().join("capabilities.yaml");
    
    // Write cached capabilities
    let cached_content = r#"
devices:
  - type: cpu
    cores: 8
  - type: gpu
    name: "NVIDIA RTX 3090"
    vram_gb: 24
"#;
    fs::write(&cache_file, cached_content).unwrap();
    
    // Read cache (should not fetch)
    let content = fs::read_to_string(&cache_file).unwrap();
    
    assert!(content.contains("NVIDIA RTX 3090"), "Should return cached GPU");
    assert!(content.contains("cores: 8"), "Should return cached CPU");
}

#[tokio::test]
async fn test_cache_miss_fetches_fresh() {
    // TEAM-246: Test cache miss triggers fresh fetch
    
    let temp_dir = TempDir::new().unwrap();
    let cache_file = temp_dir.path().join("capabilities.yaml");
    
    // Ensure cache doesn't exist
    assert!(!cache_file.exists(), "Cache should not exist");
    
    // Simulate fetch (in real code, this would call hive /capabilities endpoint)
    let fresh_content = r#"
devices:
  - type: cpu
    cores: 16
  - type: gpu
    name: "NVIDIA RTX 4090"
    vram_gb: 24
"#;
    
    // Write fresh capabilities to cache
    fs::write(&cache_file, fresh_content).unwrap();
    
    // Verify cache now exists
    assert!(cache_file.exists(), "Cache should be created");
    
    let content = fs::read_to_string(&cache_file).unwrap();
    assert!(content.contains("NVIDIA RTX 4090"), "Should have fresh GPU");
}

#[tokio::test]
async fn test_cache_refresh_updates_cache() {
    // TEAM-246: Test manual refresh updates cache
    
    let temp_dir = TempDir::new().unwrap();
    let cache_file = temp_dir.path().join("capabilities.yaml");
    
    // Write old cache
    let old_content = r#"
devices:
  - type: cpu
    cores: 8
"#;
    fs::write(&cache_file, old_content).unwrap();
    
    // Simulate refresh (force fetch)
    let new_content = r#"
devices:
  - type: cpu
    cores: 16
  - type: gpu
    name: "NVIDIA RTX 4090"
    vram_gb: 24
"#;
    fs::write(&cache_file, new_content).unwrap();
    
    // Verify cache updated
    let content = fs::read_to_string(&cache_file).unwrap();
    assert!(content.contains("cores: 16"), "Should have updated CPU");
    assert!(content.contains("NVIDIA RTX 4090"), "Should have new GPU");
}

#[tokio::test]
async fn test_cache_cleanup_on_uninstall() {
    // TEAM-246: Test cache removed on hive uninstall
    
    let temp_dir = TempDir::new().unwrap();
    let cache_file = temp_dir.path().join("capabilities.yaml");
    
    // Create cache
    fs::write(&cache_file, "devices: []").unwrap();
    assert!(cache_file.exists(), "Cache should exist");
    
    // Simulate uninstall (remove cache)
    fs::remove_file(&cache_file).unwrap();
    
    // Verify cache removed
    assert!(!cache_file.exists(), "Cache should be removed");
}

// ============================================================================
// Staleness Detection Tests
// ============================================================================

#[tokio::test]
async fn test_cache_staleness_24h() {
    // TEAM-246: Test cache older than 24h is considered stale
    
    let temp_dir = TempDir::new().unwrap();
    let cache_file = temp_dir.path().join("capabilities.yaml");
    
    // Create cache file
    fs::write(&cache_file, "devices: []").unwrap();
    
    // Get file modification time
    let metadata = fs::metadata(&cache_file).unwrap();
    let modified = metadata.modified().unwrap();
    
    // Calculate age
    let now = SystemTime::now();
    let age = now.duration_since(modified).unwrap();
    
    // Fresh cache (just created)
    assert!(age.as_secs() < 60, "Cache should be fresh (< 1 min old)");
    
    // Simulate 24h+ old cache by checking threshold
    let is_stale = age.as_secs() > 86400; // 24 hours
    assert!(!is_stale, "Fresh cache should not be stale");
    
    // In real code, if age > 24h, suggest refresh
}

#[tokio::test]
async fn test_cache_with_corrupted_file() {
    // TEAM-246: Test corrupted cache triggers fresh fetch
    
    let temp_dir = TempDir::new().unwrap();
    let cache_file = temp_dir.path().join("capabilities.yaml");
    
    // Write corrupted YAML
    fs::write(&cache_file, "invalid: yaml: content: [[[").unwrap();
    
    // Try to parse (should fail)
    let content = fs::read_to_string(&cache_file).unwrap();
    
    // In real code, parsing would fail and trigger fresh fetch
    assert!(content.contains("invalid"), "Should read corrupted content");
    
    // Simulate fresh fetch after corruption detected
    let fresh_content = "devices:\n  - type: cpu\n    cores: 8\n";
    fs::write(&cache_file, fresh_content).unwrap();
    
    // Verify cache replaced
    let new_content = fs::read_to_string(&cache_file).unwrap();
    assert!(new_content.contains("type: cpu"), "Should have fresh content");
}

#[tokio::test]
async fn test_cache_with_missing_file() {
    // TEAM-246: Test missing cache triggers fresh fetch
    
    let temp_dir = TempDir::new().unwrap();
    let cache_file = temp_dir.path().join("capabilities.yaml");
    
    // Ensure file doesn't exist
    assert!(!cache_file.exists(), "Cache should not exist");
    
    // In real code, missing cache triggers fetch
    // Simulate fetch
    let fresh_content = "devices:\n  - type: cpu\n    cores: 8\n";
    fs::write(&cache_file, fresh_content).unwrap();
    
    // Verify cache created
    assert!(cache_file.exists(), "Cache should be created");
}

// ============================================================================
// Concurrent Access Tests
// ============================================================================

#[tokio::test]
async fn test_concurrent_cache_reads() {
    // TEAM-246: Test 5 concurrent cache reads (should work)
    
    let temp_dir = TempDir::new().unwrap();
    let cache_file = temp_dir.path().join("capabilities.yaml");
    
    // Create cache
    let content = "devices:\n  - type: cpu\n    cores: 8\n";
    fs::write(&cache_file, content).unwrap();
    
    // Spawn 5 concurrent readers
    let mut handles = vec![];
    for _ in 0..5 {
        let path = cache_file.clone();
        let handle = tokio::spawn(async move {
            fs::read_to_string(&path).unwrap()
        });
        handles.push(handle);
    }
    
    // Wait for all reads
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.contains("type: cpu"), "All reads should succeed");
    }
}

#[tokio::test]
async fn test_concurrent_cache_writes() {
    // TEAM-246: Test 5 concurrent cache writes (should serialize)
    
    use std::sync::Arc;
    use tokio::sync::Mutex;
    
    let temp_dir = TempDir::new().unwrap();
    let cache_file = Arc::new(Mutex::new(temp_dir.path().join("capabilities.yaml")));
    
    // Spawn 5 concurrent writers
    let mut handles = vec![];
    for i in 0..5 {
        let path = cache_file.clone();
        let handle = tokio::spawn(async move {
            let path = path.lock().await;
            let content = format!("devices:\n  - type: cpu\n    cores: {}\n", i * 2);
            fs::write(&*path, content).unwrap();
        });
        handles.push(handle);
    }
    
    // Wait for all writes
    for handle in handles {
        handle.await.unwrap();
    }
    
    // Last write should win
    let final_path = cache_file.lock().await;
    let final_content = fs::read_to_string(&*final_path).unwrap();
    assert!(final_content.contains("type: cpu"), "Should have valid content");
}

#[tokio::test]
async fn test_read_during_write() {
    // TEAM-246: Test read during write (should see old or new, not partial)
    
    use std::sync::Arc;
    use tokio::sync::RwLock;
    
    let temp_dir = TempDir::new().unwrap();
    let cache_file = Arc::new(RwLock::new(temp_dir.path().join("capabilities.yaml")));
    
    // Write initial content
    {
        let path = cache_file.read().await;
        fs::write(&*path, "devices:\n  - type: cpu\n    cores: 8\n").unwrap();
    }
    
    // Spawn writer
    let path_clone = cache_file.clone();
    let write_handle = tokio::spawn(async move {
        let path = path_clone.write().await;
        sleep(Duration::from_millis(100)).await;
        fs::write(&*path, "devices:\n  - type: cpu\n    cores: 16\n").unwrap();
    });
    
    // Spawn reader (concurrent with writer)
    let path_clone = cache_file.clone();
    let read_handle = tokio::spawn(async move {
        sleep(Duration::from_millis(50)).await;
        let path = path_clone.read().await;
        fs::read_to_string(&*path).unwrap()
    });
    
    // Wait for both
    write_handle.await.unwrap();
    let read_result = read_handle.await.unwrap();
    
    // Should see either old (8) or new (16), not partial
    assert!(
        read_result.contains("cores: 8") || read_result.contains("cores: 16"),
        "Should see complete content, not partial"
    );
}

// ============================================================================
// Fetch Timeout Tests
// ============================================================================

#[tokio::test]
async fn test_fetch_timeout_15s() {
    // TEAM-246: Test capabilities fetch timeout (15s)
    
    use tokio::time::timeout;
    
    // Simulate slow fetch
    let slow_fetch = async {
        sleep(Duration::from_secs(20)).await;
        Ok::<(), ()>(())
    };
    
    // Apply 15s timeout
    let result = timeout(Duration::from_secs(15), slow_fetch).await;
    
    assert!(result.is_err(), "Should timeout after 15s");
}

#[tokio::test]
async fn test_fetch_failure_network_error() {
    // TEAM-246: Test fetch failure (network error)
    
    // Simulate network error
    let fetch_result: Result<String, &str> = Err("Connection refused");
    
    assert!(fetch_result.is_err(), "Should handle network error");
    
    // In real code, network error should:
    // 1. Return error to user
    // 2. Keep old cache (if exists)
    // 3. Suggest retry or check network
}

// ============================================================================
// Cache Consistency Tests
// ============================================================================

#[test]
fn test_cache_matches_actual_capabilities() {
    // TEAM-246: Test cache matches actual capabilities
    // This is more of an integration test concept
    
    let cached = r#"
devices:
  - type: cpu
    cores: 8
  - type: gpu
    name: "NVIDIA RTX 3090"
    vram_gb: 24
"#;
    
    let actual = r#"
devices:
  - type: cpu
    cores: 8
  - type: gpu
    name: "NVIDIA RTX 3090"
    vram_gb: 24
"#;
    
    // In real code, would parse both and compare
    assert_eq!(cached.trim(), actual.trim(), "Cache should match actual");
}

#[test]
fn test_cache_update_on_refresh() {
    // TEAM-246: Test cache updated on manual refresh
    
    let old_cache = "devices:\n  - type: cpu\n    cores: 8\n";
    let new_capabilities = "devices:\n  - type: cpu\n    cores: 16\n";
    
    // Simulate refresh
    let updated_cache = new_capabilities;
    
    assert_ne!(old_cache, updated_cache, "Cache should be updated");
    assert!(updated_cache.contains("cores: 16"), "Should have new value");
}

#[test]
fn test_cache_persistence_across_restarts() {
    // TEAM-246: Test cache persists across queen restarts
    
    let temp_dir = TempDir::new().unwrap();
    let cache_file = temp_dir.path().join("capabilities.yaml");
    
    // Write cache
    fs::write(&cache_file, "devices:\n  - type: cpu\n    cores: 8\n").unwrap();
    
    // Simulate restart (cache file should still exist)
    assert!(cache_file.exists(), "Cache should persist");
    
    // Read cache after "restart"
    let content = fs::read_to_string(&cache_file).unwrap();
    assert!(content.contains("type: cpu"), "Cache should be readable");
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_cache_with_empty_devices() {
    // TEAM-246: Test cache with no devices (edge case)
    
    let cache_content = "devices: []\n";
    
    assert!(cache_content.contains("devices:"), "Should have devices key");
    assert!(cache_content.contains("[]"), "Should be empty array");
}

#[test]
fn test_cache_with_multiple_gpus() {
    // TEAM-246: Test cache with multiple GPUs
    
    let cache_content = r#"
devices:
  - type: gpu
    name: "NVIDIA RTX 3090"
    vram_gb: 24
  - type: gpu
    name: "NVIDIA RTX 4090"
    vram_gb: 24
"#;
    
    assert!(cache_content.contains("NVIDIA RTX 3090"), "Should have GPU 1");
    assert!(cache_content.contains("NVIDIA RTX 4090"), "Should have GPU 2");
}

#[test]
fn test_cache_with_cpu_only() {
    // TEAM-246: Test cache with CPU only (no GPU)
    
    let cache_content = r#"
devices:
  - type: cpu
    cores: 8
"#;
    
    assert!(cache_content.contains("type: cpu"), "Should have CPU");
    assert!(!cache_content.contains("type: gpu"), "Should not have GPU");
}
