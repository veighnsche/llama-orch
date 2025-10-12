# DX Tool Performance Specifications

**Created by:** TEAM-DX-003  
**Date:** 2025-10-12

## Development Hardware

```
OS: Arch Linux x86_64
Host: NUC12WSKi5 (M46708-305)
Kernel: Linux 6.15.11-hardened1-1-hardened
CPU: 12th Gen Intel(R) Core(TM) i5-1240P (16 cores) @ 4.40 GHz
GPU: Intel Iris Xe Graphics @ 1.30 GHz [Integrated]
Memory: 62.38 GiB
Disk: 915.32 GiB NVMe SSD (ext4)
Display: 3x monitors (4K + 2x QHD)
WM: Sway 1.11 (Wayland)
```

## Timeout Configuration

### Default Timeouts (Tuned for Above Hardware)

| Operation | Timeout | Reason |
|-----------|---------|--------|
| HTTP Request | 30s | Network + server response |
| Browser Launch | 2s | Chrome startup |
| Page Load | 3s | Initial HTML + JS execution |
| SPA Render | 3s | Vue/React component mount |
| Iframe Load | 1.5s | Nested content |
| Max Browser Wait | 10s | Hard limit to prevent hangs |

### Configurable via Code

```rust
// Default (3 seconds)
let fetcher = Fetcher::new();

// Custom wait time
let fetcher = Fetcher::new().with_browser_wait(5000); // 5 seconds

// Disable browser (fast, no JS)
let fetcher = Fetcher::new().without_browser();
```

### Environment Variables

```bash
# Not yet implemented, but planned:
export DX_BROWSER_WAIT_MS=5000
export DX_HTTP_TIMEOUT_SECS=60
```

## Performance Targets

Per DX Engineering Rules:
- **Target:** < 2 seconds for simple commands
- **Actual with browser:** 5-8 seconds (acceptable for SPA support)
- **Actual without browser:** < 1 second

### Measured Performance (on dev hardware)

| Command | Time | Notes |
|---------|------|-------|
| `dx css --class-exists` | 0.8s | No browser |
| `dx html --selector` | 0.9s | No browser |
| `dx inspect button` (SPA) | 6-8s | With browser |
| `dx story-file` | 0.1s | No network |

## Why These Timeouts?

### 3 Second Default

**Tested on:**
- Histoire (Vue 3 SPA)
- Storybook (React SPA)
- Vite dev server

**Observation:**
- Most SPAs render in 2-3 seconds
- Slower machines may need 5 seconds
- Faster machines could use 2 seconds

**Decision:** 3 seconds is a good middle ground.

### 10 Second Maximum

**Reason:** If content doesn't load in 10 seconds, something is wrong:
- Server is down
- Network is slow
- Page is broken
- Infinite loading state

**Better to fail fast** than wait forever.

## Tuning for Different Hardware

### Fast Machine (Modern Desktop/Laptop)
```rust
Fetcher::new().with_browser_wait(2000) // 2 seconds
```

### Slow Machine (Old Laptop, CI)
```rust
Fetcher::new().with_browser_wait(5000) // 5 seconds
```

### Very Slow (Raspberry Pi, etc.)
```rust
Fetcher::new().with_browser_wait(10000) // 10 seconds (max)
```

### No Browser (Fast, No JS)
```rust
Fetcher::new().without_browser()
```

## Debugging Slow Performance

### Check Browser Launch Time
```bash
time google-chrome --headless --dump-dom http://localhost:6006
```

### Check Network Time
```bash
time curl http://localhost:6006
```

### Check JS Execution Time
Open DevTools → Performance → Record page load

## CI/CD Considerations

### GitHub Actions (Ubuntu)
- **CPU:** 2 cores
- **RAM:** 7 GB
- **Recommended:** 5-7 seconds

### GitLab CI (Shared Runners)
- **CPU:** 1 core
- **RAM:** 4 GB
- **Recommended:** 7-10 seconds

### Self-Hosted (Same as Dev)
- **Recommended:** 3 seconds (same as dev)

## Future Improvements

### 1. Auto-Detect Hardware
```rust
// Detect CPU cores and adjust timeout
let cores = num_cpus::get();
let wait_ms = match cores {
    1..=2 => 10000,
    3..=8 => 5000,
    _ => 3000,
};
```

### 2. Network Idle Detection
```rust
// Wait for network to be idle (no pending requests)
tab.wait_for_network_idle(Duration::from_secs(2))
```

### 3. Smart Retry
```rust
// Retry with longer timeout if first attempt fails
if content.is_empty() {
    retry_with_longer_timeout()
}
```

### 4. Progress Indicator
```bash
dx inspect button http://localhost:6006
# Output:
# ⏳ Launching browser...
# ⏳ Loading page...
# ⏳ Waiting for content...
# ✓ Inspected: button
```

## Recommendations

### For Users

1. **Use default timeouts** unless you have issues
2. **Increase timeout** if you see "Selector not found" errors
3. **Disable browser** for static HTML (faster)
4. **Check server** if timeouts persist

### For Developers

1. **Don't increase timeouts blindly** - investigate why it's slow
2. **Profile the page** - use Chrome DevTools
3. **Optimize SPA** - reduce JS bundle size, lazy load
4. **Use SSR** - pre-render content on server

## Summary

**Default: 3 seconds** - Good for most modern hardware  
**Maximum: 10 seconds** - Hard limit to prevent hangs  
**Configurable:** Yes, via code (env vars planned)  
**Measured on:** Intel i5-1240P, 62GB RAM, NVMe SSD  

**If you're waiting "forever", something else is wrong - not the timeout.**
