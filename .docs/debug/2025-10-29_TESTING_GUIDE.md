# Testing Guide for Queen UI Fixes (2025-10-29)

## Quick Verification

### 1. Test RhaiIDE Select Fix

**Start Turbo Dev:**
```bash
cd /home/vince/Projects/llama-orch
turbo dev
```

**Access Queen UI:**
- Open browser: http://localhost:7834
- Navigate to RHAI IDE section
- **Expected:** No crash, Select renders correctly
- **Expected:** See "+ New Script" option with sentinel value
- **Expected:** Clicking "+ New Script" triggers createNewScript()

**Verify in Console:**
```javascript
// Should see no errors about empty string values
// Should see Select component working normally
```

---

### 2. Test SSE Health Check (Queen Offline)

**Stop Queen (if running):**
```bash
rbee queen stop
```

**Access Queen UI:**
- Open browser: http://localhost:7834
- Open browser console (F12)

**Expected Behavior:**
- ✅ No "CORS request did not succeed" errors
- ✅ See: `🐝 [SDK] Connecting to SSE: http://localhost:7833/v1/heartbeats/stream`
- ✅ UI shows "Queen is offline" or similar error state
- ✅ No noisy console warnings

**Verify Health Check:**
```javascript
// In browser console:
const monitor = new window.rbee_sdk.HeartbeatMonitor('http://localhost:7833');
const isHealthy = await monitor.checkHealth();
console.log('Queen healthy:', isHealthy); // Should be false
```

---

### 3. Test SSE Health Check (Queen Online)

**Start Queen:**
```bash
rbee queen start
```

**Wait for Queen to be ready:**
```bash
# Check health endpoint
curl http://localhost:7833/health
# Should return: {"status":"ok"}
```

**Access Queen UI:**
- Refresh browser: http://localhost:7834
- Open browser console (F12)

**Expected Behavior:**
- ✅ Health check passes
- ✅ SSE connection established
- ✅ See: `🐝 [SDK] SSE connection OPENED`
- ✅ Heartbeat data flows to UI
- ✅ No errors in console

**Verify in Console:**
```javascript
const monitor = new window.rbee_sdk.HeartbeatMonitor('http://localhost:7833');
const isHealthy = await monitor.checkHealth();
console.log('Queen healthy:', isHealthy); // Should be true
```

---

### 4. Test Turbo Dev Stability (Install Queen)

**Setup:**
```bash
# Terminal 1: Start Turbo dev
cd /home/vince/Projects/llama-orch
turbo dev

# Wait for all dev servers to start
# Should see Vite on 7834, etc.
```

**Install Queen from Keeper:**
```bash
# Terminal 2: Install queen
rbee queen install
```

**Expected Behavior:**
- ✅ See: `⏭️  Setting RBEE_SKIP_UI_BUILD=1 for queen-rbee (prevents Turbo conflicts)`
- ✅ Cargo build completes without triggering Vite build
- ✅ Turbo dev servers remain stable (no crash)
- ✅ No high CPU/memory spikes
- ✅ Install completes successfully

**Verify Build Output:**
```bash
# Should see in cargo output:
# cargo:warning=⏭️  Skipping queen-rbee UI build (RBEE_SKIP_UI_BUILD set)
# cargo:warning=ℹ️  Debug mode: UI will be served from Vite dev server (7834)
```

**Verify Turbo Still Running:**
```bash
# Terminal 1 should still show:
# - Vite dev server on 7834
# - No crash logs
# - No "process exited" messages
```

---

### 5. Test Queen Serves UI (After Install)

**Start Queen:**
```bash
rbee queen start
```

**Access via Keeper:**
- Open Keeper GUI
- Navigate to Queen page
- **Expected:** Iframe loads http://localhost:7833
- **Expected:** UI loads from dev proxy (proxies to 7834)
- **Expected:** HMR works (make a change, see instant update)

**Verify Dev Proxy:**
```bash
# Check queen logs for proxy requests
# Should see: Proxying / to http://localhost:7834
```

---

## Manual Testing Checklist

### RhaiIDE Select
- [ ] Load with no scripts → no crash
- [ ] Load with scripts → Select works
- [ ] Select existing script → loads correctly
- [ ] Click "+ New Script" → creates new script
- [ ] No empty string values in Select items
- [ ] Placeholder shows when no selection

### SSE Health Check (Queen Offline)
- [ ] No CORS errors in console
- [ ] Health check returns false
- [ ] SSE connection not attempted
- [ ] UI shows offline state
- [ ] Error message is user-friendly

### SSE Health Check (Queen Online)
- [ ] Health check returns true
- [ ] SSE connection established
- [ ] Heartbeat events flow
- [ ] UI updates in real-time
- [ ] No errors in console

### Turbo Dev Stability
- [ ] Turbo dev starts successfully
- [ ] Install queen completes
- [ ] RBEE_SKIP_UI_BUILD=1 set
- [ ] No Vite build triggered
- [ ] Turbo dev remains stable
- [ ] No process crashes

### Dev Proxy
- [ ] Queen serves UI from 7833
- [ ] Root path proxies to 7834
- [ ] API routes work on 7833
- [ ] SSE streams work on 7833
- [ ] HMR works in dev mode

---

## Automated Testing (Future)

### Unit Tests
```typescript
// RhaiIDE.test.tsx
describe('RhaiIDE Select', () => {
  it('should not use empty string values', () => {
    const { container } = render(<RhaiIDE />);
    const selectItems = container.querySelectorAll('[role="option"]');
    selectItems.forEach(item => {
      expect(item.getAttribute('value')).not.toBe('');
    });
  });

  it('should handle new script sentinel', () => {
    const createNewScript = jest.fn();
    const { getByText } = render(<RhaiIDE createNewScript={createNewScript} />);
    fireEvent.click(getByText('+ New Script'));
    expect(createNewScript).toHaveBeenCalled();
  });
});
```

### Integration Tests
```typescript
// heartbeat.test.ts
describe('HeartbeatMonitor', () => {
  it('should check health before starting SSE', async () => {
    const monitor = new HeartbeatMonitor('http://localhost:7833');
    const healthSpy = jest.spyOn(monitor, 'checkHealth');
    
    await monitor.start(callback);
    
    expect(healthSpy).toHaveBeenCalled();
  });

  it('should not start SSE when health check fails', async () => {
    const monitor = new HeartbeatMonitor('http://offline:9999');
    const isHealthy = await monitor.checkHealth();
    
    expect(isHealthy).toBe(false);
    expect(monitor.isConnected()).toBe(false);
  });
});
```

### E2E Tests
```typescript
// install-queen.spec.ts
describe('Install Queen during dev', () => {
  it('should not crash Turbo dev servers', async () => {
    // Start Turbo dev
    const turbo = await startTurboDev();
    
    // Install queen
    await exec('rbee queen install');
    
    // Verify Turbo still running
    expect(turbo.isRunning()).toBe(true);
    expect(turbo.hasErrors()).toBe(false);
  });
});
```

---

## Troubleshooting

### RhaiIDE Still Crashes
- Check browser console for exact error
- Verify all Select items have non-empty values
- Check if sentinel value "__new__" is used
- Verify handleSelectScript is wired correctly

### CORS Errors Still Appear
- Verify SDK was rebuilt: `pnpm -F @rbee/queen-rbee-sdk build`
- Check if checkHealth method exists in WASM
- Verify useHeartbeat calls checkHealth before start
- Check browser network tab for /health request

### Turbo Still Crashes
- Verify RBEE_SKIP_UI_BUILD is set in build output
- Check if queen build.rs has the gate logic
- Verify daemon-lifecycle sets the env var
- Check for other competing processes

### Dev Proxy Not Working
- Verify queen is running: `rbee queen status`
- Check queen logs for proxy requests
- Verify Vite dev server is on 7834
- Check static_files.rs has dev proxy code

---

## Performance Verification

### Build Times
```bash
# Before (with UI build):
time cargo build -p queen-rbee
# Expected: ~30-60 seconds (includes Vite build)

# After (with RBEE_SKIP_UI_BUILD=1):
RBEE_SKIP_UI_BUILD=1 time cargo build -p queen-rbee
# Expected: ~10-20 seconds (no Vite build)
```

### Memory Usage
```bash
# Monitor during install:
watch -n 1 'ps aux | grep -E "(node|vite|cargo)" | grep -v grep'

# Expected:
# - No memory spikes
# - Node processes stable
# - Cargo completes normally
```

### CPU Usage
```bash
# Monitor during install:
top -p $(pgrep -d',' -f 'node|vite|cargo')

# Expected:
# - CPU usage normal
# - No sustained 100% CPU
# - Processes don't hang
```

---

## Rollback Plan

If any fix causes issues:

### Revert RhaiIDE Select
```bash
git checkout HEAD -- bin/10_queen_rbee/ui/app/src/components/RhaiIDE.tsx
```

### Revert Health Check
```bash
git checkout HEAD -- bin/10_queen_rbee/ui/packages/queen-rbee-sdk/src/heartbeat.rs
git checkout HEAD -- bin/10_queen_rbee/ui/packages/queen-rbee-react/src/hooks/useHeartbeat.ts
pnpm -F @rbee/queen-rbee-sdk build
```

### Revert Build Gate
```bash
git checkout HEAD -- bin/10_queen_rbee/build.rs
git checkout HEAD -- bin/99_shared_crates/daemon-lifecycle/src/build.rs
```

---

## Success Criteria

All fixes are successful when:

1. ✅ RhaiIDE loads without crashes
2. ✅ No CORS errors when queen is offline
3. ✅ SSE connects when queen is online
4. ✅ Install queen doesn't crash Turbo dev
5. ✅ Dev proxy serves UI correctly
6. ✅ HMR works in dev mode
7. ✅ Build times are reasonable
8. ✅ No memory/CPU issues

---

**Last Updated:** 2025-10-29  
**Status:** Ready for testing
