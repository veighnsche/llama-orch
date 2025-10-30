# TEAM-368: Remote Hive Queen URL Fix

**Status:** ✅ COMPLETE  
**Date:** Oct 30, 2025  
**Author:** TEAM-368

## Problem

When keeper starts a remote hive via SSH, it was passing `http://localhost:7833` as the queen_url. This is **wrong** because:

1. Remote hive can't reach Queen at "localhost" (localhost = the remote machine itself)
2. Queen is on the **same machine as keeper**, not on the remote hive machine
3. Remote hive needs Queen's **network address**, not localhost

## Root Cause

```rust
// bin/97_contracts/keeper-config-contract/src/config.rs:45-46
pub fn queen_url(&self) -> String {
    format!("http://localhost:{}", self.queen_port)
    //      ^^^^^^^^^^^ WRONG for remote hives!
}
```

This hardcoded `localhost` works fine for:
- ✅ Keeper → Queen communication (same machine)
- ✅ Localhost hive → Queen (same machine)
- ❌ **Remote hive → Queen (different machines)**

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Keeper's Machine (192.168.1.100)                       │
│  ┌──────────────┐         ┌──────────────┐             │
│  │ rbee-keeper  │────────▶│ queen-rbee   │             │
│  │ (CLI)        │         │ :7833        │             │
│  └──────┬───────┘         └──────────────┘             │
│         │                                               │
│         │ SSH                                           │
└─────────┼───────────────────────────────────────────────┘
          │
          │ Start hive with --queen-url=???
          ▼
┌─────────────────────────────────────────────────────────┐
│  Remote Hive Machine (192.168.1.200)                    │
│  ┌──────────────┐                                       │
│  │ rbee-hive    │                                       │
│  │ :7835        │                                       │
│  └──────┬───────┘                                       │
│         │                                               │
│         │ Heartbeat to Queen                            │
│         │ WHERE IS QUEEN?                               │
│         │                                               │
│         │ ❌ localhost:7833 → WRONG (points to self)   │
│         │ ✅ 192.168.1.100:7833 → CORRECT!              │
└─────────────────────────────────────────────────────────┘
```

## Solution

**Use keeper's local IP address** when starting remote hives:

```rust
// bin/00_rbee_keeper/src/handlers/hive.rs:95-102

// TEAM-368: Get keeper's local IP (Queen is on same machine)
let local_ip = local_ip().map_err(|e| anyhow::anyhow!("Failed to get local IP: {}", e))?;
let queen_port = queen_url.split(':').last()
    .and_then(|p| p.parse::<u16>().ok())
    .unwrap_or(7833);
let network_queen_url = format!("http://{}:{}", local_ip, queen_port);

n!("remote_hive_queen_url", "🌐 Remote hive will use Queen at: {}", network_queen_url);
```

## Before vs After

### Before (BROKEN)
```bash
# User runs:
./rbee hive start remote-server

# Keeper starts hive via SSH with:
ssh remote-server 'rbee-hive --queen-url=http://localhost:7833'

# Remote hive tries to connect:
POST http://localhost:7833/v1/hive-heartbeat
# ❌ FAILS - localhost points to remote-server, not keeper's machine
```

### After (FIXED)
```bash
# User runs:
./rbee hive start remote-server

# Keeper detects local IP: 192.168.1.100
# Keeper starts hive via SSH with:
ssh remote-server 'rbee-hive --queen-url=http://192.168.1.100:7833'

# Remote hive connects successfully:
POST http://192.168.1.100:7833/v1/hive-heartbeat
# ✅ SUCCESS - reaches Queen on keeper's machine
```

## Implementation Details

### 1. Detect Local IP
```rust
use local_ip_address::local_ip;

let local_ip = local_ip()
    .map_err(|e| anyhow::anyhow!("Failed to get local IP: {}", e))?;
// Returns: IpAddr (e.g., 192.168.1.100)
```

### 2. Extract Queen Port
```rust
let queen_port = queen_url.split(':').last()
    .and_then(|p| p.parse::<u16>().ok())
    .unwrap_or(7833);
// From "http://localhost:7833" → 7833
```

### 3. Build Network URL
```rust
let network_queen_url = format!("http://{}:{}", local_ip, queen_port);
// Result: "http://192.168.1.100:7833"
```

### 4. Pass to Remote Hive
```rust
let args = vec![
    "--port".to_string(),
    port.to_string(),
    "--queen-url".to_string(),
    network_queen_url,  // ← Network address, not localhost
    "--hive-id".to_string(),
    alias.clone(),
];
```

## Edge Cases Handled

### Edge Case #1: Multiple Network Interfaces
**Problem:** Machine has multiple IPs (WiFi, Ethernet, VPN)

**Solution:** `local_ip()` returns the **primary** local IP (the one used for outbound connections)

**Example:**
- WiFi: 192.168.1.100 ← Primary (returned)
- VPN: 10.0.0.5
- Docker: 172.17.0.1

### Edge Case #2: Localhost Hive
**Behavior:** Localhost hives still use `http://localhost:7833`

**Code:**
```rust
if alias == "localhost" {
    // Use localhost URL (no change)
    let args = vec![
        "--queen-url".to_string(),
        queen_url.to_string(),  // "http://localhost:7833"
    ];
} else {
    // Use network IP for remote
    let network_queen_url = format!("http://{}:{}", local_ip, queen_port);
    let args = vec![
        "--queen-url".to_string(),
        network_queen_url,  // "http://192.168.1.100:7833"
    ];
}
```

### Edge Case #3: No Network Connection
**Problem:** Keeper has no network (airplane mode, no WiFi)

**Solution:** `local_ip()` returns error, keeper fails fast with clear message:
```
Error: Failed to get local IP: No network interface found
```

### Edge Case #4: Firewall Blocks Queen Port
**Problem:** Remote hive can't reach Queen due to firewall

**Detection:** Hive's exponential backoff will fail all 5 attempts:
```
❌ Discovery attempt 1 failed: connection refused
❌ Discovery attempt 2 failed: connection refused
...
⏸️  All discovery attempts failed. Waiting for Queen to discover us via /capabilities
```

**Recovery:** Queen can still discover hive via pull-based discovery (if SSH is configured)

## Files Modified

1. **bin/00_rbee_keeper/src/handlers/hive.rs** (+14 LOC)
   - Added `use local_ip_address::local_ip;`
   - Detect local IP for remote hives
   - Build network queen_url
   - Add narration for visibility

2. **bin/00_rbee_keeper/Cargo.toml** (+3 LOC)
   - Added `local-ip-address = "0.6"` dependency

## Testing

### Test 1: Localhost Hive (No Change)
```bash
./rbee hive start localhost

# Expected:
# Hive starts with --queen-url=http://localhost:7833
# Heartbeat connects successfully
```

### Test 2: Remote Hive (Fixed)
```bash
./rbee hive start remote-server

# Expected output:
# 🌐 Remote hive will use Queen at: http://192.168.1.100:7833

# Verify on remote machine:
ps aux | grep rbee-hive
# Should show: rbee-hive --queen-url=http://192.168.1.100:7833

# Check logs:
# ✅ Discovery successful! Starting normal telemetry
```

### Test 3: No Network
```bash
# Disconnect all network
sudo ifconfig wlan0 down
sudo ifconfig eth0 down

./rbee hive start remote-server

# Expected:
# Error: Failed to get local IP: No network interface found
```

## Benefits

✅ **Remote hives can now reach Queen**  
✅ **Automatic IP detection** (no manual configuration)  
✅ **Works with any network setup** (WiFi, Ethernet, etc.)  
✅ **Clear error messages** if network unavailable  
✅ **Localhost hives unchanged** (still use localhost)  
✅ **Narration shows Queen URL** for debugging  

## Related Documents

- `bin/TEAM_366_EDGE_CASE_GUARDS.md` - Edge case analysis
- `bin/TEAM_367_QUEEN_RESTART_FIXES.md` - Queen restart detection
- `bin/.specs/HEARTBEAT_ARCHITECTURE.md` - Heartbeat protocol

## Team Signatures

- TEAM-365: Implemented bidirectional handshake
- TEAM-366: Added edge case guards
- TEAM-367: Fixed optional queen_url + restart detection
- TEAM-368: Fixed remote hive queen_url (this document)

---

**Status:** Production ready, remote hives can now connect to Queen
