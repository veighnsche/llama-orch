# TEAM-377 - Instant Hive Removal on Disconnect

## 🎯 The Problem

**User:** "Why wait 90 seconds? If the connection is closed, just remove it instantly!"

**Old Behavior:**
- ❌ Hive disconnects → Wait 90 seconds for timeout
- ❌ UI shows stale "1 hive online" for 90 seconds
- ❌ Terrible UX

**User is 100% correct!** There's no reason to wait.

---

## ✅ The Fix

**New Behavior:**
- ✅ SSE connection closes → Remove hive **immediately**
- ✅ UI updates within 2.5 seconds (next Queen heartbeat)
- ✅ Instant feedback

---

## 🔧 Implementation

**File:** `bin/10_queen_rbee/src/hive_subscriber.rs`

### Change 1: Remove on Error

```rust
Err(e) => {
    n!("hive_subscribe_error", "❌ Hive {} SSE error: {}", hive_id, e);
    
    // TEAM-377: Remove hive immediately when connection fails
    // Don't wait 90 seconds for timeout - that's ridiculous!
    n!("hive_disconnect", "🔌 Hive {} disconnected, removing from registry", hive_id);
    hive_registry.remove_hive(&hive_id);
    
    break; // Reconnect
}
```

### Change 2: Remove on Connection Close

```rust
// TEAM-377: Connection closed - remove hive before reconnecting
// If hive comes back online, it will re-register via discovery
n!("hive_connection_closed", "🔌 Hive {} connection closed, removing from registry", hive_id);
hive_registry.remove_hive(&hive_id);

// Connection closed, retry after delay
n!("hive_subscribe_reconnect", "🔄 Reconnecting to hive {} in 5s...", hive_id);
tokio::time::sleep(Duration::from_secs(5)).await;
```

---

## 📊 Before vs After

### Before (90 Second Timeout)

```
Timeline:
0s   - Hive disconnects
0s   - SSE connection closes
0s   - Queen keeps retrying connection every 5s
90s  - Heartbeat timestamp expires
90s  - count_online() excludes hive
92.5s - Next Queen heartbeat → UI updates

Result: 90+ second delay!
```

### After (Instant Removal)

```
Timeline:
0s   - Hive disconnects
0s   - SSE connection closes
0s   - remove_hive() called immediately
0s   - Hive removed from registry
2.5s - Next Queen heartbeat → UI updates

Result: ~2.5 second delay (just the heartbeat interval)
```

---

## 🎯 Why This Works

### Reconnection Still Works

When hive comes back online:
1. Hive sends POST /v1/hive/ready (discovery callback)
2. Queen receives callback
3. Queen starts new SSE subscription
4. Hive re-registers automatically

**No data loss!** The hive just re-registers when it comes back.

### No Stale Data

- Old approach: Keep stale heartbeat for 90 seconds
- New approach: Remove immediately, re-add when available

**Much cleaner!**

---

## 🧪 Testing

### Test 1: Single Hive Disconnect

```bash
# Start Queen
cd bin/10_queen_rbee && cargo run

# Start Hive
cd bin/20_rbee_hive && cargo run

# Check UI: Should show "1 hive online"

# Stop Hive (Ctrl+C)

# Check UI: Should show "0 hives online" within ~3 seconds
```

### Test 2: Multiple Hives

```bash
# Start Queen
cd bin/10_queen_rbee && cargo run

# Start Hive 1 (local)
cd bin/20_rbee_hive && cargo run

# Start Hive 2 (workstation)
ssh workstation "cd llama-orch/bin/20_rbee_hive && cargo run"

# Check UI: Should show "2 hives online"

# Stop Hive 1
# Check UI: Should show "1 hive online" within ~3 seconds

# Stop Hive 2
# Check UI: Should show "0 hives online" within ~3 seconds
```

### Test 3: Hive Restart

```bash
# Start Queen + Hive
# Stop Hive
# UI shows 0 hives

# Restart Hive
# UI should show 1 hive within ~5 seconds
# (Hive sends discovery callback on startup)
```

---

## 🎓 Why 90 Seconds Was Wrong

### The Old Design

**Assumption:** "We need a timeout to detect dead hives"

**Problem:** We already have a better signal - **the SSE connection!**

### The New Design

**Realization:** "The SSE connection IS the heartbeat"

**Benefits:**
- ✅ Instant detection (no timeout needed)
- ✅ Simpler code (no timeout logic)
- ✅ Better UX (immediate feedback)
- ✅ Still handles reconnection (via discovery)

---

## 🔄 What About Reconnection?

### Scenario: Network Blip

**Old behavior:**
- Connection drops
- Wait 90 seconds
- Hive marked offline
- Connection restores
- Hive sends telemetry
- Hive marked online again

**New behavior:**
- Connection drops
- Hive removed immediately
- Connection restores (5s retry)
- Hive sends telemetry
- Hive re-registers automatically

**Result:** Same outcome, but faster feedback during the outage!

### Scenario: Hive Restart

**Old behavior:**
- Hive stops
- Wait 90 seconds
- Hive marked offline
- Hive starts
- Hive sends discovery callback
- Hive marked online

**New behavior:**
- Hive stops
- Hive removed immediately (0s)
- Hive starts
- Hive sends discovery callback
- Hive marked online

**Result:** Much faster!

---

## 📝 Notes

### Why Keep Reconnection Loop?

Even though we remove the hive, we keep trying to reconnect because:

1. **Network blips:** Temporary connection loss
2. **Hive restarts:** Hive comes back online
3. **Discovery race:** Hive might start before Queen discovers it

The reconnection loop is harmless - it just keeps trying. When the hive comes back, it will either:
- Respond to the SSE connection (reconnection succeeds)
- Send a discovery callback (new subscription started)

Either way, the hive gets re-registered.

### Why Not Cancel Reconnection?

We could cancel the reconnection task when we remove the hive, but:
- It's more complex (need to track task handles)
- The retry loop is cheap (just sleeps)
- It handles the "hive comes back" case automatically

**Simpler to just let it retry.**

---

## ✅ Summary

**Changed:**
- Remove hive immediately when SSE connection closes
- No more 90 second timeout wait

**Benefits:**
- ✅ Instant UI feedback (~3 seconds)
- ✅ Simpler logic (no timeout needed)
- ✅ Better UX
- ✅ Still handles reconnection

**Testing:**
- Restart queen-rbee
- Test hive connect/disconnect
- Should see instant updates

---

**TEAM-377 | Instant hive removal | No more 90 second wait | Much better UX! 🎉**
