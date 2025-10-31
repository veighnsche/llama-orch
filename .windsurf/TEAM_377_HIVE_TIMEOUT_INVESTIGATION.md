# TEAM-377 - Hive Timeout Investigation

## 🐛 The Problem

**User Report:**
1. Start with 0 hives → Shows "0 hives online" ✅
2. Start local hive → Shows "1 hive online" ✅
3. Start workstation hive → Still shows "1 hive online" ❌ (should be 2)
4. Close both hives → Still shows "1 hive online" ❌ (should be 0 after 90s)

---

## 🔍 How It Should Work

### Heartbeat Timeout Logic

**File:** `bin/97_contracts/hive-contract/src/heartbeat.rs`
```rust
// Line 31: Timeout is 90 seconds
/// - **Timeout:** 90 seconds (3 missed heartbeats)

// Line 102-104: is_recent() checks timestamp
pub fn is_recent(&self) -> bool {
    self.timestamp.is_recent(HEARTBEAT_TIMEOUT_SECS)  // 90 seconds
}
```

### Queen Heartbeat Stream

**File:** `bin/10_queen_rbee/src/http/heartbeat_stream.rs`
```rust
// Line 27: Queen sends heartbeat every 2.5 seconds
let mut queen_interval = interval(Duration::from_millis(2500));

// Line 57: Each heartbeat calls count_online()
let hives_online = state.hive_registry.count_online();
```

### Count Online Logic

**File:** `bin/99_shared_crates/heartbeat-registry/src/lib.rs`
```rust
// Line 154-157: count_online() filters by is_recent()
pub fn count_online(&self) -> usize {
    let items = self.items.read().unwrap();
    items.values().filter(|hb| hb.is_recent()).count()
}
```

### Expected Behavior

```
1. Hive sends telemetry → update_hive() called → timestamp = NOW
2. Hive disconnects → No more telemetry → timestamp stays at last update
3. After 90 seconds → is_recent() returns false → count_online() excludes it
4. Queen heartbeat (every 2.5s) → UI gets updated count
```

---

## 🎯 Possible Issues

### Issue 1: Second Hive Not Registering

**Symptom:** Start 2 hives, only shows 1

**Possible Causes:**
1. **Same hive_id:** Both hives using same ID (e.g., "localhost")
   - HashMap uses hive_id as key
   - Second hive overwrites first hive
   
2. **Discovery not working:** Second hive not discovered by Queen
   - Check Queen logs for discovery messages
   - Check if second hive sends POST /v1/hive/ready
   
3. **SSE subscription not starting:** Queen doesn't subscribe to second hive
   - Check Queen logs for "Subscribing to hive" messages

### Issue 2: Stale Hives Not Expiring

**Symptom:** Close both hives, still shows 1 after 90+ seconds

**Possible Causes:**
1. **Timestamp not expiring:** Heartbeat timestamp keeps getting refreshed
   - SSE reconnection might be keeping it alive somehow
   
2. **UI not updating:** Frontend not receiving updated counts
   - Check browser console for SSE events
   - Check if `console.log({ hives, workersOnline, hivesOnline })` updates
   
3. **Cleanup not running:** Stale cleanup task not working
   - But count_online() should filter anyway

---

## 🧪 Debugging Steps

### Step 1: Check Hive IDs

**In Queen logs, look for:**
```
🐝 Hive ready callback: hive_id=XXX, url=YYY
```

**Expected:**
- First hive: `hive_id=localhost` or `hive_id=gpu-0`
- Second hive: `hive_id=<different-name>`

**If both have same ID:** That's the problem! They're overwriting each other.

### Step 2: Check Discovery

**In Queen logs, look for:**
```
🔍 Discovering hive: XXX (YYY)
✅ Discovered hive: XXX
📡 Subscribing to hive XXX SSE stream: http://...
✅ Connected to hive XXX SSE stream
```

**Expected:** See these messages for BOTH hives.

**If missing:** Second hive not discovered or not sending callback.

### Step 3: Check Telemetry

**In Queen logs, look for:**
```
Received telemetry from hive XXX: N workers
```

**Expected:** See messages for BOTH hive IDs.

**If only one:** Only one hive is sending telemetry.

### Step 4: Check Browser Console

**Open DevTools → Console, look for:**
```javascript
{ hives: [...], workersOnline: X, hivesOnline: Y }
```

**Expected:**
- After starting 2 hives: `hivesOnline: 2`
- After closing both: `hivesOnline: 0` (after 90 seconds)

**If stuck at 1:** Backend is returning wrong count.

### Step 5: Check SSE Events

**Open DevTools → Network → Filter "stream" → Click on heartbeats/stream**

**Look at EventStream messages:**
```json
{
  "type": "queen",
  "hives_online": 2,
  "hive_ids": ["localhost", "workstation"]
}
```

**Expected:**
- Should see `hives_online` change
- Should see `hive_ids` array update

---

## 🔧 Potential Fixes

### Fix 1: Ensure Unique Hive IDs

**Problem:** Both hives using same ID

**Solution:** Each hive needs unique ID

**File:** `bin/20_rbee_hive/src/main.rs`
```rust
// Check how hive_id is determined
// Should be unique per machine (e.g., hostname, IP, or config)
```

### Fix 2: Add Logging to update_hive()

**Problem:** Can't see when hives are registered/updated

**Solution:** Add narration events

**File:** `bin/10_queen_rbee/src/hive_subscriber.rs`
```rust
// After line 105
hive_registry.update_hive(HiveHeartbeat::new(hive_info));
n!("hive_registered", "✅ Registered hive {} as online", hive_id);
```

### Fix 3: Log count_online() Results

**Problem:** Can't see what count_online() returns

**Solution:** Add debug logging

**File:** `bin/10_queen_rbee/src/http/heartbeat_stream.rs`
```rust
// After line 57
let hives_online = state.hive_registry.count_online();
tracing::debug!("Hives online: {}, IDs: {:?}", hives_online, hive_ids);
```

### Fix 4: Check Timestamp Updates

**Problem:** Timestamp might be getting refreshed incorrectly

**Solution:** Log timestamp on each update

**File:** `bin/10_queen_rbee/src/hive_subscriber.rs`
```rust
// After line 105
let hb = HiveHeartbeat::new(hive_info);
tracing::debug!("Updating hive {} heartbeat, timestamp: {:?}", hive_id, hb.timestamp);
hive_registry.update_hive(hb);
```

---

## 🎯 Most Likely Issue

**HYPOTHESIS:** Both hives are using the same `hive_id` (probably "localhost" or derived from hostname).

**Why:**
1. First hive registers with ID "localhost"
2. Second hive also uses ID "localhost"
3. HashMap overwrites first entry with second
4. Result: Only 1 hive in registry

**Test:**
```bash
# Check hive_id in Queen logs
grep "Hive ready callback" queen.log

# Should see:
# 🐝 Hive ready callback: hive_id=localhost, url=http://127.0.0.1:7835
# 🐝 Hive ready callback: hive_id=workstation, url=http://192.168.1.100:7835

# If both say "localhost" → That's the bug!
```

---

## 🚀 Next Steps

1. **Check Queen logs** for hive_id values
2. **Check browser console** for `console.log` output
3. **Check SSE events** in Network tab
4. **Wait 90+ seconds** after closing hives to see if count updates
5. **Report findings** so we can implement the right fix

---

**TEAM-377 | Investigating hive timeout | Need logs to diagnose!**
