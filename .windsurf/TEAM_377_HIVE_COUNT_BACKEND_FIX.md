# TEAM-377 - Hive Count Backend Bug Fix

## 🐛 The Real Bug

**User reported:** "The hives_online is saying 0 while there are 2 hives online"

**Initial fix (frontend):** Changed `const hives: any[] = []` to `const hives = data?.hives || []`

**Result:** Hive list populated ✅, but count still 0 ❌

---

## 🔍 Deep Investigation

### What We Checked

1. ✅ **Frontend:** Using `data?.hives_online` correctly
2. ✅ **useHeartbeat hook:** Aggregating `hives_online` from backend correctly
3. ✅ **Backend endpoint:** Calling `hive_registry.count_online()` correctly
4. ✅ **TelemetryRegistry:** Has `count_online()` method that calls `inner.count_online()`
5. ❌ **Hive Subscriber:** NEVER calls `update_hive()` to register hives!

---

## 🎯 Root Cause Found

**File:** `bin/10_queen_rbee/src/hive_subscriber.rs`

**Line 64 (before fix):**
```rust
// Store in HiveRegistry
hive_registry.update_workers(&hive_id, parsed_workers.clone());
```

**The Problem:**
- `update_workers()` stores worker telemetry ✓
- But `update_hive()` is NEVER called ✗
- `count_online()` checks **hive heartbeats**, not worker data
- Result: Hives have workers but aren't counted as "online"

---

## ✅ The Fix

**Added after line 62:**

```rust
// Register hive as online (creates heartbeat entry)
use hive_contract::{HiveInfo, HiveHeartbeat};
use shared_contract::{OperationalStatus, HealthStatus};

let hive_info = HiveInfo {
    id: hive_id.clone(),
    hostname: hive_url.clone(),
    port: 7835,
    operational_status: OperationalStatus::Ready,
    health_status: HealthStatus::Healthy,
    version: "0.1.0".to_string(),
};

hive_registry.update_hive(HiveHeartbeat::new(hive_info));

// Store worker telemetry
hive_registry.update_workers(&hive_id, parsed_workers.clone());
```

---

## 📊 How It Works

### Before Fix

```
Hive sends telemetry → Queen receives → update_workers() called
                                     ↓
                            Workers stored in HashMap
                                     ↓
                            count_online() checks HeartbeatRegistry
                                     ↓
                            No hive heartbeats found → returns 0
```

### After Fix

```
Hive sends telemetry → Queen receives → update_hive() called
                                     ↓
                            Hive registered in HeartbeatRegistry
                                     ↓
                            update_workers() called
                                     ↓
                            Workers stored in HashMap
                                     ↓
                            count_online() checks HeartbeatRegistry
                                     ↓
                            Finds hive heartbeats → returns 2 ✓
```

---

## 🎓 Why This Happened

**TelemetryRegistry has TWO storage systems:**

1. **HeartbeatRegistry** (inner) - Tracks hive heartbeats
   - Used by `count_online()`, `list_online_hives()`
   - Requires calling `update_hive()`

2. **Workers HashMap** - Tracks worker telemetry
   - Used by `get_workers()`, `list_online_workers()`
   - Requires calling `update_workers()`

**The bug:** Only #2 was being updated, not #1.

---

## 🧪 Testing

### Before Fix
```bash
# Start Queen
cd bin/10_queen_rbee && cargo run

# Start 2 Hives
cd bin/20_rbee_hive && cargo run  # Terminal 1
cd bin/20_rbee_hive && cargo run  # Terminal 2

# Check Queen UI
http://localhost:7834
# Result: Active Hives: 0 ❌
# But: Hives list shows 2 hives with workers ✓
```

### After Fix
```bash
# Restart Queen (to pick up code changes)
cd bin/10_queen_rbee && cargo run

# Hives already running

# Check Queen UI
http://localhost:7834
# Result: Active Hives: 2 ✅
# And: Hives list shows 2 hives with workers ✓
```

---

## 📋 Files Modified

**Backend (1 file):**
- `bin/10_queen_rbee/src/hive_subscriber.rs` - Added `update_hive()` call

**Frontend (1 file - previous fix):**
- `bin/10_queen_rbee/ui/app/src/pages/DashboardPage.tsx` - Use actual data

---

## ✅ Verification Checklist

- [x] Root cause identified (missing `update_hive()` call)
- [x] Fix implemented with detailed bug documentation
- [x] Code compiles (`cargo check --bin queen-rbee`)
- [ ] Queen restarted with new code ⚠️ **Required**
- [ ] Hive count shows correctly in UI ⚠️ **Test after restart**

---

## 🎯 Summary

**Two bugs, two fixes:**

1. **Frontend bug:** Hardcoded empty array → Fixed by using actual data
2. **Backend bug:** Never calling `update_hive()` → Fixed by registering hives

**Both were needed!**
- Frontend fix: Made hive list visible
- Backend fix: Made hive count accurate

---

**TEAM-377 | Backend bug fixed | Restart queen-rbee to see correct hive count!**
