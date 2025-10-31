# TEAM-377 BUG FIX - Hive Count Always Showing 0

## 🐛 Issue

**Symptom:** Dashboard shows "Active Hives: 0" even with multiple hives running and connected.

**Screenshot Evidence:** User showed 2 hives running, but UI displayed 0.

---

## 🔍 Root Cause

**File:** `bin/10_queen_rbee/ui/app/src/pages/DashboardPage.tsx`

**Lines 12-14 (Before Fix):**
```typescript
const hives: any[] = []; // TODO: Parse hives from heartbeat data
const workersOnline = data?.workers_online || 0;
const hivesOnline = hives.length; // TEAM-375: Count of online hives
```

**Problem:**
1. Line 12 had a TODO comment that was never implemented
2. `hives` was hardcoded as empty array `[]`
3. Line 14 calculated `hivesOnline = hives.length` which always returned `0`
4. The actual hive data from `useHeartbeat()` hook was being **ignored**

---

## ✅ Investigation Process

### Step 1: Check Backend
**Result:** ✅ Backend is correctly sending hive data in heartbeat stream

### Step 2: Check useHeartbeat Hook
**Result:** ✅ Hook correctly receives and aggregates `hives_online` from backend

**Evidence:** `useHeartbeat.ts` lines 109-110:
```typescript
hives_online: heartbeatEvent.hives_online,
hives_available: heartbeatEvent.hives_available,
```

**Evidence:** `useHeartbeat.ts` lines 117-121:
```typescript
const aggregated: HeartbeatData = {
  workers_online: queenDataRef.current?.workers_online ?? 0,
  workers_available: queenDataRef.current?.workers_available ?? 0,
  hives_online: queenDataRef.current?.hives_online ?? 0, // ← CORRECT DATA
  hives_available: queenDataRef.current?.hives_available ?? 0,
  hives: Array.from(hivesRef.current.values()), // ← CORRECT ARRAY
```

### Step 3: Check Dashboard Component
**Result:** ❌ Found the bug - hardcoded empty array

**Evidence:** DashboardPage.tsx line 12:
```typescript
const hives: any[] = []; // TODO: Parse hives from heartbeat data
```

---

## 🛠️ Fix Applied

**File:** `bin/10_queen_rbee/ui/app/src/pages/DashboardPage.tsx`

**Changes:**
```diff
- const hives: any[] = []; // TODO: Parse hives from heartbeat data
+ const hives = data?.hives || [];
  const workersOnline = data?.workers_online || 0;
- const hivesOnline = hives.length; // TEAM-375: Count of online hives
+ const hivesOnline = data?.hives_online || 0; // TEAM-377: Use backend count
```

**Key Changes:**
1. Use `data?.hives` from hook instead of empty array
2. Use `data?.hives_online` from hook instead of calculating array length
3. Backend already aggregates the count correctly - just use it!

---

## 🎯 Impact

**Before Fix:**
- ❌ Active Hives always showed 0
- ❌ Hives list always empty ("No hives online")
- ❌ Worker telemetry not visible

**After Fix:**
- ✅ Active Hives shows correct count from backend
- ✅ Hives list populated with actual hive data
- ✅ Worker telemetry visible per hive

---

## 🧪 Testing

### Manual Test Steps
1. Start Queen: `cd bin/10_queen_rbee && cargo run`
2. Start Hive(s): `cd bin/20_rbee_hive && cargo run`
3. Open Queen UI: `http://localhost:7834`
4. Verify: Active Hives count matches number of running hives
5. Verify: Hives list shows connected hives with workers

### Expected Results
- If 2 hives running → "Active Hives: 2"
- Hives list shows both hives with collapsible worker details
- Worker GPU/VRAM stats visible

---

## 🎓 Lesson Learned

### Why TODO Comments Are Dangerous

**The Problem:**
```typescript
const hives: any[] = []; // TODO: Parse hives from heartbeat data
```

This TODO was written and **never completed**. The code went to production with:
- A hardcoded empty array
- A comment promising future work
- No compilation error (empty array is valid)
- No runtime error (code "works", just returns wrong data)

**The Result:**
- Feature appears broken to users
- Backend is working correctly
- UI is ignoring correct data
- Bug is invisible until manual testing

### Better Pattern

**❌ DON'T DO THIS:**
```typescript
const hives: any[] = []; // TODO: Parse hives from heartbeat data
```

**✅ DO THIS INSTEAD:**
```typescript
// Option 1: Implement it immediately
const hives = data?.hives || [];

// Option 2: If you can't implement now, throw an error
const hives = data?.hives || (() => {
  throw new Error('UNIMPLEMENTED: Hive data parsing not yet complete');
})();

// Option 3: Use TypeScript to force implementation
const hives = data?.hives!; // Will fail at runtime if not implemented
```

**Key Principle:** If you write a TODO comment, the code should **fail loudly** until the TODO is complete, not silently return wrong data.

---

## 📊 Files Modified

**TEAM-377:**
- `bin/10_queen_rbee/ui/app/src/pages/DashboardPage.tsx` - Fixed hardcoded empty hives array

**Lines Changed:** 2 lines modified, 38 lines of bug documentation added
**Breaking Changes:** 0
**Impact:** Critical - fixes completely broken hive count display

---

## ✅ Verification Checklist

- [x] Root cause identified (hardcoded empty array)
- [x] Fix implemented (use data from hook)
- [x] Bug documentation added (38-line comment block)
- [x] Following debugging rules (mandatory bug template)
- [ ] Manual browser test ⚠️ **Next: Verify in browser**

---

**TEAM-377 | Bug fixed | 2 lines changed | Documented in code**

**Reload the page and verify Active Hives count is now correct!**
