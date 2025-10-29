# 🚨 CRITICAL FIX: Remove refreshCapabilities (TEAM-358)

**Date:** 2025-10-29  
**Team:** TEAM-358  
**Severity:** CRITICAL  
**Issue:** Internal Queen operation exposed in Keeper UI

---

## **The Problem**

`refreshCapabilities` is an **internal Queen operation** for managing hive capabilities. It should **NEVER** be exposed in the Bee Keeper UI.

### **Why This Is Wrong:**

1. **Architecture violation** - Keeper should not know about Queen's internal operations
2. **Confusing UX** - Users don't understand "refresh capabilities"
3. **Wrong abstraction** - Keeper needs "rebuild", not "refresh capabilities"

### **User Feedback:**
> "What is refresh capabilities doing in the bee keeper front-end??? That is a huge mistake that we completely need to remove. The refresh capabilities is a Queens internal action for the hive. I am seething seeing refreshCapabilities being exposed in the keeper"

---

## **The Fix**

Replaced `refreshCapabilities` with `rebuild` throughout the Keeper UI.

### **Backend Already Has It:**
```rust
// bin/00_rbee_keeper/src/tauri_commands.rs
#[tauri::command]
pub async fn hive_rebuild(alias: String) -> Result<String, String> {
    // Calls daemon-lifecycle rebuild.rs
    // Process: build → stop → install → start
}
```

### **What `rebuild` Does:**
1. Build binary locally
2. Stop running daemon (if running)
3. Install new binary
4. Start daemon with new binary

**This is what users want** - not "refresh capabilities"!

---

## **Changes Made**

### **1. hiveStore.ts**

**Before (WRONG):**
```typescript
interface SshHivesState {
  refreshCapabilities: (hiveId: string) => Promise<void>  // ❌ Internal Queen op
}

refreshCapabilities: async (hiveId: string) => {
  await withCommandExecution(
    () => commands.hiveRefreshCapabilities(hiveId),  // ❌ Wrong command
    () => get().invalidate(hiveId),
    "Hive refresh capabilities",  // ❌ Confusing
  );
}

export function useHiveActions() {
  return {
    refreshCapabilities: store.refreshCapabilities,  // ❌ Exposed
  };
}
```

**After (CORRECT):**
```typescript
interface SshHivesState {
  rebuild: (hiveId: string) => Promise<void>  // ✅ User-facing operation
}

rebuild: async (hiveId: string) => {
  await withCommandExecution(
    () => commands.hiveRebuild(hiveId),  // ✅ Correct command
    () => get().invalidate(hiveId),
    "Hive rebuild",  // ✅ Clear
  );
}

export function useHiveActions() {
  return {
    rebuild: store.rebuild,  // ✅ Correct abstraction
  };
}
```

### **2. HiveCard.tsx**

**Before:**
```typescript
const { refreshCapabilities } = useHiveActions()  // ❌
actions={{ rebuild: (id) => refreshCapabilities(id!) }}  // ❌ Wrong mapping
```

**After:**
```typescript
const { rebuild } = useHiveActions()  // ✅
actions={{ rebuild: (id) => rebuild(id!) }}  // ✅ Direct mapping
```

### **3. LocalhostHive.tsx**

**Before:**
```typescript
const { refreshCapabilities } = useHiveActions()  // ❌
rebuild: (id) => refreshCapabilities(id!)  // ❌ Wrong mapping
```

**After:**
```typescript
const { rebuild } = useHiveActions()  // ✅
rebuild: (id) => rebuild(id!)  // ✅ Direct mapping
```

---

## **Verification**

```bash
# No more refreshCapabilities in UI code
grep -r "refreshCapabilities" bin/00_rbee_keeper/ui/src/
# Should only find in generated bindings (can't remove from there)

# All uses replaced with rebuild
grep -r "rebuild" bin/00_rbee_keeper/ui/src/store/hiveStore.ts
# Should see: rebuild: (hiveId: string) => Promise<void>

grep -r "rebuild" bin/00_rbee_keeper/ui/src/components/cards/
# Should see: const { rebuild } = useHiveActions()
```

---

## **Architecture Boundary**

### **Keeper UI (User-Facing):**
- ✅ `rebuild` - Rebuild and hot-reload hive
- ✅ `start` - Start hive
- ✅ `stop` - Stop hive
- ✅ `install` - Install hive
- ✅ `uninstall` - Uninstall hive

### **Queen Internal (NOT User-Facing):**
- ❌ `refreshCapabilities` - Internal Queen operation
- ❌ Should NEVER appear in Keeper UI

---

## **Why This Matters**

### **Before (Broken Abstraction):**
```
User → Keeper UI → "Refresh Capabilities" → ??? (confusing)
                       ↓
                   Queen internal operation (wrong layer)
```

### **After (Correct Abstraction):**
```
User → Keeper UI → "Rebuild" → hive_rebuild command
                       ↓
                   Build → Stop → Install → Start (clear)
```

---

## **Files Changed**

1. ✅ `src/store/hiveStore.ts` - Replaced refreshCapabilities with rebuild
2. ✅ `src/components/cards/HiveCard.tsx` - Uses rebuild
3. ✅ `src/components/cards/LocalhostHive.tsx` - Uses rebuild

---

## **Summary**

✅ **Removed `refreshCapabilities` from Keeper UI**  
✅ **Replaced with `rebuild` (correct abstraction)**  
✅ **Backend already has `hive_rebuild` command**  
✅ **Architecture boundary restored**  

**`refreshCapabilities` is a Queen internal operation and should NEVER be exposed in Keeper UI.**

**This fix restores the correct abstraction layer.**
