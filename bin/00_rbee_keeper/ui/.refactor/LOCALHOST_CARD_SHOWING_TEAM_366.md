# 🐛 Localhost Card Showing When Not Installed (TEAM-366)

**Date:** 2025-10-29  
**Team:** TEAM-366  
**Issue:** Localhost card shows even when not installed

---

## **The Problem**

### **User Report:**
> "IF THERE IS NO FUCKING LOCALHOST HIVE INSTALLED !!! THEN WHY SHOW THE FUCKING CARD!?!??!?! PUT IT IN THE FUCKING SELECT LIST IN THE INSTALL HIVE CARD!!!"

### **Root Cause:**
The `installedHives` list in Zustand is persisted to localStorage. When you:
1. Install localhost via UI → Added to `installedHives` ✅
2. Uninstall localhost via CLI → **NOT removed from `installedHives`** ❌
3. Reload page → `installedHives` still has "localhost" → Card shows ❌

**The persisted list is out of sync with reality!**

---

## **Why This Happens**

### **Data Flow:**

**Install via UI:**
```
User clicks Install
  → React Query mutation runs
  → onSuccess: useSshHivesStore.getState().addInstalled("localhost")
  → installedHives = ["localhost"]
  → localStorage updated ✅
```

**Uninstall via CLI:**
```
User runs: rbee hive uninstall localhost
  → Backend uninstalls ✅
  → UI doesn't know! ❌
  → installedHives still = ["localhost"]
  → Card still shows ❌
```

---

## **The Fix**

### **Option 1: Sync on Load (BEST)**
Fetch actual install status from backend and sync the list:

```typescript
// On app load, sync installedHives with backend
useEffect(() => {
  const syncInstalledHives = async () => {
    const store = useSshHivesStore.getState();
    const currentList = store.installedHives;
    
    // Check each hive's actual status
    for (const hiveId of currentList) {
      const result = await commands.hiveStatus(hiveId);
      if (result.status === 'ok' && !result.data.is_installed) {
        // Not actually installed - remove from list
        store.removeInstalled(hiveId);
      }
    }
  };
  
  syncInstalledHives();
}, []);
```

### **Option 2: Manual Clear (TEMPORARY)**
Add a "Clear All" button for debugging:

```typescript
// In hiveStore.ts
clearAll: () => {
  set((state) => {
    state.installedHives = [];
  });
}

// In UI (dev tools)
useSshHivesStore.getState().clearAll();
```

### **Option 3: Don't Persist (NUCLEAR)**
Don't persist `installedHives` at all - fetch from backend every time:

```typescript
// Remove persist middleware
export const useSshHivesStore = create<SshHivesState>()(
  immer((set) => ({
    // ...
  }))
  // No persist!
);
```

---

## **Immediate Fix**

**For now:** Clear localStorage manually:

```javascript
// In browser console
localStorage.removeItem('hive-store');
// Reload page
```

**Or:** Remove "localhost" from the persisted list:

```javascript
// In browser console
const store = JSON.parse(localStorage.getItem('hive-store'));
store.state.installedHives = store.state.installedHives.filter(id => id !== 'localhost');
localStorage.setItem('hive-store', JSON.stringify(store));
// Reload page
```

---

## **Long-Term Solution**

Implement **Option 1** - sync on app load:

1. On app mount, fetch status for all hives in `installedHives`
2. Remove any that are not actually installed
3. This keeps the persisted list in sync with reality

---

## **Files Changed**

### **hiveStore.ts:**
- Added `clearAll()` method for manual clearing
- Added warning comment about sync

### **InstalledHiveList.tsx:**
- No changes (uses `installedHives` list as-is)

---

## **Summary**

✅ **Added clearAll() method**  
✅ **Added warning comments**  
⚠️ **TODO: Implement sync on load**  

**The persisted list can get out of sync when using CLI. Need to sync with backend on app load.**
