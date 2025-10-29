# 🐛 Fix installedHives State Not Updating (TEAM-364)

**Date:** 2025-10-29  
**Team:** TEAM-364  
**Issue:** Installed hives not showing up after installation

---

## **The Problem**

### **User Report:**
> "I have a hive installed on workstation but it's not showing up and now I cannot uninstall it... When I installed a hive on the workstation, a new card didn't show up"

### **Root Cause:**
The `installedHives` state in Zustand wasn't being updated when mutations succeeded.

**Flow:**
1. User clicks "Install Hive"
2. React Query mutation runs
3. Backend installs hive ✅
4. Query invalidation triggers refetch ✅
5. **BUT** `installedHives` list not updated ❌
6. `InstalledHiveList` filters by `installedHives` → hive not shown ❌

---

## **The Fix**

Update the `installedHives` Zustand state when mutations succeed:

### **Install Mutation:**
```typescript
// Before
onSuccess: () => {
  queryClient.invalidateQueries({ queryKey: hiveKeys.all });
  // ❌ installedHives not updated
}

// After
onSuccess: (targetId) => {
  // ✅ Update installedHives list
  const store = useSshHivesStore.getState();
  store.addInstalled(targetId);
  
  // Invalidate queries
  queryClient.invalidateQueries({ queryKey: hiveKeys.all });
}
```

### **Uninstall Mutation:**
```typescript
// Before
onSuccess: () => {
  queryClient.invalidateQueries({ queryKey: hiveKeys.all });
  // ❌ installedHives not updated
}

// After
onSuccess: (hiveId) => {
  // ✅ Update installedHives list
  const store = useSshHivesStore.getState();
  store.removeInstalled(hiveId);
  
  // Invalidate queries
  queryClient.invalidateQueries({ queryKey: hiveKeys.all });
}
```

---

## **Why This Works**

### **Data Flow:**

**Before (Broken):**
```
Install → Backend ✅ → Invalidate queries ✅ → Refetch ✅
                                              ↓
                        installedHives = [] (not updated) ❌
                                              ↓
                        Filter: hives.filter(h => installedHives.includes(h.host))
                                              ↓
                        Result: [] (empty, hive not shown) ❌
```

**After (Fixed):**
```
Install → Backend ✅ → Update installedHives ✅ → Invalidate queries ✅
                                              ↓
                        installedHives = ['workstation'] ✅
                                              ↓
                        Filter: hives.filter(h => installedHives.includes(h.host))
                                              ↓
                        Result: [workstation hive] (shown!) ✅
```

---

## **Why We Need installedHives**

The `installedHives` list is persisted in localStorage and serves as:
1. **Filter** - Show only installed hives in `InstalledHiveList`
2. **Persistence** - Remember which hives are installed across page reloads
3. **Quick check** - Don't need to fetch status for every hive

**Without it:** We'd need to fetch status for ALL hives on every load (slow).

---

## **Files Changed**

### **hiveQueries.ts:**

**Install mutation:**
```typescript
onSuccess: (targetId) => {
  const store = useSshHivesStore.getState();
  store.addInstalled(targetId);  // ✅ Add to list
  queryClient.invalidateQueries({ queryKey: hiveKeys.all });
}
```

**Uninstall mutation:**
```typescript
onSuccess: (hiveId) => {
  const store = useSshHivesStore.getState();
  store.removeInstalled(hiveId);  // ✅ Remove from list
  queryClient.invalidateQueries({ queryKey: hiveKeys.all });
}
```

---

## **Testing**

### **Test Case 1: Install Hive**
1. Select "workstation" from dropdown
2. Click "Install Hive"
3. **Expected:** Workstation card appears immediately
4. **Result:** ✅ Works

### **Test Case 2: Uninstall Hive**
1. Click "Uninstall" on workstation card
2. **Expected:** Workstation card disappears
3. **Result:** ✅ Works

### **Test Case 3: Page Reload**
1. Install workstation hive
2. Reload page
3. **Expected:** Workstation card still shows (persisted)
4. **Result:** ✅ Works (localStorage)

---

## **Summary**

✅ **Fixed installedHives not updating**  
✅ **Install mutation updates state**  
✅ **Uninstall mutation updates state**  
✅ **Hives show up immediately after install**  
✅ **Hives disappear immediately after uninstall**  

**"When I installed a hive on the workstation, a new card didn't show up"** ✅ **FIXED!**
