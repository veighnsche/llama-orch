# 🎯 Unified Service Action Button (TEAM-357)

**Date:** 2025-10-29  
**Team:** TEAM-357  
**Purpose:** Create single action button component for ALL services

---

## **Problem**

User feedback from screenshots showed:
1. ❌ **Duplicate actions** - Start/Stop showing twice in dropdown
2. ❌ **Inconsistent naming** - "Refresh" vs "Update" vs "Rebuild"
3. ❌ **Wrong logic** - Uninstall available when running
4. ❌ **Different components** - HiveActionButton, QueenActionButton, manual buttons

**User requirement:** "They can all have the exact same split button. And there are no exceptions."

---

## **Solution**

Created **ONE unified component** that works for Queen, Localhost, and all SSH Hives.

### **ServiceActionButton.tsx** (95 LOC)

```typescript
interface ServiceActionButtonProps {
  serviceId: string  // "localhost", hiveId, or "queen"
  isInstalled: boolean
  isRunning: boolean
  isExecuting: boolean
  actions: {
    start: (id?: string) => Promise<void>
    stop: (id?: string) => Promise<void>
    install?: (id?: string) => Promise<void>
    rebuild?: (id?: string) => Promise<void>
    uninstall?: (id?: string) => Promise<void>
  }
}
```

### **Logic Rules:**

#### **Main Button:**
- Not installed → **Install** (if action provided)
- Installed + Running → **Stop** (red/destructive)
- Installed + Stopped → **Start** (default)

#### **Dropdown Menu:**
1. **Rebuild** - Always shown when installed (works even when running)
2. **Uninstall** - Only when stopped (separator before it)

**That's it!** No Start/Stop in dropdown (they're the main button).

---

## **Key Features**

### **1. No Duplicates** ✅
- Main button is Start OR Stop (never both)
- Dropdown only has Rebuild + Uninstall
- No duplicate actions anywhere

### **2. Consistent Naming** ✅
- **Rebuild** everywhere (not "Refresh" or "Update")
- Same terminology across Queen, Localhost, SSH Hives

### **3. Correct Logic** ✅
- Uninstall only when stopped (can't uninstall running service)
- Rebuild available when running (backend handles stop/rebuild/start)

### **4. Universal** ✅
- Works for Queen (no hiveId needed)
- Works for Localhost (has uninstall now!)
- Works for SSH Hives (same as always)

---

## **Files Changed**

### **Deleted (Rule Zero):**
- ❌ `HiveActionButton.tsx` (125 LOC)
- ❌ `QueenActionButton.tsx` (120 LOC)

### **Created:**
- ✅ `ServiceActionButton.tsx` (95 LOC)

### **Updated:**
- ✅ `HiveCard.tsx` - Uses ServiceActionButton
- ✅ `LocalhostHive.tsx` - Uses ServiceActionButton (now has uninstall!)
- ✅ `QueenCard.tsx` - Uses ServiceActionButton

---

## **Comparison**

### **Before (3 different components):**

| Component | Main Actions | Dropdown | Issues |
|-----------|-------------|----------|--------|
| HiveActionButton | Install/Start/Stop | Start, Stop, Install, Refresh, Uninstall | Duplicates, "Refresh" |
| QueenActionButton | Install/Start/Stop | Start, Stop, Install, Update, Uninstall | Duplicates, "Update" |
| LocalhostHive | Start/Stop (manual) | None | No dropdown |

### **After (1 unified component):**

| Component | Main Actions | Dropdown | Issues |
|-----------|-------------|----------|--------|
| ServiceActionButton | Install/Start/Stop | Rebuild, Uninstall (when stopped) | **None** ✅ |

---

## **Usage Examples**

### **Queen:**
```typescript
<ServiceActionButton
  serviceId="queen"
  isInstalled={isInstalled}
  isRunning={isRunning}
  isExecuting={isExecuting}
  actions={{
    start: () => queenActions.start(),
    stop: () => queenActions.stop(),
    install: () => queenActions.install(),
    rebuild: () => queenActions.rebuild(),
    uninstall: () => queenActions.uninstall(),
  }}
/>
```

### **Localhost:**
```typescript
<ServiceActionButton
  serviceId="localhost"
  isInstalled={true}  // Always installed
  isRunning={isRunning}
  isExecuting={isExecuting}
  actions={{
    start: (id) => start(id!),
    stop: (id) => stop(id!),
    rebuild: (id) => refreshCapabilities(id!),
    uninstall: (id) => uninstall(id!),  // ← Now has uninstall!
  }}
/>
```

### **SSH Hive:**
```typescript
<ServiceActionButton
  serviceId={hiveId}
  isInstalled={isInstalled}
  isRunning={isRunning}
  isExecuting={isExecuting}
  actions={{
    start: (id) => actions.start(id!),
    stop: (id) => actions.stop(id!),
    install: (id) => actions.install(id!),
    rebuild: (id) => actions.refreshCapabilities(id!),
    uninstall: (id) => actions.uninstall(id!),
  }}
/>
```

---

## **UI States**

### **Not Installed:**
- **Main Button:** Install (default)
- **Dropdown:** Empty (nothing to rebuild/uninstall)

### **Installed + Stopped:**
- **Main Button:** Start (default)
- **Dropdown:** Rebuild, Uninstall

### **Installed + Running:**
- **Main Button:** Stop (destructive/red)
- **Dropdown:** Rebuild (no Uninstall - can't uninstall while running)

---

## **Benefits**

### **1. Consistency** ✅
- **Same UI** across Queen, Localhost, SSH Hives
- **Same logic** everywhere
- **Same terminology** (Rebuild, not Refresh/Update)

### **2. Simplicity** ✅
- **1 component** instead of 3
- **95 LOC** instead of 245 LOC
- **Easier to maintain** (change once, applies everywhere)

### **3. Correctness** ✅
- **No duplicates** (Start/Stop only in main button)
- **Uninstall only when stopped** (can't uninstall running service)
- **Rebuild when running** (backend handles it)

### **4. User Satisfaction** ✅
- **Addresses all user complaints** from screenshots
- **"No exceptions"** - truly unified

---

## **Code Reduction**

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Components** | 3 files | 1 file | **-2 files** |
| **Total LOC** | 245 LOC | 95 LOC | **-150 LOC (61% reduction)** |
| **Consistency** | 3 different patterns | 1 pattern | **100% consistent** |

---

## **Verification**

```bash
# Only one action button component
ls src/components/cards/*ActionButton.tsx
# ServiceActionButton.tsx

# All cards use it
grep -r "ServiceActionButton" src/components/cards/
# HiveCard.tsx, LocalhostHive.tsx, QueenCard.tsx

# No old components
grep -r "HiveActionButton\|QueenActionButton" src/
# Should be empty
```

---

## **Summary**

✅ **Created single unified ServiceActionButton**  
✅ **Deleted 2 old components (Rule Zero)**  
✅ **150 LOC removed (61% reduction)**  
✅ **Fixed all user-reported issues:**
   - No duplicate Start/Stop
   - Consistent "Rebuild" terminology
   - Uninstall only when stopped
   - Same component for all services

**"They can all have the exact same split button. And there are no exceptions."** ✅ DONE!
