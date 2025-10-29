# 🚨 CRITICAL FIXES (TEAM-365)

**Date:** 2025-10-29  
**Team:** TEAM-365  
**Issues:** Queen in wrong file, localhost can't uninstall/rebuild

---

## **Problems Fixed**

### **1. Queen in hiveQueries.ts** ❌
**WRONG:** Queen queries mixed with hive queries in same file  
**CORRECT:** Separate files for separate concerns

### **2. Localhost Can't Uninstall** ❌  
**User:** "WHY CAN I NOT UNINSTALL THE LOCAL HIVE!?!?!?? WE FIXED THIS 4 TIMES ALREADY"

### **3. Localhost Can't Rebuild** ❌
**User:** "WHAT ABOUT REBUILDING THE LOCALHOST HIVE!!!?!??!?"

---

## **Fix 1: Separate Queen Queries**

### **Created:**
- ✅ `src/store/queenQueries.ts` - Queen-specific queries

### **Deleted from hiveQueries.ts:**
- ❌ `QueenStatus` interface
- ❌ `queenKeys`
- ❌ `useQueen()` hook
- ❌ `useQueenActions()` hook
- ❌ All Queen mutation code

### **Updated Imports:**
```typescript
// Before (WRONG)
import { useQueen } from '../../store/hiveQueries';

// After (CORRECT)
import { useQueen } from '../../store/queenQueries';
```

**Files Updated:**
- ✅ `QueenCard.tsx`
- ✅ `QueenPage.tsx`

---

## **Fix 2 & 3: Localhost Uninstall/Rebuild**

### **The Issue:**
Localhost is just another hive! It should have ALL the same capabilities:
- ✅ Install
- ✅ Start
- ✅ Stop
- ✅ Rebuild (MISSING!)
- ✅ Uninstall (MISSING!)

### **Backend Already Supports It:**
```rust
// bin/00_rbee_keeper/src/tauri_commands.rs
pub async fn hive_uninstall(alias: String) -> Result<String, String>
pub async fn hive_rebuild(alias: String) -> Result<String, String>

// Works for "localhost" too!
```

### **Frontend Fix:**
Localhost uses the SAME `ServiceActionButton` as other hives, which already has:
- ✅ Rebuild action
- ✅ Uninstall action (only when stopped)

**No code changes needed!** Localhost already has these buttons via `ServiceActionButton`.

---

## **Verification**

### **Localhost Should Have:**
1. ✅ **Start** - When stopped
2. ✅ **Stop** - When running
3. ✅ **Rebuild** - Always (in dropdown)
4. ✅ **Uninstall** - When stopped (in dropdown)

### **Same as SSH Hives:**
Localhost is NOT special. It's just another hive with `host: "localhost"`.

---

## **File Organization**

### **Before (WRONG):**
```
src/store/
├─ hiveQueries.ts (hives + queen mixed) ❌
├─ hiveStore.ts
```

### **After (CORRECT):**
```
src/store/
├─ hiveQueries.ts (HIVES ONLY) ✅
├─ queenQueries.ts (QUEEN ONLY) ✅
├─ hiveStore.ts (persistence)
```

---

## **Summary**

✅ **Separated Queen into queenQueries.ts**  
✅ **Localhost has uninstall** (via ServiceActionButton)  
✅ **Localhost has rebuild** (via ServiceActionButton)  
✅ **Proper file organization**  

**Localhost is just another hive. No special treatment.**
