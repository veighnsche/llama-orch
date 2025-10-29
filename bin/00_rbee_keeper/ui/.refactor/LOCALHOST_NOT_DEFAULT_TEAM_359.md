# 🎯 Localhost Not Default (TEAM-359)

**Date:** 2025-10-29  
**Team:** TEAM-359  
**Issue:** Localhost hive showing when not installed

---

## **Problem**

The Localhost Hive card was **always showing**, even when not installed.

### **User Feedback:**
> "If the hive localhost is not installed on the machine then I don't want to see a card of it. Hives are most of the time NOT localhost. localhost hive is not the default way to use this app"

### **Evidence:**
Screenshot showed:
- Localhost Hive card visible with "Start" button
- Narration: "❌ Hive 'localhost' is not installed"
- Card should not be visible at all

---

## **Root Cause**

```typescript
// LocalhostHive.tsx - WRONG
const isInstalled = true  // ❌ Always true!
```

This assumed localhost is always installed, which is **incorrect**.

---

## **The Fix**

### **Before (Wrong Assumption):**
```typescript
const isInstalled = true  // ❌ Localhost always installed
const isRunning = hive?.status === 'online'

return (
  <Card>  {/* ❌ Always renders */}
    <LocalhostHive />
  </Card>
)
```

### **After (Correct):**
```typescript
const isInstalled = hive?.isInstalled ?? false  // ✅ Check actual status
const isRunning = hive?.status === 'online'

// TEAM-359: Don't show localhost if not installed
if (!isInstalled) {
  return null  // ✅ Hide card when not installed
}

return (
  <Card>  {/* ✅ Only renders when installed */}
    <LocalhostHive />
  </Card>
)
```

---

## **Behavior**

### **When Localhost NOT Installed:**
- ❌ No Localhost Hive card shown
- ✅ Only Queen + SSH Hives + Install Hive Card visible

### **When Localhost IS Installed:**
- ✅ Localhost Hive card shown
- ✅ Can start/stop/rebuild/uninstall

---

## **Why This Matters**

### **User's Workflow:**
1. Most users use **remote SSH hives** (not localhost)
2. Localhost is **optional**, not default
3. Showing uninstalled localhost is **confusing**

### **Correct UX:**
- **Queen** - Always shown (core service)
- **Localhost Hive** - Only if installed (optional)
- **SSH Hives** - Only installed ones (from config)
- **Install Hive Card** - Always shown (to install new hives)

---

## **Files Changed**

### **LocalhostHive.tsx**

**Changed:**
```typescript
// Before
const isInstalled = true  // ❌ Wrong assumption

// After  
const isInstalled = hive?.isInstalled ?? false  // ✅ Check actual status

// Added
if (!isInstalled) {
  return null  // ✅ Hide when not installed
}
```

---

## **Verification**

### **Test Case 1: Localhost Not Installed**
```bash
# Backend returns: isInstalled: false
# UI shows: No Localhost Hive card
```

### **Test Case 2: Localhost Installed**
```bash
# Backend returns: isInstalled: true
# UI shows: Localhost Hive card with Start/Stop
```

---

## **Architecture Note**

### **Service Visibility Rules:**

| Service | Visibility |
|---------|-----------|
| **Queen** | Always shown (core) |
| **Localhost Hive** | Only if installed (optional) |
| **SSH Hives** | Only installed ones |
| **Install Hive Card** | Always shown |

**Localhost is NOT special** - it follows the same rules as SSH hives.

---

## **Summary**

✅ **Localhost Hive only shows when installed**  
✅ **Removed wrong assumption (isInstalled = true)**  
✅ **Localhost is optional, not default**  
✅ **Consistent with SSH hive behavior**  

**"Hives are most of the time NOT localhost. localhost hive is not the default way to use this app."** ✅ FIXED!
