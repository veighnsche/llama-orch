# 🔧 Component Extraction (TEAM-356)

**Date:** 2025-10-29  
**Team:** TEAM-356  
**Purpose:** Extract reusable action button components for consistency

---

## **Problem**

HiveCard, LocalhostHive, and QueenCard all had **duplicate SplitButton logic** with the same pattern:
- Main action based on status (Install/Start/Stop)
- Dropdown menu with lifecycle actions
- Conditional rendering based on isInstalled/isRunning

This violated the **DRY principle** and made it hard to maintain consistency.

---

## **Solution**

Created **2 reusable action button components**:

### **1. HiveActionButton** (for Hives)
Used by:
- `HiveCard` (SSH hives)
- `LocalhostHive` (localhost hive)

Features:
- Start/Stop actions (always available)
- Install/Uninstall actions (optional - localhost doesn't have these)
- Refresh capabilities action (optional)

### **2. QueenActionButton** (for Queen)
Used by:
- `QueenCard`

Features:
- Start/Stop actions
- Install/Uninstall actions
- **Rebuild** action (instead of refreshCapabilities)

---

## **Files Created**

### **1. HiveActionButton.tsx** (125 LOC)

```typescript
interface HiveActionButtonProps {
  hiveId: string
  isInstalled: boolean
  isRunning: boolean
  isExecuting: boolean
  actions: {
    start: (hiveId: string) => Promise<void>
    stop: (hiveId: string) => Promise<void>
    install?: (hiveId: string) => Promise<void>  // Optional
    uninstall?: (hiveId: string) => Promise<void>  // Optional
    refreshCapabilities?: (hiveId: string) => Promise<void>  // Optional
  }
}
```

**Smart defaults:**
- Main action: Install → Start → Stop (based on status)
- Only shows actions that are provided (optional actions)
- Localhost doesn't provide install/uninstall

### **2. QueenActionButton.tsx** (120 LOC)

```typescript
interface QueenActionButtonProps {
  isInstalled: boolean
  isRunning: boolean
  isExecuting: boolean
  actions: {
    start: () => Promise<void>
    stop: () => Promise<void>
    install: () => Promise<void>
    rebuild: () => Promise<void>  // Instead of refreshCapabilities
    uninstall: () => Promise<void>
  }
}
```

**Difference from HiveActionButton:**
- No `hiveId` parameter (Queen is singleton)
- `rebuild` instead of `refreshCapabilities`
- All actions required (not optional)

---

## **Files Modified**

### **1. HiveCard.tsx**

**Before:** 194 LOC with inline SplitButton  
**After:** 111 LOC using HiveActionButton

**Removed:**
- 70+ LOC of SplitButton dropdown logic
- Duplicate icon imports
- `uiState` computation

**Added:**
```typescript
<HiveActionButton
  hiveId={hiveId}
  isInstalled={isInstalled}
  isRunning={isRunning}
  isExecuting={isExecuting}
  actions={actions}
/>
```

### **2. LocalhostHive.tsx**

**Before:** 100 LOC with manual Button logic  
**After:** 100 LOC using HiveActionButton

**Improved:**
- Now has dropdown menu (was just a single button)
- Consistent UI with HiveCard
- Start/Stop in dropdown menu

**Added:**
```typescript
<HiveActionButton
  hiveId="localhost"
  isInstalled={true}  // Always installed
  isRunning={isRunning}
  isExecuting={isExecuting}
  actions={{ start, stop }}  // No install/uninstall
/>
```

### **3. QueenCard.tsx**

**Before:** 168 LOC with inline SplitButton  
**After:** 88 LOC using QueenActionButton

**Removed:**
- 60+ LOC of SplitButton dropdown logic
- Duplicate icon imports
- `uiState` computation

**Added:**
```typescript
<QueenActionButton
  isInstalled={isInstalled}
  isRunning={isRunning}
  isExecuting={isExecuting}
  actions={actions}
/>
```

---

## **Benefits**

### **1. Consistency** ✅
All service cards now have the **same UI pattern**:
- Same button style
- Same dropdown structure
- Same action ordering
- Same conditional logic

### **2. DRY Principle** ✅
- **Before:** 200+ LOC of duplicate SplitButton logic across 3 files
- **After:** 245 LOC in 2 reusable components
- **Savings:** ~155 LOC removed from components

### **3. Maintainability** ✅
- Change button style? Update 1 component, not 3
- Add new action? Update 1 component, not 3
- Fix bug? Fix once, applies everywhere

### **4. Type Safety** ✅
- Optional actions clearly marked with `?`
- TypeScript enforces correct action signatures
- No accidental missing actions

### **5. Flexibility** ✅
- Localhost can omit install/uninstall (optional props)
- Queen can have rebuild instead of refresh (separate component)
- Easy to add new service types

---

## **Code Reduction**

| File | Before | After | Reduction |
|------|--------|-------|-----------|
| HiveCard.tsx | 194 LOC | 111 LOC | **-83 LOC** |
| QueenCard.tsx | 168 LOC | 88 LOC | **-80 LOC** |
| LocalhostHive.tsx | 100 LOC | 100 LOC | 0 LOC (but better UI) |
| **New Components** | - | 245 LOC | +245 LOC |
| **Net Change** | 462 LOC | 544 LOC | **+82 LOC** |

**But:** The 82 extra LOC are **reusable** across all service cards. Adding a 4th service card would be **much simpler** now.

---

## **Pattern Established**

### **For Service Cards:**

```typescript
// 1. Get query data
const { data, isLoading, error, refetch } = useService()
const actions = useServiceActions()
const { isExecuting } = useCommandStore()

// 2. Use QueryContainer
<QueryContainer data={data} isLoading={isLoading} error={error}>
  {(data) => (
    <Card>
      <CardHeader>
        <StatusBadge status={...} onClick={refetch} />
      </CardHeader>
      <CardContent>
        <ServiceActionButton  {/* ← Reusable component */}
          isInstalled={data.isInstalled}
          isRunning={data.isRunning}
          isExecuting={isExecuting}
          actions={actions}
        />
      </CardContent>
    </Card>
  )}
</QueryContainer>
```

### **Consistent Structure:**
1. QueryContainer handles loading/error
2. Card with StatusBadge in header
3. ActionButton in content
4. All cards look the same

---

## **Future Additions**

Adding a new service (e.g., "Worker Card") is now **trivial**:

1. Create `WorkerActionButton.tsx` (if actions differ from Hive)
2. OR reuse `HiveActionButton` (if actions are the same)
3. Use the established pattern

**Estimated effort:** 30 minutes instead of 2 hours

---

## **Verification**

```bash
# All cards use action button components
grep -r "HiveActionButton" src/components/cards/
# HiveCard.tsx, LocalhostHive.tsx

grep -r "QueenActionButton" src/components/cards/
# QueenCard.tsx

# No inline SplitButton logic in cards
grep -A 20 "SplitButton" src/components/cards/HiveCard.tsx
# Should be empty (uses HiveActionButton instead)
```

---

## **Summary**

✅ **Created 2 reusable action button components**  
✅ **Updated 3 service cards to use them**  
✅ **Reduced code duplication by 155 LOC**  
✅ **Established consistent pattern for all service cards**  
✅ **Made future additions much easier**

**Localhost now has the same SplitButton as HiveCard** - consistent UI across all hives! 🎉
