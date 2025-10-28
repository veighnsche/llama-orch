# TEAM-338: Status Badge Component

**Date:** Oct 28, 2025  
**Status:** ✅ COMPLETE

## Summary

Added clickable `StatusBadge` component to Queen card header that manually refreshes status when clicked. Uses Card's `CardAction` slot for proper positioning.

## Changes

### `/bin/00_rbee_keeper/ui/src/components/StatusBadge.tsx`

**Updated from static badge to clickable button:**

**Before:**
- Static `<span>` with status text
- Status values: `"online" | "offline" | "unknown"`
- No interaction

**After:**
- Polymorphic component (`<button>` or `<span>` based on `onClick` prop)
- Status values: `"running" | "stopped" | "unknown"`
- Clickable with hover/active states
- Loading state with pulsing dot animation
- Disabled state when loading

**Props:**
```typescript
interface StatusBadgeProps {
  status: "running" | "stopped" | "unknown";
  onClick?: () => void;        // Makes it a button
  isLoading?: boolean;          // Shows loading state
}
```

**Visual States:**
- **Running:** Green badge with green dot
- **Stopped:** Red badge with red dot
- **Unknown:** Gray badge with gray dot
- **Loading:** Pulsing dot animation, disabled state
- **Hover:** Slightly brighter background (when clickable)
- **Active:** Scale down to 95% (when clickable)

### `/bin/00_rbee_keeper/ui/src/components/QueenCard.tsx`

**Added StatusBadge to CardHeader:**

```tsx
<CardHeader>
  <CardTitle>Queen</CardTitle>
  <CardDescription>Smart API server</CardDescription>
  <CardAction>
    <StatusBadge
      status={badgeStatus}
      onClick={fetchStatus}
      isLoading={isLoading}
    />
  </CardAction>
</CardHeader>
```

**Badge Status Logic:**
```typescript
const badgeStatus = !isInstalled
  ? "unknown"      // Not installed → gray
  : isRunning
    ? "running"    // Running → green
    : "stopped";   // Stopped → red
```

## Card Layout

The `CardAction` component from rbee-ui positions the badge in the top-right corner of the card header:

```
┌─────────────────────────────────────────────┐
│ Queen                        [Running] ← Badge│
│ Smart API server                            │
├─────────────────────────────────────────────┤
│ Job router that dispatches inference...     │
│ [Start ▼]                                   │
└─────────────────────────────────────────────┘
```

**CSS Grid Layout (from Card.tsx):**
```css
grid-cols-[1fr_auto]  /* Title/Description | Action */
col-start-2           /* Action in second column */
row-span-2            /* Spans both title and description rows */
row-start-1           /* Starts at first row */
self-start            /* Align to top */
justify-self-end      /* Align to right */
```

## User Interaction Flow

1. **User clicks badge** → `onClick={fetchStatus}` fires
2. **Store updates** → `isLoading: true`
3. **Badge shows loading** → Pulsing dot, disabled state
4. **Backend call** → `commands.queenStatus()`
5. **Store updates** → `status: { isRunning, isInstalled }`
6. **Badge updates** → New color/text, loading stops

## Why This Pattern?

### Clickable Badge vs Refresh Button

**Considered alternatives:**
- ❌ Separate refresh icon button → Takes more space
- ❌ Refresh in dropdown menu → Hidden, not discoverable
- ✅ **Clickable badge** → Space-efficient, discoverable, intuitive

**Benefits:**
1. **Space-efficient** - Badge already shows status, why add another button?
2. **Discoverable** - Hover state indicates it's clickable
3. **Intuitive** - Click status to refresh status (natural mapping)
4. **Consistent** - Same pattern can be used for Hive cards

### Polymorphic Component Pattern

```typescript
const Component = onClick ? "button" : "span";
```

**Why?**
- Semantic HTML - `<button>` when interactive, `<span>` when static
- Accessibility - Screen readers announce buttons correctly
- No wrapper divs - Clean DOM structure

## Accessibility

**When clickable (button):**
- ✅ Keyboard accessible (Tab + Enter)
- ✅ Screen reader announces as button
- ✅ Disabled state prevents interaction
- ✅ Visual feedback (hover, active, disabled)

**When static (span):**
- ✅ No tab stop (not interactive)
- ✅ Screen reader reads as text

## Status Mapping

| Backend State | Badge Status | Badge Color | Badge Text |
|--------------|--------------|-------------|------------|
| Not installed | `unknown` | Gray | "Unknown" |
| Installed, not running | `stopped` | Red | "Stopped" |
| Installed, running | `running` | Green | "Running" |

## Loading States

### Initial Load (Suspense)
```tsx
<QueenDataProvider>  {/* Shows loading fallback */}
  <QueenCard />
</QueenDataProvider>
```

### Manual Refresh (Badge)
```tsx
<StatusBadge
  status="running"
  onClick={fetchStatus}
  isLoading={true}  {/* Pulsing dot, disabled */}
/>
```

**Different purposes:**
- **Suspense** - First load, no data yet
- **Badge loading** - Refresh, data exists but updating

## CSS Classes

### Base Styles (All States)
```css
inline-flex items-center rounded-full
px-2 py-1 text-xs font-medium
transition-colors
```

### Status Colors
```css
/* Running */
bg-green-500/10 text-green-500

/* Stopped */
bg-red-500/10 text-red-500

/* Unknown */
bg-gray-500/10 text-gray-500
```

### Interactive States (Button Only)
```css
/* Hover */
hover:bg-opacity-20 cursor-pointer

/* Active */
active:scale-95

/* Disabled */
opacity-50 cursor-not-allowed
```

### Loading Animation
```css
/* Dot */
animate-pulse  /* Tailwind's built-in pulse animation */
```

## Future Enhancements

Consider adding:
1. **Tooltip** - "Click to refresh" on hover
2. **Last updated time** - "Updated 5s ago"
3. **Auto-refresh** - Refresh every 30s
4. **Error state** - Red badge with error icon
5. **Transition animations** - Smooth color changes

## Related Components

**Similar pattern can be applied to:**
- `HiveCard` - Show hive status badge
- `WorkerCard` - Show worker status badge
- `ModelCard` - Show model download status

**Reusable StatusBadge props:**
```typescript
// Hive card
<StatusBadge
  status={hiveStatus}
  onClick={() => refreshHiveStatus(alias)}
  isLoading={isRefreshing}
/>

// Worker card
<StatusBadge
  status={workerStatus}
  onClick={() => refreshWorkerStatus(workerId)}
  isLoading={isRefreshing}
/>
```

---

**Pattern:** Clickable status badges in card headers provide space-efficient, discoverable status refresh functionality.
