---
trigger: always_on
---

# 🚨 MANDATORY FRONTEND RULES (React/TypeScript)

**Version:** 1.0 | **Date:** 2025-10-28 | **Status:** MANDATORY

## ⚠️ CRITICAL: READ THIS FIRST

Violations result in: **REJECTED work**, **DELETED code**.

---

## 🔥 RULE ZERO: UPDATE EXISTING COMPONENTS, DON'T CREATE NEW ONES

**THE #1 CAUSE OF FRONTEND ENTROPY: ORPHANED COMPONENTS**

❌ **BANNED:**
- Creating `ComponentV2.tsx`, `ComponentNew.tsx`, `ComponentFixed.tsx`
- Leaving old components orphaned without `@deprecated`

✅ **REQUIRED:**
- **UPDATE THE EXISTING COMPONENT** - Let TypeScript find all usages
- **DELETE deprecated components immediately**
- **Fix TypeScript errors** - That's what the compiler is for
- **One component per purpose**

**Refinements are IMPROVEMENTS, not NEW COMPONENTS.** If refining `QueenCard`, update `QueenCard.tsx`, not create `QueenCardRefined.tsx`.

| Scenario | ❌ WRONG | ✅ RIGHT |
|----------|---------|---------|
| Add prop | Create `ComponentWithProp.tsx` | Add prop to `Component.tsx` |
| Fix styling | Create `ComponentFixed.tsx` | Fix in `Component.tsx` |
| Better UX | Create `ComponentV2.tsx` | Update `Component.tsx` |

---

## 1. SEPARATION OF CONCERNS: COMPONENTS vs CONTAINERS

**Components = Presentation. Containers = Business logic. NEVER MIX.**

**Components** (`src/components/`):
✅ JSX, CSS, props, local UI state, event handlers
❌ API calls, business logic, global state, useEffect with APIs, auth, navigation, WebSocket/SSE

**✅ CORRECT Component:**
```tsx
interface QueenCardProps {
  status: 'running' | 'stopped' | 'error';
  onStart: () => void;
  error?: string;
}

export function QueenCard({ status, onStart, error }: QueenCardProps) {
  return (
    <Card>
      <CardHeader><h2>Queen Status</h2></CardHeader>
      <CardContent>
        {error && <ErrorMessage>{error}</ErrorMessage>}
        <Button onClick={onStart}>Start</Button>
      </CardContent>
    </Card>
  );
}
```

**❌ WRONG Component:**
```tsx
export function QueenCard() {
  const [status, setStatus] = useState('stopped');
  useEffect(() => { fetch('/api/queen/status')... }, []); // ❌ API in component
  return <Card>...</Card>;
}
```

**Containers** (`src/containers/`):
✅ API calls, business logic, global state, useEffect, auth, navigation, WebSocket/SSE
❌ Complex JSX, CSS classes, styling, detailed markup

**✅ CORRECT Container:**
```tsx
export function QueenContainer() {
  const { status, error, start } = useQueenStore();
  return <QueenCard status={status} onStart={start} error={error} />;
}
```

**❌ WRONG Container:**
```tsx
export function QueenContainer() {
  const { status } = useQueenStore();
  return <div className="p-4 bg-white">...</div>; // ❌ Styling in container
}
```

**File Structure:**
```
src/
├── components/   # Presentation
├── containers/   # Business logic
├── hooks/        # Reusable logic
└── services/     # API clients
```

---

## 2. REFINEMENTS UPDATE EXISTING COMPONENTS

**When refining/improving:**
1. Find existing component
2. Update in place
3. Fix TypeScript errors
4. Delete old code

**Create NEW component ONLY if:**
- Completely different purpose
- New feature (not improvement)
- Refactoring large component

**Deprecation:**
```tsx
/** @deprecated Use NewComponent instead */
export function OldComponent() {}
```
Then update all imports and delete within 24 hours.

---

## 3. PROPS vs STATE MANAGEMENT

**Props:** 1-2 levels deep, specific to component tree
**Zustand:** 3+ levels, across trees, global data

```tsx
// Zustand store
export const useQueenStore = create<QueenStore>((set) => ({
  status: 'stopped',
  start: async () => {
    await queenService.start();
    set({ status: 'running' });
  },
}));
```

---

## 4. TYPESCRIPT

❌ NO `any`, NO unnecessary type assertions
✅ Use `unknown`, type guards, `import type`

```tsx
// ✅ CORRECT
import type { FC } from 'react';
const el = document.getElementById('foo');
if (el instanceof HTMLElement) { /* use el */ }

// ❌ WRONG
const data: any = await fetch('/api');
const el = document.getElementById('foo') as HTMLElement;
```

---

## 5. IMPORT ORDER

```tsx
// 1. React
import { useState } from 'react';
import type { FC } from 'react';

// 2. Third-party
import { Card } from '@/components/ui/card';

// 3. Internal (absolute)
import { useQueenStore } from '@/stores/queenStore';

// 4. Relative
import { StatusBadge } from './StatusBadge';

// 5. Styles
import './QueenCard.css';
```

---

## 6. ERROR HANDLING

```tsx
// ✅ CORRECT
const handleStart = async () => {
  try {
    await queenService.start();
  } catch (err) {
    setError(err instanceof Error ? err.message : 'Unknown error');
  }
};

// ❌ WRONG
const handleStart = async () => {
  await queenService.start(); // No error handling
};
```

---

## 7. NAMING CONVENTIONS

- **Components:** `QueenCard.tsx` (PascalCase, descriptive)
- **Containers:** `QueenContainer.tsx` (ends with Container)
- **Hooks:** `useQueenService.ts` (starts with use)
- **Services:** `queenService.ts` (ends with Service)

---

## 8. TESTING REQUIREMENTS

**Every component must:**
1. Accept props (no hardcoded data)
2. Have clear TypeScript interface
3. Be pure (same props = same output)
4. Have NO side effects (API calls, timers)

**Containers handle side effects, components don't.**

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────┐
│ FRONTEND RULES QUICK REFERENCE                          │
├─────────────────────────────────────────────────────────┤
│ 🔥 RULE ZERO:                                           │
│   ✅ UPDATE existing components, don't create new ones │
│   ❌ NO ComponentV2, ComponentNew, ComponentFixed      │
│                                                         │
│ Separation of Concerns:                                 │
│   ✅ Components = presentation only                    │
│   ✅ Containers = business logic only                  │
│   ❌ NO mixing logic and presentation                  │
│                                                         │
│ TypeScript:                                             │
│   ✅ Strict types, no any                              │
│   ✅ Type-only imports                                 │
│   ✅ Proper error handling                             │
│                                                         │
│ File Structure:                                         │
│   src/components/  → Presentation                      │
│   src/containers/  → Business logic                    │
│   src/hooks/       → Reusable logic                    │
│   src/services/    → API clients                       │
└─────────────────────────────────────────────────────────┘
```

---

## Checklist Before Committing

- [ ] No business logic in components
- [ ] No presentation in containers
- [ ] All props properly typed
- [ ] No `any` types
- [ ] Error handling in place
- [ ] Imports organized correctly
- [ ] No orphaned components
- [ ] No ComponentV2/ComponentNew files
- [ ] Updated existing components instead of creating new ones

---

**Update existing components. Don't create new ones.**

**Components = presentation. Containers = logic. NEVER MIX.**
