# Data Fetching Pattern with React 19 `use()` Hook

**Author:** TEAM-296, TEAM-338  
**Date:** October 28, 2025  
**Status:** Active Pattern

## 🚨 CRITICAL: WHAT THIS PATTERN IS

This pattern creates **COMPONENT-AGNOSTIC DATA PROVIDERS** that use **RENDER PROPS**.

## The Split

### 1. **Container** (`*Container.tsx`) - DATA LAYER ONLY
- ✅ **ONLY** data fetching logic
- ✅ **ONLY** promise caching
- ✅ **ONLY** error boundary
- ✅ **ONLY** Suspense wrapper
- ✅ Exports `*DataProvider` component with render prop pattern
- ✅ Exports type definitions
- ❌ **NO** presentation components
- ❌ **NO** UI imports (Card, Table, Button, etc.)
- ❌ **NO** loading fallbacks
- ❌ **NO** JSX except error boundary

### 2. **Presentation Component** (`*Card.tsx`, `*Table.tsx`) - UI ONLY
- ✅ Pure presentation logic
- ✅ Receives data as props
- ✅ Includes loading fallback component
- ✅ All UI/styling logic
- ❌ **NO** data fetching
- ❌ **NO** `use()` hook
- ❌ **NO** Suspense

## 🔥 RULE ZERO: CONTAINERS ARE COMPONENT AGNOSTIC

**Containers provide DATA via render props. Consumers provide PRESENTATION.**

```tsx
// ❌ WRONG - Container includes presentation
export function SshHivesContainer() {
  return (
    <SshHivesDataProvider>
      {(hives, onRefresh) => (
        <SshHivesTable hives={hives} onRefresh={onRefresh} />
      )}
    </SshHivesDataProvider>
  );
}

// ✅ RIGHT - Container is component agnostic
export function SshHivesDataProvider({ children, fallback }) {
  // ... data fetching logic ...
  return (
    <ErrorBoundary>
      <Suspense fallback={fallback}>
        {children(data, refresh)}
      </Suspense>
    </ErrorBoundary>
  );
}
```

## File Structure

```
components/
├── QueenCard.tsx           # ✅ Presentation + Loading fallback
containers/
└── QueenContainer.tsx      # ✅ Data fetching ONLY (component agnostic)
pages/
└── ServicesPage.tsx        # ✅ Consumer wires them together
```

## Reference Implementation

**See:** `QueenContainer.tsx` (data layer) + `QueenCard.tsx` (presentation) + `ServicesPage.tsx` (usage)

## Implementation Guide

### 1. Container (`*Container.tsx`) - DATA LAYER ONLY

```tsx
// QueenContainer.tsx
import { use, useState, Suspense, useCallback, Component, type ReactNode } from "react";
import { commands } from "@/generated/bindings";
import { AlertCircle } from "lucide-react";
import { Button } from "@rbee/ui/atoms";

// Re-export type from presentation component
export type { QueenStatus } from "../components/QueenCard";

// Promise cache
const promiseCache = new Map<string, Promise<any>>();

// Fetch function
async function fetchQueenStatus(key: string): Promise<any> {
  if (!promiseCache.has(key)) {
    const promise = commands.queenStatus(); // Your Tauri command
    promiseCache.set(key, promise);
  }
  return promiseCache.get(key)!;
}

// Error boundary (generic error UI)
class QueenErrorBoundary extends Component<
  { children: ReactNode; onReset: () => void },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: { children: ReactNode; onReset: () => void }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center p-8 space-y-4">
          <AlertCircle className="h-12 w-12 text-destructive" />
          <div className="text-center space-y-2">
            <h3 className="text-lg font-semibold">Failed to load data</h3>
            <p className="text-sm text-muted-foreground">
              {this.state.error?.message || "Unknown error"}
            </p>
          </div>
          <Button onClick={() => {
            this.setState({ hasError: false, error: null });
            this.props.onReset();
          }}>
            Try Again
          </Button>
        </div>
      );
    }
    return this.props.children;
  }
}

// COMPONENT AGNOSTIC DATA PROVIDER
export function QueenDataProvider({
  children,
  fallback,
}: {
  children: (status: any, refresh: () => void) => ReactNode;
  fallback?: ReactNode;
}) {
  const [refreshKey, setRefreshKey] = useState(0);

  const handleRefresh = useCallback(() => {
    const newKey = refreshKey + 1;
    setRefreshKey(newKey);
    promiseCache.delete(`queen-${refreshKey}`);
  }, [refreshKey]);

  return (
    <QueenErrorBoundary onReset={handleRefresh}>
      <Suspense fallback={fallback}>
        <QueenContentWrapper promiseKey={`queen-${refreshKey}`}>
          {(status) => children(status, handleRefresh)}
        </QueenContentWrapper>
      </Suspense>
    </QueenErrorBoundary>
  );
}

// Wrapper to use() the promise
function QueenContentWrapper({
  promiseKey,
  children,
}: {
  promiseKey: string;
  children: (status: any) => ReactNode;
}) {
  const status = use(fetchQueenStatus(promiseKey));
  return <>{children(status)}</>;
}
```

### 2. Presentation Component (`*Card.tsx`) - UI ONLY

```tsx
// QueenCard.tsx
import { Card, CardContent, CardHeader, CardTitle, Button } from "@rbee/ui/atoms";
import { Play, Loader2 } from "lucide-react";
import { commands } from "@/generated/bindings";

// Export the data type
export interface QueenStatus {
  isRunning: boolean;
  isInstalled: boolean;
}

// Export the loading fallback
export function LoadingQueen() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Queen</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-center justify-center py-4">
          <Loader2 className="h-6 w-6 animate-spin" />
        </div>
      </CardContent>
    </Card>
  );
}

// Export the presentation component
export interface QueenCardProps {
  status: QueenStatus;
  onRefresh: () => void;
}

export function QueenCard({ status, onRefresh }: QueenCardProps) {
  const handleStart = async () => {
    await commands.queenStart();
    onRefresh();
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Queen</CardTitle>
      </CardHeader>
      <CardContent>
        <Button onClick={handleStart} icon={<Play />}>
          Start
        </Button>
      </CardContent>
    </Card>
  );
}
```

### 3. Usage in Pages - WIRE THEM TOGETHER

```tsx
// pages/ServicesPage.tsx
import { QueenDataProvider } from "../containers/QueenContainer";
import { QueenCard, LoadingQueen } from "../components/QueenCard";

export default function ServicesPage() {
  return (
    <div>
      <h2>Services</h2>
      
      {/* Consumer provides presentation via render prop */}
      <QueenDataProvider fallback={<LoadingQueen />}>
        {(status, onRefresh) => (
          <QueenCard status={status} onRefresh={onRefresh} />
        )}
      </QueenDataProvider>
    </div>
  );
}
```

**Key Points:**
- ✅ Container exports `QueenDataProvider` (data layer)
- ✅ Presentation exports `QueenCard` + `LoadingQueen` (UI layer)
- ✅ Page wires them together with render prop
- ✅ Consumer controls what UI to render
- ✅ Same data provider can power different UIs (table, card, list, etc.)

## Key Concepts

### React 19 `use()` Hook

The `use()` hook is a new React 19 primitive that:
- Unwraps promises directly in render
- Suspends component rendering until promise resolves
- Eliminates need for `useEffect` and manual loading states
- Handles race conditions automatically

**Before (React 18):**
```tsx
function Component() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    fetchData().then(setData).finally(() => setLoading(false));
  }, []);
  
  if (loading) return <Loading />;
  return <Table data={data} />;
}
```

**After (React 19):**
```tsx
function Container() {
  const [promise, setPromise] = useState(() => fetchData());
  
  return (
    <Suspense fallback={<Loading />}>
      <Table data={use(promise)} />
    </Suspense>
  );
}
```

### Suspense Boundary

- Wraps the component that uses `use()`
- Provides `fallback` prop for loading state
- Automatically shows fallback while promise is pending
- Automatically shows content when promise resolves

### Promise State Management

```tsx
const [dataPromise, setDataPromise] = useState(() => fetchData());

const handleRefresh = () => {
  setDataPromise(fetchData()); // Create new promise to trigger re-fetch
};
```

- Store the **promise itself**, not the data
- Create new promise to trigger re-fetch
- React handles the rest automatically

## Benefits

### ✅ Separation of Concerns
- Presentation logic isolated from data fetching
- Each file has single responsibility
- Easy to reason about

### ✅ Testability
- Presentation component easily tested with mock data
- No need to mock Tauri invoke calls in presentation tests
- Container can be tested separately

### ✅ Reusability
- Presentation component can be used with different data sources
- Easy to create Storybook stories
- Can be used in different contexts

### ✅ Cleaner Code
- No `useEffect` boilerplate
- No manual loading state management
- No race condition handling
- Automatic error boundaries (with ErrorBoundary wrapper)

### ✅ Better UX
- Suspense provides consistent loading states
- Automatic handling of async operations
- Smooth transitions between loading and loaded states

## Common Patterns

### Refresh/Reload

```tsx
const [promise, setPromise] = useState(() => fetchData());

const handleRefresh = () => {
  setPromise(fetchData()); // New promise = new fetch
};
```

### Error Handling

Wrap the Suspense boundary with an ErrorBoundary:

```tsx
<ErrorBoundary fallback={<ErrorComponent />}>
  <Suspense fallback={<LoadingComponent />}>
    <ComponentTable items={use(dataPromise)} onRefresh={handleRefresh} />
  </Suspense>
</ErrorBoundary>
```

### Conditional Fetching

```tsx
const [promise, setPromise] = useState<Promise<Data[]> | null>(null);

useEffect(() => {
  if (shouldFetch) {
    setPromise(fetchData());
  }
}, [shouldFetch]);

if (!promise) return <EmptyState />;

return (
  <Suspense fallback={<Loading />}>
    <Component data={use(promise)} />
  </Suspense>
);
```

## Migration Guide

### From useEffect Pattern

**Old Pattern:**
```tsx
function Component() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    fetchData()
      .then(setData)
      .catch(setError)
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <Loading />;
  if (error) return <Error error={error} />;
  return <Table data={data} />;
}
```

**New Pattern:**
```tsx
// Container.tsx
function Container() {
  const [promise, setPromise] = useState(() => fetchData());
  return (
    <ErrorBoundary fallback={<Error />}>
      <Suspense fallback={<Loading />}>
        <Table data={use(promise)} onRefresh={() => setPromise(fetchData())} />
      </Suspense>
    </ErrorBoundary>
  );
}

// Table.tsx
function Table({ data, onRefresh }) {
  return <div>{/* render data */}</div>;
}
```

## 🚨 Anti-Patterns (VIOLATIONS)

### ❌ VIOLATION: Container includes presentation

```tsx
// ❌ WRONG - Container has UI components inside
export function SshHivesContainer() {
  return (
    <SshHivesDataProvider>
      {(hives, onRefresh) => (
        <SshHivesTable hives={hives} onRefresh={onRefresh} />
      )}
    </SshHivesDataProvider>
  );
}
```

**Why wrong:** Container is NOT component agnostic. It hardcodes `SshHivesTable`. What if consumer wants a dropdown? A list? A grid?

### ❌ VIOLATION: Container imports UI components

```tsx
// ❌ WRONG - Container imports presentation
import { SshHivesTable, LoadingHives } from "./SshHivesTable";
import { Card, Table, Button } from "@rbee/ui/atoms";
```

**Why wrong:** Containers should ONLY import data-related things (commands, types). NO UI imports.

### ❌ VIOLATION: Presentation component fetches data

```tsx
// ❌ WRONG - Presentation component uses use() hook
export function ComponentTable() {
  const data = use(fetchData()); // ❌ Wrong!
  return <Table data={data} />;
}
```

**Why wrong:** Presentation should receive data as props. NO data fetching.

### ❌ VIOLATION: Storing data in state instead of promise

```tsx
// ❌ WRONG - Store promise, not data
const [data, setData] = useState([]);
const promise = fetchData();
promise.then(setData); // ❌ Wrong!
```

**Why wrong:** React 19 `use()` hook expects promises, not data.

### ❌ VIOLATION: Conditional `use()` call

```tsx
// ❌ WRONG - use() must be called unconditionally
if (shouldFetch) {
  const data = use(promise); // ❌ Wrong!
}
```

**Why wrong:** React hooks must be called unconditionally.

## Tauri Command Bindings (Recommended)

**TEAM-296:** We now use auto-generated TypeScript bindings from `tauri-plugin-typegen`.

### Using Generated Bindings

```tsx
import { hive_list } from "@/generated/commands";

// ✅ Best - Fully typed, auto-generated, always in sync
const result = await hive_list();
```

**Benefits:**
- Full type safety (parameters + return types)
- Auto-generated from Rust code
- Always in sync (regenerates on build)
- Zero manual maintenance
- IDE autocomplete for everything

**Location:** `src/generated/` (auto-generated, do not edit)

### Alternative: Command Registry Pattern (Legacy)

If you need to use the manual registry for any reason:

```tsx
import { COMMANDS } from "../api/commands.registry";

// ✅ Good - Type-safe command names
const result = await invoke<string>(COMMANDS.HIVE_LIST);

// ❌ Bad - Typo-prone, no autocomplete
const result = await invoke<string>("hive_list");
```

**Note:** The generated bindings approach is preferred as it provides full type safety, not just command names.

**See:** `TAURI_TYPEGEN_SETUP.md` for complete setup guide

## ✅ Checklist for New Components

### Container (`*Container.tsx`)
- [ ] Create container file in `/containers/`
- [ ] Import ONLY: `react`, `@/generated/bindings`, `lucide-react` (AlertCircle), `@rbee/ui/atoms` (Button for error boundary)
- [ ] ❌ NO UI component imports (Card, Table, etc.)
- [ ] Define promise cache: `const promiseCache = new Map<string, Promise<any>>()`
- [ ] Implement fetch function with cache
- [ ] Implement error boundary (generic error UI)
- [ ] Export `*DataProvider` component with render prop signature: `children: (data: any, refresh: () => void) => ReactNode`
- [ ] Export type re-exports: `export type { YourType } from "../components/YourComponent"`
- [ ] ❌ NO presentation components inside container

### Presentation Component (`*Card.tsx` / `*Table.tsx`)
- [ ] Create presentation file in `/components/`
- [ ] Export data type interface
- [ ] Export loading fallback component
- [ ] Export presentation component with props interface
- [ ] ❌ NO data fetching
- [ ] ❌ NO `use()` hook
- [ ] ❌ NO Suspense

### Page (Wire them together)
- [ ] Import `*DataProvider` from container
- [ ] Import presentation components from components
- [ ] Use render prop pattern:
  ```tsx
  <YourDataProvider fallback={<LoadingComponent />}>
    {(data, onRefresh) => (
      <YourComponent data={data} onRefresh={onRefresh} />
    )}
  </YourDataProvider>
  ```

## Resources

- [React 19 `use()` Hook Documentation](https://react.dev/reference/react/use)
- [Suspense Documentation](https://react.dev/reference/react/Suspense)
- **Reference Implementation:** 
  - Container: `containers/QueenContainer.tsx` (data layer)
  - Presentation: `components/QueenCard.tsx` (UI layer)
  - Usage: `pages/ServicesPage.tsx` (wiring)

## 🎯 TL;DR

**CONTAINERS = DATA ONLY. NO UI.**

**PRESENTATION = UI ONLY. NO DATA FETCHING.**

**PAGES = WIRE THEM TOGETHER WITH RENDER PROPS.**

If you put a `<Card>` or `<Table>` in a container, **YOU FUCKED UP.**

If you put `use()` or `Suspense` in a presentation component, **YOU FUCKED UP.**

**Containers are COMPONENT AGNOSTIC. They provide DATA via render props. Consumers provide PRESENTATION.**
