# Data Fetching Pattern with React 19 `use()` Hook

**Author:** TEAM-296  
**Date:** October 26, 2025  
**Status:** Active Pattern

## Overview

This document describes our component architecture pattern for data fetching using React 19's `use()` hook with Suspense. This pattern separates presentation logic from data fetching logic, creating clean, testable, and maintainable components.

## Pattern Architecture

We split components into **two files**:

1. **Presentation Component** (`*Table.tsx`, `*List.tsx`, etc.)
   - Pure presentation logic
   - Receives data as props
   - No data fetching
   - Includes loading fallback component
   - Easily testable with mock data
   - Storybook-ready

2. **Container Component** (`*Container.tsx`)
   - Handles data fetching
   - Manages promise state
   - Provides Suspense boundary
   - Uses React 19 `use()` hook
   - Minimal logic

## Reference Implementation

See `SshHivesContainer.tsx` and `SshHivesTable.tsx` for a complete working example.

## File Structure

```
components/
├── SshHivesTable.tsx       # Presentation + Loading fallback
└── SshHivesContainer.tsx   # Data fetching + Container
```

## Implementation Guide

### 1. Presentation Component (`ComponentTable.tsx`)

```tsx
// ComponentTable.tsx
import { Table, Button } from "@rbee/ui/atoms";
import { RefreshCw } from "lucide-react";

// Export the data type
export interface DataItem {
  id: string;
  name: string;
  status: "online" | "offline";
}

// Export the loading fallback
export function LoadingComponent() {
  return (
    <div className="rounded-lg border border-border bg-card">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Name</TableHead>
            <TableHead className="text-right">
              <Button variant="ghost" size="icon-sm" disabled>
                <RefreshCw className="animate-spin" />
              </Button>
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          <TableRow>
            <TableCell className="text-center text-muted-foreground">
              Loading...
            </TableCell>
          </TableRow>
        </TableBody>
      </Table>
    </div>
  );
}

// Export the presentation component
export interface ComponentTableProps {
  items: DataItem[];
  onRefresh: () => void;
}

export function ComponentTable({ items, onRefresh }: ComponentTableProps) {
  return (
    <div className="rounded-lg border border-border bg-card">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Name</TableHead>
            <TableHead className="text-right">
              <Button onClick={onRefresh} variant="ghost" size="icon-sm">
                <RefreshCw />
              </Button>
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {items.length === 0 ? (
            <TableRow>
              <TableCell className="text-center text-muted-foreground">
                No items found
              </TableCell>
            </TableRow>
          ) : (
            items.map((item) => (
              <TableRow key={item.id}>
                <TableCell>{item.name}</TableCell>
                <TableCell>{item.status}</TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}
```

### 2. Container Component (`ComponentContainer.tsx`)

```tsx
// ComponentContainer.tsx
import { invoke } from "@tauri-apps/api/core";
import { use, useState, Suspense } from "react";
import type { CommandResponse } from "../api/types";
import { COMMANDS } from "../api/commands.registry";
import { ComponentTable, LoadingComponent, type DataItem } from "./ComponentTable";

// Data fetching function
async function fetchData(): Promise<DataItem[]> {
  const result = await invoke<string>(COMMANDS.YOUR_COMMAND);
  const response: CommandResponse = JSON.parse(result);

  if (response.success && response.data) {
    return JSON.parse(response.data) as DataItem[];
  }

  throw new Error(response.message || "Failed to load data");
}

// Container component
export function ComponentContainer() {
  const [dataPromise, setDataPromise] = useState(() => fetchData());

  const handleRefresh = () => {
    setDataPromise(fetchData());
  };

  return (
    <Suspense fallback={<LoadingComponent />}>
      <ComponentTable items={use(dataPromise)} onRefresh={handleRefresh} />
    </Suspense>
  );
}
```

### 3. Usage in Pages

```tsx
// pages/SomePage.tsx
import { ComponentContainer } from "../components/ComponentContainer";

export default function SomePage() {
  return (
    <div>
      <h2>My Data</h2>
      <ComponentContainer />
    </div>
  );
}
```

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

## Anti-Patterns

### ❌ Don't use `use()` in presentation component

```tsx
// BAD - Presentation component shouldn't fetch data
export function ComponentTable() {
  const data = use(fetchData()); // ❌ Wrong!
  return <Table data={data} />;
}
```

### ❌ Don't store data in state

```tsx
// BAD - Store promise, not data
const [data, setData] = useState([]);
const promise = fetchData();
promise.then(setData); // ❌ Wrong!
```

### ❌ Don't call `use()` conditionally

```tsx
// BAD - use() must be called unconditionally
if (shouldFetch) {
  const data = use(promise); // ❌ Wrong!
}
```

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

## Checklist for New Components

- [ ] Create presentation component file (`*Table.tsx`)
- [ ] Export data type interface
- [ ] Export loading fallback component
- [ ] Export presentation component with props interface
- [ ] Create container component file (`*Container.tsx`)
- [ ] Import command from `@/generated/commands` (e.g., `import { hive_list } from "@/generated/commands"`)
- [ ] Implement data fetching function using generated command
- [ ] Use `useState(() => fetchData())` for promise
- [ ] Wrap with `<Suspense fallback={<Loading />}>`
- [ ] Use `use(promise)` inline in JSX
- [ ] Implement refresh handler that creates new promise
- [ ] Update page imports to use container component

## Resources

- [React 19 `use()` Hook Documentation](https://react.dev/reference/react/use)
- [Suspense Documentation](https://react.dev/reference/react/Suspense)
- **Reference Implementation:** `SshHivesContainer.tsx` + `SshHivesTable.tsx`

## Questions?

If you're unsure about the pattern, look at `SshHivesContainer.tsx` and `SshHivesTable.tsx` for a complete, working example of this pattern in action.
