# Data Fetching Pattern with Zustand

**Author:** TEAM-296, TEAM-338  
**Date:** October 28, 2025  
**Status:** Active Pattern (Zustand + CommandStore)

## 🚨 CRITICAL: ZUSTAND + COMMANDSTORE PATTERN

This pattern uses **ZUSTAND STORES** for state management with **GLOBAL COMMAND EXECUTION STATE**.

## The Split

### 1. **Store** (`store/*Store.ts`) - STATE MANAGEMENT
- ✅ **ONLY** state definitions
- ✅ **ONLY** async actions (fetch, commands)
- ✅ **ONLY** state updates
- ✅ Imports `commandStore` internally (NOT exposed to components)
- ✅ Uses `withCommandExecution` helper for all commands
- ✅ Exports custom hook (`useQueenStore`, `useHiveStore`)
- ✅ Exports type definitions
- ❌ **NO** React components
- ❌ **NO** UI imports
- ❌ **NO** JSX

### 2. **Component** (`components/*Card.tsx`) - UI + HOOK USAGE
- ✅ Uses store hook (`useQueenStore`, `useHiveStore`)
- ✅ Reads `isExecuting` from `commandStore` (for button disabled state)
- ✅ Handles loading/error states in UI
- ✅ Calls store actions directly (e.g., `start()`, `stop()`, `install()`)
- ✅ All UI/styling logic
- ❌ **NO** direct state management
- ❌ **NO** useState for data
- ❌ **NO** manual fetch logic
- ❌ **NO** manual `setIsExecuting` calls (store handles this)

## 🔥 RULE ZERO: STORES MANAGE STATE + COMMANDS INTERNALLY

**Stores provide STATE and ACTIONS. Components consume via hooks. CommandStore is INTERNAL to stores.**

```tsx
// ❌ WRONG - Component manages isExecuting
export function QueenCard() {
  const [isExecuting, setIsExecuting] = useState(false);
  const { start } = useQueenStore();
  
  const handleStart = async () => {
    setIsExecuting(true);
    try {
      await start();
    } finally {
      setIsExecuting(false);
    }
  };
  
  return <Button onClick={handleStart} disabled={isExecuting}>Start</Button>;
}

// ✅ RIGHT - Store manages isExecuting internally, component just reads it
export function QueenCard() {
  const { start } = useQueenStore();
  const { isExecuting } = useCommandStore();
  
  return <Button onClick={start} disabled={isExecuting}>Start</Button>;
}
```

## File Structure

```
store/
├── commandStore.ts         # ✅ Global isExecuting state (imported by other stores)
├── queenStore.ts           # ✅ Zustand store with state + actions (imports commandStore)
└── hiveStore.ts            # ✅ Zustand store with state + actions (imports commandStore)
components/
├── QueenCard.tsx           # ✅ Component using queenStore + commandStore hooks
└── InstallHiveCard.tsx     # ✅ Component using hiveStore + commandStore hooks
pages/
└── ServicesPage.tsx        # ✅ Just renders components
```

## Reference Implementation

**See:** 
- `store/commandStore.ts` - Global command execution state
- `store/queenStore.ts` - Queen service state + actions (imports commandStore)
- `store/hiveStore.ts` - Hive state + actions (imports commandStore)
- `components/QueenCard.tsx` - UI using queenStore + commandStore
- `components/InstallHiveCard.tsx` - UI using hiveStore + commandStore
- `pages/ServicesPage.tsx` - Page composition

## Implementation Guide

### 1. Store (`store/*Store.ts`) - STATE MANAGEMENT

```tsx
// store/queenStore.ts
// Imports commandStore internally to manage global isExecuting state
import { create } from 'zustand';
import { commands } from '@/generated/bindings';
import { useCommandStore } from './commandStore';

export interface QueenStatus {
  isRunning: boolean;
  isInstalled: boolean;
}

interface QueenState {
  status: QueenStatus | null;
  isLoading: boolean;
  error: string | null;
  
  // Actions
  fetchStatus: () => Promise<void>;
  start: () => Promise<void>;
  stop: () => Promise<void>;
  install: () => Promise<void>;
  rebuild: () => Promise<void>;
  uninstall: () => Promise<void>;
  reset: () => void;
}

// Helper to wrap commands with global isExecuting state
const withCommandExecution = async (commandFn: () => Promise<unknown>, refreshFn: () => Promise<void>) => {
  const { setIsExecuting } = useCommandStore.getState();
  setIsExecuting(true);
  try {
    await commandFn();
    await refreshFn();
  } catch (error) {
    console.error('Command failed:', error);
    throw error;
  } finally {
    setIsExecuting(false);
  }
};

export const useQueenStore = create<QueenState>((set, get) => ({
  status: null,
  isLoading: false,
  error: null,

  fetchStatus: async () => {
    set({ isLoading: true, error: null });
    try {
      const status = await commands.queenStatus();
      set({ status, isLoading: false });
    } catch (error) {
      set({ 
        error: error instanceof Error ? error.message : 'Failed to fetch status',
        isLoading: false 
      });
    }
  },

  start: async () => {
    await withCommandExecution(() => commands.queenStart(), get().fetchStatus);
  },

  stop: async () => {
    await withCommandExecution(() => commands.queenStop(), get().fetchStatus);
  },

  install: async () => {
    await withCommandExecution(() => commands.queenInstall(null), get().fetchStatus);
  },

  rebuild: async () => {
    await withCommandExecution(() => commands.queenRebuild(false), get().fetchStatus);
  },

  uninstall: async () => {
    await withCommandExecution(() => commands.queenUninstall(), get().fetchStatus);
  },

  reset: () => {
    set({ status: null, isLoading: false, error: null });
  },
}));
```

**Key Points:**
- ✅ Store imports `commandStore` internally via `useCommandStore.getState()`
- ✅ `withCommandExecution` helper wraps all commands with `setIsExecuting(true/false)`
- ✅ Commands auto-refresh data after execution (e.g., `get().fetchStatus`)
- ✅ Components just call simple functions like `start()`, `stop()`, `install()`
- ✅ Components read `isExecuting` from `commandStore` for button disabled state
- ✅ NO manual `setIsExecuting` calls in components

### 2. Component (`components/*Card.tsx`) - UI + HOOK USAGE

```tsx
// components/QueenCard.tsx
import { useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle, Button } from "@rbee/ui/atoms";
import { Play, Loader2, AlertCircle } from "lucide-react";
import { useQueenStore } from "../store/queenStore";
import { useCommandStore } from "../store/commandStore";

// Re-export type from store
export type { QueenStatus } from "../store/queenStore";

export function QueenCard() {
  const { status, isLoading, error, fetchStatus, start, stop, install } = useQueenStore();
  const { isExecuting } = useCommandStore();

  // Fetch status on mount
  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  // Loading state
  if (isLoading && !status) {
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

  // Error state
  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Queen</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2 text-destructive">
            <AlertCircle className="h-4 w-4" />
            <p className="text-sm">{error}</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Queen</CardTitle>
      </CardHeader>
      <CardContent>
        <Button onClick={start} disabled={isExecuting} icon={<Play />}>
          Start
        </Button>
        <Button onClick={stop} disabled={isExecuting}>
          Stop
        </Button>
        <Button onClick={install} disabled={isExecuting}>
          Install
        </Button>
      </CardContent>
    </Card>
  );
}
```

**Key Points:**
- ✅ Store exposes command functions directly (`start`, `stop`, `install`, etc.)
- ✅ Component just calls functions - no command wrapping needed
- ✅ Component ONLY reads `isExecuting` from `commandStore` (NEVER calls `setIsExecuting`)
- ✅ Store handles all command execution logic internally via `withCommandExecution`
- ✅ Clean, simple component code - just call actions and read state

### 3. Usage in Pages - SIMPLE IMPORT

```tsx
// pages/ServicesPage.tsx
import { QueenCard } from "../components/QueenCard";

export default function ServicesPage() {
  return (
    <div>
      <h2>Services</h2>
      <QueenCard />
    </div>
  );
}
```

**Key Points:**
- ✅ Store manages all state and actions
- ✅ Component handles UI and calls store actions
- ✅ Page just renders component (no wiring needed)
- ✅ Store is globally accessible via hook
- ✅ Multiple components can share same store

## Key Concepts

### Zustand Store Pattern

Zustand provides simple, hook-based state management:
- No providers/context needed
- Direct state access via hooks
- Actions are just functions in the store
- Automatic re-renders when state changes

**Benefits:**
- ✅ Less boilerplate than Redux
- ✅ No context providers needed
- ✅ TypeScript-friendly
- ✅ Easy to test
- ✅ Minimal bundle size

### State Management

```tsx
export const useQueenStore = create<QueenState>((set, get) => ({
  // State
  status: null,
  isLoading: false,
  
  // Actions
  fetchStatus: async () => {
    set({ isLoading: true });
    const status = await commands.queenStatus();
    set({ status, isLoading: false });
  },
}));
```

- `set()` updates state
- `get()` reads current state
- Actions can be async
- State updates trigger re-renders

### Component Usage

```tsx
export function QueenCard() {
  const { status, isLoading, fetchStatus } = useQueenStore();
  
  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);
  
  if (isLoading) return <Loading />;
  return <Card>{status.isRunning ? 'Running' : 'Stopped'}</Card>;
}
```

- Destructure only what you need
- Call actions directly
- Component re-renders when used state changes

## Benefits

### ✅ Separation of Concerns
- State management isolated in store
- UI logic isolated in component
- Each file has single responsibility
- Easy to reason about

### ✅ Testability
- Store can be tested independently
- Component can be tested with mock store
- No need to mock Tauri commands in component tests

### ✅ Reusability
- Store can be used by multiple components
- Component can be reused in different contexts
- Easy to create Storybook stories

### ✅ Cleaner Code
- No boilerplate providers/context
- Simple hook-based API
- TypeScript-friendly
- Minimal bundle size

### ✅ Better UX
- Centralized state management
- Consistent loading/error states
- Easy to share state across components

## Common Patterns

### Refresh/Reload

```tsx
// In store
fetchStatus: async () => {
  set({ isLoading: true });
  const status = await commands.queenStatus();
  set({ status, isLoading: false });
}

// In component
const { fetchStatus } = useQueenStore();
<Button onClick={fetchStatus}>Refresh</Button>
```

### Error Handling

```tsx
// In store
fetchStatus: async () => {
  set({ isLoading: true, error: null });
  try {
    const status = await commands.queenStatus();
    set({ status, isLoading: false });
  } catch (error) {
    set({ 
      error: error instanceof Error ? error.message : 'Failed',
      isLoading: false 
    });
  }
}

// In component
const { error } = useQueenStore();
if (error) return <ErrorMessage>{error}</ErrorMessage>;
```

### Selective Re-renders

```tsx
// Only re-render when status changes (not when isLoading changes)
const status = useQueenStore(state => state.status);
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
// store/dataStore.ts
export const useDataStore = create<DataState>((set) => ({
  data: [],
  isLoading: false,
  error: null,
  
  fetchData: async () => {
    set({ isLoading: true, error: null });
    try {
      const data = await fetchData();
      set({ data, isLoading: false });
    } catch (error) {
      set({ error: error.message, isLoading: false });
    }
  },
}));

// Component.tsx
function Component() {
  const { data, isLoading, error, fetchData } = useDataStore();
  
  useEffect(() => {
    fetchData();
  }, [fetchData]);
  
  if (isLoading) return <Loading />;
  if (error) return <Error error={error} />;
  return <Table data={data} />;
}
```

## 🚨 Anti-Patterns (VIOLATIONS)

### ❌ VIOLATION: Component manages state

```tsx
// ❌ WRONG - Component has useState for data
export function QueenCard() {
  const [status, setStatus] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  
  useEffect(() => {
    fetchStatus().then(setStatus);
  }, []);
}
```

**Why wrong:** State should be in Zustand store, not component.

### ❌ VIOLATION: Direct Tauri calls in component

```tsx
// ❌ WRONG - Component calls Tauri directly
export function QueenCard() {
  const handleStart = async () => {
    await commands.queenStart();
  };
}
```

**Why wrong:** Commands should be wrapped in store actions.

### ❌ VIOLATION: Store has UI logic

```tsx
// ❌ WRONG - Store imports React components
import { toast } from "@rbee/ui/atoms";

export const useQueenStore = create<QueenState>((set) => ({
  fetchStatus: async () => {
    try {
      const status = await commands.queenStatus();
      toast.success("Status loaded"); // ❌ Wrong!
    } catch (error) {
      toast.error("Failed"); // ❌ Wrong!
    }
  },
}));
```

**Why wrong:** Store should only manage state, not UI.

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

### Store (`store/*Store.ts`)
- [ ] Create store file in `/store/`
- [ ] Import `commandStore`: `import { useCommandStore } from './commandStore'`
- [ ] Define state interface with data, isLoading, error fields
- [ ] Define actions interface (fetch, command actions, reset)
- [ ] Create `withCommandExecution` helper:
  ```tsx
  const withCommandExecution = async (
    commandFn: () => Promise<unknown>,
    refreshFn: () => Promise<void>
  ) => {
    const { setIsExecuting } = useCommandStore.getState();
    setIsExecuting(true);
    try {
      await commandFn();
      await refreshFn();
    } catch (error) {
      console.error('Command failed:', error);
      throw error;
    } finally {
      setIsExecuting(false);
    }
  };
  ```
- [ ] Implement fetch action with isLoading/error handling
- [ ] Implement command actions using `withCommandExecution`
- [ ] Export store hook: `export const useYourStore = create<YourState>(...)`
- [ ] Export type definitions
- [ ] ❌ NO React imports
- [ ] ❌ NO UI logic

### Component (`components/*Card.tsx`)
- [ ] Create component file in `/components/`
- [ ] Import store hook: `import { useYourStore } from '../store/yourStore'`
- [ ] Import commandStore: `import { useCommandStore } from '../store/commandStore'`
- [ ] Destructure state and actions from store
- [ ] Destructure `isExecuting` from commandStore
- [ ] Call fetch action in `useEffect` on mount
- [ ] Handle loading state (show spinner)
- [ ] Handle error state (show error message)
- [ ] Render UI with data
- [ ] Pass `isExecuting` to button `disabled` prop
- [ ] Call store actions directly on button clicks
- [ ] ❌ NO useState for data
- [ ] ❌ NO manual setIsExecuting calls
- [ ] ❌ NO try/catch around store actions

### Page (Composition)
- [ ] Import components from `/components/`
- [ ] Render components directly (no providers needed)
- [ ] Store is globally accessible via hooks
- [ ] Multiple components can share same store

## Resources

- [Zustand Documentation](https://github.com/pmndrs/zustand)
- **Reference Implementation:** 
  - Global State: `store/commandStore.ts` (isExecuting state)
  - Service Store: `store/queenStore.ts` (state + actions with commandStore)
  - Service Store: `store/hiveStore.ts` (state + actions with commandStore)
  - Component: `components/QueenCard.tsx` (UI using stores)
  - Component: `components/InstallHiveCard.tsx` (UI using stores)
  - Page: `pages/ServicesPage.tsx` (composition)

## 🎯 TL;DR

**STORES = STATE + ACTIONS. IMPORT COMMANDSTORE INTERNALLY.**

**COMPONENTS = UI + STORE HOOKS. READ isExecuting, NEVER SET IT.**

**PAGES = JUST RENDER COMPONENTS.**

If you call `setIsExecuting` in a component, **YOU FUCKED UP.**

If you wrap store actions in try/catch in a component, **YOU FUCKED UP.**

If you use `useState` for data that should be in a store, **YOU FUCKED UP.**

**Stores manage EVERYTHING internally. Components just call actions and read state.**
