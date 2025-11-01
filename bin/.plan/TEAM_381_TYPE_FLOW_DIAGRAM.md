# TEAM-381: Type Flow Diagram

## Type Hierarchy (Single Source of Truth)

```
┌─────────────────────────────────────────────────────────────┐
│                    Backend (Rust)                           │
│                                                             │
│  - Model (in operations_contract)                          │
│  - Operations (ModelList, ModelLoad, etc.)                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
                    WASM Bindings
                            ↓
┌─────────────────────────────────────────────────────────────┐
│          @rbee/rbee-hive-sdk (Source of Truth)             │
│                                                             │
│  export interface Model {                                  │
│    id: string                                              │
│    name: string                                            │
│    size: number                                            │
│    status: string                                          │
│    loaded?: boolean                                        │
│    vram_mb?: number                                        │
│  }                                                         │
│                                                             │
│  export interface HFModel {                                │
│    id: string                                              │
│    modelId: string                                         │
│    author: string                                          │
│    downloads: number                                       │
│    likes: number                                           │
│    tags: string[]                                          │
│    private: boolean                                        │
│    gated: boolean | string                                 │
│  }                                                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
                      Re-export
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              @rbee/rbee-hive-react                         │
│                                                             │
│  // Re-export from SDK                                     │
│  export type { Model, HFModel } from '@rbee/rbee-hive-sdk' │
│                                                             │
│  // Local types                                            │
│  export interface Worker {                                 │
│    pid: number                                             │
│    model: string                                           │
│    device: string                                          │
│  }                                                         │
│                                                             │
│  // Hooks                                                  │
│  export function useModels(): { models: Model[], ... }     │
│  export function useModelOperations(): { ... }             │
└─────────────────────────────────────────────────────────────┘
                            ↓
                      Re-export
                            ↓
┌─────────────────────────────────────────────────────────────┐
│         ModelManagement/types.ts (UI Layer)                │
│                                                             │
│  // Re-export from React layer                             │
│  export type { Model, HFModel } from '@rbee/rbee-hive-react'│
│                                                             │
│  // UI-specific types                                      │
│  export type ViewMode = 'downloaded' | 'loaded' | 'search' │
│                                                             │
│  export interface FilterState {                            │
│    formats: string[]                                       │
│    architectures: string[]                                 │
│    maxSize: string                                         │
│    openSourceOnly: boolean                                 │
│    sortBy: 'downloads' | 'likes' | 'recent'                │
│  }                                                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
                      Import
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              UI Components                                  │
│                                                             │
│  import type { Model, HFModel, ViewMode, FilterState }     │
│    from './types'                                          │
│                                                             │
│  - DownloadedModelsView.tsx                                │
│  - LoadedModelsView.tsx                                    │
│  - SearchResultsView.tsx                                   │
│  - FilterPanel.tsx                                         │
│  - ModelDetailsPanel.tsx                                   │
└─────────────────────────────────────────────────────────────┘
```

## Type Ownership

| Type | Owner | Re-exported By | Used By |
|------|-------|----------------|---------|
| `Model` | `rbee-hive-sdk` | `rbee-hive-react`, UI | All layers |
| `HFModel` | `rbee-hive-sdk` | `rbee-hive-react`, UI | All layers |
| `Worker` | `rbee-hive-react` | - | React hooks, UI |
| `ViewMode` | UI (`types.ts`) | - | UI components |
| `FilterState` | UI (`types.ts`) | - | UI components |
| `ProcessStats` | `rbee-hive-sdk` | - | Heartbeat monitor |
| `HiveInfo` | `rbee-hive-sdk` | - | Heartbeat monitor |

## Import Patterns

### ✅ CORRECT

```typescript
// In UI components
import type { Model, HFModel } from '@rbee/rbee-hive-react'

// In React hooks
import type { Model, HFModel } from '@rbee/rbee-hive-sdk'

// In UI types.ts
export type { Model, HFModel } from '@rbee/rbee-hive-react'
```

### ❌ WRONG

```typescript
// DON'T define types locally
interface Model {
  id: string
  // ...
}

// DON'T import SDK types in UI (skip React layer)
import type { Model } from '@rbee/rbee-hive-sdk'
```

## Data Flow

```
Backend (Rust)
    ↓ JSON over SSE
WASM SDK (rbee-hive-sdk)
    ↓ Parse JSON → Model[]
React Hooks (rbee-hive-react)
    ↓ useModels() → { models: Model[] }
UI Components
    ↓ Display models
User
```

## Type Safety Benefits

### 1. Compile-Time Checks
```typescript
// If backend changes Model.size → Model.size_bytes
// TypeScript catches ALL usages:
const size = model.size // ❌ Error: Property 'size' does not exist
const size = model.size_bytes // ✅ Correct
```

### 2. Autocomplete
```typescript
const model: Model = // ...
model. // ← IDE shows: id, name, size, status, loaded, vram_mb
```

### 3. Refactoring Safety
```typescript
// Rename Model → LocalModel in SDK
// TypeScript finds ALL imports and usages
// No manual search-and-replace needed
```

## Migration Checklist

- [x] Define `Model` and `HFModel` in SDK
- [x] Remove duplicate `Model` from React layer
- [x] Re-export from SDK in React layer
- [x] Remove duplicate types from UI layer
- [x] Re-export from React in UI layer
- [x] Update all imports to use re-exports
- [ ] Rebuild SDK (`pnpm build`)
- [ ] Rebuild React hooks (`pnpm build`)
- [ ] Test in UI

## Summary

**Before:**
```
UI defines Model ❌
React defines Model ❌
SDK has no Model ❌
→ Duplication, inconsistency
```

**After:**
```
SDK defines Model ✅
React re-exports Model ✅
UI re-exports Model ✅
→ Single source of truth
```

**The type hierarchy is now correct!** 🎉
