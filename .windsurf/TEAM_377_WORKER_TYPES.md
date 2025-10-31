# TEAM-377 - Worker Types Exposed for Frontend

## ✅ Mission Accomplished

**Exposed Rust WorkerType enum to frontend with proper TypeScript types and select component support.**

---

## 🎯 Problem

User noticed hardcoded worker type in `useHiveOperations`:
```typescript
// ❌ HARDCODED
const op = OperationBuilder.workerSpawn(hiveId, modelId, 'cuda', 0)
```

**Questions:**
1. What worker types are available?
2. Where are they defined in Rust?
3. How can frontend show them in a select component?

---

## 🔍 Investigation

### Found Rust Enum

**File:** `bin/25_rbee_hive_crates/worker-catalog/src/types.rs`

```rust
/// Worker type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerType {
    /// CPU-based LLM worker
    CpuLlm,
    /// CUDA-based LLM worker
    CudaLlm,
    /// Metal-based LLM worker (macOS)
    MetalLlm,
}
```

**Binary names:**
- `CpuLlm` → `llm-worker-rbee-cpu`
- `CudaLlm` → `llm-worker-rbee-cuda`
- `MetalLlm` → `llm-worker-rbee-metal`

**String values (for API):**
- `cpu`
- `cuda`
- `metal`

---

## ✅ Solution Implemented

### 1. TypeScript Type Definition

**File:** `bin/20_rbee_hive/ui/packages/rbee-hive-react/src/hooks/useHiveOperations.ts`

```typescript
// TEAM-377: Worker types (matches Rust enum in worker-catalog/src/types.rs)
export type WorkerType = 'cpu' | 'cuda' | 'metal'

export const WORKER_TYPES: readonly WorkerType[] = ['cpu', 'cuda', 'metal'] as const

export interface WorkerTypeOption {
  value: WorkerType
  label: string
  description: string
}

export const WORKER_TYPE_OPTIONS: readonly WorkerTypeOption[] = [
  { value: 'cpu', label: 'CPU', description: 'CPU-based LLM worker' },
  { value: 'cuda', label: 'CUDA', description: 'NVIDIA GPU-based LLM worker' },
  { value: 'metal', label: 'Metal', description: 'Apple Metal GPU-based LLM worker (macOS)' },
] as const
```

---

### 2. Updated Hook API

**Before (Hardcoded):**
```typescript
export interface UseHiveOperationsResult {
  spawnWorker: (modelId: string) => void
  // ...
}

// Usage
spawnWorker('llama-3.2-1b')  // Always uses 'cuda', device 0
```

**After (Configurable):**
```typescript
export interface SpawnWorkerParams {
  modelId: string
  workerType?: WorkerType  // Optional, defaults to 'cuda'
  deviceId?: number        // Optional, defaults to 0
}

export interface UseHiveOperationsResult {
  spawnWorker: (params: SpawnWorkerParams) => void
  // ...
}

// Usage
spawnWorker({ 
  modelId: 'llama-3.2-1b',
  workerType: 'cuda',
  deviceId: 0
})
```

---

### 3. Exported for UI Components

**File:** `bin/20_rbee_hive/ui/packages/rbee-hive-react/src/index.ts`

```typescript
export { 
  useHiveOperations, 
  WORKER_TYPE_OPTIONS,  // For select components
  WORKER_TYPES          // For validation
} from './hooks/useHiveOperations'

export type { 
  UseHiveOperationsResult, 
  WorkerType,           // Type definition
  WorkerTypeOption,     // Option shape
  SpawnWorkerParams     // Hook params
} from './hooks/useHiveOperations'
```

---

## 🎨 Frontend Select Component Example

### Using Radix UI Select

```tsx
import { useState } from 'react'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@rbee/ui/atoms'
import { useHiveOperations, WORKER_TYPE_OPTIONS, type WorkerType } from '@rbee/rbee-hive-react'

export function WorkerSpawnForm() {
  const [modelId, setModelId] = useState('llama-3.2-1b')
  const [workerType, setWorkerType] = useState<WorkerType>('cuda')
  const [deviceId, setDeviceId] = useState(0)
  
  const { spawnWorker, isPending } = useHiveOperations()

  const handleSpawn = () => {
    spawnWorker({ modelId, workerType, deviceId })
  }

  return (
    <div className="space-y-4">
      {/* Model Selection */}
      <div>
        <label>Model</label>
        <input 
          value={modelId} 
          onChange={(e) => setModelId(e.target.value)}
        />
      </div>

      {/* Worker Type Selection */}
      <div>
        <label>Worker Type</label>
        <Select value={workerType} onValueChange={setWorkerType}>
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {WORKER_TYPE_OPTIONS.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                <div>
                  <div className="font-medium">{option.label}</div>
                  <div className="text-sm text-muted-foreground">
                    {option.description}
                  </div>
                </div>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Device ID Selection */}
      <div>
        <label>Device ID</label>
        <input 
          type="number" 
          value={deviceId} 
          onChange={(e) => setDeviceId(Number(e.target.value))}
          min={0}
        />
      </div>

      {/* Spawn Button */}
      <button onClick={handleSpawn} disabled={isPending}>
        {isPending ? 'Spawning...' : 'Spawn Worker'}
      </button>
    </div>
  )
}
```

---

### Using Native HTML Select

```tsx
import { useState } from 'react'
import { useHiveOperations, WORKER_TYPE_OPTIONS, type WorkerType } from '@rbee/rbee-hive-react'

export function SimpleWorkerSpawn() {
  const [workerType, setWorkerType] = useState<WorkerType>('cuda')
  const { spawnWorker, isPending } = useHiveOperations()

  return (
    <div>
      <select 
        value={workerType} 
        onChange={(e) => setWorkerType(e.target.value as WorkerType)}
      >
        {WORKER_TYPE_OPTIONS.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label} - {option.description}
          </option>
        ))}
      </select>
      
      <button 
        onClick={() => spawnWorker({ 
          modelId: 'llama-3.2-1b', 
          workerType 
        })}
        disabled={isPending}
      >
        Spawn Worker
      </button>
    </div>
  )
}
```

---

## 📊 Mapping: Rust ↔ TypeScript

| Rust Enum | String Value | TypeScript | Label | Description |
|-----------|--------------|------------|-------|-------------|
| `WorkerType::CpuLlm` | `"cpu"` | `'cpu'` | CPU | CPU-based LLM worker |
| `WorkerType::CudaLlm` | `"cuda"` | `'cuda'` | CUDA | NVIDIA GPU-based LLM worker |
| `WorkerType::MetalLlm` | `"metal"` | `'metal'` | Metal | Apple Metal GPU-based LLM worker (macOS) |

---

## 🎯 Benefits

### 1. Type Safety
```typescript
// ✅ Type-safe
const workerType: WorkerType = 'cuda'  // OK
const workerType: WorkerType = 'invalid'  // ❌ TypeScript error

// ✅ Autocomplete in IDE
spawnWorker({ 
  modelId: 'llama-3.2-1b',
  workerType: '...'  // IDE shows: 'cpu' | 'cuda' | 'metal'
})
```

### 2. Single Source of Truth
- Rust enum defines available types
- TypeScript types match exactly
- UI components use same constants
- No hardcoded strings scattered around

### 3. Easy to Extend
**To add a new worker type:**

1. **Update Rust enum:**
```rust
pub enum WorkerType {
    CpuLlm,
    CudaLlm,
    MetalLlm,
    RocmLlm,  // ← New AMD GPU worker
}
```

2. **Update TypeScript:**
```typescript
export type WorkerType = 'cpu' | 'cuda' | 'metal' | 'rocm'

export const WORKER_TYPE_OPTIONS = [
  // ... existing options
  { value: 'rocm', label: 'ROCm', description: 'AMD GPU-based LLM worker' },
] as const
```

3. **UI automatically gets new option** - no other changes needed!

---

## ✅ Verification

### Check Hook API
```typescript
import { useHiveOperations } from '@rbee/rbee-hive-react'

const { spawnWorker } = useHiveOperations()

// ✅ All valid
spawnWorker({ modelId: 'llama-3.2-1b', workerType: 'cpu' })
spawnWorker({ modelId: 'llama-3.2-1b', workerType: 'cuda' })
spawnWorker({ modelId: 'llama-3.2-1b', workerType: 'metal' })

// ✅ Defaults work
spawnWorker({ modelId: 'llama-3.2-1b' })  // Uses 'cuda', device 0
```

### Check Exports
```typescript
import { 
  WORKER_TYPE_OPTIONS,  // ✅ Available
  WORKER_TYPES,         // ✅ Available
  type WorkerType,      // ✅ Available
  type WorkerTypeOption // ✅ Available
} from '@rbee/rbee-hive-react'
```

---

## 📚 Files Modified

**rbee-hive-react (2 files):**
- `src/hooks/useHiveOperations.ts` - Added WorkerType definitions and updated API
- `src/index.ts` - Exported WorkerType constants and types

---

## 🚀 Next Steps

### Recommended: Create Shared WorkerTypeSelect Component

**File:** `frontend/packages/rbee-ui/src/molecules/WorkerTypeSelect/WorkerTypeSelect.tsx`

```typescript
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../../atoms'
import { WORKER_TYPE_OPTIONS, type WorkerType } from '@rbee/rbee-hive-react'

export interface WorkerTypeSelectProps {
  value: WorkerType
  onValueChange: (value: WorkerType) => void
  disabled?: boolean
}

export function WorkerTypeSelect({ value, onValueChange, disabled }: WorkerTypeSelectProps) {
  return (
    <Select value={value} onValueChange={onValueChange} disabled={disabled}>
      <SelectTrigger>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {WORKER_TYPE_OPTIONS.map((option) => (
          <SelectItem key={option.value} value={option.value}>
            <div>
              <div className="font-medium">{option.label}</div>
              <div className="text-sm text-muted-foreground">
                {option.description}
              </div>
            </div>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}
```

Then use it anywhere:
```typescript
import { WorkerTypeSelect } from '@rbee/ui/molecules'

<WorkerTypeSelect 
  value={workerType} 
  onValueChange={setWorkerType} 
/>
```

---

**TEAM-377 | Worker types exposed | 3 types available | Type-safe | Ready for select components! 🎉**
