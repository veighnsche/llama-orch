# Phase 3: Simplify Containers (1 day)

**Team:** TEAM-352  
**Duration:** 1 day  
**Status:** 🔴 NOT STARTED  
**Dependencies:** Phase 2 must be complete

---

## **Goal**

Create generic `QueryContainer<T>` component. Delete `DaemonContainer.tsx` (Rule Zero).

---

## **Background**

**Current (Broken):**
- `DaemonContainer` manages promises, ErrorBoundary, caching
- Complex (161 LOC), hard to understand
- Type-unsafe (`Promise<void>` loses type info)

**Target:**
- `QueryContainer<T>` is dumb (just UI)
- Type-safe (enforces data type)
- Simple (~40 LOC)

---

## **Tasks**

### **Task 1: Create QueryContainer** (2 hours)

**File:** `src/containers/QueryContainer.tsx`

```typescript
// TEAM-352: Generic query container (replaces DaemonContainer)
// Dumb component - just renders loading/error/data states
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Button,
  Alert,
  AlertDescription,
  AlertTitle,
} from '@rbee/ui/atoms'
import { AlertCircle, Loader2 } from 'lucide-react'
import type { ReactNode } from 'react'

interface QueryContainerProps<T> {
  isLoading: boolean
  error: string | null
  data: T | null
  children: (data: T) => ReactNode
  loadingFallback?: ReactNode
  errorFallback?: (error: string, retry: () => void) => ReactNode
  onRetry?: () => void
}

export function QueryContainer<T>({
  isLoading,
  error,
  data,
  children,
  loadingFallback,
  errorFallback,
  onRetry,
}: QueryContainerProps<T>) {
  // Loading state
  if (isLoading && !data) {
    return loadingFallback ?? <DefaultLoadingUI />
  }
  
  // Error state
  if (error && !data) {
    if (errorFallback) {
      return <>{errorFallback(error, onRetry ?? (() => {}))}</>
    }
    return <DefaultErrorUI error={error} onRetry={onRetry} />
  }
  
  // No data state
  if (!data) {
    return null
  }
  
  // Success state - render children with data
  return <>{children(data)}</>
}

// Default loading UI
function DefaultLoadingUI() {
  return (
    <Card>
      <CardContent>
        <div className="flex items-center justify-center py-8">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      </CardContent>
    </Card>
  )
}

// Default error UI
function DefaultErrorUI({ error, onRetry }: { error: string; onRetry?: () => void }) {
  return (
    <Card>
      <CardContent>
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
        {onRetry && (
          <Button onClick={onRetry} className="mt-4 w-full">
            Try Again
          </Button>
        )}
      </CardContent>
    </Card>
  )
}
```

**Key Points:**
- ✅ Type-safe: `<QueryContainer<SshHive>>` enforces children receive `SshHive`
- ✅ Dumb: No fetching, no state, just renders based on props
- ✅ Simple: ~40 LOC vs 161 LOC
- ✅ Composable: Custom loading/error UI via props

---

### **Task 2: Update HiveCard** (1 hour)

**File:** `src/components/cards/HiveCard.tsx` (REWRITE)

```typescript
// TEAM-352: Simplified HiveCard using QueryContainer
import { QueryContainer } from '@/containers/QueryContainer'
import { useHive } from '@/hooks/useHive'
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardAction,
  SplitButton,
} from '@rbee/ui/atoms'
import { Play, Square, Trash2 } from 'lucide-react'

interface HiveCardProps {
  hiveId: string
  title: string
  description: string
}

export function HiveCard({ hiveId, title, description }: HiveCardProps) {
  const { hive, isLoading, error, refetch, start, stop, uninstall } = useHive(hiveId)
  
  return (
    <QueryContainer
      isLoading={isLoading}
      error={error}
      data={hive}
      onRetry={refetch}
    >
      {(hive) => {
        const isRunning = hive.status === 'online'
        
        return (
          <Card>
            <CardHeader>
              <CardTitle>{title} Hive</CardTitle>
              <CardDescription>{description}</CardDescription>
              <CardAction>
                <StatusBadge
                  status={isRunning ? 'running' : 'stopped'}
                  onClick={refetch}
                />
              </CardAction>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <p className="text-sm text-muted-foreground">{description}</p>
                {isRunning ? (
                  <SplitButton
                    variant="destructive"
                    icon={<Square className="h-4 w-4" />}
                    onClick={() => stop(hiveId)}
                    className="w-full"
                  >
                    Stop
                  </SplitButton>
                ) : (
                  <SplitButton
                    icon={<Play className="h-4 w-4" />}
                    onClick={() => start(hiveId)}
                    className="w-full"
                    dropdownContent={
                      <>
                        <DropdownMenuItem
                          onClick={() => uninstall(hiveId)}
                          variant="destructive"
                        >
                          <Trash2 className="mr-2 h-4 w-4" />
                          Uninstall
                        </DropdownMenuItem>
                      </>
                    }
                  >
                    Start
                  </SplitButton>
                )}
              </div>
            </CardContent>
          </Card>
        )
      }}
    </QueryContainer>
  )
}
```

**Code Reduction:**
- Old: 186 LOC (with DaemonContainer wrapper)
- New: ~80 LOC (with QueryContainer)
- **Savings: 106 LOC (57% reduction)**

---

### **Task 3: Update QueenCard** (1 hour)

**File:** `src/components/cards/QueenCard.tsx` (REWRITE)

```typescript
// TEAM-352: Simplified QueenCard using QueryContainer
import { QueryContainer } from '@/containers/QueryContainer'
import { useQueen } from '@/hooks/useQueen'
import { Card, CardHeader, CardTitle, CardContent, SplitButton } from '@rbee/ui/atoms'
import { Play, Square, Download } from 'lucide-react'

export function QueenCard() {
  const { queen, isLoading, error, refetch, start, stop, install } = useQueen()
  
  return (
    <QueryContainer
      isLoading={isLoading}
      error={error}
      data={queen}
      onRetry={refetch}
    >
      {(queen) => {
        const { isRunning, isInstalled } = queen
        
        return (
          <Card>
            <CardHeader>
              <CardTitle>Queen</CardTitle>
              <CardDescription>Job router and scheduler</CardDescription>
              <CardAction>
                <StatusBadge
                  status={!isInstalled ? 'unknown' : isRunning ? 'running' : 'stopped'}
                  onClick={refetch}
                />
              </CardAction>
            </CardHeader>
            <CardContent>
              {!isInstalled ? (
                <Button onClick={install} className="w-full">
                  <Download className="h-4 w-4 mr-2" />
                  Install Queen
                </Button>
              ) : isRunning ? (
                <Button
                  variant="destructive"
                  onClick={stop}
                  className="w-full"
                >
                  <Square className="h-4 w-4 mr-2" />
                  Stop Queen
                </Button>
              ) : (
                <Button onClick={start} className="w-full">
                  <Play className="h-4 w-4 mr-2" />
                  Start Queen
                </Button>
              )}
            </CardContent>
          </Card>
        )
      }}
    </QueryContainer>
  )
}
```

---

### **Task 4: Delete DaemonContainer** (30 min)

**RULE ZERO: DELETE COMPLEXITY**

```bash
# Delete old container
rm src/containers/DaemonContainer.tsx

# Delete old SSH container (if exists)
rm src/containers/SshHivesContainer.tsx
```

**Files to update:**
- Remove imports from all components
- Update any remaining references

---

### **Task 5: Update InstalledHiveList** (1 hour)

**File:** `src/components/InstalledHiveList.tsx` (REWRITE)

```typescript
// TEAM-352: Simplified hive list using QueryContainer
import { QueryContainer } from '@/containers/QueryContainer'
import { useSshHives } from '@/hooks/useSshHives'
import { HiveCard } from './cards/HiveCard'

export function InstalledHiveList() {
  const { hives, isLoading, error, refetch } = useSshHives()
  
  return (
    <QueryContainer
      isLoading={isLoading}
      error={error}
      data={hives}
      onRetry={refetch}
    >
      {(hives) => {
        // Filter to only installed SSH hives (exclude localhost)
        const installedSshHives = hives.filter(
          (hive) => hive.isInstalled && hive.host !== 'localhost'
        )
        
        if (installedSshHives.length === 0) {
          return (
            <div className="col-span-full">
              <p className="text-sm text-muted-foreground text-center py-8">
                No SSH hives installed yet.
              </p>
            </div>
          )
        }
        
        return (
          <>
            {installedSshHives.map((hive) => (
              <HiveCard
                key={hive.host}
                hiveId={hive.host}
                title={hive.host}
                description={`${hive.user}@${hive.hostname}:${hive.port}`}
              />
            ))}
          </>
        )
      }}
    </QueryContainer>
  )
}
```

---

## **Checklist**

- [ ] Create `QueryContainer<T>` component
- [ ] Rewrite `HiveCard.tsx` to use QueryContainer
- [ ] Rewrite `QueenCard.tsx` to use QueryContainer
- [ ] Rewrite `InstalledHiveList.tsx` to use QueryContainer
- [ ] Delete `DaemonContainer.tsx` (Rule Zero)
- [ ] Delete `SshHivesContainer.tsx` (if exists)
- [ ] Remove all DaemonContainer imports
- [ ] Test all cards render correctly
- [ ] Test loading states
- [ ] Test error states
- [ ] All tests pass

---

## **Success Criteria**

✅ `QueryContainer<T>` is type-safe  
✅ All cards use QueryContainer  
✅ DaemonContainer deleted (no compatibility layer)  
✅ 40% code reduction verified  
✅ Loading/error UI works  
✅ Type safety enforced (TypeScript errors if wrong type)

---

## **Code Reduction**

| Component | Before | After | Savings |
|-----------|--------|-------|---------|
| DaemonContainer | 161 LOC | 0 LOC | **161 LOC** |
| QueryContainer | 0 LOC | 40 LOC | -40 LOC |
| HiveCard | 186 LOC | 80 LOC | **106 LOC** |
| QueenCard | ~150 LOC | ~70 LOC | **80 LOC** |
| **Total** | **497 LOC** | **190 LOC** | **307 LOC (62%)** |
