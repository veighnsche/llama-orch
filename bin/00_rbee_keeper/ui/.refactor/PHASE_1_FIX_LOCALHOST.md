# Phase 1: Fix Localhost (1 day)

**Team:** TEAM-350  
**Duration:** 1 day  
**Status:** 🔴 NOT STARTED

---

## **Goal**

Separate localhost from SSH hives. Localhost is always available (no installation needed).

---

## **Background**

**Current (Broken):**
- Frontend treats localhost as SSH target needing installation
- Backend treats localhost as special case (always available)
- Result: Confusing UX, broken workflow

**Target:**
- Localhost = separate component (no install/uninstall)
- SSH hives = targets that require installation
- Clear separation in UI

---

## **Tasks**

### **Task 1: Create LocalhostHive Component** (2 hours)

**File:** `src/components/cards/LocalhostHive.tsx`

```typescript
// TEAM-350: Localhost hive card (no installation workflow)
import { useHive } from '@/store/hiveStore'
import { Card, CardHeader, CardTitle, CardDescription, CardContent, Button } from '@rbee/ui/atoms'
import { Play, Square } from 'lucide-react'

export function LocalhostHive() {
  const { hive, isLoading, error, refetch } = useHive('localhost')
  const { start, stop } = useHiveActions()
  
  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Localhost Hive</CardTitle>
          <CardDescription>This machine</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin" />
          </div>
        </CardContent>
      </Card>
    )
  }
  
  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Localhost Hive</CardTitle>
        </CardHeader>
        <CardContent>
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
          <Button onClick={refetch} className="mt-4">Try Again</Button>
        </CardContent>
      </Card>
    )
  }
  
  const isRunning = hive?.status === 'online'
  
  return (
    <Card>
      <CardHeader>
        <CardTitle>Localhost Hive</CardTitle>
        <CardDescription>This machine</CardDescription>
        <CardAction>
          <StatusBadge 
            status={isRunning ? 'running' : 'stopped'} 
            onClick={refetch} 
          />
        </CardAction>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Manage workers and models on this computer.
          </p>
          {isRunning ? (
            <Button
              variant="destructive"
              onClick={() => stop('localhost')}
              className="w-full"
            >
              <Square className="h-4 w-4 mr-2" />
              Stop Hive
            </Button>
          ) : (
            <Button
              onClick={() => start('localhost')}
              className="w-full"
            >
              <Play className="h-4 w-4 mr-2" />
              Start Hive
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
```

**Key Points:**
- ✅ No "Install" button (localhost is always available)
- ✅ No "Uninstall" button (can't remove localhost)
- ✅ Only Start/Stop actions
- ✅ Uses `useHive('localhost')` hook (from Phase 2)

---

### **Task 2: Update ServicesPage** (30 min)

**File:** `src/pages/ServicesPage.tsx`

```typescript
// TEAM-350: Show localhost separately from SSH hives
import { LocalhostHive } from '@/components/cards/LocalhostHive'
import { SshHiveList } from '@/components/SshHiveList'
import { InstallHiveCard } from '@/components/cards/InstallHiveCard'

export default function ServicesPage() {
  return (
    <PageContainer title="Services" description="Manage Queen and Hive services">
      <div className="grid gap-4 sm:gap-6 grid-cols-1 md:grid-cols-2">
        <QueenCard />
        <LocalhostHive />  {/* ← Always shown, no install needed */}
      </div>
      
      <SshHiveList />      {/* ← Installed SSH hives */}
      <InstallHiveCard />  {/* ← Install new SSH hive */}
    </PageContainer>
  )
}
```

---

### **Task 3: Remove Localhost from SSH Lists** (1 hour)

**File:** `src/components/InstalledHiveList.tsx`

```typescript
// TEAM-350: Remove localhost logic (handled by LocalhostHive component)
function InstalledHiveCards() {
  const { hives, installedHives } = useSshHivesStore()
  
  // TEAM-350: Filter OUT localhost (shown separately)
  const installedSshHives = hives.filter((hive) =>
    installedHives.includes(hive.host) && hive.host !== 'localhost'
  )
  
  if (installedSshHives.length === 0) {
    return (
      <div className="col-span-full">
        <p className="text-sm text-muted-foreground text-center py-8">
          No SSH hives installed yet. Install one below.
        </p>
      </div>
    )
  }
  
  return (
    <>
      {installedSshHives.map((hive) => (
        <HiveCard key={hive.host} hiveId={hive.host} /* ... */ />
      ))}
    </>
  )
}
```

**File:** `src/components/cards/InstallHiveCard.tsx`

```typescript
// TEAM-350: Remove localhost from install dropdown
<SelectContent>
  {/* TEAM-350: No localhost option (always available) */}
  {availableHives
    .filter((hive) => hive.host !== 'localhost')  // ← Filter out localhost
    .map((hive) => (
      <SelectItem key={hive.host} value={hive.host}>
        <SshTargetItem name={hive.host} subtitle={`${hive.user}@${hive.hostname}`} />
      </SelectItem>
    ))}
</SelectContent>
```

---

### **Task 4: Backend Verification** (2 hours)

**Verify:** Localhost commands work without `hives.conf`

**Test:**
```bash
# Start keeper
cargo run --bin rbee-keeper

# In UI: Click "Start Hive" on localhost card
# Should work WITHOUT creating hives.conf

# Verify in terminal
curl http://localhost:7835/health
# Should return 200 OK
```

**If broken:** Check `bin/00_rbee_keeper/src/ssh_resolver.rs`

```rust
// Should bypass SSH for localhost
pub fn resolve_ssh_config(host_alias: &str) -> Result<SshConfig> {
    if host_alias == "localhost" {
        return Ok(SshConfig::localhost());  // ← Must work without hives.conf
    }
    // ...
}
```

---

## **Checklist**

- [ ] Create `LocalhostHive.tsx` component
- [ ] Update `ServicesPage.tsx` to show LocalhostHive
- [ ] Remove localhost from `InstalledHiveList.tsx`
- [ ] Remove localhost from `InstallHiveCard.tsx` dropdown
- [ ] Test localhost start/stop (no install needed)
- [ ] Verify backend works without `hives.conf`
- [ ] Update sidebar (if localhost shown there)
- [ ] All tests pass
- [ ] Code reviewed
- [ ] Merged to main

---

## **Success Criteria**

✅ Localhost hive card always visible on Services page  
✅ No "Install" button on localhost card  
✅ Start/Stop work without installation  
✅ SSH hive list excludes localhost  
✅ Install dropdown excludes localhost  
✅ Backend accepts localhost without hives.conf

---

## **Dependencies**

**Blocks:** Phase 2 (needs this to be clean before query pattern)  
**Blocked By:** None

---

## **Notes**

- Localhost is NOT an SSH target - it's local
- Don't try to "fix" localhost in SSH config - separate it
- If backend needs changes, update ssh_resolver.rs
- This phase is foundational - don't skip it
