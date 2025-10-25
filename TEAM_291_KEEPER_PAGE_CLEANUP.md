# TEAM-291: Bee Keeper Page Cleanup

**Status:** ✅ COMPLETE

**Mission:** Remove Worker and Model operations from Bee Keeper page, keeping only operations that cannot be done through Queen HTTP API.

## Rationale

Worker and Model operations can be done through the Queen HTTP API (already forwarded to hives). The Bee Keeper page should only show direct CLI operations for:
- Queen lifecycle management (start, stop, restart, status, build)
- Hive lifecycle management (local and SSH)
- Git operations on remote hosts (SSH)

## Operations Removed

### ❌ Worker Operations (Removed)
- List Workers
- Spawn Worker
- Retire Worker

**Reason:** These are forwarded to hives via Queen HTTP API

### ❌ Model Operations (Removed)
- List Models
- Download Model
- Delete Model

**Reason:** These are forwarded to hives via Queen HTTP API

## Operations Kept

### ✅ Queen Operations
**Purpose:** Direct lifecycle management of the central orchestrator

**Commands:**
1. Start Queen
2. Stop Queen
3. Restart Queen
4. Queen Status
5. Build Queen

**Why:** These cannot be done through HTTP API (Queen manages itself)

### ✅ Hive Operations (Local)
**Purpose:** Manage local hive instances

**Commands:**
1. Start Hive
2. Stop Hive
3. Restart Hive
4. Hive Status
5. Build Hive

**Why:** Direct CLI access for local development/debugging

### ✅ Hive Operations (SSH)
**Purpose:** Manage remote hive instances via SSH

**Commands:**
1. Install Hive (SSH)
2. Start Hive (SSH)
3. Stop Hive (SSH)
4. Restart Hive (SSH)
5. Hive Status (SSH)
6. Build Hive (SSH)

**Why:** SSH operations for remote deployment and management

### ✅ Git Operations (SSH)
**Purpose:** Download and update git repositories on remote hosts

**Commands:**
1. Clone Repository
2. Pull Updates
3. Check Status

**Why:** Remote git management for deployment

## Layout Changes

### Before (4 Cards)
```
┌─────────────────┬─────────────────┐
│ Queen Ops (3)   │ Hive Ops (4)    │
├─────────────────┼─────────────────┤
│ Worker Ops (3)  │ Model Ops (3)   │ ← Removed
└─────────────────┴─────────────────┘
```

### After (4 Cards)
```
┌─────────────────┬─────────────────┐
│ Queen Ops (5)   │ Hive Ops        │
│                 │ Local (5)       │
├─────────────────┼─────────────────┤
│ Hive Ops        │ Git Ops         │
│ SSH (6)         │ SSH (3)         │
└─────────────────┴─────────────────┘
```

## New Organization

### Card 1: Queen Operations (5 commands)
- Start, Stop, Restart, Status, Build
- Lifecycle management for central orchestrator

### Card 2: Hive Operations (Local) (5 commands)
- Start, Stop, Restart, Status, Build
- Local hive instance management

### Card 3: Hive Operations (SSH) (6 commands)
- Install, Start, Stop, Restart, Status, Build
- Remote hive management via SSH

### Card 4: Git Operations (SSH) (3 commands)
- Clone Repository, Pull Updates, Check Status
- Remote git repository management

## Command Count

**Before:**
- Queen: 3 commands
- Hive: 4 commands
- Worker: 3 commands (removed)
- Model: 3 commands (removed)
- **Total: 13 commands**

**After:**
- Queen: 5 commands (+2)
- Hive (Local): 5 commands (+1)
- Hive (SSH): 6 commands (new)
- Git (SSH): 3 commands (new)
- **Total: 19 commands**

## Architecture Alignment

### Queen HTTP API (Dashboard)
```
Dashboard → Queen HTTP API → Hive HTTP API
  ↓
- Worker operations (spawn, list, retire)
- Model operations (download, list, delete)
- Inference operations
```

### Bee Keeper (Direct CLI)
```
Bee Keeper → rbee CLI → Direct execution
  ↓
- Queen lifecycle (start, stop, build)
- Hive lifecycle (local and SSH)
- Git operations (SSH)
```

## Separation of Concerns

### Dashboard
- **Purpose:** Monitor and manage via HTTP API
- **Operations:** Worker, Model, Inference
- **Method:** HTTP requests to Queen
- **Audience:** End users

### Bee Keeper
- **Purpose:** Direct CLI operations for infrastructure
- **Operations:** Queen, Hive, Git
- **Method:** Direct CLI execution
- **Audience:** Operators, DevOps

## Files Changed

### Modified
1. **`src/app/keeper/page.tsx`**
   - Removed Worker Operations card
   - Removed Model Operations card
   - Added Restart and Build to Queen Operations
   - Split Hive Operations into Local and SSH
   - Added Git Operations (SSH) card
   - Updated page description

## Command Mapping

### Queen Operations
```bash
./rbee queen start
./rbee queen stop
./rbee queen restart
./rbee queen status
./rbee queen build
```

### Hive Operations (Local)
```bash
./rbee hive start -a localhost
./rbee hive stop -a localhost
./rbee hive restart -a localhost
./rbee hive status -a localhost
./rbee hive build -a localhost
```

### Hive Operations (SSH)
```bash
./rbee hive install -a workstation
./rbee hive start -a workstation
./rbee hive stop -a workstation
./rbee hive restart -a workstation
./rbee hive status -a workstation
./rbee hive build -a workstation
```

### Git Operations (SSH)
```bash
# Clone repository on remote host
./rbee git clone -a workstation --repo <url>

# Pull updates on remote host
./rbee git pull -a workstation

# Check git status on remote host
./rbee git status -a workstation
```

## Future Implementation

### Phase 1: Command Execution
- Wire up buttons to execute CLI commands
- Stream output to Command Output card
- Handle errors and display

### Phase 2: Input Forms
- Add dialogs for commands requiring parameters
- Hive alias selection
- Repository URL input
- Build options

### Phase 3: Command History
- Show previous commands
- Re-run commands
- Save command presets

### Phase 4: Batch Operations
- Execute multiple commands
- Command sequences
- Conditional execution

## Benefits

### Clarity
- ✅ Clear separation between HTTP API and CLI operations
- ✅ Focused on infrastructure management
- ✅ No confusion about which operations to use

### Maintainability
- ✅ Aligned with backend architecture
- ✅ No duplicate functionality
- ✅ Single source of truth for each operation type

### User Experience
- ✅ Dashboard for day-to-day operations
- ✅ Bee Keeper for infrastructure management
- ✅ Clear purpose for each interface

## Testing Checklist

- ✅ Page loads without errors
- ✅ All 4 cards visible
- ✅ 19 total buttons displayed
- ✅ Command Output card at bottom
- ✅ Layout responsive (2 columns)
- ✅ Sidebar navigation works
- ✅ Active state on Bee Keeper

## Engineering Rules Compliance

- ✅ Layout only (no logic yet)
- ✅ Clear separation of concerns
- ✅ Aligned with architecture
- ✅ No breaking changes
- ✅ Team signature (TEAM-291)

---

**TEAM-291 COMPLETE** - Bee Keeper page cleaned up, focused on CLI operations only.
