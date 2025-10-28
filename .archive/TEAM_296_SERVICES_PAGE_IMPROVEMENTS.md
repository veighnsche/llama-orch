# TEAM-296: Services Page Improvements - Clearer Descriptions

**Status:** ✅ COMPLETE  
**Date:** Oct 26, 2025

## Problem

The service card descriptions were too generic and didn't explain what Queen and Hive actually do:

**Before:**
- **Queen:** "Manage the Queen orchestrator service" ❌ (vague)
- **Hive:** "Manage the local Hive service" ❌ (vague)

Users wouldn't understand:
- What Queen does (job routing)
- What Hive does (worker/catalog management)
- Why they need to start a hive to see their models
- How SSH targets relate to remote hives

## Solution

Updated card descriptions to be specific and added helper text explaining the architecture:

### Card Descriptions (Concise)

**Queen:**
```
Job router that dispatches inference requests to workers in the correct hive
```

**Hive (localhost):**
```
Manages workers and catalogs (models, worker binaries) on this machine
```

### Helper Text (Detailed)

Added an "About Services" section below the cards:

```
Queen routes inference jobs to the right worker in the right hive. 
Start Queen first to enable job routing.

Hive manages worker lifecycle and catalogs (models from HuggingFace, 
worker binaries). Start localhost hive to see local models and workers. 
Use SSH targets below to start remote hives and access their catalogs.
```

## Architecture Explained

### Queen (Job Router)
- Routes inference requests to workers
- Determines which hive has the right worker
- Dispatches jobs to the correct worker in the correct hive
- **Start first** to enable job routing

### Hive (Worker & Catalog Manager)
- Manages worker lifecycle (spawn, stop, monitor)
- Maintains model catalog (HuggingFace downloads)
- Maintains worker catalog (binaries from GitHub/local builds)
- Each machine runs its own hive
- **Start localhost hive** to see local models/workers
- **Start remote hive via SSH** to access remote catalogs

### Workflow Example

```
User wants to run inference:
1. Start Queen (job router)
2. Start Hive (localhost or remote via SSH)
3. Hive shows available models in catalog
4. User submits inference job to Queen
5. Queen routes job to worker in correct hive
6. Worker processes inference
```

## UI Changes

### Before
```tsx
<Card>
  <CardHeader>
    <CardTitle>Queen</CardTitle>
  </CardHeader>
  <CardContent>
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        Manage the Queen orchestrator service
      </p>
      <ServiceActionButtons ... />
    </div>
  </CardContent>
</Card>
```

### After
```tsx
<Card>
  <CardHeader>
    <CardTitle>Queen</CardTitle>
  </CardHeader>
  <CardContent>
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        Job router that dispatches inference requests to workers in
        the correct hive
      </p>
      <ServiceActionButtons ... />
    </div>
  </CardContent>
</Card>

{/* NEW: Helper Text */}
<div className="rounded-lg border bg-muted/50 p-4">
  <h3 className="text-sm font-medium mb-2">About Services</h3>
  <div className="space-y-2 text-sm text-muted-foreground">
    <p>
      <strong className="text-foreground">Queen</strong> routes
      inference jobs to the right worker in the right hive. Start Queen
      first to enable job routing.
    </p>
    <p>
      <strong className="text-foreground">Hive</strong> manages worker
      lifecycle and catalogs (models from HuggingFace, worker binaries).
      Start localhost hive to see local models and workers. Use SSH
      targets below to start remote hives and access their catalogs.
    </p>
  </div>
</div>
```

## Files Changed

1. **`bin/00_rbee_keeper/ui/src/pages/ServicesPage.tsx`**
   - Updated Queen card description (lines 91-94)
   - Updated Hive card description (lines 110-113)
   - Added "About Services" helper section (lines 124-140)

## Benefits

1. **Clear Purpose:** Users understand what each service does
2. **Actionable Info:** Explains when/why to start each service
3. **Catalog Discovery:** Users know how to find their models (start hive)
4. **Remote Access:** Explains SSH targets → remote hives → remote catalogs
5. **Not Overwhelming:** Cards stay concise, details in helper section

## User Understanding

### Before (Confusion)
- "What does Queen do?" 🤷
- "Why do I need a Hive?" 🤷
- "Where are my models?" 🤷
- "What are SSH targets for?" 🤷

### After (Clarity)
- "Queen routes jobs to workers" ✅
- "Hive manages workers and catalogs" ✅
- "Start hive to see models" ✅
- "SSH targets start remote hives" ✅

## Design Decisions

### Why Concise Card Descriptions?
- Cards are for quick scanning
- Too much text overwhelms
- Key info only: what it does

### Why Helper Section?
- Provides context without cluttering cards
- Explains relationships (Queen → Hive → Workers)
- Actionable guidance (start Queen first)
- Connects to SSH targets below

### Why "About Services" Title?
- Clear section purpose
- Not "Help" (too generic)
- Not "Documentation" (too formal)
- "About" = informative, friendly

## Visual Layout

```
┌─────────────────────────────────────────────────────────┐
│ Services                                                │
│ Manage Queen, Hive, and SSH connections                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ ┌─────────────────────┐  ┌─────────────────────┐      │
│ │ Queen               │  │ Hive (localhost)    │      │
│ │ Job router that...  │  │ Manages workers...  │      │
│ │ [Start] [Stop] ...  │  │ [Start] [Stop] ...  │      │
│ └─────────────────────┘  └─────────────────────┘      │
│                                                         │
│ ┌─────────────────────────────────────────────────────┐│
│ │ About Services                                      ││
│ │ Queen routes inference jobs...                      ││
│ │ Hive manages worker lifecycle...                    ││
│ └─────────────────────────────────────────────────────┘│
│                                                         │
│ SSH Hives                                               │
│ ┌─────────────────────────────────────────────────────┐│
│ │ [SSH Targets Table]                                 ││
│ └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

## Related Work

- **TEAM-296:** Queen lifecycle implementation
- **TEAM-296:** Install/uninstall fixes
- **TEAM-296:** Services page improvements (this work)

---

**TEAM-296: Updated service card descriptions to be specific and actionable. Added helper text explaining Queen (job router) and Hive (worker/catalog manager) architecture.**
