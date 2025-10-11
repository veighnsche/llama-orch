# TEAM-FE-000: Workspace Setup Complete

**Team:** TEAM-FE-000 (Project Manager)  
**Date:** 2025-10-11  
**Task:** Configure pnpm workspace for side-by-side comparison  
**Status:** Complete ✅

---

## Completed Work

### 1. Updated pnpm Workspace Configuration

✅ **Added to `pnpm-workspace.yaml`:**
- `frontend/bin/commercial-frontend-v2` - New Vue 3 implementation
- `frontend/reference/v0` - React reference for comparison

**Before:**
```yaml
packages:
  - frontend/bin/commercial-frontend
  - frontend/bin/d3-sim-frontend
  - frontend/libs/storybook
  - frontend/libs/frontend-tooling
```

**After:**
```yaml
packages:
  - frontend/bin/commercial-frontend
  - frontend/bin/commercial-frontend-v2  # ✨ NEW
  - frontend/bin/d3-sim-frontend
  - frontend/libs/storybook
  - frontend/libs/frontend-tooling
  - frontend/reference/v0                # ✨ NEW
```

### 2. Created Documentation

✅ **`frontend/WORKSPACE_GUIDE.md`**
- Complete workspace structure
- Installation instructions
- Running all projects
- Side-by-side comparison workflow
- Common commands
- Troubleshooting guide
- Project comparison table

✅ **`frontend/start-comparison.sh`**
- Quick start script
- Starts all 3 servers at once:
  - React reference (port 3000)
  - Storybook (port 6006)
  - Vue v2 (port 5173)

---

## Benefits

### 1. Single Install Command

**Before:** Install each project separately
```bash
cd frontend/reference/v0 && pnpm install
cd frontend/libs/storybook && pnpm install
cd frontend/bin/commercial-frontend-v2 && pnpm install
```

**After:** Install everything at once
```bash
cd /home/vince/Projects/llama-orch
pnpm install
```

### 2. Shared Dependencies

Workspace packages are linked automatically:
- `orchyra-storybook` available in all projects
- `orchyra-frontend-tooling` available in all projects
- No need to publish to npm
- Changes reflect immediately

### 3. Easy Comparison

Run React and Vue side-by-side:
```bash
# Quick start (all 3 servers)
./frontend/start-comparison.sh

# Or manually
pnpm --filter frontend/reference/v0 dev        # React (port 3000)
pnpm --filter orchyra-storybook story:dev      # Storybook (port 6006)
pnpm --filter commercial-frontend-v2 dev       # Vue (port 5173)
```

### 4. Consistent Commands

Use pnpm filters from root:
```bash
# Dev
pnpm --filter commercial-frontend-v2 dev

# Build
pnpm --filter commercial-frontend-v2 build

# Type check
pnpm --filter commercial-frontend-v2 type-check

# Lint
pnpm --filter commercial-frontend-v2 lint
```

---

## Usage Instructions

### Quick Start (Recommended)

```bash
cd /home/vince/Projects/llama-orch

# 1. Install all dependencies
pnpm install

# 2. Start all 3 servers for comparison
./frontend/start-comparison.sh
```

**Opens:**
- React reference: http://localhost:3000
- Storybook: http://localhost:6006
- Vue v2: http://localhost:5173

### Manual Start (Individual Servers)

```bash
# Terminal 1: React reference
cd /home/vince/Projects/llama-orch
pnpm --filter frontend/reference/v0 dev

# Terminal 2: Storybook
pnpm --filter orchyra-storybook story:dev

# Terminal 3: Vue v2
pnpm --filter commercial-frontend-v2 dev
```

---

## Workflow: Porting a Component

### Step 1: Compare Visually

1. **Start React reference:**
   ```bash
   pnpm --filter frontend/reference/v0 dev
   ```
   Open: http://localhost:3000

2. **Navigate to the component** you want to port

### Step 2: Read React Code

```bash
# Example: Button component
cat frontend/reference/v0/components/ui/button.tsx
```

### Step 3: Port to Vue in Storybook

1. **Start Storybook:**
   ```bash
   pnpm --filter orchyra-storybook story:dev
   ```
   Open: http://localhost:6006

2. **Edit Vue component:**
   ```bash
   # Edit: frontend/libs/storybook/stories/atoms/Button/Button.vue
   ```

3. **Edit story:**
   ```bash
   # Edit: frontend/libs/storybook/stories/atoms/Button/Button.story.ts
   ```

4. **Test in Storybook** (http://localhost:6006)

### Step 4: Use in Vue App

1. **Start Vue v2:**
   ```bash
   pnpm --filter commercial-frontend-v2 dev
   ```
   Open: http://localhost:5173

2. **Import component:**
   ```vue
   <script setup>
   import { Button } from 'orchyra-storybook/stories'
   </script>
   
   <template>
     <Button>Click me</Button>
   </template>
   ```

3. **Compare side-by-side:**
   - React (port 3000) vs Vue (port 5173)

---

## Files Created/Modified

### Modified
```
pnpm-workspace.yaml  # Added commercial-frontend-v2 and reference/v0
```

### Created
```
frontend/WORKSPACE_GUIDE.md                                    # Complete guide
frontend/start-comparison.sh                                   # Quick start script
frontend/bin/commercial-frontend-v2/.handoffs/TEAM-FE-000-WORKSPACE-SETUP.md  # This file
```

---

## Verification Checklist

- [x] `pnpm-workspace.yaml` updated
- [x] `commercial-frontend-v2` added to workspace
- [x] `reference/v0` added to workspace
- [x] Workspace guide created
- [x] Quick start script created
- [x] Script is executable
- [ ] Dependencies installed (run `pnpm install`)
- [ ] React reference runs (run `pnpm --filter frontend/reference/v0 dev`)
- [ ] Storybook runs (run `pnpm --filter orchyra-storybook story:dev`)
- [ ] Vue v2 runs (run `pnpm --filter commercial-frontend-v2 dev`)

---

## Next Steps

### 1. Install Dependencies

```bash
cd /home/vince/Projects/llama-orch
pnpm install
```

This will install dependencies for:
- ✅ React reference (Next.js 15)
- ✅ Storybook (Histoire)
- ✅ Vue v2 (Vite)
- ✅ All workspace packages

### 2. Test React Reference

```bash
pnpm --filter frontend/reference/v0 dev
```

Open http://localhost:3000 - Should see the React site

### 3. Test Storybook

```bash
pnpm --filter orchyra-storybook story:dev
```

Open http://localhost:6006 - Should see all 121 components

### 4. Test Vue v2

```bash
pnpm --filter commercial-frontend-v2 dev
```

Open http://localhost:5173 - Should see the Vue site (minimal for now)

### 5. Start Porting!

Use the side-by-side comparison to port components from React to Vue.

---

## Troubleshooting

### "pnpm: command not found"

Install pnpm:
```bash
npm install -g pnpm
```

### "Cannot find module 'orchyra-storybook'"

Install workspace dependencies:
```bash
cd /home/vince/Projects/llama-orch
pnpm install
```

### React reference won't start

Check if Next.js dependencies are installed:
```bash
cd frontend/reference/v0
ls node_modules/next  # Should exist
```

If not:
```bash
cd /home/vince/Projects/llama-orch
pnpm install
```

### Port already in use

Kill the process:
```bash
# Port 3000 (React)
lsof -ti:3000 | xargs kill -9

# Port 6006 (Storybook)
lsof -ti:6006 | xargs kill -9

# Port 5173 (Vue)
lsof -ti:5173 | xargs kill -9
```

---

## Summary

✅ **Workspace configured** - All projects in pnpm workspace  
✅ **Single install** - `pnpm install` installs everything  
✅ **Side-by-side comparison** - Run React and Vue together  
✅ **Quick start script** - `./frontend/start-comparison.sh`  
✅ **Documentation** - Complete guide in `WORKSPACE_GUIDE.md`

**Ready to compare and port!** 🚀

---

## Signatures

```
// Created by: TEAM-FE-000
// Date: 2025-10-11
// Task: Workspace setup for side-by-side comparison
// Status: Complete ✅
```

---

**Next:** Run `pnpm install` and start comparing React vs Vue!
