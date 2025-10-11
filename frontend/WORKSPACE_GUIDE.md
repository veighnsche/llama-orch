# Frontend Workspace Guide

**Created by:** TEAM-FE-000  
**Date:** 2025-10-11

---

## 📦 Workspace Structure

All frontend projects are now in the pnpm workspace:

```
frontend/
├── bin/
│   ├── commercial-frontend/        # Vue 3 commercial frontend ✨
│   └── d3-sim-frontend/            # D3 simulation frontend
├── libs/
│   ├── storybook/                  # Shared component library
│   └── frontend-tooling/           # Shared tooling (ESLint, Prettier, etc.)
└── reference/
    └── v0/                         # React reference (for comparison) 🔍
```

---

## 🚀 Installation

From the **project root**:

```bash
cd /home/vince/Projects/llama-orch
pnpm install
```

This installs dependencies for **all** frontend projects at once!

---

## 🎯 Running Projects

### 1. React Reference (v0) - For Comparison

```bash
# From root
pnpm --filter frontend/reference/v0 dev

# Or from the v0 directory
cd frontend/reference/v0
pnpm dev
```

**URL:** http://localhost:3000  
**Purpose:** Compare with Vue implementation

### 2. Commercial Frontend (Vue 3)

```bash
# From root
pnpm --filter rbee-commercial-frontend dev

# Or from the directory
cd frontend/bin/commercial-frontend
pnpm dev
```

**URL:** http://localhost:5173  
**Purpose:** Vue 3 commercial frontend

### 3. Storybook (Component Library)

```bash
# From root
pnpm --filter rbee-storybook story:dev

# Or from the directory
cd frontend/libs/storybook
pnpm story:dev
```

**URL:** http://localhost:6006  
**Purpose:** Develop and test components in isolation


---

## 🔄 Side-by-Side Comparison Workflow

### Step 1: Run React Reference
```bash
# Terminal 1
cd /home/vince/Projects/llama-orch/frontend/reference/v0
pnpm dev
```
Open: http://localhost:3000

### Step 2: Run Storybook
```bash
# Terminal 2
cd /home/vince/Projects/llama-orch/frontend/libs/storybook
pnpm story:dev
```
Open: http://localhost:6006

### Step 3: Run Vue Frontend
```bash
# Terminal 3
cd /home/vince/Projects/llama-orch/frontend/bin/commercial-frontend
pnpm dev
```
Open: http://localhost:5173

### Step 4: Compare!
- **React (port 3000)** - Original design
- **Storybook (port 6006)** - Component development
- **Vue (port 5173)** - New implementation

---

## 📝 Common Commands

### Install All Dependencies
```bash
cd /home/vince/Projects/llama-orch
pnpm install
```

### Install for Specific Project
```bash
# React reference
pnpm --filter frontend/reference/v0 install

# Vue v2
pnpm --filter commercial-frontend-v2 install

# Storybook
pnpm --filter orchyra-storybook install
```

### Build All Projects
```bash
pnpm -r build
```

### Build Specific Project
```bash
# React reference
pnpm --filter frontend/reference/v0 build

# Vue v2
pnpm --filter commercial-frontend-v2 build

# Storybook
pnpm --filter orchyra-storybook story:build
```

### Type Check
```bash
# Vue v2
pnpm --filter commercial-frontend-v2 type-check

# Storybook
cd frontend/libs/storybook && pnpm type-check
```

### Lint
```bash
# Vue v2
pnpm --filter commercial-frontend-v2 lint

# Storybook
pnpm --filter orchyra-storybook lint
```

---

## 🎨 Development Workflow

### Porting a Component

1. **Start React reference** (for visual comparison)
   ```bash
   cd frontend/reference/v0
   pnpm dev
   ```

2. **Start Storybook** (for component development)
   ```bash
   cd frontend/libs/storybook
   pnpm story:dev
   ```

3. **Read React component**
   ```bash
   cat frontend/reference/v0/components/ui/button.tsx
   ```

4. **Port to Vue in Storybook**
   - Edit: `frontend/libs/storybook/stories/atoms/Button/Button.vue`
   - Edit: `frontend/libs/storybook/stories/atoms/Button/Button.story.ts`

5. **Test in Storybook**
   - Open: http://localhost:6006
   - Navigate to: atoms/Button
   - Test all variants

6. **Use in Vue app**
   ```vue
   <script setup>
   import { Button } from 'orchyra-storybook/stories'
   </script>
   
   <template>
     <Button>Click me</Button>
   </template>
   ```

7. **Compare side-by-side**
   - React: http://localhost:3000
   - Vue: http://localhost:5173

---

## 🐛 Troubleshooting

### "Cannot find module 'orchyra-storybook'"

Run install from root:
```bash
cd /home/vince/Projects/llama-orch
pnpm install
```

### Port Already in Use

Kill the process:
```bash
# Find process on port 3000
lsof -ti:3000 | xargs kill -9

# Find process on port 5173
lsof -ti:5173 | xargs kill -9

# Find process on port 6006
lsof -ti:6006 | xargs kill -9
```

### React Reference Won't Start

Check if Next.js dependencies are installed:
```bash
cd frontend/reference/v0
pnpm install
pnpm dev
```

### Storybook Won't Start

Check if Histoire dependencies are installed:
```bash
cd frontend/libs/storybook
pnpm install
pnpm story:dev
```

### Vue v2 Won't Start

Check if Vite dependencies are installed:
```bash
cd frontend/bin/commercial-frontend-v2
pnpm install
pnpm dev
```

---

## 📊 Project Comparison

| Feature | React Reference (v0) | Vue v2 (commercial-frontend-v2) |
|---------|---------------------|----------------------------------|
| **Framework** | Next.js 15 + React 19 | Vite + Vue 3 |
| **UI Library** | shadcn/ui (Radix UI) | Radix Vue |
| **Styling** | Tailwind CSS v4 | Tailwind CSS v4 |
| **Icons** | Lucide React | Lucide Vue Next |
| **State** | React hooks | Vue Composables |
| **Router** | Next.js App Router | Vue Router 4 |
| **Dev Server** | Next.js (port 3000) | Vite (port 5173) |
| **Build** | Next.js | Vite |
| **Purpose** | Reference implementation | Production implementation |

---

## 🎯 Quick Reference

### Ports
- **3000** - React reference (Next.js)
- **5173** - Vue v2 (Vite)
- **6006** - Storybook (Histoire)

### Key Directories
- **React components:** `frontend/reference/v0/components/`
- **Vue components:** `frontend/libs/storybook/stories/`
- **Vue pages:** `frontend/bin/commercial-frontend-v2/src/views/`

### Key Files
- **React page:** `frontend/reference/v0/app/page.tsx`
- **Vue page:** `frontend/bin/commercial-frontend-v2/src/views/HomeView.vue`
- **Storybook exports:** `frontend/libs/storybook/stories/index.ts`

---

## 📚 Documentation

- **Port Plan:** `frontend/bin/commercial-frontend-v2/REACT_TO_VUE_PORT_PLAN.md`
- **Developer Checklist:** `frontend/libs/storybook/DEVELOPER_CHECKLIST.md`
- **Dependencies Guide:** `frontend/bin/commercial-frontend-v2/DEPENDENCIES_GUIDE.md`
- **Atomic Design:** `frontend/libs/storybook/ATOMIC_DESIGN_PHILOSOPHY.md`
- **Scaffolding:** `frontend/libs/storybook/SCAFFOLDING_COMPLETE.md`

---

**Created by:** TEAM-FE-000  
**Workspace configured!** ✅  
**Ready to compare React and Vue side-by-side!** 🚀
