# Part 1: Keeper GUI Setup

**TEAM-293: Set up rbee-keeper GUI structure**

## Goal

Set up the new GUI directory structure and configure Tauri to run from the repository root.

## Current Structure

```
bin/00_rbee_keeper/
├── ui/                    # ❌ OLD (deleted by user)
├── src/
├── tauri.conf.json
└── Cargo.toml
```

## New Structure

```
frontend/apps/00_rbee_keeper/  # ✅ NEW (moved from bin/00_rbee_keeper/GUI)
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   │   ├── components/
│   │   │   ├── Sidebar.tsx
│   │   │   └── IframeHost.tsx
│   │   ├── pages/
│   │   │   ├── KeeperDashboard.tsx
│   │   │   ├── KeeperInstall.tsx
│   │   │   └── KeeperSettings.tsx
│   │   └── lib/
│   │       ├── heartbeat.ts
│   │       └── tauri-commands.ts
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── tailwind.config.js
├── src/                   # Rust code
├── tauri.conf.json
└── Cargo.toml
```

## Step 1: Update pnpm-workspace.yaml

**File:** `/home/vince/Projects/llama-orch/pnpm-workspace.yaml`

```yaml
packages:
  - frontend/apps/commercial
  - frontend/apps/user-docs
  - frontend/apps/00_rbee_keeper    # ✅ Keeper GUI
  - frontend/apps/10_queen_rbee     # Renamed from web-ui
  - frontend/apps/20_rbee_hive      # Hive UI
  - frontend/packages/rbee-ui
  - frontend/packages/rbee-sdk
  - frontend/packages/rbee-react
  - frontend/packages/tailwind-config
```

## Step 2: Create GUI package.json

**File:** `frontend/apps/00_rbee_keeper/package.json`

```json
{
  "name": "@rbee/00-keeper-gui",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0"
  },
  "dependencies": {
    "@rbee-ui/styles": "workspace:*",
    "@rbee-ui/stories": "workspace:*",
    "@tauri-apps/api": "^1.5.3",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.22.0",
    "zustand": "^4.5.0"
  },
  "// NOTE": "Keeper uses Tauri commands, NOT SDK packages",
  "devDependencies": {
    "@types/react": "^18.2.56",
    "@types/react-dom": "^18.2.19",
    "@typescript-eslint/eslint-plugin": "^7.0.2",
    "@typescript-eslint/parser": "^7.0.2",
    "@vitejs/plugin-react": "^4.2.1",
    "autoprefixer": "^10.4.17",
    "eslint": "^8.56.0",
    "eslint-plugin-react-hooks": "^4.6.0",
    "eslint-plugin-react-refresh": "^0.4.5",
    "postcss": "^8.4.35",
    "tailwindcss": "^3.4.1",
    "typescript": "^5.2.2",
    "vite": "^5.1.4"
  }
}
```

## Step 3: Create GUI vite.config.ts

**File:** `frontend/apps/00_rbee_keeper/vite.config.ts`

```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  clearScreen: false,
  server: {
    port: 5173,
    strictPort: true,
    watch: {
      ignored: ['**/target/**'],
    },
  },
  envPrefix: ['VITE_', 'TAURI_'],
  build: {
    target: 'esnext',
    outDir: 'dist',
    assetsDir: 'assets',
    minify: 'esbuild',
    sourcemap: false,
  },
});
```

## Step 4: Create GUI tsconfig.json

**File:** `bin/00_rbee_keeper/GUI/tsconfig.json`

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

## Step 5: Create GUI tailwind.config.js

**File:** `bin/00_rbee_keeper/GUI/tailwind.config.js`

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
    '../../../frontend/packages/rbee-ui/**/*.{vue,js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {},
  },
  plugins: [],
};
```

## Step 6: Create GUI index.html

**File:** `bin/00_rbee_keeper/GUI/index.html`

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>rbee Keeper</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

## Step 7: Update tauri.conf.json

**File:** `bin/00_rbee_keeper/tauri.conf.json`

```json
{
  "$schema": "https://schema.tauri.app/config/1",
  "build": {
    "devPath": "http://localhost:5173",
    "distDir": "../../../frontend/apps/00_rbee_keeper/dist"
  },
  "package": {
    "productName": "rbee Keeper",
    "version": "0.1.0"
  },
  "tauri": {
    "allowlist": {
      "all": false,
      "shell": {
        "all": false,
        "open": false
      },
      "window": {
        "all": false,
        "close": true,
        "hide": true,
        "show": true,
        "maximize": true,
        "minimize": true,
        "unmaximize": true,
        "unminimize": true,
        "startDragging": true
      }
    },
    "bundle": {
      "active": false,
      "identifier": "com.rbee.keeper"
    },
    "security": {
      "csp": null
    },
    "windows": [
      {
        "fullscreen": false,
        "resizable": true,
        "title": "rbee Keeper",
        "width": 1400,
        "height": 900,
        "minWidth": 1200,
        "minHeight": 700
      }
    ]
  }
}
```

## Step 8: Enable cargo tauri run from Root

**Goal:** Run `cargo tauri run` from `/home/vince/Projects/llama-orch/` and have it work

### Option A: Create Workspace Alias (Recommended)

**File:** `/home/vince/Projects/llama-orch/.cargo/config.toml`

```toml
[alias]
tauri = "tauri --manifest-path bin/00_rbee_keeper/Cargo.toml"
```

**Usage:**
```bash
cd /home/vince/Projects/llama-orch
cargo tauri dev
cargo tauri build
```

### Option B: Shell Script Wrapper

**File:** `/home/vince/Projects/llama-orch/run-keeper-gui.sh`

```bash
#!/bin/bash
cd bin/00_rbee_keeper
cargo tauri "$@"
```

**Usage:**
```bash
./run-keeper-gui.sh dev
./run-keeper-gui.sh build
```

### Option C: Makefile

**File:** `/home/vince/Projects/llama-orch/Makefile`

```makefile
.PHONY: keeper-gui keeper-gui-dev keeper-gui-build

keeper-gui-dev:
	cd bin/00_rbee_keeper && cargo tauri dev

keeper-gui-build:
	cd bin/00_rbee_keeper && cargo tauri build

keeper-gui: keeper-gui-dev
```

**Usage:**
```bash
make keeper-gui-dev
make keeper-gui-build
```

## Step 9: Install Dependencies

```bash
# From root
cd /home/vince/Projects/llama-orch

# Install GUI dependencies
pnpm install

# Verify
pnpm --filter @rbee/keeper-gui list
```

## Step 10: Verify Setup

```bash
# Start frontend dev server
cd frontend/apps/00_rbee_keeper
pnpm dev

# In another terminal, start Tauri
cd bin/00_rbee_keeper
cargo tauri dev

# Or from root (if using alias/script)
cd /home/vince/Projects/llama-orch
cargo tauri dev
```

## Expected Output

```
✅ GUI dev server running on http://localhost:5173
✅ Tauri window opens
✅ Empty React app visible
✅ No errors in console
```

## File Checklist

- [ ] `pnpm-workspace.yaml` updated
- [ ] `GUI/package.json` created
- [ ] `GUI/vite.config.ts` created
- [ ] `GUI/tsconfig.json` created
- [ ] `GUI/tailwind.config.js` created
- [ ] `GUI/index.html` created
- [ ] `tauri.conf.json` updated
- [ ] `.cargo/config.toml` created (for root command)
- [ ] Dependencies installed
- [ ] Dev server runs
- [ ] Tauri app opens

## Next Steps

Once this setup is complete:
1. **Next:** `02_RENAME_WEB_UI.md` - Rename web-ui to ui-queen-rbee
2. **Then:** `03_EXTRACT_KEEPER_PAGE.md` - Extract keeper page to GUI

---

**Status:** 📋 READY TO IMPLEMENT
