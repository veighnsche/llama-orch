# TEAM-351 Step 4: Create @rbee/dev-utils Package

**Estimated Time:** 20 minutes  
**Priority:** LOW  
**Previous Step:** TEAM_351_STEP_3_IFRAME_BRIDGE.md  
**Next Step:** TEAM_351_STEP_5_INTEGRATION.md

---

## Mission

Create the `@rbee/dev-utils` package - environment detection and logging utilities.

**Why This Matters:**
- Consistent startup logging across all UIs
- Reusable environment detection
- Clean, informative console output

---

## Deliverables Checklist

- [ ] Package structure created
- [ ] package.json created
- [ ] tsconfig.json created
- [ ] src/environment.ts created
- [ ] src/logging.ts created
- [ ] src/index.ts created
- [ ] README.md created
- [ ] Package builds successfully

---

## Step 1: Create Package Structure

```bash
mkdir -p frontend/packages/dev-utils/src
cd frontend/packages/dev-utils
```

---

## Step 2: Create package.json

```bash
cat > package.json << 'EOF'
{
  "name": "@rbee/dev-utils",
  "version": "0.1.0",
  "type": "module",
  "main": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": {
      "import": "./dist/index.js",
      "types": "./dist/index.d.ts"
    }
  },
  "scripts": {
    "build": "tsc",
    "dev": "tsc --watch"
  },
  "devDependencies": {
    "typescript": "^5.0.0"
  }
}
EOF
```

---

## Step 3: Create tsconfig.json

```bash
cat > tsconfig.json << 'EOF'
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ES2020",
    "moduleResolution": "node",
    "declaration": true,
    "outDir": "./dist",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
EOF
```

---

## Step 4: Create src/environment.ts

```bash
cat > src/environment.ts << 'EOF'
export function isDevelopment(): boolean {
  return import.meta.env.DEV
}

export function isProduction(): boolean {
  return import.meta.env.PROD
}

export function getCurrentPort(): number {
  return parseInt(window.location.port, 10) || 80
}

export function isRunningOnPort(port: number): boolean {
  return getCurrentPort() === port
}
EOF
```

---

## Step 5: Create src/logging.ts

```bash
cat > src/logging.ts << 'EOF'
export function logStartupMode(
  serviceName: string,
  isDev: boolean,
  port?: number
): void {
  const emoji = isDev ? '🔧' : '🚀'
  const mode = isDev ? 'DEVELOPMENT' : 'PRODUCTION'
  
  console.log(`${emoji} [${serviceName}] Running in ${mode} mode`)
  
  if (isDev && port) {
    console.log(`   - Vite dev server active (hot reload enabled)`)
    console.log(`   - Running on: http://localhost:${port}`)
  } else if (!isDev) {
    console.log(`   - Serving embedded static files`)
  }
}
EOF
```

---

## Step 6: Create src/index.ts

```bash
cat > src/index.ts << 'EOF'
export * from './environment'
export * from './logging'
EOF
```

---

## Step 7: Create README.md

```bash
cat > README.md << 'EOF'
# @rbee/dev-utils

Development utilities for environment detection and logging.

## Installation

```bash
pnpm add @rbee/dev-utils
```

## Usage

### Startup Logging

```typescript
import { logStartupMode, isDevelopment, getCurrentPort } from '@rbee/dev-utils'

logStartupMode('QUEEN UI', isDevelopment(), getCurrentPort())
```

**Output (dev mode):**
```
🔧 [QUEEN UI] Running in DEVELOPMENT mode
   - Vite dev server active (hot reload enabled)
   - Running on: http://localhost:7834
```

**Output (prod mode):**
```
🚀 [QUEEN UI] Running in PRODUCTION mode
   - Serving embedded static files
```

### Environment Detection

```typescript
import { isDevelopment, isProduction, getCurrentPort, isRunningOnPort } from '@rbee/dev-utils'

if (isDevelopment()) {
  console.log('Dev mode')
}

if (isRunningOnPort(7834)) {
  console.log('Running on Queen dev port')
}
```

## Features

- ✅ Environment detection (dev/prod)
- ✅ Port detection
- ✅ Consistent startup logging
- ✅ Clean console output
EOF
```

---

## Step 8: Build and Test

```bash
pnpm install
pnpm build
```

---

## Verification Checklist

- [ ] `dist/` folder created
- [ ] All files compiled
- [ ] No TypeScript errors
- [ ] All functions exported

---

## Expected Output

```
dist/
├── index.js
├── index.d.ts
├── environment.js
├── environment.d.ts
├── logging.js
└── logging.d.ts
```

---

## Next Step

✅ **Step 4 Complete!**

**Next:** `TEAM_351_STEP_5_INTEGRATION.md` - Integrate all packages into workspace

---

**TEAM-351 Step 4: Development utilities!** 🛠️
