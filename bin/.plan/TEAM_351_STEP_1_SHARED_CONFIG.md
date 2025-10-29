# TEAM-351 Step 1: Create @rbee/shared-config Package

**Estimated Time:** 30-45 minutes  
**Priority:** CRITICAL  
**Next Step:** TEAM_351_STEP_2_NARRATION_CLIENT.md

---

## Mission

Create the `@rbee/shared-config` package - the single source of truth for all port configuration.

**Why This Matters:**
- Eliminates hardcoded ports across all UIs
- Generates Rust constants from TypeScript
- One place to update when adding services

---

## Deliverables Checklist

- [ ] Package structure created
- [ ] package.json created
- [ ] tsconfig.json created
- [ ] src/ports.ts created (port configuration)
- [ ] src/index.ts created (exports)
- [ ] Rust code generator script created
- [ ] README.md created
- [ ] Package builds successfully
- [ ] Rust constants generated

---

## Step 1: Create Package Structure

```bash
mkdir -p frontend/packages/shared-config/src
mkdir -p frontend/packages/shared-config/scripts
cd frontend/packages/shared-config
```

---

## Step 2: Create package.json

```bash
cat > package.json << 'EOF'
{
  "name": "@rbee/shared-config",
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
    "dev": "tsc --watch",
    "generate:rust": "node scripts/generate-rust.js"
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

## Step 4: Create src/ports.ts

```bash
cat > src/ports.ts << 'EOF'
/**
 * SINGLE SOURCE OF TRUTH for all port configurations
 * 
 * CRITICAL: When adding a new service:
 * 1. Add to PORTS constant
 * 2. Update PORT_CONFIGURATION.md
 * 3. Run `pnpm generate:rust`
 * 4. Update backend Cargo.toml default port
 * 
 * @packageDocumentation
 */

/**
 * Port configuration for each service
 */
export const PORTS = {
  keeper: {
    dev: 5173,
    prod: null,  // Tauri app, no HTTP port
  },
  queen: {
    dev: 7834,      // Vite dev server
    prod: 7833,     // Embedded in backend
    backend: 7833,  // Backend HTTP server
  },
  hive: {
    dev: 7836,
    prod: 7835,
    backend: 7835,
  },
  worker: {
    dev: 7837,
    prod: 8080,
    backend: 8080,
  },
} as const

export type ServiceName = keyof typeof PORTS

/**
 * Generate allowed origins for postMessage listener
 * Automatically includes all dev and prod ports
 */
export function getAllowedOrigins(): string[] {
  const origins: string[] = []
  
  for (const [service, ports] of Object.entries(PORTS)) {
    if (service === 'keeper') continue  // Keeper doesn't send messages
    
    if (ports.dev) {
      origins.push(`http://localhost:${ports.dev}`)
    }
    if (ports.prod) {
      origins.push(`http://localhost:${ports.prod}`)
    }
  }
  
  return origins
}

/**
 * Get iframe URL for a service
 * @param service - Service name
 * @param isDev - Development mode flag
 */
export function getIframeUrl(
  service: ServiceName,
  isDev: boolean
): string {
  const ports = PORTS[service]
  const port = isDev ? ports.dev : ports.prod
  return port ? `http://localhost:${port}` : ''
}

/**
 * Get parent origin for postMessage
 * Detects environment based on current port
 * @param currentPort - Current window port
 */
export function getParentOrigin(currentPort: number): string {
  // If we're on a dev port, send to keeper dev
  const isDevPort = Object.values(PORTS).some(p => p.dev === currentPort)
  
  return isDevPort
    ? `http://localhost:${PORTS.keeper.dev}`  // Dev: Keeper Vite
    : '*'                                       // Prod: Tauri app
}

/**
 * Get service URL for HTTP requests
 * @param service - Service name
 * @param mode - 'dev' or 'prod'
 */
export function getServiceUrl(
  service: ServiceName,
  mode: 'dev' | 'prod' = 'dev'
): string {
  const ports = PORTS[service]
  const port = mode === 'dev' ? ports.dev : ports.prod
  return port ? `http://localhost:${port}` : ''
}
EOF
```

---

## Step 5: Create src/index.ts

```bash
cat > src/index.ts << 'EOF'
export * from './ports'
EOF
```

---

## Step 6: Create Rust Code Generator

```bash
cat > scripts/generate-rust.js << 'EOF'
#!/usr/bin/env node

import { writeFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

// Import the ports config
const PORTS = {
  keeper: { dev: 5173, prod: null },
  queen: { dev: 7834, prod: 7833, backend: 7833 },
  hive: { dev: 7836, prod: 7835, backend: 7835 },
  worker: { dev: 7837, prod: 8080, backend: 8080 },
}

const rustCode = `// AUTO-GENERATED from frontend/packages/shared-config/src/ports.ts
// DO NOT EDIT MANUALLY - Run 'pnpm generate:rust' in shared-config package to update
// 
// This file provides port constants for Rust build.rs scripts
// Last generated: ${new Date().toISOString()}

// TEAM-351: Shared port configuration constants

pub const KEEPER_DEV_PORT: u16 = ${PORTS.keeper.dev};

pub const QUEEN_DEV_PORT: u16 = ${PORTS.queen.dev};
pub const QUEEN_PROD_PORT: u16 = ${PORTS.queen.prod};
pub const QUEEN_BACKEND_PORT: u16 = ${PORTS.queen.backend};

pub const HIVE_DEV_PORT: u16 = ${PORTS.hive.dev};
pub const HIVE_PROD_PORT: u16 = ${PORTS.hive.prod};
pub const HIVE_BACKEND_PORT: u16 = ${PORTS.hive.backend};

pub const WORKER_DEV_PORT: u16 = ${PORTS.worker.dev};
pub const WORKER_PROD_PORT: u16 = ${PORTS.worker.prod};
pub const WORKER_BACKEND_PORT: u16 = ${PORTS.worker.backend};
`

const outputPath = join(__dirname, '..', '..', '..', 'shared-constants.rs')
writeFileSync(outputPath, rustCode, 'utf8')

console.log('✅ Generated Rust constants at:', outputPath)
EOF

chmod +x scripts/generate-rust.js
```

---

## Step 7: Create README.md

```bash
cat > README.md << 'EOF'
# @rbee/shared-config

Single source of truth for port configuration across the entire rbee project.

## Installation

```bash
pnpm add @rbee/shared-config
```

## Usage

### Get iframe URL

```typescript
import { getIframeUrl } from '@rbee/shared-config'

const isDev = import.meta.env.DEV
const queenUrl = getIframeUrl('queen', isDev)
// Dev: http://localhost:7834
// Prod: http://localhost:7833
```

### Get allowed origins

```typescript
import { getAllowedOrigins } from '@rbee/shared-config'

const allowedOrigins = getAllowedOrigins()
// ["http://localhost:7833", "http://localhost:7834", ...]
```

### Get parent origin

```typescript
import { getParentOrigin } from '@rbee/shared-config'

const currentPort = parseInt(window.location.port)
const parentOrigin = getParentOrigin(currentPort)
```

## Generate Rust Constants

```bash
pnpm generate:rust
```

This creates `frontend/shared-constants.rs` for use in build.rs scripts.

## Adding a New Service

1. Update `src/ports.ts` - Add service to PORTS
2. Run `pnpm generate:rust`
3. Update `PORT_CONFIGURATION.md`
4. Update backend Cargo.toml with default port
EOF
```

---

## Step 8: Build and Test

```bash
# Install dependencies
pnpm install

# Build the package
pnpm build

# Generate Rust constants
pnpm generate:rust
```

---

## Verification Checklist

- [ ] `dist/` folder created with JS and .d.ts files
- [ ] `frontend/shared-constants.rs` created
- [ ] No TypeScript errors
- [ ] Rust file contains all port constants (KEEPER, QUEEN, HIVE, WORKER)
- [ ] All helper functions exported (getAllowedOrigins, getIframeUrl, etc.)

---

## Expected Output

### After `pnpm build`
```
dist/
├── index.js
├── index.d.ts
├── ports.js
└── ports.d.ts
```

### After `pnpm generate:rust`
```
frontend/shared-constants.rs
```

**Contents should include:**
```rust
pub const QUEEN_DEV_PORT: u16 = 7834;
pub const QUEEN_PROD_PORT: u16 = 7833;
// ... etc
```

---

## Troubleshooting

### Issue: TypeScript not found

```bash
cd frontend/packages/shared-config
pnpm install
```

### Issue: Rust file not generated

```bash
# Make script executable
chmod +x scripts/generate-rust.js

# Run manually
node scripts/generate-rust.js
```

### Issue: Wrong output path

Check the path calculation in `generate-rust.js`:
```javascript
const outputPath = join(__dirname, '..', '..', '..', 'shared-constants.rs')
```

Should resolve to: `frontend/shared-constants.rs`

---

## Next Step

✅ **Step 1 Complete!**

**Next:** `TEAM_351_STEP_2_NARRATION_CLIENT.md` - Create the narration client package

---

**TEAM-351 Step 1: Single source of truth for ports!** 🎯
