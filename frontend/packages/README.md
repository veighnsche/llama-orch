# Shared Packages

This directory contains shared configuration and component packages used across all rbee UIs.

## Configuration Packages

### @repo/typescript-config

Shared TypeScript configurations for consistent type checking across all apps.

**Configs:**
- `base.json` - Base TypeScript config
- `react-app.json` - React + Vite app config
- `vite.json` - Vite config file config

**Usage:**
```json
{
  "extends": "@repo/typescript-config/react-app.json",
  "compilerOptions": {
    "tsBuildInfoFile": "./node_modules/.tmp/tsconfig.app.tsbuildinfo"
  },
  "include": ["src"]
}
```

### @repo/eslint-config

Shared ESLint configurations for consistent code quality.

**Configs:**
- `react.js` - React + TypeScript + Vite config

**Usage:**
```js
import sharedConfig from '@repo/eslint-config/react.js';

export default sharedConfig;
```

### @repo/vite-config

Shared Vite configuration with React + Tailwind CSS v4 setup.

**Usage:**
```js
import { createViteConfig } from '@repo/vite-config';

export default createViteConfig({
  // Optional overrides
  plugins: [/* additional plugins */],
});
```

### @repo/tailwind-config

Shared Tailwind CSS configuration and design tokens.

**Usage:**
```css
@import "tailwindcss";
@import "@repo/tailwind-config";
```

## Component Packages

### @rbee/ui

Shared component library with atoms, molecules, organisms, and templates.

**Usage:**
```tsx
import { Button } from '@rbee/ui/atoms';
import { Card } from '@rbee/ui/molecules';
```

## SDK Packages

Each binary has its own specialized SDK packages:

- **@rbee/queen-rbee-sdk** + **@rbee/queen-rbee-react** - Queen (scheduler) SDK
- **@rbee/rbee-hive-sdk** + **@rbee/rbee-hive-react** - Hive (model/worker manager) SDK
- **@rbee/llm-worker-sdk** + **@rbee/llm-worker-react** - LLM worker SDK

## Benefits

1. **Consistency** - All apps use the same configuration
2. **Maintainability** - Update once, applies everywhere
3. **Type Safety** - Shared TypeScript configs ensure consistency
4. **Code Quality** - Shared ESLint rules across all apps
5. **Performance** - Shared Vite config with optimizations
6. **Design System** - Shared Tailwind config and components

## Adding a New App

1. Add to `pnpm-workspace.yaml`
2. Install shared configs:
   ```json
   {
     "devDependencies": {
       "@repo/eslint-config": "workspace:*",
       "@repo/typescript-config": "workspace:*",
       "@repo/vite-config": "workspace:*"
     }
   }
   ```
3. Use shared configs in your config files
4. Run `pnpm install`

## Turborepo Integration

All packages are managed by Turborepo for optimal caching and parallel execution.

See `turbo.json` in the root for task configuration.
