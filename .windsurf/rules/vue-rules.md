---
trigger: glob
globs: src/**/*.vue
---

# Vue Rules

1. Design tokens
   - Always import design tokens from the Storybook workspace package.
   - Allowed:
     - In app entry styles (e.g., [src/assets/main.css](cci:7://file:///home/vince/Projects/llama-orch/frontend/bin/commercial-frontend/src/assets/main.css:0:0-0:0)):
       `@import 'orchyra-storybook/styles/tokens.css';`
   - Disallowed:
     - Absolute/relative paths into other workspaces:
       - `/home/.../frontend/storybook/styles/tokens.css`
       - `../../../../frontend/libs/storybook/styles/tokens.css`

2. Reusable components
   - Import shared UI components ONLY via the published workspace package.
   - Allowed:
     - `import { Button, Badge } from 'orchyra-storybook/stories'`
     - `import { NavbarShell, Brand, NavLinks, Drawer, DrawerTrigger, DrawerPanel } from 'orchyra-storybook/stories'`
   - Disallowed:
     - `import { Button } from '../../../../../libs/storybook/stories'`
     - `import { Button } from '../../../frontend/libs/storybook/stories'`
     - `import { Button } from 'consumers/storybook/stories'`

3. File naming
   - Vue component filenames must follow `CamelCase.vue`.

4. Workspace boundaries
   - This is a pnpm workspace. Do NOT cross-import code from sibling libraries/packages using relative paths.
   - Always import dependencies through their declared package entry points (e.g., `orchyra-storybook`), never via `../../` into another package’s `src/` or `stories/`.
   - If a shared building block is missing:
     - Add it under `frontend/libs/storybook/stories/`.
     - Export it from [frontend/libs/storybook/stories/index.ts](cci:7://file:///home/vince/Projects/llama-orch/frontend/libs/storybook/stories/index.ts:0:0-0:0).
     - Consumers must import it exclusively via `orchyra-storybook/stories`.

5. Offending pattern (for quick grep and migration)
   - Bad (found and fixed):
     - `../../../../../libs/storybook/stories`
     - `../../../../libs/storybook/stories`
     - `../../../frontend/libs/storybook/stories`
   - Good (use this instead):
     - `orchyra-storybook/stories`

6. ESLint enforcement (recommended)
   - Add a `no-restricted-imports` rule to reject cross-workspace relative imports.
   - If you own the shared ESLint config (`orchyra-frontend-tooling/eslint.config.js`), add an override there; otherwise add this to your local config.

   Example override:
   ```js
   // eslint.config.js (override or local)
   export default [
     // ...existing config...
     {
       files: ['src/**/*.{vue,ts,tsx,js,jsx}'],
       rules: {
         'no-restricted-imports': ['error', {
           patterns: [
             '**/libs/storybook/**',
             '**/frontend/libs/storybook/**',
             '**/consumers/storybook/**',
             // optionally block any deep relative that goes up 3+ levels
             '../../**/libs/**',
             '../../../**/libs/**',
             '../../../../**/libs/**',
           ],
         }],
       },
     },
   ]
