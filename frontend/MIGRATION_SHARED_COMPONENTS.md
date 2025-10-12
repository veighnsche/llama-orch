# Migration Guide: Shared Components Library

**Created by:** TEAM-FE-DX-006  
**Date:** 2025-10-12

## Overview

Created `@orchyra/shared-components` library in `frontend/libs/shared-components` to share atoms and molecules between `commercial` and `user-docs` frontends.

## Structure

```
frontend/
├── bin/
│   ├── commercial/          # Commercial frontend (Nuxt)
│   └── user-docs/           # User docs frontend (Nuxt)
└── libs/
    ├── frontend-tooling/    # Shared tooling (ESLint, Prettier, tsconfig)
    └── shared-components/   # ✨ NEW: Shared UI components
        ├── atoms/           # Atomic components (Button, Input, Card, etc.)
        ├── molecules/       # Molecular components (FormField, SearchBar, etc.)
        ├── styles/          # Design tokens and CSS
        ├── lib/             # Utilities (cn helper)
        └── index.ts         # Barrel export
```

## What's Included

### Atoms (67 components)
- **Core UI:** Button, Input, Badge, Card, Alert, Avatar, Checkbox, etc.
- **Advanced UI:** Tabs, Dialog, Dropdown, Select, Accordion, etc.
- **Specialized:** Form, Sidebar, Calendar, Chart, etc.

### Molecules (13 components)
- FormField, SearchBar, PasswordInput
- NavItem, BreadcrumbItem
- StatCard, FeatureCard, TestimonialCard, PricingCard
- ConfirmDialog, DropdownAction, TabPanel

### Styles
- `tokens.css` - Design tokens
- `tokens-base.css` - Base token definitions
- `fonts.css` - Font imports

### Utils
- `cn()` - Tailwind class merger utility

## Usage

### In Commercial Frontend

```vue
<script setup lang="ts">
import { Button, Card, CardHeader, CardTitle, CardContent } from '@orchyra/shared-components'
</script>

<template>
  <Card>
    <CardHeader>
      <CardTitle>Welcome</CardTitle>
    </CardHeader>
    <CardContent>
      <Button>Get Started</Button>
    </CardContent>
  </Card>
</template>
```

### In User Docs Frontend

```vue
<script setup lang="ts">
import { Alert, AlertTitle, AlertDescription, Button } from '@orchyra/shared-components'
</script>

<template>
  <Alert>
    <AlertTitle>Note</AlertTitle>
    <AlertDescription>This is important information.</AlertDescription>
  </Alert>
  <Button variant="outline">Learn More</Button>
</template>
```

## Configuration

### pnpm-workspace.yaml
```yaml
packages:
  - frontend/libs/shared-components  # Added
```

### package.json (both bins)
```json
{
  "dependencies": {
    "@orchyra/shared-components": "workspace:*"
  }
}
```

### nuxt.config.ts (both bins)
```typescript
// No alias needed - workspace package resolution works automatically
export default defineNuxtConfig({
  // ... other config
})
```

## Installation

```bash
# From project root
pnpm install

# This will link @orchyra/shared-components to both bins via workspace protocol
# No aliases needed - pnpm handles workspace package resolution automatically
```

## Next Steps

### For Commercial Frontend
1. **Keep organisms local** - Page-specific organisms stay in `commercial/app/stories/organisms/`
2. **Keep templates local** - Page templates stay in `commercial/app/stories/templates/`
3. **Import from shared** - Use `@orchyra/shared-components` for atoms/molecules
4. **Optional cleanup** - Can remove duplicate atoms/molecules from commercial after migration

### For User Docs Frontend
1. **Import atoms/molecules** - Use `@orchyra/shared-components` for all shared components
2. **Create local organisms** - Build doc-specific organisms in `user-docs/app/stories/organisms/`
3. **Import styles** - Import design tokens from shared library

## Benefits

✅ **Single source of truth** - Atoms and molecules defined once  
✅ **Consistent design** - Both frontends use same components  
✅ **Easier maintenance** - Update component once, applies everywhere  
✅ **Type safety** - Full TypeScript support  
✅ **Tree shaking** - Only import what you need  
✅ **Independent deployment** - Each bin can deploy separately

## Design Principles

- **Atoms** = Basic building blocks (Button, Input)
- **Molecules** = Simple combinations of atoms (FormField = Label + Input + Error)
- **Organisms** = Complex, page-specific components (stay in bin projects)
- **Templates** = Page layouts (stay in bin projects)

## Dependencies

The shared library includes:
- Vue 3.5.22
- Tailwind CSS 4.1.14
- Radix Vue 1.9.11 (headless primitives)
- CVA 0.7.1 (class-variance-authority)
- Lucide Vue Next 0.454.0 (icons)

## Notes

- Organisms and templates remain in `commercial` bin (not shared)
- Each bin can have its own organisms/templates
- Design tokens are shared via `styles/tokens.css`
- The `cn()` utility is exported for custom component styling
