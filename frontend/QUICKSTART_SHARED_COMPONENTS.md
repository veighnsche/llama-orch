# Quickstart: Using Shared Components

**Created by:** TEAM-FE-DX-006

## Installation

```bash
# From project root
pnpm install
```

This automatically links `@orchyra/shared-components` to both `commercial` and `user-docs`.

## Usage in Commercial Frontend

```vue
<script setup lang="ts">
import { Button, Card, CardHeader, CardTitle, CardContent } from '@orchyra/shared-components'
</script>

<template>
  <Card>
    <CardHeader>
      <CardTitle>Welcome to Orchyra</CardTitle>
    </CardHeader>
    <CardContent>
      <Button>Get Started</Button>
    </CardContent>
  </Card>
</template>
```

## Usage in User Docs Frontend

```vue
<script setup lang="ts">
import { Alert, AlertTitle, AlertDescription, Badge } from '@orchyra/shared-components'
</script>

<template>
  <Alert>
    <AlertTitle>Documentation</AlertTitle>
    <AlertDescription>Learn how to use Orchyra.</AlertDescription>
  </Alert>
  <Badge>v1.0</Badge>
</template>
```

## Available Components

### Core Atoms
- `Button`, `Input`, `Textarea`, `Label`, `Checkbox`, `Switch`, `Slider`
- `Badge`, `Avatar`, `Separator`, `Spinner`, `Skeleton`, `Progress`
- `Card`, `CardHeader`, `CardTitle`, `CardDescription`, `CardContent`, `CardFooter`
- `Alert`, `AlertTitle`, `AlertDescription`

### Advanced Atoms
- `Tabs`, `TabsList`, `TabsTrigger`, `TabsContent`
- `Accordion`, `AccordionItem`, `AccordionTrigger`, `AccordionContent`
- `Dialog`, `Sheet`, `Popover`, `Tooltip`, `HoverCard`
- `DropdownMenu`, `Select`, `Command`, `NavigationMenu`

### Molecules
- `FormField`, `SearchBar`, `PasswordInput`
- `StatCard`, `FeatureCard`, `TestimonialCard`, `PricingCard`
- `ConfirmDialog`, `DropdownAction`

### Utilities
```typescript
import { cn } from '@orchyra/shared-components'

// Merge Tailwind classes
const classes = cn('px-4 py-2', 'bg-blue-500', props.className)
```

## Design Tokens

Shared design tokens are automatically imported from `@orchyra/shared-components/styles/tokens.css`.

## Development

```bash
# Run commercial frontend
cd frontend/bin/commercial
pnpm dev

# Run user-docs frontend
cd frontend/bin/user-docs
pnpm dev
```

## Notes

- **Organisms and templates** stay in each bin (not shared)
- **Atoms and molecules** are shared via `@orchyra/shared-components`
- Changes to shared components apply to both frontends
- Full TypeScript support with auto-completion
