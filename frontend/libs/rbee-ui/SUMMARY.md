# @rbee/ui - Shared Component Library

## What Was Created

A new shared component library at `frontend/libs/rbee-ui` that provides:

1. **Design Tokens** - Shared CSS variables and theme configuration
2. **Global Styles** - Complete Tailwind CSS setup with utilities and animations
3. **Atoms** - 74 atomic UI components (Button, Badge, Input, Select, Dialog, etc.)
4. **Molecules** - 45+ molecular components (FeatureCard, TestimonialCard, etc.)
5. **Organisms** - 30+ complex components (Hero sections, Navigation, etc.)
6. **Patterns** - 4 pattern components (BeeGlyph, HoneycombPattern, etc.)
7. **Utilities** - Helper functions (cn for class merging)
8. **Storybook** - Interactive component documentation

**Total: 150+ components migrated from commercial site**

## Project Structure

```
frontend/libs/rbee-ui/
├── .storybook/
│   ├── main.ts              # Storybook configuration
│   ├── preview.ts           # Global settings
│   └── preview-head.html    # Dark mode script
├── src/
│   ├── tokens/
│   │   ├── styles.css       # Lightweight design tokens (for Nextra)
│   │   ├── globals.css      # Full Tailwind + utilities (for apps)
│   │   └── index.ts         # TypeScript exports for tokens
│   ├── atoms/
│   │   ├── Button.tsx       # Button component
│   │   ├── Button.stories.tsx
│   │   ├── Badge.tsx        # Badge component
│   │   ├── Badge.stories.tsx
│   │   └── index.ts         # Atom exports
│   ├── molecules/
│   │   ├── Card.tsx         # Card component system
│   │   ├── Card.stories.tsx
│   │   └── index.ts         # Molecule exports
│   ├── utils/
│   │   └── index.ts         # cn() utility function
│   └── Introduction.mdx     # Storybook welcome page
├── package.json             # Package configuration
├── tsconfig.json            # TypeScript config
├── vite.config.ts           # Vite + Tailwind config
├── README.md                # Usage guide
├── INTEGRATION.md           # Integration guide
├── MIGRATION_PLAN.md        # Migration strategy
├── MIGRATION_COMPLETE.md    # globals.css migration report
├── DESIGN_SYSTEM.md         # Design system documentation
├── STORYBOOK.md             # Storybook documentation
├── SUMMARY.md               # This file
└── .gitignore
```

## What's Integrated

### User Docs (Nextra) ✅
- Added `@rbee/ui` dependency
- Imports `@rbee/ui/styles` in root layout
- Created example page at `/docs/components` demonstrating all components
- Maintains Nextra theme while using shared design tokens

### Commercial Site ✅
- Added `@rbee/ui` dependency
- **Migrated globals.css** - Now imports from `@rbee/ui/globals`
- All design tokens, utilities, and animations now sourced from shared package
- Ready to gradually migrate existing components

## Design System Highlights

### Primary Color
- **Honeycomb Yellow/Gold**: #f59e0b (hue: 45)
- Reflects bee theme and Dutch heritage
- Used for CTAs, links, and accents

### CSS Variables
All tokens use `--rbee-*` prefix:
- `--rbee-primary`, `--rbee-background`, `--rbee-foreground`
- `--rbee-card`, `--rbee-border`, `--rbee-muted`
- `--rbee-radius`, `--rbee-radius-sm`, `--rbee-radius-lg`

### Dark Mode
Automatic switching via `.dark` class on root element. All CSS variables have light and dark mode values.

## Components Available

### Atoms

#### Button
```tsx
import { Button } from '@rbee/ui/atoms';

<Button variant="default">Primary</Button>
<Button variant="secondary">Secondary</Button>
<Button variant="outline">Outline</Button>
<Button variant="ghost">Ghost</Button>
<Button variant="destructive">Delete</Button>

<Button size="sm">Small</Button>
<Button size="md">Medium</Button>
<Button size="lg">Large</Button>
```

#### Badge
```tsx
import { Badge } from '@rbee/ui/atoms';

<Badge>Default</Badge>
<Badge variant="secondary">Beta</Badge>
<Badge variant="outline">New</Badge>
<Badge variant="destructive">Deprecated</Badge>
```

### Molecules

#### Card System
```tsx
import { 
  Card, 
  CardHeader, 
  CardTitle, 
  CardDescription, 
  CardContent, 
  CardFooter 
} from '@rbee/ui/molecules';

<Card>
  <CardHeader>
    <CardTitle>Title</CardTitle>
    <CardDescription>Description</CardDescription>
  </CardHeader>
  <CardContent>
    <p>Content here</p>
  </CardContent>
  <CardFooter>
    <Button>Action</Button>
  </CardFooter>
</Card>
```

### Utilities

#### cn() - Class Name Merger
```tsx
import { cn } from '@rbee/ui/utils';

<div className={cn(
  'base-class',
  condition && 'conditional-class',
  className // Allow overrides
)} />
```

## How to Use

### 1. Add Dependency (Already Done)
```json
{
  "dependencies": {
    "@rbee/ui": "workspace:*"
  }
}
```

### 2. Import Styles in Root Layout
```tsx
// app/layout.tsx
import "@rbee/ui/styles";
```

### 3. Use Components
```tsx
import { Button } from '@rbee/ui/atoms';
import { Card, CardHeader } from '@rbee/ui/molecules';
```

## Benefits

### Immediate
✅ Consistent design tokens across all applications  
✅ Single source of truth for colors and spacing  
✅ Dark mode works consistently everywhere  
✅ Shared components reduce duplication

### Long-term
- Faster feature development
- Easier to maintain design consistency
- Better collaboration between apps
- Scalable design system

## Next Steps

### Short Term
1. **Test the components** in both apps
2. **Add more atoms** as needed (Input, Label, etc.)
3. **Document patterns** as they emerge

### Medium Term
1. **Migrate existing components** gradually from commercial site
2. **Add documentation-specific components** (Callout, CodeBlock)
3. **Expand molecule library** (Alert, Dialog, Dropdown)

### Long Term
1. **Build organism patterns** for shared layouts
2. **Create component playground** for testing
3. **Add visual regression testing**

## Important Notes

### Not a Public Library
This is internal tooling for rbee applications only. Not designed for external consumption.

### Not Replacing Everything
- Keep app-specific components in their respective apps
- Only share truly reusable, pure presentational components
- Business logic stays in apps

### Gradual Migration
Don't rush to migrate everything. Move components to @rbee/ui when:
- They're needed by both apps
- They're pure presentational components
- They have no app-specific logic

## Files You Should Review

1. **README.md** - Basic usage guide
2. **INTEGRATION.md** - Step-by-step integration
3. **MIGRATION_PLAN.md** - Migration strategy and priorities
4. **DESIGN_SYSTEM.md** - Complete design system documentation

## Demo

Visit `http://localhost:3100/docs/components` to see all components in action in the user-docs app.

## Storybook

Interactive component documentation is available via Storybook:

```bash
cd frontend/libs/rbee-ui
pnpm storybook
```

Visit `http://localhost:6006` to:
- Preview all components
- Test interactive controls
- Toggle dark mode
- View auto-generated documentation
- Copy code examples

See `STORYBOOK.md` for detailed Storybook documentation.

## Questions?

Refer to the documentation files in this package, or check how components are used in:
- `/frontend/bin/user-docs/app/docs/components/page.mdx` (example usage)
- `/frontend/bin/commercial` (existing component patterns)
- Storybook at `http://localhost:6006` (interactive preview)
