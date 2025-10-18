# FeatureInfoCard Refactoring

## Problem

We had duplicate card implementations across the codebase:

1. **Inline card in `HomeSolutionSection`** - A simple card with icon, title, and body
2. **`ProblemCard` in `ProblemSection`** - A more complex card with icon, title, body, tag, and tone variants

Both cards had the same structure and purpose but were implemented separately, violating DRY principles and making maintenance difficult.

## Solution

Created a new reusable molecule: **`FeatureInfoCard`**

### Key Features

- ✅ **Uses Card atom** - Built on top of the base `Card` component
- ✅ **CVA-powered variants** - Uses `class-variance-authority` for type-safe styling
- ✅ **Flexible icon handling** - Supports both component types and ReactNode
- ✅ **Multiple tone variants** - `default`, `primary`, `destructive`, `muted`
- ✅ **Optional tag/badge** - For displaying loss amounts or other metadata
- ✅ **Animation support** - Accepts delay classes for staggered animations
- ✅ **Fully typed** - Complete TypeScript support with proper types
- ✅ **Composable variants** - Export individual variant functions for advanced use cases

### Tone Variants

```tsx
// Default - neutral card for general features
<FeatureInfoCard tone="default" />

// Primary - positive features/benefits
<FeatureInfoCard tone="primary" />

// Destructive - problems/risks
<FeatureInfoCard tone="destructive" tag="Loss €50/mo" />

// Muted - secondary information
<FeatureInfoCard tone="muted" />
```

## Changes Made

### 1. Created New Molecule

**Location:** `frontend/packages/rbee-ui/src/molecules/FeatureInfoCard/`

Files:
- `FeatureInfoCard.tsx` - Main component with full documentation
- `FeatureInfoCard.stories.tsx` - Storybook stories with examples
- `index.ts` - Barrel export

### 2. Updated `HomeSolutionSection`

**Before:**
```tsx
<div className="group rounded-lg border border-border bg-card p-6 ...">
  <IconPlate icon={benefit.icon} size="lg" tone="primary" className="mb-4" />
  <h3 className="mb-2 text-lg font-semibold text-card-foreground">{benefit.title}</h3>
  <p className="text-balance text-sm leading-relaxed text-muted-foreground">{benefit.body}</p>
</div>
```

**After:**
```tsx
<FeatureInfoCard
  icon={benefit.icon}
  title={benefit.title}
  body={benefit.body}
  tone="primary"
/>
```

### 3. Updated `ProblemSection`

**Before:**
- Had inline `ProblemCard` function component
- Duplicated tone mapping logic
- Duplicated icon handling logic

**After:**
```tsx
<FeatureInfoCard
  icon={item.icon}
  title={item.title}
  body={item.body}
  tag={item.tag}
  tone={item.tone || 'destructive'}
  delay={['delay-75', 'delay-150', 'delay-200'][idx]}
  className="min-h-[220px]"
/>
```

### 4. Removed Duplicate Code

- ❌ Removed inline card div in `HomeSolutionSection`
- ❌ Removed `ProblemCard` function component
- ❌ Removed duplicate `toneMap` object
- ❌ Removed duplicate icon handling logic

## Benefits

1. **Single Source of Truth** - One component for all feature/problem/benefit cards
2. **CVA-Powered Variants** - Type-safe styling with `class-variance-authority`
3. **Easier Maintenance** - Changes only need to be made in one place
4. **Consistent Design** - All cards look and behave the same way
5. **Better Reusability** - Can be used anywhere in the app
6. **Proper Atomic Design** - Follows the atoms → molecules → organisms pattern
7. **Type Safety** - Full TypeScript support with proper prop types
8. **Composable** - Export individual variant functions for advanced customization

## Usage Examples

### Benefits Grid
```tsx
<FeatureInfoCard
  icon={DollarSign}
  title="Zero ongoing costs"
  body="Pay only for electricity. No API bills, no per-token surprises."
  tone="primary"
/>
```

### Problems Grid
```tsx
<FeatureInfoCard
  icon={Lock}
  title="The provider shuts down"
  body="APIs get deprecated. Your AI-built code becomes unmaintainable overnight."
  tone="destructive"
  tag="Loss €2,400/mo"
/>
```

### Features Grid
```tsx
<FeatureInfoCard
  icon={Shield}
  title="Security first"
  body="Built with security best practices from the ground up."
  tone="muted"
/>
```

## Migration Guide

If you have other components using similar card patterns:

1. Import `FeatureInfoCard` from `@rbee/ui/molecules`
2. Replace inline card divs with `<FeatureInfoCard />`
3. Map your props to the component's API:
   - `icon` → icon component or ReactNode
   - `title` → card heading
   - `body` → card description
   - `tone` → visual variant (default, primary, destructive, muted)
   - `tag` → optional badge text
   - `delay` → animation delay class

## Files Changed

- ✅ Created: `frontend/packages/rbee-ui/src/molecules/FeatureInfoCard/FeatureInfoCard.tsx`
- ✅ Created: `frontend/packages/rbee-ui/src/molecules/FeatureInfoCard/FeatureInfoCard.stories.tsx`
- ✅ Created: `frontend/packages/rbee-ui/src/molecules/FeatureInfoCard/index.ts`
- ✅ Updated: `frontend/packages/rbee-ui/src/molecules/index.ts`
- ✅ Updated: `frontend/packages/rbee-ui/src/organisms/Home/SolutionSection/HomeSolutionSection.tsx`
- ✅ Updated: `frontend/packages/rbee-ui/src/organisms/ProblemSection/ProblemSection.tsx`

## Verification

Run Storybook to see the new component:
```bash
cd frontend/packages/rbee-ui
pnpm storybook
```

Navigate to: **Molecules → FeatureInfoCard**

## Next Steps

Consider using `FeatureInfoCard` in other places where similar card patterns exist:
- Feature sections
- Use case cards
- Benefit displays
- Problem statements
- Solution highlights
