# HowItWorksSection Migration

## Changes Made

Moved `HowItWorksSection` from `Home/HowItWorksSection` to standalone `HowItWorksSection` organism and removed default steps.

### Before

**Location:** `/organisms/Home/HowItWorksSection/`  
**Props:** Had default title and default steps (4 hardcoded steps)

```tsx
export function HowItWorksSection({
  title = 'From zero to AI infrastructure in 15 minutes',
  subtitle,
  steps = DEFAULT_STEPS, // ❌ Hardcoded defaults
  id,
  className,
}: HowItWorksSectionProps)
```

### After

**Location:** `/organisms/HowItWorksSection/`  
**Props:** Required title and steps (no defaults)

```tsx
export type HowItWorksSectionProps = {
  title: string        // ✅ Required
  subtitle?: string
  steps: Array<{...}>  // ✅ Required
  id?: string
  className?: string
}
```

## Breaking Changes

### 1. Title is now required
**Before:** `title?: string` with default  
**After:** `title: string` (required)

### 2. Steps are now required
**Before:** `steps?: Array<...>` with DEFAULT_STEPS  
**After:** `steps: Array<...>` (required)

### 3. Import path changed
**Before:** `import { HowItWorksSection } from '@rbee/ui/organisms/Home/HowItWorksSection'`  
**After:** `import { HowItWorksSection } from '@rbee/ui/organisms'`

## Migration Guide

### Home Page (`/app/page.tsx`)

**Before:**
```tsx
<HowItWorksSection />
```

**After:**
```tsx
<HowItWorksSection
  title="From zero to AI infrastructure in 15 minutes"
  steps={[
    {
      label: 'Install rbee',
      block: {
        kind: 'terminal',
        title: 'terminal',
        lines: (
          <>
            <div>curl -sSL https://rbee.dev/install.sh | sh</div>
            <div className="text-[var(--syntax-comment)]">rbee-keeper daemon start</div>
          </>
        ),
        copyText: 'curl -sSL https://rbee.dev/install.sh | sh\nrbee-keeper daemon start',
      },
    },
    // ... 3 more steps
  ]}
/>
```

### Developers Page (`/app/developers/page.tsx`)

Already using explicit steps - no changes needed! ✅

## Benefits

✅ **Reusable** - No longer coupled to Home folder  
✅ **Flexible** - Each page can customize steps  
✅ **Explicit** - No hidden defaults  
✅ **Consistent** - Same pattern as other organisms  
✅ **Type-safe** - Required props prevent missing data  

## Files Changed

- ✅ Created `/organisms/HowItWorksSection/HowItWorksSection.tsx`
- ✅ Created `/organisms/HowItWorksSection/index.ts`
- ✅ Updated `/organisms/index.ts` - Removed Home export, added standalone export
- ✅ Updated `/apps/commercial/app/page.tsx` - Added explicit steps
- ✅ `/apps/commercial/app/developers/page.tsx` - Already correct ✅

## Old Location

The old `/organisms/Home/HowItWorksSection/` can be deleted once all references are confirmed working.

## Component Features

- Uses `SectionContainer` for consistent styling
- Supports 3 block types: `terminal`, `code`, `note`
- Numbered steps with animated badges
- Copy-to-clipboard for code blocks
- Responsive design
- Staggered animations

## Usage Example

```tsx
<HowItWorksSection
  title="Get started in minutes"
  subtitle="Simple setup process"
  steps={[
    {
      label: 'Step 1',
      number: 1, // Optional, defaults to index + 1
      block: {
        kind: 'terminal',
        title: 'terminal',
        lines: <div>command here</div>,
        copyText: 'command here',
      },
    },
    {
      label: 'Step 2',
      block: {
        kind: 'code',
        title: 'TypeScript',
        language: 'ts',
        lines: <div>code here</div>,
        copyText: 'code here',
      },
    },
    {
      label: 'Note',
      block: {
        kind: 'note',
        content: <p>Important information</p>,
      },
    },
  ]}
/>
```

## Result

`HowItWorksSection` is now a standalone, reusable organism that can be used across all pages with custom steps.
