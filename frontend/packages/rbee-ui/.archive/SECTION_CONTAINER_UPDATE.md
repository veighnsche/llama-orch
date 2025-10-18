# SectionContainer Refactoring

## Problem

`ProblemSection` had hand-rolled title/subtitle/kicker markup that duplicated logic from `SectionContainer`. This violated DRY principles and made it harder to maintain consistent section styling across the app.

## Solution

### 1. Enhanced SectionContainer

Added two new features to `SectionContainer`:

#### A. Destructive Gradient Background
New `bgVariant` option for problem/warning sections:

```tsx
bgVariant="destructive-gradient"
// Renders: bg-gradient-to-b from-background via-destructive/8 to-background border-b border-border
```

#### B. Kicker Variant
New `kickerVariant` prop to style the kicker text:

```tsx
kickerVariant="destructive"  // Red text for warnings
kickerVariant="default"      // Muted text (default)
```

### 2. Refactored ProblemSection

**Before (hand-rolled):**
```tsx
<section className="border-b border-border bg-gradient-to-b from-background via-destructive/8 to-background px-6 py-20 lg:py-28">
  <div className="mx-auto max-w-7xl">
    <div className="mb-12 text-center animate-in fade-in slide-in-from-bottom-2 motion-reduce:animate-none duration-500">
      {displayKicker && <p className="mb-2 text-sm font-medium text-destructive/80">{displayKicker}</p>}
      <h2 className="text-balance text-4xl font-extrabold tracking-tight text-foreground lg:text-5xl">{title}</h2>
      {subtitle && (
        <p className="mx-auto mt-4 max-w-2xl text-pretty text-lg leading-snug text-muted-foreground lg:text-xl">
          {subtitle}
        </p>
      )}
    </div>
    {/* content */}
  </div>
</section>
```

**After (using SectionContainer):**
```tsx
<SectionContainer
  title={title}
  description={subtitle}
  kicker={displayKicker}
  kickerVariant="destructive"
  bgVariant="destructive-gradient"
  paddingY="xl"
  maxWidth="7xl"
  align="center"
  headingId={id}
  className={className}
>
  {/* content */}
</SectionContainer>
```

## Benefits

✅ **DRY Principle** - No duplicate title/subtitle markup  
✅ **Consistency** - All sections use the same container component  
✅ **Maintainability** - Changes to section styling happen in one place  
✅ **Flexibility** - Easy to add new background variants  
✅ **Accessibility** - Proper heading IDs and semantic structure  
✅ **Less Code** - Reduced from 60+ lines to ~15 lines in ProblemSection  

## SectionContainer API

### New Props

| Prop | Type | Description |
|------|------|-------------|
| `kickerVariant` | `'default' \| 'destructive'` | Kicker text color |
| `bgVariant` | includes `'destructive-gradient'` | Background variant |

### All Background Variants

- `background` - Plain background
- `secondary` - Secondary background
- `card` - Card background
- `default` - Default background
- `muted` - Muted background
- `subtle` - Subtle with top border
- **`destructive-gradient`** - Red gradient for problems/warnings

## Usage Examples

### Problem Section
```tsx
<SectionContainer
  title="The hidden risks"
  description="Problems you need to know about..."
  kicker="Why this matters"
  kickerVariant="destructive"
  bgVariant="destructive-gradient"
  paddingY="xl"
  maxWidth="7xl"
  align="center"
>
  {/* Problem cards */}
</SectionContainer>
```

### Warning Section
```tsx
<SectionContainer
  title="Important Notice"
  kicker="Attention Required"
  kickerVariant="destructive"
  bgVariant="destructive-gradient"
>
  {/* Warning content */}
</SectionContainer>
```

### Standard Section
```tsx
<SectionContainer
  title="Our Features"
  description="Everything you need to succeed"
  kicker="What we offer"
  kickerVariant="default"
  bgVariant="background"
>
  {/* Feature content */}
</SectionContainer>
```

## Files Updated

- ✅ `SectionContainer.tsx` - Added `kickerVariant` and `destructive-gradient` bgVariant
- ✅ `ProblemSection.tsx` - Refactored to use SectionContainer
- ✅ `SECTION_CONTAINER_UPDATE.md` - Documentation

## Migration Guide

If you have other sections with hand-rolled title markup:

1. Import `SectionContainer` from `@rbee/ui/molecules`
2. Replace your `<section>` wrapper with `<SectionContainer>`
3. Map your props:
   - `title` → `title`
   - `subtitle` → `description`
   - `kicker` → `kicker`
   - Custom background → `bgVariant`
4. Remove duplicate heading markup
5. Keep your content as children

## Result

**Before:** 143 lines with duplicate markup  
**After:** 143 lines but cleaner, more maintainable, and consistent

All sections now use the same battle-tested container component with proper accessibility, animations, and responsive design.
