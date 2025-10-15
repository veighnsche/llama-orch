# FeatureInfoCard CVA Usage

This component uses [class-variance-authority (CVA)](https://cva.style/docs) for type-safe variant management.

## Exported Variants

The component exports four CVA variant functions:

```tsx
import {
  FeatureInfoCard,
  featureInfoCardVariants,
  iconContainerVariants,
  iconVariants,
  tagVariants,
} from '@rbee/ui/molecules'
```

## Variant Functions

### `featureInfoCardVariants`
Controls the card container styling (border, background, hover states).

```tsx
featureInfoCardVariants({ tone: 'primary' })
// Returns: "group transition-all animate-in fade-in slide-in-from-bottom-2 motion-reduce:animate-none border-primary/40 bg-gradient-to-b from-primary/15 to-background backdrop-blur supports-[backdrop-filter]:bg-background/60 hover:border-primary/50"
```

### `iconContainerVariants`
Controls the icon container background.

```tsx
iconContainerVariants({ tone: 'destructive' })
// Returns: "mb-4 flex h-11 w-11 items-center justify-center rounded-xl bg-destructive/10"
```

### `iconVariants`
Controls the icon color.

```tsx
iconVariants({ tone: 'primary' })
// Returns: "h-6 w-6 text-primary"
```

### `tagVariants`
Controls the optional tag/badge styling.

```tsx
tagVariants({ tone: 'destructive' })
// Returns: "mt-3 inline-flex rounded-full px-2.5 py-1 text-xs tabular-nums bg-destructive/10 text-destructive"
```

## Tone Variants

All variant functions support four tones:

| Tone | Use Case | Example |
|------|----------|---------|
| `default` | Neutral features | General information cards |
| `primary` | Positive features | Benefits, solutions, advantages |
| `destructive` | Problems/risks | Issues, warnings, losses |
| `muted` | Secondary info | Less important features |

## Advanced Usage

### Custom Styling with Variants

You can use the exported variants to create custom components:

```tsx
import { featureInfoCardVariants, iconVariants } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'

function CustomCard() {
  return (
    <div className={cn(
      featureInfoCardVariants({ tone: 'primary' }),
      'custom-additional-classes'
    )}>
      <MyIcon className={iconVariants({ tone: 'primary' })} />
      {/* ... */}
    </div>
  )
}
```

### Extending Variants

If you need additional tones, you can extend the variants:

```tsx
import { featureInfoCardVariants } from '@rbee/ui/molecules'
import { cva } from 'class-variance-authority'

const extendedVariants = cva(featureInfoCardVariants.base, {
  variants: {
    tone: {
      ...featureInfoCardVariants.variants.tone,
      success: 'border-green-500/40 bg-gradient-to-b from-green-500/15 to-background',
      warning: 'border-yellow-500/40 bg-gradient-to-b from-yellow-500/15 to-background',
    },
  },
})
```

## Type Safety

CVA provides full TypeScript support:

```tsx
import type { VariantProps } from 'class-variance-authority'
import type { featureInfoCardVariants } from '@rbee/ui/molecules'

// Extract the tone type
type Tone = VariantProps<typeof featureInfoCardVariants>['tone']
// Type: 'default' | 'primary' | 'destructive' | 'muted' | null | undefined

// Use in your own components
interface MyProps {
  cardTone: Tone
}
```

## Benefits of CVA

1. **Type Safety** - TypeScript autocomplete and validation for variants
2. **Composability** - Easily combine and extend variants
3. **Maintainability** - Centralized variant definitions
4. **Performance** - Optimized class name generation
5. **DX** - Better developer experience with IntelliSense

## Migration from Object-based Variants

**Before (object-based):**
```tsx
const toneMap = {
  primary: {
    border: 'border-primary/40',
    bg: 'bg-gradient-to-b from-primary/15 to-background',
    // ...
  },
}

const styles = toneMap[tone]
className={cn(styles.border, styles.bg)}
```

**After (CVA):**
```tsx
const variants = cva('base-classes', {
  variants: {
    tone: {
      primary: 'border-primary/40 bg-gradient-to-b from-primary/15 to-background',
    },
  },
})

className={variants({ tone })}
```

## Resources

- [CVA Documentation](https://cva.style/docs)
- [CVA GitHub](https://github.com/joe-bell/cva)
- [Tailwind CSS Variants](https://tailwindcss.com/docs/hover-focus-and-other-states)
