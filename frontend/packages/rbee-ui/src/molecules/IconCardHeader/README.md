# IconCardHeader

Reusable card header molecule combining an icon, title, and optional subtitle.

## Usage

```tsx
import { IconCardHeader } from '@rbee/ui/molecules'
import { Lock } from 'lucide-react'

<IconCardHeader
  icon={<Lock className="h-6 w-6" />}
  title="ISO 27001"
  subtitle="International Standard"
  titleId="compliance-iso27001"
/>
```

## Props

- **icon**: ReactNode - Icon element (required)
- **title**: string - Card title (required)
- **subtitle**: string (optional) - Subtitle/description
- **titleId**: string (optional) - ID for the title element (for aria-labelledby)
- **iconSize**: 'sm' | 'md' | 'lg' (default: 'lg') - IconPlate size
- **iconTone**: 'primary' | 'secondary' | 'accent' | 'muted' (default: 'primary') - IconPlate tone
- **titleClassName**: string (default: 'text-2xl') - Title size class
- **className**: string (optional) - Additional CSS classes for the header wrapper

## Composition

- **CardHeader** (from atoms) - Wrapper with `mb-6 p-0`
- **IconPlate** (from molecules) - Icon container
- **CardTitle** (from atoms) - Title text
- **CardDescription** (from atoms) - Subtitle text (if provided)

## When to Use

- Card headers that need an icon alongside title/subtitle
- Compliance cards, feature cards, status cards
- Any card that benefits from visual icon identification
