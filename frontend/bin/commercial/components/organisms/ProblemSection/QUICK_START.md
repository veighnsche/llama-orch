# ProblemSection Quick Start Guide

## 🚀 Using the Component

### Basic Usage (Home Page)
```tsx
import { ProblemSection } from '@/components/organisms/ProblemSection/ProblemSection'
import { AlertTriangle, DollarSign, Lock } from 'lucide-react'

<ProblemSection
  kicker="Why this matters"
  title="The Problem We're Solving"
  subtitle="A brief description of the problem space."
  items={[
    {
      icon: AlertTriangle,
      title: 'Problem A',
      body: 'Description of the first problem.',
      tone: 'destructive',
      tag: 'High impact',
    },
    {
      icon: DollarSign,
      title: 'Problem B',
      body: 'Description of the second problem.',
      tone: 'primary',
      tag: 'Cost: $500/mo',
    },
    {
      icon: Lock,
      title: 'Problem C',
      body: 'Description of the third problem.',
      tone: 'destructive',
    },
  ]}
  ctaPrimary={{ label: 'Get Started', href: '/signup' }}
  ctaSecondary={{ label: 'Learn More', href: '#details' }}
  ctaCopy="Every day you wait is another day of lost opportunity."
/>
```

### Creating a Vertical Wrapper
```tsx
// components/organisms/MyVertical/my-vertical-problem.tsx
import { TrendingDown, Zap, AlertCircle } from 'lucide-react'
import { ProblemSection } from '@/components/organisms/ProblemSection/ProblemSection'

export function MyVerticalProblem() {
  return (
    <ProblemSection
      kicker="Your Custom Kicker"
      title="Your Custom Title"
      subtitle="Your custom subtitle explaining the problem."
      items={[
        {
          icon: TrendingDown,
          title: 'Specific Problem 1',
          body: 'Detailed description for this vertical.',
          tag: 'Loss: €100/mo',
          tone: 'destructive',
        },
        // ... more items
      ]}
      ctaPrimary={{ label: 'Your CTA', href: '/your-path' }}
      ctaSecondary={{ label: 'Secondary Action', href: '#anchor' }}
      ctaCopy="Your persuasive CTA banner copy."
    />
  )
}
```

## 🎨 Tone System

### Visual Themes
Choose the tone that matches the severity/nature of each problem:

```tsx
// Destructive (red) - Urgent problems, losses, risks
{ tone: 'destructive' }  
// → Red borders, red icons, red tags

// Primary (brand) - Important but not urgent, opportunities
{ tone: 'primary' }
// → Brand color borders/icons/tags

// Muted (gray) - Less critical, informational
{ tone: 'muted' }
// → Subtle gray styling
```

## 🏷️ Loss Tags

Add optional monetary/impact tags to cards:

```tsx
{
  icon: TrendingDown,
  title: 'Wasted Resources',
  body: 'Your hardware sits idle 90% of the time.',
  tag: 'Potential earnings €50-200/mo',  // ← Displays as badge
  tone: 'destructive',
}
```

**Best Practices:**
- Use `tabular-nums` automatically applied
- Include currency symbols (€, $)
- Keep short: "Loss €100/mo" or "Cost: 10x"
- Optional - omit if not relevant

## 🎯 CTA Banner

### Both Buttons
```tsx
ctaPrimary={{ label: 'Start Earning', href: '/signup' }}
ctaSecondary={{ label: 'Learn More', href: '#details' }}
ctaCopy="Turn this problem into opportunity."
```

### Primary Only
```tsx
ctaPrimary={{ label: 'Get Started', href: '/signup' }}
ctaCopy="Ready to solve this problem?"
```

### No CTA Banner
```tsx
// Omit all CTA props - banner won't render
```

## 📐 Grid Layouts

### Default (3 columns)
```tsx
<ProblemSection items={[...]} />
// → md:grid-cols-3 (default)
```

### Custom Grid (e.g., 4 columns for Enterprise)
```tsx
<ProblemSection
  items={[...]}  // 4 items
  gridClassName="md:grid-cols-2 lg:grid-cols-4"
/>
```

### Two Columns
```tsx
<ProblemSection
  items={[...]}  // 2 items
  gridClassName="md:grid-cols-2"
/>
```

## 🎭 Icons

### As Component (preferred)
```tsx
import { AlertTriangle } from 'lucide-react'

{
  icon: AlertTriangle,  // ← Component reference
  title: 'Problem',
  // ...
}
```

### As JSX (also supported)
```tsx
{
  icon: <AlertTriangle className="h-6 w-6 text-destructive" />,
  title: 'Problem',
  // ...
}
```

**Note:** Component reference is preferred - the component handles sizing and color based on tone.

## 🔧 Props Reference

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `kicker` | `string?` | - | Small label above title |
| `title` | `string?` | Developer default | Main headline |
| `subtitle` | `string?` | Developer default | Subtitle/description |
| `items` | `ProblemItem[]?` | Developer defaults | Array of problem cards |
| `ctaPrimary` | `{label, href}?` | - | Primary CTA button |
| `ctaSecondary` | `{label, href}?` | - | Secondary CTA button |
| `ctaCopy` | `string?` | - | CTA banner copy |
| `id` | `string?` | - | Section anchor ID |
| `className` | `string?` | - | Section wrapper classes |
| `gridClassName` | `string?` | `md:grid-cols-3` | Grid layout override |
| `eyebrow` | `string?` | - | Legacy alias for `kicker` |

## 📋 ProblemItem Reference

| Prop | Type | Required | Description |
|------|------|----------|-------------|
| `title` | `string` | ✅ | Card headline |
| `body` | `string` | ✅ | Card description |
| `icon` | `Component \| ReactNode` | ✅ | Lucide icon or JSX |
| `tag` | `string?` | - | Loss/impact badge |
| `tone` | `'primary' \| 'destructive' \| 'muted'?` | - | Visual theme (default: `destructive`) |

## ✅ Quick Checklist

Before shipping a new problem section:

- [ ] Kicker text is concise (2-5 words)
- [ ] Title is benefit/problem-oriented
- [ ] Subtitle explains context (1-2 sentences)
- [ ] 3-4 cards (max 4 for readability)
- [ ] Each card has icon, title, body
- [ ] Tags include currency/numbers where relevant
- [ ] CTA buttons have clear, action-oriented labels
- [ ] CTA copy is persuasive (1-2 sentences)
- [ ] Grid layout appropriate for card count
- [ ] Tone matches problem severity

## 🎯 Examples

See live implementations:
- **GPU Providers**: `/components/organisms/Providers/providers-problem.tsx`
- **Developers**: `/components/organisms/Developers/developers-problem.tsx`
- **Enterprise**: `/components/organisms/Enterprise/enterprise-problem.tsx`
- **Home**: `/app/page.tsx`

## 📚 Full Documentation

For complete API reference and migration guide, see:
- `PROBLEM_SECTION_UNIFICATION.md` - Overall architecture
- `MIGRATION_GUIDE.md` - Detailed migration examples
