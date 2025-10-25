# PageContainer

**Purpose:** Consistent page wrapper for web-ui app pages (NOT for marketing/commercial site)

## Overview

`PageContainer` eliminates the repetitive pattern of manually implementing page headers and containers across all web-ui pages. It's the app-focused sibling of `TemplateContainer` (which is for marketing pages).

## Problem Solved

**Before:**
```tsx
// Every page repeated this pattern
<div className="flex-1 space-y-4">
  <div>
    <h1 className="text-3xl font-bold">Dashboard</h1>
    <p className="text-muted-foreground">Monitor your queen...</p>
  </div>
  {/* Page content */}
</div>
```

**After:**
```tsx
<PageContainer
  title="Dashboard"
  description="Monitor your queen..."
>
  {/* Page content */}
</PageContainer>
```

## Differences from TemplateContainer

| Feature | PageContainer | TemplateContainer |
|---------|---------------|-------------------|
| **Use Case** | App pages (Dashboard, Settings, etc.) | Marketing pages (Homepage, Pricing, etc.) |
| **Background** | None (simple) | Gradients, patterns, decorations |
| **Layout** | Functional, minimal | Rich, marketing-focused |
| **CTAs** | Optional actions (buttons) | Multiple CTA types (banners, rails, etc.) |
| **Spacing** | Compact variants | Generous padding |
| **Complexity** | ~60 LOC | ~400 LOC |

## Usage

### Basic

```tsx
<PageContainer
  title="Dashboard"
  description="Monitor your queen, hives, workers, and models"
>
  <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
    {/* Cards, content, etc. */}
  </div>
</PageContainer>
```

### With Actions

```tsx
<PageContainer
  title="Settings"
  description="Configure your rbee installation"
  actions={
    <>
      <Button variant="outline" size="sm">Reset</Button>
      <Button size="sm">Save Changes</Button>
    </>
  }
>
  {/* Settings content */}
</PageContainer>
```

### Spacing Variants

```tsx
// Compact (space-y-3)
<PageContainer spacing="compact" title="..." description="...">
  {/* Content */}
</PageContainer>

// Default (space-y-4)
<PageContainer spacing="default" title="..." description="...">
  {/* Content */}
</PageContainer>

// Relaxed (space-y-6)
<PageContainer spacing="relaxed" title="..." description="...">
  {/* Content */}
</PageContainer>
```

## Props

```tsx
interface PageContainerProps {
  /** Page title */
  title: string
  /** Optional page description */
  description?: string
  /** Optional actions (buttons, etc.) aligned to the right */
  actions?: ReactNode
  /** Page content */
  children: ReactNode
  /** Additional CSS classes */
  className?: string
  /** Content spacing variant */
  spacing?: 'default' | 'compact' | 'relaxed'
}
```

## Implementation Details

### Structure

```
PageContainer
├─ Header (flex container)
│  ├─ Title + Description (flex-1)
│  └─ Actions (shrink-0, optional)
└─ Content (children)
```

### Responsive Behavior

- **Mobile:** Actions wrap below title/description
- **Desktop (sm+):** Actions align to the right on same line

### Accessibility

- Semantic HTML (`<h1>` for title)
- Proper heading hierarchy
- ARIA-friendly structure

## Pages Using PageContainer

✅ **DashboardPage** - Monitor queen, hives, workers  
✅ **SettingsPage** - Configuration  
✅ **HelpPage** - Documentation links  
❌ **KeeperPage** - Uses custom layout (sidebar + terminal)

## Design Consistency

This component enforces the user's consistency requirements:

1. ✅ **No mixed patterns** - All pages use same structure
2. ✅ **Standardized spacing** - Controlled via `spacing` prop
3. ✅ **No style variations** - Single source of truth for page headers
4. ✅ **Consistent structure** - Same pattern everywhere

## When NOT to Use

- **Marketing pages** → Use `TemplateContainer`
- **Custom layouts** → Build your own (e.g., KeeperPage with sidebar)
- **Full-screen views** → Build your own (e.g., terminal-only views)

## Related Components

- **TemplateContainer** - Marketing page wrapper (rich features)
- **IconCardHeader** - Card-level headers (not page-level)
- **PageHeader** - Empty directory (was planned but not needed)
