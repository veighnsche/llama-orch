# CoreFeaturesTabs Component

## Overview

The `CoreFeaturesTabs` component is a reusable tabbed interface for showcasing core features with a sticky sidebar navigation and animated content panels.

## Architecture

The component is organized into:

1. **CoreFeaturesTabs.tsx** - The main reusable component
2. **DefaultCoreFeaturesTabs.tsx** - Pre-configured wrapper with default content
3. **tabConfigs.tsx** - Default tab configuration using existing reusable components

## Usage

### Basic Usage (Default Configuration)

```tsx
import { DefaultCoreFeaturesTabs } from '@rbee/ui/organisms'

export default function Page() {
  return <DefaultCoreFeaturesTabs />
}
```

### Custom Configuration

```tsx
import { CoreFeaturesTabs, type TabConfig } from '@rbee/ui/organisms'
import { CodeBlock } from '@rbee/ui/molecules/CodeBlock'
import { GPUUtilizationBar } from '@rbee/ui/molecules/GPUUtilizationBar'
import { Code, Cpu } from 'lucide-react'

const customTabs: TabConfig[] = [
  {
    value: 'api',
    icon: Code,
    label: 'API',
    mobileLabel: 'API',
    subtitle: 'RESTful API',
    badge: 'New',
    description: 'Our powerful API',
    content: (
      <CodeBlock
        code="curl https://api.example.com"
        language="bash"
        copyable={true}
      />
    ),
    highlight: {
      text: 'Fast and reliable',
      variant: 'primary',
    },
    benefits: [
      { text: 'Easy to use' },
      { text: 'Well documented' },
      { text: 'Scalable' },
    ],
  },
  // ... more tabs
]

export default function Page() {
  return (
    <CoreFeaturesTabs
      title="Our Features"
      description="Explore what we offer"
      tabs={customTabs}
      defaultTab="api"
      sectionId="features"
      bgClassName="bg-gradient-to-b from-primary to-secondary"
    />
  )
}
```

## Props

### CoreFeaturesTabsProps

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `title` | `string` | Required | Main heading for the section |
| `description` | `string` | Required | Subheading text |
| `tabs` | `TabConfig[]` | Required | Array of tab configurations |
| `defaultTab` | `string` | First tab value | Initially active tab |
| `sectionId` | `string` | `"feature-list"` | HTML id for the section |
| `bgClassName` | `string` | `"bg-gradient-to-b from-secondary to-background"` | Background styling |

### TabConfig

| Property | Type | Description |
|----------|------|-------------|
| `value` | `string` | Unique identifier for the tab |
| `icon` | `LucideIcon` | Icon component from lucide-react |
| `label` | `string` | Full label text (shown on desktop) |
| `mobileLabel` | `string` | Shortened label (shown on mobile) |
| `subtitle` | `string` | Small text below the tab trigger |
| `badge` | `string` | Badge text shown next to the title |
| `description` | `string` | Description text in the content panel |
| `content` | `ReactNode` | Custom content for the tab panel |
| `highlight` | `{ text: string, variant: 'primary' \| 'secondary' \| 'success' }` | Highlighted message box |
| `benefits` | `Array<{ text: string }>` | List of benefit items |

## Features

- **Responsive Design**: Adapts to mobile, tablet, and desktop viewports
- **Sticky Navigation**: Left sidebar stays visible while scrolling on desktop
- **Smooth Animations**: Fade and slide transitions between tabs
- **Accessible**: Proper ARIA labels and keyboard navigation
- **Customizable**: All content and styling can be overridden

## Using Existing Reusable Components

The tab content uses existing components from the design system:

### CodeBlock (from molecules)

```tsx
import { CodeBlock } from '@rbee/ui/molecules/CodeBlock'

<CodeBlock
  code="npm install rbee"
  language="bash"
  copyable={true}
/>
```

### GPUUtilizationBar (from molecules)

```tsx
import { GPUUtilizationBar } from '@rbee/ui/molecules/GPUUtilizationBar'

<GPUUtilizationBar label="RTX 4090" percentage={92} />
<GPUUtilizationBar label="CPU" percentage={34} variant="secondary" />
```

### TerminalWindow (from molecules)

```tsx
import { TerminalWindow } from '@rbee/ui/molecules/TerminalWindow'

<TerminalWindow showChrome={false} copyable={true}>
  $ npm install rbee
</TerminalWindow>
```

## Current Usage

- **Home Page** (`/frontend/apps/commercial/app/page.tsx`): Uses `DefaultCoreFeaturesTabs`
- **Features Page** (`/frontend/apps/commercial/app/features/page.tsx`): Uses `DefaultCoreFeaturesTabs`

## Exports

```tsx
// Main reusable component
export { CoreFeaturesTabs, type CoreFeaturesTabsProps, type TabConfig } from './CoreFeaturesTabs'

// Pre-configured wrapper
export { DefaultCoreFeaturesTabs } from './DefaultCoreFeaturesTabs'

// Default tab data
export { defaultTabConfigs } from './tabConfigs'
```
