# @rbee/shared-components

Shared React component library for Next.js applications.

## Structure

```
shared-components/
├── ui/             # UI components (shadcn/ui based)
├── lib/            # Utilities (cn helper)
└── index.ts        # Main export barrel
```

## Usage

```tsx
import { Button, Card, Alert } from '@rbee/shared-components'

export default function Page() {
  return (
    <Card>
      <Alert>Welcome to Orchyra</Alert>
      <Button>Get Started</Button>
    </Card>
  )
}
```

## Components

All components from shadcn/ui are available, including:
- Button, Card, Alert, Badge
- Input, Label, Checkbox, Switch
- Tabs, Accordion, Dialog
- Dropdown Menu, Select, Tooltip
- And more...

## Development

Components work with Next.js 15+ and React 19.
HMR works automatically with `transpilePackages` in next.config.
