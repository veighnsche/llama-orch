# Quickstart: Next.js + Shared Components

## Start Dev Server

```bash
cd frontend/bin/commercial
pnpm dev
```

Open http://localhost:3000

## Test HMR

1. Open `frontend/libs/shared-components/ui/button.tsx`
2. Change line 14 from:
   ```tsx
   "inline-flex items-center justify-center gap-2..."
   ```
   To:
   ```tsx
   "inline-flex items-center justify-center gap-2 bg-gradient-to-r from-purple-500 to-pink-500..."
   ```
3. Save the file
4. **Watch the browser update instantly** - no refresh needed!

## Import Shared Components

```tsx
// In any page or component
import { Button, Card, CardHeader, CardTitle, CardContent } from '@rbee/shared-components'

export default function MyPage() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Welcome</CardTitle>
      </CardHeader>
      <CardContent>
        <Button>Get Started</Button>
      </CardContent>
    </Card>
  )
}
```

## Available Components

All from shadcn/ui:
- `Button`, `Card`, `Alert`, `Badge`
- `Input`, `Label`, `Checkbox`, `Switch`
- `Tabs`, `Accordion`, `Dialog`
- `DropdownMenu`, `Select`, `Tooltip`
- And 40+ more in `ui/` directory

## Build for Production

```bash
pnpm build
pnpm start
```

## Deploy

The app is configured for Cloudflare Pages (from reference/v0).
See `wrangler.jsonc` for deployment config.
