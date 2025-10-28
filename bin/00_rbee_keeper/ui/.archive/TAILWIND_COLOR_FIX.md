# Tailwind Color Utilities Fix

## Problem
Action buttons using `text-success`, `text-danger`, `bg-success-muted`, and `bg-danger-muted` were appearing white/invisible because these Tailwind utility classes weren't being generated.

## Root Cause
Tailwind v4's JIT compiler only generates utility classes that are:
1. Used in files scanned by `@source` directive, OR
2. Explicitly defined via `@utility` directive

Since these color utilities weren't used in the rbee-ui package and weren't explicitly defined, they weren't available to the Tauri app.

## Solution

**Define custom utilities directly in the Tauri app's `globals.css`** using the `@utility` directive.

### 1. Updated `globals.css` with custom utilities
**File:** `bin/00_rbee_keeper/ui/src/globals.css`

```css
@import "tailwindcss";
@import "@repo/tailwind-config";           // Shared theme with color mappings
@import "@rbee/ui/tokens/theme-tokens.css"; // Design tokens (CSS variables)

/* Scan this app's source files for Tailwind classes */
@source "../src/**/*.{ts,tsx}";

/* Custom action color utilities for this app */
@utility text-success {
  color: var(--color-success);
}

@utility text-danger {
  color: var(--color-danger);
}

@utility bg-success-muted {
  background-color: var(--color-success-muted);
}

@utility bg-danger-muted {
  background-color: var(--color-danger-muted);
}
```

### 2. Kept component styles import
**File:** `bin/00_rbee_keeper/ui/src/main.tsx`

```tsx
import "./globals.css";        // Custom utilities + theme
import "@rbee/ui/styles.css";  // Component styles (Button, Table, etc.)
```

**Important:** Both imports are needed:
- `globals.css` provides custom utilities and theme configuration
- `@rbee/ui/styles.css` provides component styles (borders, padding, backgrounds)

### 3. Added missing dependencies
**File:** `bin/00_rbee_keeper/ui/package.json`

```json
{
  "dependencies": {
    "next-themes": "^0.4.6"  // ← Added
  },
  "devDependencies": {
    "@repo/tailwind-config": "workspace:*"  // ← Added
  }
}
```

### 4. Exported theme-tokens.css from rbee-ui
**File:** `frontend/packages/rbee-ui/package.json`

```json
{
  "exports": {
    "./tokens/theme-tokens.css": "./src/tokens/theme-tokens.css"  // ← Added
  }
}
```

### 5. Updated component to use Tailwind utilities
**File:** `bin/00_rbee_keeper/ui/src/components/SshTargetsTable.tsx`

```diff
- className="text-[var(--success)] hover:text-[var(--success)] hover:bg-[var(--success-muted)]"
+ className="text-success hover:text-success hover:bg-success-muted"

- className="text-[var(--danger)] hover:text-[var(--danger)] hover:bg-[var(--danger-muted)]"
+ className="text-danger hover:text-danger hover:bg-danger-muted"
```

## How It Works Now

1. **Tailwind v4 Vite plugin** processes `globals.css` at build/dev time
2. **@source directive** scans all `.ts` and `.tsx` files in the Tauri app
3. **@repo/tailwind-config** provides color mappings:
   - `--color-success: var(--success)` → enables `text-success` utility
   - `--color-danger: var(--danger)` → enables `text-danger` utility
   - `--color-success-muted: var(--success-muted)` → enables `bg-success-muted` utility
   - `--color-danger-muted: var(--danger-muted)` → enables `bg-danger-muted` utility
4. **theme-tokens.css** defines the actual color values:
   - `--success: #10b981` (emerald-500)
   - `--danger: #dc2626` (red-600)
   - `--success-muted: #d1fae5` (emerald-100)
   - `--danger-muted: #fee2e2` (red-100)
5. **JIT compilation** generates only the utilities actually used in the app

## Color Utilities Available

### Success Colors
- `text-success` - Green text (#10b981)
- `text-success-foreground` - White text on success bg
- `bg-success` - Green background
- `bg-success-muted` - Light green background (#d1fae5)
- `border-success` - Green border
- `hover:text-success`, `hover:bg-success-muted`, etc.

### Danger Colors
- `text-danger` - Red text (#dc2626)
- `text-danger-foreground` - White text on danger bg
- `bg-danger` - Red background
- `bg-danger-muted` - Light red background (#fee2e2)
- `border-danger` - Red border
- `hover:text-danger`, `hover:bg-danger-muted`, etc.

## Dark Mode Support
All colors automatically adapt to dark mode via CSS variables defined in `theme-tokens.css`:
- Light mode: `--success: #10b981` (emerald-500)
- Dark mode: `--success: #059669` (emerald-600, darker)

## Testing
After running `pnpm install`, the dev server will compile Tailwind with the correct utilities:

```bash
pnpm --filter @rbee/keeper-ui dev
```

The action buttons should now display with proper colors:
- Play button: Green (#10b981)
- Stop button: Red (#dc2626)
- Hover states: Light green/red backgrounds

## Files Changed
1. `bin/00_rbee_keeper/ui/src/globals.css` - Added theme imports and @source
2. `bin/00_rbee_keeper/ui/src/main.tsx` - Removed pre-compiled CSS import
3. `bin/00_rbee_keeper/ui/package.json` - Added dependencies
4. `bin/00_rbee_keeper/ui/src/components/SshTargetsTable.tsx` - Updated to use utilities
5. `bin/00_rbee_keeper/ui/src/pages/KeeperPage.tsx` - Fixed unused variable
6. `frontend/packages/rbee-ui/package.json` - Exported theme-tokens.css
7. `frontend/packages/tailwind-config/shared-styles.css` - Added comment for clarity
