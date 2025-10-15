# Storybook Dark Mode Toggle

## Overview

The Storybook instance now includes a **dark mode toggle** using the official `@storybook/addon-themes` addon. This allows you to preview all components in both light and dark modes without creating component variants.

## How to Use

1. **Start Storybook:**
   ```bash
   pnpm run storybook
   ```

2. **Toggle Dark Mode:**
   - Look for the **theme icon** (🎨 or similar) in the Storybook toolbar (top-right area)
   - Click it to open the theme selector dropdown
   - Select **"dark"** or **"light"** from the dropdown
   - All components and the background will automatically update to show the selected theme

## Implementation Details

### Addon Configuration

The dark mode toggle is implemented using:
- **Addon:** `@storybook/addon-themes`
- **Method:** `withThemeByClassName` decorator
- **CSS Class:** Toggles the `.dark` class on the root element

### Files Modified

1. **`.storybook/main.ts`**
   - Added `@storybook/addon-themes` to the addons array

2. **`.storybook/preview.ts`**
   - Imported `withThemeByClassName` from the addon
   - Configured the decorator with light/dark themes
   - Light theme: no class (default)
   - Dark theme: adds `.dark` class to `<html>`
   - Set `parentSelector: 'html'` to match next-themes behavior
   - Disabled default Storybook backgrounds

3. **`.storybook/preview-head.html`**
   - Added CSS to ensure html/body/root elements respect theme tokens
   - Applies `var(--background)` and `var(--foreground)` to all root elements

4. **`package.json`**
   - Added `@storybook/addon-themes` as a dev dependency

### How It Works

The addon works by:
1. Adding a theme switcher button to the Storybook toolbar
2. Toggling the `.dark` class on the root `<html>` element
3. Your existing CSS in `src/tokens/theme-tokens.css` already defines `.dark` styles
4. All components automatically respond to the theme change via CSS custom properties

### CSS Token System

Your design tokens in `theme-tokens.css` use the pattern:
```css
:root {
  --background: #ffffff;
  --foreground: #0f172a;
  /* ... more tokens */
}

.dark {
  --background: #0f172a;
  --foreground: #f1f5f9;
  /* ... dark mode overrides */
}
```

When the `.dark` class is applied, all CSS custom properties automatically switch to their dark mode values.

## Syntax Highlighting Colors

The terminal and code examples now use theme-aware colors:

- **Keywords** (`import`, `const`, `await`, `export`): `text-chart-2` (blue) or `text-chart-4` (purple)
- **Strings**: `text-primary` (amber/gold)
- **Functions**: `text-chart-3` (green)
- **Comments/Secondary text**: `text-muted-foreground`

These colors automatically adjust between light and dark modes via the CSS custom properties defined in `theme-tokens.css`.

## Benefits

✅ **No component variants needed** - Dark mode is handled at the CSS level  
✅ **Idiomatic Storybook pattern** - Uses official addon  
✅ **Works with Tailwind v4** - Compatible with your existing setup  
✅ **Turborepo friendly** - No additional build complexity  
✅ **Next.js compatible** - Same pattern used in production apps  
✅ **Theme-aware syntax highlighting** - Code examples respect light/dark mode  

## Troubleshooting

If the dark mode toggle doesn't appear:
1. Ensure Storybook is fully restarted after the changes
2. Check that `dist/index.css` exists (run `pnpm run build:styles`)
3. Verify the addon is listed in `.storybook/main.ts`
4. Check browser console for any errors

## Related Files

- `.storybook/main.ts` - Addon registration
- `.storybook/preview.ts` - Theme configuration
- `src/tokens/theme-tokens.css` - Light/dark theme tokens
- `src/tokens/globals.css` - Global CSS imports
