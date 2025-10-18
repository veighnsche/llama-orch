# Serif Font Set as Default Across All Sites

## Change Made

**Source Serif 4 is now the default font** instead of Geist Sans.

### What Changed

**File:** `packages/tailwind-config/shared-styles.css`

```css
/* Fonts - Keep utilities working correctly */
--font-sans: var(--font-geist-sans);
--font-serif: var(--font-source-serif);
--font-mono: var(--font-geist-mono);

/* Override default to serif (but utilities still work) */
--default-font-family: var(--font-source-serif);
```

## Impact

### Before
- **Default font (no class):** Geist Sans (sans-serif)
- **`font-sans` class:** Geist Sans
- **`font-serif` class:** Source Serif 4
- **`font-mono` class:** Geist Mono

### After
- **Default font (no class):** Source Serif 4 (serif)
- **`font-sans` class:** Geist Sans ✅ STILL WORKS
- **`font-serif` class:** Source Serif 4 ✅ STILL WORKS
- **`font-mono` class:** Geist Mono ✅ STILL WORKS

## How to Use

### Default (Serif)
```tsx
<h1>This uses serif</h1>
<p>This also uses serif</p>
```

### Explicit Sans-Serif
```tsx
<h1 className="font-sans">This uses sans-serif</h1>
<p className="font-sans">This also uses sans-serif</p>
```

### Explicit Mono
```tsx
<code className="font-mono">This uses monospace</code>
```

## Components Affected

**All components now default to serif unless they explicitly use `font-sans`:**

- ✅ Hero headlines → Serif
- ✅ Body text → Serif
- ✅ Card content → Serif
- ✅ Navigation → Serif (unless changed)
- ✅ Buttons → Serif (unless changed)

**Components that should probably stay sans-serif:**
- Navigation menus
- Buttons
- Form inputs
- UI controls
- Terminal/console windows

## Next Steps

You'll likely want to add `font-sans` to specific UI components:

```tsx
// Navigation
<nav className="font-sans">...</nav>

// Buttons
<Button className="font-sans">...</Button>

// Forms
<input className="font-sans" />

// Terminal
<TerminalWindow className="font-mono">...</TerminalWindow>
```

## Verification

Rebuild and check:
```bash
cd packages/rbee-ui
pnpm build:styles

cd apps/commercial
pnpm dev
```

All text should now render in Source Serif 4 by default.

## Design Rationale

Serif fonts are traditionally used for:
- ✅ Editorial/publishing content
- ✅ Long-form reading
- ✅ Formal/elegant branding
- ✅ Print-inspired designs

This gives the site a more editorial, sophisticated feel compared to the tech-focused sans-serif default.

## Related Files

- `packages/tailwind-config/shared-styles.css` - Default font override
- `packages/rbee-ui/src/tokens/fonts.css` - Font loading
- `frontend/FONT_CENTRALIZATION_COMPLETE.md` - Font system architecture
- `frontend/FONT_CDN_FIX.md` - CDN loading approach
