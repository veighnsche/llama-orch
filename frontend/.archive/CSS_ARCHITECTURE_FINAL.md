# CSS Architecture - FINAL CLEAN STATE

## Single Source of Truth

**`packages/rbee-ui/src/tokens/theme-tokens.css`** - THE ONLY place defining CSS variables

Contains:
- Brand color: `--primary: #f59e0b`
- All semantic colors (background, foreground, card, etc.)
- Chart colors (5 colors)
- Terminal colors (3 colors)
- Spacing scale (12 levels)
- Typography scale (11 sizes)
- Icon sizes (5 levels)
- Border radius (6 variants)
- Shadows (6 levels)

## File Structure

```
packages/rbee-ui/
├── src/tokens/
│   ├── globals.css (entry point)
│   └── theme-tokens.css (CSS VARIABLES - ONLY SOURCE)
└── dist/
    └── index.css (built output)

apps/commercial/
└── app/
    ├── layout.tsx (imports '@rbee/ui/styles.css' then './globals.css')
    └── globals.css (app-specific animations & utilities ONLY)

apps/user-docs/
└── app/
    ├── layout.tsx (imports '@rbee/ui/styles.css' then './globals.css')
    └── globals.css (app-specific styles ONLY)
```

## Rules

### ✅ DO
- Define CSS variables in `packages/rbee-ui/src/tokens/theme-tokens.css`
- Rebuild after changes: `cd packages/rbee-ui && pnpm run build:styles`
- Let apps inherit variables automatically

### ❌ DON'T
- Define CSS variables in app globals.css files
- Create multiple globals.css files
- Duplicate tokens anywhere

## App CSS Files

Apps should only contain:
- Animations (@keyframes)
- App-specific utilities (@layer utilities)
- Base layer overrides (@layer base)

**NO CSS VARIABLES** - they come from the UI package.

## How It Works

1. UI package defines tokens in `theme-tokens.css`
2. Build step: `postcss globals.css → dist/index.css`
3. Apps import `@rbee/ui/styles.css`
4. Variables are inherited everywhere
5. Apps import their own `globals.css` for app-specific CSS only

---

**Status**: CLEAN ✅  
**Files**: Minimal  
**Sources**: ONE  
**Confusion**: NONE
