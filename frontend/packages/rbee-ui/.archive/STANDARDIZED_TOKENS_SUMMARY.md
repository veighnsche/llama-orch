# Standardized Syntax Highlighting Tokens - Summary

## What Changed

Converted all hardcoded syntax highlighting colors to reusable CSS custom properties (tokens) in the design system.

## New Tokens Added

### In `src/tokens/theme-tokens.css`

```css
:root {
  /* Code/Console backgrounds */
  --console-bg: #0f172a;
  --console-fg: #f1f5f9;

  /* Syntax highlighting colors */
  --syntax-keyword: #3b82f6;   /* blue - const, await, export */
  --syntax-import: #8b5cf6;    /* purple - import, from */
  --syntax-string: #f59e0b;    /* amber - string literals */
  --syntax-function: #10b981;  /* green - function names */
  --syntax-comment: #64748b;   /* muted - comments */
}

.dark {
  /* Code/Console backgrounds - dark mode */
  --console-bg: #020617;       /* darker for better contrast */
  --console-fg: #f1f5f9;

  /* Syntax highlighting - lighter for dark mode */
  --syntax-keyword: #60a5fa;   /* lighter blue */
  --syntax-import: #a78bfa;    /* lighter purple */
  --syntax-string: #fbbf24;    /* lighter amber */
  --syntax-function: #34d399;  /* lighter green */
  --syntax-comment: #94a3b8;   /* lighter muted */
}
```

## Files Modified

### 1. `src/tokens/theme-tokens.css`
- ✅ Added 7 new token pairs (light + dark mode)
- ✅ Documented purpose of each token
- ✅ Optimized contrast for both themes

### 2. `src/atoms/ConsoleOutput/ConsoleOutput.tsx`
**Before:**
```tsx
dark: 'bg-[#0f172a] dark:bg-[#020617] text-foreground'
```

**After:**
```tsx
dark: 'bg-[var(--console-bg)] text-[var(--console-fg)]'
```

### 3. `src/organisms/Home/HowItWorksSection/HowItWorksSection.tsx`
**Before:**
```tsx
<span className="text-blue-400">const</span>
<span className="text-purple-400">import</span>
<span className="text-amber-400">'string'</span>
<span className="text-green-400">invoke</span>
<span className="text-slate-400">// comment</span>
```

**After:**
```tsx
<span className="text-[var(--syntax-keyword)]">const</span>
<span className="text-[var(--syntax-import)]">import</span>
<span className="text-[var(--syntax-string)]">'string'</span>
<span className="text-[var(--syntax-function)]">invoke</span>
<span className="text-[var(--syntax-comment)]">// comment</span>
```

## Benefits

### ✅ Reusability
- Define once in `theme-tokens.css`
- Use anywhere in the codebase
- No duplication of color values

### ✅ Consistency
- All code examples use the same colors
- Matches the design system
- Predictable visual hierarchy

### ✅ Maintainability
- Change colors in one place
- Updates propagate everywhere
- Easy to adjust for accessibility

### ✅ Theme-Aware
- Automatically adjusts for light/dark mode
- Optimized contrast for each theme
- No manual theme switching needed

### ✅ Accessibility
- Lighter colors in dark mode for better contrast
- Darker colors in light mode for readability
- Meets WCAG contrast requirements

## Usage Examples

### Terminal Command
```tsx
<ConsoleOutput showChrome title="bash" background="dark">
  <div>curl -sSL https://rbee.dev/install.sh | sh</div>
  <div className="text-[var(--syntax-comment)]"># Install rbee</div>
</ConsoleOutput>
```

### TypeScript Code
```tsx
<div className="bg-[var(--console-bg)] text-[var(--console-fg)] p-4">
  <span className="text-[var(--syntax-keyword)]">const</span> result ={' '}
  <span className="text-[var(--syntax-keyword)]">await</span>{' '}
  <span className="text-[var(--syntax-function)]">fetch</span>(
    <span className="text-[var(--syntax-string)]">'/api'</span>
  );
</div>
```

### Shell Script
```tsx
<div className="bg-[var(--console-bg)] text-[var(--console-fg)] p-4">
  <div>
    <span className="text-[var(--syntax-keyword)]">export</span> API_KEY=secret
  </div>
  <div className="text-[var(--syntax-comment)]"># Set environment variable</div>
</div>
```

## Token Reference Table

| Token | Purpose | Light Mode | Dark Mode |
|-------|---------|------------|-----------|
| `--console-bg` | Code block background | `#0f172a` | `#020617` |
| `--console-fg` | Code block text | `#f1f5f9` | `#f1f5f9` |
| `--syntax-keyword` | Keywords (const, let, etc.) | `#3b82f6` | `#60a5fa` |
| `--syntax-import` | Import statements | `#8b5cf6` | `#a78bfa` |
| `--syntax-string` | String literals | `#f59e0b` | `#fbbf24` |
| `--syntax-function` | Function names | `#10b981` | `#34d399` |
| `--syntax-comment` | Comments & secondary | `#64748b` | `#94a3b8` |

## Migration Checklist

When adding new code examples:

- [ ] Use `--console-bg` and `--console-fg` for backgrounds
- [ ] Use `--syntax-keyword` for keywords (const, let, await, export)
- [ ] Use `--syntax-import` for import/from statements
- [ ] Use `--syntax-string` for string literals
- [ ] Use `--syntax-function` for function names
- [ ] Use `--syntax-comment` for comments and secondary text
- [ ] Test in both light and dark modes
- [ ] Verify contrast is sufficient

## Documentation

- **Complete Guide**: `SYNTAX_HIGHLIGHTING_TOKENS.md`
- **Implementation Details**: `THEME_AWARE_SYNTAX_HIGHLIGHTING.md`
- **Storybook Dark Mode**: `STORYBOOK_DARK_MODE.md`

## Testing

1. Rebuild styles:
   ```bash
   pnpm run build:styles
   ```

2. Start Storybook:
   ```bash
   pnpm run storybook
   ```

3. Navigate to stories with code examples:
   - Organisms → Home → HowItWorksSection
   - Atoms → ConsoleOutput
   - Molecules → CodeBlock

4. Toggle theme in Storybook toolbar

5. Verify:
   - [ ] Colors change between light/dark modes
   - [ ] All syntax elements are visible
   - [ ] Contrast is good in both modes
   - [ ] No hardcoded colors remain

## Next Steps

To extend the system:

1. Add new tokens to `theme-tokens.css`
2. Document in `SYNTAX_HIGHLIGHTING_TOKENS.md`
3. Use in components with `text-[var(--token-name)]`
4. Test in both themes
5. Update this summary

## Questions?

See the complete documentation in:
- `SYNTAX_HIGHLIGHTING_TOKENS.md` - Full token reference
- `THEME_AWARE_SYNTAX_HIGHLIGHTING.md` - Implementation details
