# Syntax Highlighting Tokens - Design System

## Overview

Standardized CSS custom properties for syntax highlighting and code/console displays. These tokens ensure consistent, theme-aware colors across all code examples, terminals, and console outputs.

## Token Reference

### Console/Code Backgrounds

```css
/* Light mode */
--console-bg: #0f172a;  /* Dark slate background for code blocks */
--console-fg: #f1f5f9;  /* Light text on dark background */

/* Dark mode */
--console-bg: #020617;  /* Even darker background for better contrast */
--console-fg: #f1f5f9;  /* Same light text */
```

**Usage:**
```tsx
<div className="bg-[var(--console-bg)] text-[var(--console-fg)]">
  // Code content
</div>
```

### Syntax Highlighting Colors

| Token | Light Mode | Dark Mode | Purpose | Example |
|-------|------------|-----------|---------|---------|
| `--syntax-keyword` | `#3b82f6` (blue) | `#60a5fa` (lighter blue) | Keywords | `const`, `await`, `export`, `let` |
| `--syntax-import` | `#8b5cf6` (purple) | `#a78bfa` (lighter purple) | Import statements | `import`, `from`, `require` |
| `--syntax-string` | `#f59e0b` (amber) | `#fbbf24` (lighter amber) | String literals | `'text'`, `"text"`, `` `template` `` |
| `--syntax-function` | `#10b981` (green) | `#34d399` (lighter green) | Function names | `invoke()`, `fetch()`, `map()` |
| `--syntax-comment` | `#64748b` (muted) | `#94a3b8` (lighter muted) | Comments & secondary | `// comment`, `# comment` |

**Usage:**
```tsx
<span className="text-[var(--syntax-keyword)]">const</span>
<span className="text-[var(--syntax-string)]">'hello'</span>
<span className="text-[var(--syntax-function)]">invoke</span>
<span className="text-[var(--syntax-comment)]">// comment</span>
```

## Color Philosophy

### Light Mode
- **Darker, saturated colors** for good contrast on light backgrounds
- Base colors from the chart palette for consistency
- Muted colors for comments to reduce visual noise

### Dark Mode
- **Lighter, brighter colors** for better contrast on dark backgrounds
- Increased luminosity (e.g., `#3b82f6` → `#60a5fa`)
- Maintains the same hue family for brand consistency

## Complete Example

### Terminal Output
```tsx
import { ConsoleOutput } from '@rbee/ui/atoms'

<ConsoleOutput showChrome title="bash" background="dark">
  <div>curl -sSL https://rbee.dev/install.sh | sh</div>
  <div className="text-[var(--syntax-comment)]">rbee-keeper daemon start</div>
</ConsoleOutput>
```

### TypeScript Code
```tsx
<div className="bg-[var(--console-bg)] text-[var(--console-fg)] p-4 rounded-lg">
  <div>
    <span className="text-[var(--syntax-import)]">import</span> {'{'} invoke {'}'}{' '}
    <span className="text-[var(--syntax-import)]">from</span>{' '}
    <span className="text-[var(--syntax-string)]">'@llama-orch/utils'</span>;
  </div>
  <div className="mt-2">
    <span className="text-[var(--syntax-keyword)]">const</span> code ={' '}
    <span className="text-[var(--syntax-keyword)]">await</span>{' '}
    <span className="text-[var(--syntax-function)]">invoke</span>({'({'});
  </div>
  <div className="pl-4">
    prompt: <span className="text-[var(--syntax-string)]">'Generate API'</span>,
  </div>
  <div>{'});'}</div>
</div>
```

### Bash/Shell Commands
```tsx
<div className="bg-[var(--console-bg)] text-[var(--console-fg)] p-4 rounded-lg">
  <div>
    <span className="text-[var(--syntax-keyword)]">export</span> API_KEY=abc123
  </div>
  <div className="text-[var(--syntax-comment)]"># Set environment variable</div>
</div>
```

## Migration Guide

### Before (Hardcoded Colors)
```tsx
// ❌ Don't use hardcoded Tailwind colors
<span className="text-blue-400">const</span>
<span className="text-purple-400">import</span>
<span className="text-amber-400">'string'</span>
<span className="text-green-400">function</span>
<span className="text-slate-400">// comment</span>
```

### After (Semantic Tokens)
```tsx
// ✅ Use semantic tokens
<span className="text-[var(--syntax-keyword)]">const</span>
<span className="text-[var(--syntax-import)]">import</span>
<span className="text-[var(--syntax-string)]">'string'</span>
<span className="text-[var(--syntax-function)]">function</span>
<span className="text-[var(--syntax-comment)]">// comment</span>
```

## Benefits

✅ **Theme-aware** - Automatically adjusts between light and dark modes  
✅ **Consistent** - Same colors across all code examples  
✅ **Maintainable** - Change once in tokens, updates everywhere  
✅ **Accessible** - Optimized contrast ratios for both themes  
✅ **Reusable** - Use in any component without duplication  
✅ **Type-safe** - CSS custom properties are validated at build time  

## Token Location

All tokens are defined in:
```
frontend/packages/rbee-ui/src/tokens/theme-tokens.css
```

## Components Using These Tokens

- `ConsoleOutput` - Terminal/console output displays
- `HowItWorksSection` - Step-by-step code examples
- `CodeBlock` - Standalone code blocks
- `TerminalWindow` - Terminal window chrome
- `CodeExamplesSection` - Interactive code examples

## Extending the System

To add new syntax highlighting tokens:

1. Add to `theme-tokens.css`:
```css
:root {
  --syntax-variable: #ec4899; /* pink for variables */
}

.dark {
  --syntax-variable: #f9a8d4; /* lighter pink for dark mode */
}
```

2. Use in components:
```tsx
<span className="text-[var(--syntax-variable)]">myVar</span>
```

## Testing

After making changes:
1. Run `pnpm run build:styles` to rebuild CSS
2. Start Storybook: `pnpm run storybook`
3. Toggle between light and dark themes
4. Verify all syntax colors are visible and have good contrast
