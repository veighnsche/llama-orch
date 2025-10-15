# CodeBlock Molecule

**Prism-powered syntax highlighting with dark/light adaptive themes.**

## Features

- ✅ **First-class syntax highlighting** via `prism-react-renderer`
- ✅ **Theme switching**: Night Owl (dark) / Night Owl Light (light) via `next-themes`
- ✅ **Language support**: Python, TypeScript, JavaScript, Bash, JSON, YAML, Rust, Go, SQL, and more
- ✅ **Atomic design**: Reuses `atoms/Button` for copy action
- ✅ **Accessibility**: Screen reader announcements for copy feedback
- ✅ **Line highlighting**: Visual emphasis with `bg-primary/10` and left border accent
- ✅ **Responsive**: Adapts to mobile/desktop with proper scrolling
- ✅ **No build-step**: Client-side rendering, no async highlighter needed

## API

```tsx
interface CodeBlockProps {
  /** Code content */
  code: string
  /** Programming language */
  language?: string
  /** Optional title */
  title?: string
  /** Show copy button */
  copyable?: boolean
  /** Show line numbers */
  showLineNumbers?: boolean
  /** Line numbers to highlight */
  highlight?: number[]
  /** Additional CSS classes */
  className?: string
}
```

## Usage

### Basic Example

```tsx
import { CodeBlock } from '@rbee/ui/molecules/CodeBlock'

<CodeBlock
  code={`console.log('Hello, world!')`}
  language="typescript"
  title="example.ts"
/>
```

### With Line Numbers

```tsx
<CodeBlock
  code={pythonCode}
  language="python"
  showLineNumbers={true}
/>
```

### With Line Highlighting

```tsx
<CodeBlock
  code={typescriptCode}
  language="typescript"
  showLineNumbers={true}
  highlight={[3, 4, 5]}  // Highlight lines 3-5
/>
```

## Language Support

The `resolveLang` utility maps common language aliases to Prism languages:

- **TypeScript/JavaScript**: `typescript`, `ts`, `tsx`, `javascript`, `js`, `jsx`
- **Python**: `python`, `py`
- **Bash**: `bash`, `sh`, `shell`
- **JSON**: `json`
- **YAML**: `yaml`, `yml`
- **Markdown**: `markdown`, `md`
- **CSS**: `css`
- **HTML/XML**: `html`, `xml`
- **Rust**: `rust`, `rs`
- **Go**: `go`
- **SQL**: `sql`

Unknown languages default to `tsx` and render gracefully.

## Theming

The component uses **CSS custom properties** for syntax highlighting colors, automatically adapting to light/dark mode:

### CSS Variables (defined in `globals.css`)

```css
:root {
  --code-string: #4876d6;      /* Blue for strings */
  --code-variable: #c96765;    /* Red for variables */
  --code-number: #aa0982;      /* Magenta for numbers */
  --code-function: #005cc5;    /* Deep blue for functions */
  --code-punctuation: #994cc3; /* Purple for punctuation */
  --code-class: #111111;       /* Dark for class names */
  --code-keyword: #0077aa;     /* Teal for keywords */
  --code-property: #0077aa;    /* Teal for properties */
}

.dark {
  --code-string: #addb67;      /* Green for strings */
  --code-variable: #d6deeb;    /* Light gray for variables */
  --code-number: #f78c6c;      /* Orange for numbers */
  --code-function: #82aaff;    /* Light blue for functions */
  --code-punctuation: #c792ea; /* Light purple for punctuation */
  --code-class: #ffcb8b;       /* Yellow for class names */
  --code-keyword: #7fdbca;     /* Cyan for keywords */
  --code-property: #80cbc4;    /* Cyan for properties */
}
```

Colors are based on the **Night Owl** theme palette and meet WCAG AA contrast requirements.

## Layout

### Without Line Numbers
Single-column layout with full-width code lines.

### With Line Numbers
2-column CSS grid:
- **Left column**: Line numbers (right-aligned, muted, non-selectable)
- **Right column**: Syntax-highlighted code

### Line Highlighting
Highlighted lines receive:
- `bg-primary/10` background
- `border-l-2 border-l-primary/60` left border accent

## Accessibility

- **Copy button**: Uses `atoms/Button` with proper `aria-label`
- **Copy feedback**: `aria-live="polite"` region announces "Code copied to clipboard"
- **Animated feedback**: `animate-in fade-in zoom-in-95 duration-200` on "Copied" label
- **Keyboard navigation**: Full keyboard support via Button atom

## Scrolling

Custom scrollbar styling (in `globals.css`):
- Height: 10px
- Track: `var(--secondary)`
- Thumb: `var(--border)` with rounded corners
- Hover: `var(--muted-foreground)`

## Dependencies

- `prism-react-renderer`: Syntax highlighting engine
- `next-themes`: Theme detection
- `lucide-react`: Copy/Check icons
- `@rbee/ui/atoms/Button`: Copy button component

## Implementation Notes

1. **CSS token-based theming**: Uses CSS custom properties instead of hardcoded colors
2. **No `defaultProps`**: prism-react-renderer v2.x removed `defaultProps` export
3. **No `dark:` prefix**: Theme switching handled via `.dark` class on CSS variables
4. **Grid layout**: Uses CSS grid with `theme(spacing.10)` for line number column
5. **Tab size**: `tab-size-[2]` for consistent indentation
6. **Foreground rendering**: All rendering happens in `<Highlight>` render-props

## Edge Cases

- **Long lines**: `overflow-x-auto` on `<pre>` prevents layout jitter
- **No language**: Defaults to `tsx`, language badge only shows if set
- **Empty code**: Renders empty block gracefully
- **SSR/Client**: Component is `'use client'`, safe for CSR

## Used In

- **FeaturesSection**: Displays API examples in multiple languages
- **Documentation pages**: Code snippets and tutorials
- **API reference**: Request/response examples

## QA Checklist

- ✅ Python, Bash, TS examples show token colors in both light/dark
- ✅ Screen reader announces "Code copied to clipboard"
- ✅ Line highlight visible in both themes with adequate contrast
- ✅ No horizontal scroll on header; code scrolls independently
- ✅ Works inside FeaturesSection organism without layout shifts
- ✅ Type-safe with full TypeScript support
