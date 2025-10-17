# Inline Markdown Guide

**When to use:** Occasional markdown formatting in UI strings (card descriptions, tooltips, short copy)  
**When NOT to use:** Long-form content, blog posts, documentation pages

---

## Quick Start

```tsx
import { parseInlineMarkdown, InlineMarkdown } from '@rbee/ui/utils'

// Option 1: Function (for dynamic content)
<p>{parseInlineMarkdown('Power **your** GPUs')}</p>

// Option 2: Component (for JSX)
<InlineMarkdown>Power **your** GPUs with *zero* API fees</InlineMarkdown>
```

---

## Supported Syntax

### Bold
```tsx
'Power **your** GPUs'
// Renders: Power <strong>your</strong> GPUs
```

### Italic
```tsx
'Deploy *any* model'
// Renders: Deploy <em>any</em> model
```

### Links
```tsx
'Read the [documentation](https://docs.example.com)'
// Renders: Read the <a href="...">documentation</a>
// External links automatically get target="_blank" rel="noopener noreferrer"
```

---

## Best Practices

### ✅ DO Use For:
- **Card descriptions** with occasional emphasis
- **Tooltip text** with bold keywords
- **Short UI copy** (1-2 sentences)
- **Feature lists** with inline formatting

### ❌ DON'T Use For:
- **Long paragraphs** (use a proper markdown library)
- **Complex formatting** (headings, lists, code blocks)
- **User-generated content** (security risk - no sanitization)
- **Blog posts or documentation** (use MDX or react-markdown)

---

## Examples

### Card Description (Current Use Case)
```tsx
// In HomePageProps.tsx
description: 'Power Zed, Cursor, and your own agents on **your** GPUs. OpenAI-compatible - drop-in, zero API fees.'

// Renders with "your" in bold
```

### Tooltip
```tsx
<Tooltip>
  <InlineMarkdown>
    Press **Cmd+K** to open the command palette
  </InlineMarkdown>
</Tooltip>
```

### Feature List
```tsx
features: [
  'Deploy **any** open model',
  'Keep data *private*',
  'Zero ongoing costs',
]
```

---

## Implementation Details

### How It Works
1. Parses string with regex patterns
2. Splits text into parts (plain text + React elements)
3. Returns array of ReactNode elements
4. Automatically adds keys for React reconciliation

### Styling
- **Bold (`<strong>`)**: Inherits font-weight from parent
- **Italic (`<em>`)**: Inherits font-style from parent
- **Links (`<a>`)**: Uses `brandLink` styling (amber underline, hover states)

### Performance
- ✅ Lightweight (no dependencies)
- ✅ Runs at render time (minimal overhead)
- ✅ Memoizable if needed (wrap in `useMemo`)

---

## When to Add a Full Markdown Library

Consider adding `react-markdown` or `MDX` if you need:
- Headings (h1-h6)
- Lists (ul, ol)
- Code blocks with syntax highlighting
- Tables
- Blockquotes
- Images
- User-generated content (with sanitization)

For now, the lightweight inline parser is perfect for occasional UI formatting.

---

## Migration Path

If you later need full markdown:

```bash
pnpm add react-markdown remark-gfm
```

Then update components:
```tsx
import ReactMarkdown from 'react-markdown'

<ReactMarkdown>{description}</ReactMarkdown>
```

But for 95% of UI strings, `parseInlineMarkdown` is the right choice.

---

## Security Note

⚠️ **This parser does NOT sanitize HTML**. Only use with:
- Trusted content (your own copy)
- Static strings in props files
- Content you control

**Never** use with user-generated content without proper sanitization.

---

## Related Files

- **Implementation**: `src/utils/parse-inline-markdown.tsx`
- **Usage Example**: `src/organisms/AudienceCard/AudienceCard.tsx`
- **Props Example**: `src/pages/HomePage/HomePageProps.tsx`

---

**Version:** 1.0.0  
**Last Updated:** October 17, 2025
