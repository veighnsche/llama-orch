# CodeSnippet Component

A component for displaying inline or small block code snippets with proper monospace font (Geist Mono).

## Features

- ✅ **Geist Mono font** - Professional monospace typography
- ✅ **Inline variant** - For use within paragraphs
- ✅ **Block variant** - For standalone code snippets
- ✅ **Proper styling** - Muted background with border
- ✅ **Copy-friendly** - Text selection works perfectly

## Usage

### Inline Code

```tsx
import { CodeSnippet } from '@/components/atoms'

;<p>
  Run <CodeSnippet>npm install</CodeSnippet> to get started.
</p>
```

### Block Code

```tsx
<CodeSnippet variant="block">curl -sSL rbee.dev/install.sh | sh</CodeSnippet>
```

### Custom Styling

```tsx
<CodeSnippet variant="block" className="text-xs text-primary">
  export OPENAI_API_BASE=http://localhost:8080/v1
</CodeSnippet>
```

## Props

| Prop        | Type                  | Default    | Description            |
| ----------- | --------------------- | ---------- | ---------------------- |
| `children`  | `ReactNode`           | -          | Code content           |
| `variant`   | `'inline' \| 'block'` | `'inline'` | Display variant        |
| `className` | `string`              | -          | Additional CSS classes |

## When to Use

- **CodeSnippet (inline)** - For commands or code within text: "Run `npm install`"
- **CodeSnippet (block)** - For short, standalone commands without terminal chrome
- **ConsoleOutput** - For terminal output with chrome and multiple lines
- **CodeBlock** - For larger code blocks with line numbers and syntax highlighting

## Examples

### Installation Command

```tsx
<div className="space-y-2">
  <p className="text-muted-foreground">Install rbee with a single command:</p>
  <CodeSnippet variant="block">curl -sSL rbee.dev/install.sh | sh</CodeSnippet>
</div>
```

### Environment Variable

```tsx
<p>
  Set <CodeSnippet>OPENAI_API_BASE</CodeSnippet> to point to your local instance.
</p>
```

### API Endpoint

```tsx
<CodeSnippet variant="block" className="text-sm">
  http://localhost:8080/v1/chat/completions
</CodeSnippet>
```
