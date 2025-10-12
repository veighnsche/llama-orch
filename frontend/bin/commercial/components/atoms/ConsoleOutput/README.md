# ConsoleOutput Component

A component for displaying terminal/console output with proper monospace font (Geist Mono).

## Features

- ✅ **Geist Mono font** - Professional monospace typography
- ✅ **Terminal chrome** - Optional macOS-style traffic lights and title bar
- ✅ **Multiple variants** - Terminal, code, or output styles
- ✅ **Background options** - Dark, light, or card backgrounds
- ✅ **Proper overflow** - Horizontal scrolling for long lines
- ✅ **Syntax highlighting support** - Use inline spans for colors

## Usage

### Basic Console Output

```tsx
import { ConsoleOutput } from '@/components/atoms'

;<ConsoleOutput>$ npm install rbee ✓ Installation complete</ConsoleOutput>
```

### With Terminal Chrome

```tsx
<ConsoleOutput showChrome title="bash">
  $ curl -sSL https://rbee.dev/install.sh | sh Downloading rbee... ✓ Installation complete
</ConsoleOutput>
```

### Dark Background (Default)

```tsx
<ConsoleOutput showChrome title="terminal" background="dark">
  <div>$ git clone https://github.com/veighnsche/llama-orch</div>
  <div className="text-slate-400">Cloning into 'llama-orch'...</div>
</ConsoleOutput>
```

### With Syntax Highlighting

```tsx
<ConsoleOutput showChrome title="TypeScript" background="dark">
  <div>
    <span className="text-purple-400">import</span> <span className="text-amber-400">'react'</span>
  </div>
  <div>
    <span className="text-blue-400">const</span> App = () =&gt; {'{'}
  </div>
</ConsoleOutput>
```

## Props

| Prop         | Type                               | Default      | Description                      |
| ------------ | ---------------------------------- | ------------ | -------------------------------- |
| `children`   | `ReactNode`                        | -            | Console content                  |
| `showChrome` | `boolean`                          | `false`      | Show terminal window chrome      |
| `title`      | `string`                           | -            | Terminal title (shown in chrome) |
| `variant`    | `'terminal' \| 'code' \| 'output'` | `'terminal'` | Visual variant                   |
| `background` | `'dark' \| 'light' \| 'card'`      | `'dark'`     | Background style                 |
| `className`  | `string`                           | -            | Additional CSS classes           |

## Color Palette for Syntax Highlighting

Use Tailwind classes for syntax highlighting:

- **Keywords**: `text-purple-400` or `text-blue-400`
- **Strings**: `text-amber-400` or `text-green-400`
- **Functions**: `text-green-400` or `text-cyan-400`
- **Comments**: `text-slate-400` or `text-slate-500`
- **Numbers**: `text-orange-400`
- **Operators**: `text-pink-400`

## Related Components

- **CodeSnippet** - For inline or small block code snippets
- **CodeBlock** - For larger code blocks with line numbers
- **TerminalWindow** - Legacy component (consider migrating to ConsoleOutput)
