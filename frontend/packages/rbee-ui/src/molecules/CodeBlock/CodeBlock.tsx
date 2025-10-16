'use client'

import { cn } from '@rbee/ui/utils'
import { Check, Copy } from 'lucide-react'
import { Highlight, type PrismTheme } from 'prism-react-renderer'
import { useState } from 'react'
import { resolveLang } from './prism'

// Theme using CSS design tokens (adapts to light/dark automatically)
const tokenBasedTheme: PrismTheme = {
  plain: {
    color: 'var(--foreground)',
    backgroundColor: 'var(--background)',
  },
  styles: [
    {
      types: ['comment'],
      style: {
        color: 'var(--muted-foreground)',
        fontStyle: 'italic',
      },
    },
    {
      types: ['string', 'url', 'attr-value'],
      style: {
        color: 'var(--code-string)',
      },
    },
    {
      types: ['variable', 'parameter'],
      style: {
        color: 'var(--code-variable)',
      },
    },
    {
      types: ['number', 'boolean', 'constant'],
      style: {
        color: 'var(--code-number)',
      },
    },
    {
      types: ['builtin', 'char', 'function', 'method'],
      style: {
        color: 'var(--code-function)',
      },
    },
    {
      types: ['punctuation', 'operator'],
      style: {
        color: 'var(--code-punctuation)',
      },
    },
    {
      types: ['class-name', 'type'],
      style: {
        color: 'var(--code-class)',
      },
    },
    {
      types: ['tag', 'keyword', 'selector'],
      style: {
        color: 'var(--code-keyword)',
      },
    },
    {
      types: ['property', 'attr-name'],
      style: {
        color: 'var(--code-property)',
      },
    },
    {
      types: ['deleted'],
      style: {
        color: 'var(--destructive)',
        fontStyle: 'italic',
      },
    },
    {
      types: ['inserted'],
      style: {
        color: 'var(--code-string)',
        fontStyle: 'italic',
      },
    },
    {
      types: ['changed'],
      style: {
        color: 'var(--code-function)',
        fontStyle: 'italic',
      },
    },
  ],
}

export interface CodeBlockProps {
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

export function CodeBlock({
  code,
  language,
  title,
  copyable = true,
  showLineNumbers = false,
  highlight = [],
  className,
}: CodeBlockProps) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className={cn('border border-border bg-card overflow-hidden', className)}>
      {(title || language || copyable) && (
        <div className="flex items-center justify-between gap-1 bg-muted/50 px-4 py-2">
          <div className="flex items-center gap-3">
            {title && <span className="font-mono text-xs text-muted-foreground">{title}</span>}
            {language && !title && (
              <span className="font-mono text-xs text-muted-foreground">
                {language}
              </span>
            )}
          </div>
          {copyable && (
            <>
              <button
                onClick={handleCopy}
                className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium font-sans text-muted-foreground hover:text-foreground transition-colors rounded-md hover:bg-accent/50"
                aria-label="Copy code to clipboard"
              >
                {copied ? (
                  <>
                    <Check className="h-3.5 w-3.5" />
                    <span>Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="h-3.5 w-3.5" />
                    <span>Copy</span>
                  </>
                )}
              </button>
              <div className="sr-only" aria-live="polite">
                {copied ? 'Code copied to clipboard' : ''}
              </div>
            </>
          )}
        </div>
      )}
      <Highlight code={code} language={resolveLang(language)} theme={tokenBasedTheme}>
        {({ className: prismClass, style, tokens, getLineProps, getTokenProps }) => (
          <pre className={cn('overflow-x-auto px-0 py-6', prismClass)} style={style}>
            <code
              className={cn('block text-[13px] sm:text-sm leading-6 font-mono', showLineNumbers ? 'grid' : 'block')}
              style={showLineNumbers ? { gridTemplateColumns: 'theme(spacing.10) 1fr' } : undefined}
            >
              {tokens.map((line, i) => {
                const lineNumber = i + 1
                const isHighlighted = highlight.includes(lineNumber)
                const lineProps = getLineProps({ line })

                if (!showLineNumbers) {
                  // Single column: just render tokens
                  return (
                    <div
                      key={i}
                      {...lineProps}
                      className={cn(
                        'px-6 py-0.5 whitespace-pre tab-size-[2]',
                        isHighlighted && 'bg-primary/10 border-l-2 border-l-primary/60',
                      )}
                    >
                      {line.map((token, key) => (
                        <span key={key} {...getTokenProps({ token })} />
                      ))}
                    </div>
                  )
                }

                // Two columns: line number + code
                return (
                  <div
                    key={i}
                    {...lineProps}
                    className={cn(
                      'grid gap-4 px-6 min-w-full tabular-nums',
                      '[grid-template-columns:theme(spacing.10)_1fr]',
                      isHighlighted && 'bg-primary/10 border-l-2 border-l-primary/60',
                    )}
                  >
                    <span className="select-none text-muted-foreground/80 text-xs text-right pr-2 py-0.5">
                      {String(lineNumber).padStart(2, ' ')}
                    </span>
                    <span className="whitespace-pre tab-size-[2] py-0.5">
                      {line.map((token, key) => (
                        <span key={key} {...getTokenProps({ token })} />
                      ))}
                    </span>
                  </div>
                )
              })}
            </code>
          </pre>
        )}
      </Highlight>
    </div>
  )
}
