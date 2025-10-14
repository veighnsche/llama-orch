'use client'

import { cn } from '@rbee/ui/utils'
import { useState } from 'react'
import { Check, Copy } from 'lucide-react'

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
  const lines = code.split('\n')

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className={cn('rounded-xl border bg-card/60 shadow-sm overflow-hidden', className)}>
      {(title || language || copyable) && (
        <div className="flex items-center justify-between px-4 py-2.5 border-b bg-card/80">
          <div className="flex items-center gap-3">
            {title && <span className="text-sm font-medium text-foreground">{title}</span>}
            {language && (
              <span className="text-[11px] uppercase tracking-wide text-muted-foreground font-medium">{language}</span>
            )}
          </div>
          {copyable && (
            <button
              onClick={handleCopy}
              className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors rounded-md hover:bg-accent/50"
              aria-label="Copy code"
            >
              {copied ? (
                <>
                  <Check className="h-3.5 w-3.5" />
                  Copied
                </>
              ) : (
                <>
                  <Copy className="h-3.5 w-3.5" />
                  Copy
                </>
              )}
            </button>
          )}
        </div>
      )}
      <div className="p-4 sm:p-6">
        <pre className="overflow-x-auto text-sm font-mono leading-relaxed">
          <code>
            {showLineNumbers
              ? lines.map((line, index) => (
                  <div key={index} className={cn('flex gap-4', highlight.includes(index + 1) && 'bg-primary/10')}>
                    <span className="text-muted-foreground select-none">{(index + 1).toString().padStart(2, ' ')}</span>
                    <span>{line}</span>
                  </div>
                ))
              : code}
          </code>
        </pre>
      </div>
    </div>
  )
}
