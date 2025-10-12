import { cn } from '@/lib/utils'

export interface CodeBlockProps {
  /** Code content */
  code: string
  /** Programming language */
  language?: string
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
  showLineNumbers = false,
  highlight = [],
  className,
}: CodeBlockProps) {
  const lines = code.split('\n')

  return (
    <div
      className={cn(
        'bg-card border border-border rounded-lg p-6 font-mono text-sm',
        className
      )}
    >
      <pre className="overflow-x-auto">
        <code>
          {showLineNumbers ? (
            lines.map((line, index) => (
              <div
                key={index}
                className={cn(
                  'flex gap-4',
                  highlight.includes(index + 1) && 'bg-primary/10'
                )}
              >
                <span className="text-muted-foreground select-none">
                  {(index + 1).toString().padStart(2, ' ')}
                </span>
                <span>{line}</span>
              </div>
            ))
          ) : (
            code
          )}
        </code>
      </pre>
    </div>
  )
}
