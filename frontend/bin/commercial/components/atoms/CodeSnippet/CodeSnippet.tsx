// Created by: TEAM-AI-ASSISTANT
import { cn } from '@/lib/utils'
import type { ReactNode } from 'react'

export interface CodeSnippetProps {
  /** Code content */
  children: ReactNode
  /** Visual variant */
  variant?: 'inline' | 'block'
  /** Additional CSS classes */
  className?: string
}

/**
 * CodeSnippet - A component for displaying inline or small block code snippets
 * 
 * Features:
 * - Uses Geist Mono font for proper code display
 * - Inline variant for use within text
 * - Block variant for standalone snippets
 * - Proper text selection and copy support
 * 
 * @example
 * ```tsx
 * // Inline usage
 * <p>Run <CodeSnippet>npm install</CodeSnippet> to get started</p>
 * 
 * // Block usage
 * <CodeSnippet variant="block">
 *   curl -sSL rbee.dev/install.sh | sh
 * </CodeSnippet>
 * ```
 */
export function CodeSnippet({
  children,
  variant = 'inline',
  className,
}: CodeSnippetProps) {
  if (variant === 'inline') {
    return (
      <code
        className={cn(
          'rounded-md bg-muted px-1.5 py-0.5 text-sm text-foreground font-mono',
          className
        )}
      >
        {children}
      </code>
    )
  }

  return (
    <div
      className={cn(
        'overflow-x-auto rounded-lg border border-border bg-muted p-3',
        className
      )}
    >
      <code className="block text-sm text-foreground font-mono">
        {children}
      </code>
    </div>
  )
}
