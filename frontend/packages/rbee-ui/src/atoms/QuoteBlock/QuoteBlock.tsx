import { cn } from '@rbee/ui/utils'
import type { ReactNode } from 'react'

export interface QuoteBlockProps {
  /** Quote text content */
  children: ReactNode
  /** Show closing quote mark */
  showClosingQuote?: boolean
  /** Text size variant */
  size?: 'sm' | 'base' | 'lg'
  /** Additional CSS classes */
  className?: string
  /** Schema.org itemProp (e.g., 'reviewBody') */
  itemProp?: string
}

const sizeClasses = {
  sm: 'text-sm leading-6',
  base: 'text-base leading-relaxed',
  lg: 'text-lg leading-relaxed',
} as const

/**
 * QuoteBlock - Semantic blockquote with decorative quote marks
 *
 * @example
 * ```tsx
 * <QuoteBlock>This is a great product!</QuoteBlock>
 * <QuoteBlock showClosingQuote>Amazing experience.</QuoteBlock>
 * <QuoteBlock size="lg" itemProp="reviewBody">Five stars!</QuoteBlock>
 * ```
 */
export function QuoteBlock({
  children,
  showClosingQuote = false,
  size = 'base',
  className,
  itemProp,
}: QuoteBlockProps) {
  return (
    <blockquote>
      <p className={cn('text-muted-foreground text-pretty', sizeClasses[size], className)} itemProp={itemProp}>
        <span className="mr-1 text-primary">&ldquo;</span>
        {children}
        {showClosingQuote && <span className="ml-1 text-primary">&rdquo;</span>}
      </p>
    </blockquote>
  )
}
