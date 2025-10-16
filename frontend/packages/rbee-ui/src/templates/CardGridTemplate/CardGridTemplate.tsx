import type * as React from 'react'

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

export type CardGridTemplateProps = {
  children: React.ReactNode
  className?: string
}

// ────────────────────────────────────────────────────────────────────────────
// Main Component
// ────────────────────────────────────────────────────────────────────────────

/**
 * CardGridTemplate - Generic 2-column grid layout for cards
 *
 * @example
 * ```tsx
 * <CardGridTemplate>
 *   {items.map((item, i) => <Card key={i} {...item} />)}
 * </CardGridTemplate>
 * ```
 */
export function CardGridTemplate({ children, className }: CardGridTemplateProps) {
  return (
    <div className={className}>
      {/* Grid - centered with max-width and 2 columns */}
      <div className="mx-auto max-w-[60%] grid gap-6 grid-cols-2">{children}</div>
    </div>
  )
}
