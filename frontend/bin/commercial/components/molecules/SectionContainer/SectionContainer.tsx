import { cn } from '@/lib/utils'
import type { ReactNode } from 'react'

export interface SectionContainerProps {
  /** Section title (null to skip rendering) */
  title: string | ReactNode | null
  /** Optional subtitle */
  subtitle?: string | ReactNode
  /** Background variant */
  bgVariant?: 'background' | 'secondary' | 'card' | 'default'
  /** Center the content */
  centered?: boolean
  /** Maximum width of content */
  maxWidth?: 'xl' | '2xl' | '3xl' | '4xl' | '5xl' | '6xl' | '7xl'
  /** Section content */
  children: ReactNode
  /** Additional CSS classes */
  className?: string
  /** Optional ID for the heading (for aria-labelledby) */
  headingId?: string
}

export function SectionContainer({
  title,
  subtitle,
  bgVariant = 'background',
  centered = true,
  maxWidth = '4xl',
  children,
  className,
  headingId,
}: SectionContainerProps) {
  const bgClasses = {
    background: 'bg-background',
    secondary: 'bg-secondary',
    card: 'bg-card',
    default: 'bg-background',
  }

  const maxWidthClasses = {
    xl: 'max-w-xl',
    '2xl': 'max-w-2xl',
    '3xl': 'max-w-3xl',
    '4xl': 'max-w-4xl',
    '5xl': 'max-w-5xl',
    '6xl': 'max-w-6xl',
    '7xl': 'max-w-7xl',
  }

  return (
    <section className={cn('py-24', bgClasses[bgVariant], className)} aria-labelledby={title ? headingId : undefined}>
      <div className="container mx-auto px-4">
        {title && (
          <div className={cn(maxWidthClasses[maxWidth], 'mx-auto mb-16', centered && 'text-center')}>
            <h2 id={headingId} className="text-4xl md:text-5xl font-semibold tracking-tight text-foreground mb-6 text-balance leading-tight">{title}</h2>
            {subtitle && <div className="text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">{subtitle}</div>}
          </div>
        )}
        {children}
      </div>
    </section>
  )
}
