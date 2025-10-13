import { cn } from '@/lib/utils'
import type { ReactNode } from 'react'

export interface SectionContainerProps {
  /** Section title */
  title: string | ReactNode
  /** Optional subtitle */
  subtitle?: string
  /** Background variant */
  bgVariant?: 'background' | 'secondary' | 'card'
  /** Center the content */
  centered?: boolean
  /** Maximum width of content */
  maxWidth?: 'xl' | '2xl' | '3xl' | '4xl' | '5xl' | '6xl' | '7xl'
  /** Section content */
  children: ReactNode
  /** Additional CSS classes */
  className?: string
}

export function SectionContainer({
  title,
  subtitle,
  bgVariant = 'background',
  centered = true,
  maxWidth = '4xl',
  children,
  className,
}: SectionContainerProps) {
  const bgClasses = {
    background: 'bg-background',
    secondary: 'bg-secondary',
    card: 'bg-card',
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
    <section className={cn('py-24', bgClasses[bgVariant], className)}>
      <div className="container mx-auto px-4">
        <div className={cn(maxWidthClasses[maxWidth], 'mx-auto mb-16', centered && 'text-center')}>
          <h2 className="text-4xl md:text-5xl font-semibold tracking-tight text-foreground mb-6 text-balance leading-tight">{title}</h2>
          {subtitle && <p className="text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">{subtitle}</p>}
        </div>
        {children}
      </div>
    </section>
  )
}
