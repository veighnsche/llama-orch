import { IconPlate } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import type * as React from 'react'

export interface StatItem {
  /** Stat value (e.g., "100%", "€50-200", "24/7") */
  value: string | number
  /** Stat label/description */
  label: string
  /** Optional rendered icon component */
  icon?: React.ReactNode
  /** Optional help text for accessibility */
  helpText?: string
  /** Optional color tone for the value */
  valueTone?: 'default' | 'primary' | 'foreground'
}

export interface StatsGridProps {
  /** Array of stat items to display */
  stats: StatItem[]
  /** Visual variant */
  variant?: 'pills' | 'tiles' | 'cards' | 'inline' | 'strip'
  /** Number of columns (responsive) */
  columns?: 2 | 3 | 4
  /** Additional CSS classes */
  className?: string
}

/**
 * StatsGrid molecule - unified stat display component
 * Consolidates stat pills, tiles, cards, and inline stats patterns
 * Used across heroes, CTAs, testimonials, and feature sections
 */
export function StatsGrid({ stats, variant = 'cards', columns = 3, className }: StatsGridProps) {
  const gridClasses = {
    2: 'grid-cols-1 sm:grid-cols-2',
    3: 'grid-cols-1 sm:grid-cols-3',
    4: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-4',
  }

  if (variant === 'pills') {
    return (
      <div className={cn('grid gap-3', gridClasses[columns], className)}>
        {stats.map((stat, idx) => (
          <div
            key={idx}
            className="group rounded border/70 bg-background/60 p-4 backdrop-blur transition-all supports-[backdrop-filter]:bg-background/50 hover:border-primary/40 hover:bg-background/80"
          >
            <div className="flex items-center gap-2.5">
              {stat.icon && (
                <IconPlate
                  icon={stat.icon}
                  size="md"
                  tone="primary"
                  className="transition-colors group-hover:bg-primary/20"
                />
              )}
              <div>
                <div
                  className={cn(
                    'tabular-nums text-xl font-bold',
                    stat.valueTone === 'primary' ? 'text-primary' : 'text-foreground',
                  )}
                >
                  {stat.value}
                </div>
                <div className="text-xs text-muted-foreground">{stat.label}</div>
              </div>
            </div>
            {stat.helpText && <span className="sr-only">{stat.helpText}</span>}
          </div>
        ))}
      </div>
    )
  }

  if (variant === 'tiles') {
    return (
      <div className={cn('grid gap-4', gridClasses[columns], className)}>
        {stats.map((stat, idx) => (
          <div
            key={idx}
            className="h-full rounded-md border/70 bg-card/50 p-5 transition-all duration-200 hover:border-border hover:bg-card/70"
            role="group"
            aria-label={`${stat.value} ${stat.label}`}
          >
            {stat.icon && <div className="mb-3">{stat.icon}</div>}
            <div
              className={cn(
                'mb-2 text-3xl font-bold',
                stat.valueTone === 'foreground' ? 'text-foreground' : 'text-primary',
              )}
            >
              {stat.value}
            </div>
            <div className="text-sm leading-snug text-muted-foreground">{stat.label}</div>
            {stat.helpText && <span className="sr-only">{stat.helpText}</span>}
          </div>
        ))}
      </div>
    )
  }

  if (variant === 'inline') {
    return (
      <div className={cn('grid gap-5 text-sm text-muted-foreground', gridClasses[columns], className)}>
        {stats.map((stat, idx) => (
          <div key={idx} className="rounded-md border/60 bg-card/40 p-4">
            {stat.icon && (
              <div className="mb-2 flex justify-center">
                <IconPlate icon={stat.icon} size="sm" tone="primary" />
              </div>
            )}
            <div className={cn('font-medium', stat.valueTone === 'primary' ? 'text-primary' : 'text-foreground')}>
              {stat.value}
            </div>
            <div className="text-xs">{stat.label}</div>
            {stat.helpText && <span className="sr-only">{stat.helpText}</span>}
          </div>
        ))}
      </div>
    )
  }

  if (variant === 'strip') {
    return (
      <div className={cn('grid gap-6 text-center text-sm', gridClasses[columns], className)}>
        {stats.map((stat, idx) => (
          <div key={idx}>
            <div className="font-semibold text-foreground">{stat.value}</div>
            <div className="text-muted-foreground">{stat.label}</div>
            {stat.helpText && <span className="sr-only">{stat.helpText}</span>}
          </div>
        ))}
      </div>
    )
  }

  // Default: 'cards' variant
  return (
    <div className={cn('grid gap-4', gridClasses[columns], className)}>
      {stats.map((stat, idx) => (
        <div key={idx} className="text-center">
          {stat.icon && <div className="mb-2 flex justify-center">{stat.icon}</div>}
          <div
            className={cn(
              'mb-2 text-4xl font-bold',
              stat.valueTone === 'foreground' ? 'text-foreground' : 'text-primary',
            )}
          >
            {stat.value}
          </div>
          <div className="text-sm text-muted-foreground">{stat.label}</div>
          {stat.helpText && <span className="sr-only">{stat.helpText}</span>}
        </div>
      ))}
    </div>
  )
}
