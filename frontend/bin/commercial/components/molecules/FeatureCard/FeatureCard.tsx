import type { LucideIcon } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface FeatureCardProps {
  /** Lucide icon component */
  icon: LucideIcon
  /** Card title */
  title: string
  /** Card description */
  description: string
  /** Icon background color (Tailwind class) */
  iconColor?: string
  /** Enable hover effect */
  hover?: boolean
  /** Card size variant */
  size?: 'sm' | 'md' | 'lg'
  /** Additional CSS classes */
  className?: string
  /** Optional footer content (e.g., micro-metrics) */
  children?: React.ReactNode
  /** Optional bullet points list */
  bullets?: string[]
  /** Optional mini-stat for header */
  stat?: { label: string; value: string }
  /** Optional ID for anchor linking */
  id?: string
}

export function FeatureCard({
  icon: Icon,
  title,
  description,
  iconColor = 'primary',
  hover = false,
  size = 'md',
  className,
  children,
  bullets,
  stat,
  id,
}: FeatureCardProps) {
  const sizeClasses = {
    sm: 'p-4 space-y-2',
    md: 'p-6 space-y-3',
    lg: 'p-8 space-y-4',
  }

  const iconSizeClasses = {
    sm: 'h-8 w-8',
    md: 'h-10 w-10',
    lg: 'h-12 w-12',
  }

  const iconInnerSizeClasses = {
    sm: 'h-4 w-4',
    md: 'h-5 w-5',
    lg: 'h-6 w-6',
  }

  const titleSizeClasses = {
    sm: 'text-sm',
    md: 'text-base',
    lg: 'text-lg',
  }

  const descriptionSizeClasses = {
    sm: 'text-xs',
    md: 'text-sm',
    lg: 'text-base',
  }

  const colorClasses = {
    primary: { bg: 'bg-primary/10', text: 'text-primary' },
    'chart-1': { bg: 'bg-chart-1/10', text: 'text-chart-1' },
    'chart-2': { bg: 'bg-chart-2/10', text: 'text-chart-2' },
    'chart-3': { bg: 'bg-chart-3/10', text: 'text-chart-3' },
    'chart-4': { bg: 'bg-chart-4/10', text: 'text-chart-4' },
    'chart-5': { bg: 'bg-chart-5/10', text: 'text-chart-5' },
  }

  const colors = colorClasses[iconColor as keyof typeof colorClasses] || colorClasses.primary

  // Map iconColor to border accent color
  const borderAccentClasses = {
    primary: 'border-t-primary',
    'chart-1': 'border-t-chart-1',
    'chart-2': 'border-t-chart-2',
    'chart-3': 'border-t-chart-3',
    'chart-4': 'border-t-chart-4',
    'chart-5': 'border-t-chart-5',
  }
  const borderAccent = borderAccentClasses[iconColor as keyof typeof borderAccentClasses] || borderAccentClasses.primary

  return (
    <div
      id={id}
      className={cn(
        'bg-card border border-border rounded-lg flex flex-col border-t-2',
        borderAccent,
        sizeClasses[size],
        hover && 'transition-all hover:border-primary/50 hover:bg-card/80',
        className,
      )}
    >
      <div className="flex items-start justify-between gap-4">
        <div className={cn('rounded-lg flex items-center justify-center shrink-0', iconSizeClasses[size], colors.bg)} aria-hidden="true">
          <Icon aria-hidden="true" focusable="false" className={cn(iconInnerSizeClasses[size], colors.text)} />
        </div>
        {stat && (
          <div className="text-right">
            <div className="text-xs text-muted-foreground">{stat.label}</div>
            <div className="text-sm font-semibold text-foreground">{stat.value}</div>
          </div>
        )}
      </div>
      <h3 id={id ? `${id}-title` : undefined} className={cn('font-semibold text-card-foreground', titleSizeClasses[size])}>
        {title}
      </h3>
      <p className={cn('text-muted-foreground leading-6', descriptionSizeClasses[size])} aria-describedby={id ? `${id}-title` : undefined}>
        {description}
      </p>
      {bullets && bullets.length > 0 && (
        <ul role="list" className="mt-4 space-y-2 text-sm">
          {bullets.map((bullet, index) => (
            <li key={index} className="flex items-start gap-2">
              <span className="text-ring/60 mt-1.5 block h-1 w-1 rounded-full bg-current shrink-0" aria-hidden="true" />
              <span className="text-muted-foreground">{bullet}</span>
            </li>
          ))}
        </ul>
      )}
      {children && <div className="mt-auto pt-2">{children}</div>}
    </div>
  )
}
