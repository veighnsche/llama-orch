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
}

export function FeatureCard({
  icon: Icon,
  title,
  description,
  iconColor = 'primary',
  hover = false,
  size = 'md',
  className,
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
    sm: 'text-base',
    md: 'text-lg',
    lg: 'text-xl',
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

  return (
    <div
      className={cn(
        'bg-card border border-border rounded-lg',
        sizeClasses[size],
        hover && 'transition-all hover:border-primary/50 hover:bg-card/80',
        className,
      )}
    >
      <div className={cn('rounded-lg flex items-center justify-center', iconSizeClasses[size], colors.bg)}>
        <Icon className={cn(iconInnerSizeClasses[size], colors.text)} />
      </div>
      <h3 className={cn('font-bold text-card-foreground', titleSizeClasses[size])}>{title}</h3>
      <p className={cn('text-muted-foreground leading-relaxed', descriptionSizeClasses[size])}>{description}</p>
    </div>
  )
}
