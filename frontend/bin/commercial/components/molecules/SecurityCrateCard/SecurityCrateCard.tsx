import type { LucideIcon } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface SecurityCrateCardProps {
  icon: LucideIcon
  title: string
  subtitle: string
  description: string
  features: string[]
  className?: string
}

export function SecurityCrateCard({
  icon: Icon,
  title,
  subtitle,
  description,
  features,
  className,
}: SecurityCrateCardProps) {
  return (
    <div className={cn('rounded-lg border border-border bg-card p-8', className)}>
      <div className="mb-4 flex items-center gap-3">
        <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
          <Icon className="h-6 w-6 text-primary" />
        </div>
        <div>
          <h3 className="text-xl font-bold text-foreground">{title}</h3>
          <p className="text-sm text-muted-foreground">{subtitle}</p>
        </div>
      </div>

      <p className="mb-4 leading-relaxed text-muted-foreground">{description}</p>

      <div className="space-y-2">
        {features.map((feature, index) => (
          <div key={index} className="flex items-start gap-2 text-sm text-muted-foreground">
            <span className="text-chart-3">✓</span>
            <span>{feature}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
