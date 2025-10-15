import { IconPlate } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import type { LucideIcon } from 'lucide-react'

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
    <div className={cn('rounded-lg border bg-card p-8', className)}>
      <div className="mb-4 flex items-center gap-3">
        <IconPlate icon={Icon} size="lg" tone="primary" />
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
