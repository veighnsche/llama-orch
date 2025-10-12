import { cn } from '@/lib/utils'
import { Button } from '@/components/atoms/Button/Button'
import { Check } from 'lucide-react'

export interface PricingTierProps {
  title: string
  price: string | number
  period?: string
  features: string[]
  ctaText: string
  ctaVariant?: 'default' | 'outline'
  highlighted?: boolean
  badge?: string
  className?: string
}

export function PricingTier({
  title,
  price,
  period,
  features,
  ctaText,
  ctaVariant = 'default',
  highlighted = false,
  badge,
  className,
}: PricingTierProps) {
  return (
    <div
      className={cn(
        'bg-card rounded-lg p-8 space-y-6',
        highlighted ? 'border-2 border-primary' : 'border-2 border-border',
        className,
      )}
    >
      {badge && (
        <div className="inline-flex items-center px-3 py-1 bg-primary/10 text-primary text-xs font-medium rounded-full">
          {badge}
        </div>
      )}

      <div>
        <h3 className="text-2xl font-bold text-foreground">{title}</h3>
        <div className="mt-4">
          <span className="text-4xl font-bold text-foreground">{price}</span>
          {period && <span className="text-muted-foreground ml-2">{period}</span>}
        </div>
      </div>

      <ul className="space-y-3">
        {features.map((feature, index) => (
          <li key={index} className="flex items-start gap-2">
            <Check className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5" />
            <span className="text-muted-foreground">{feature}</span>
          </li>
        ))}
      </ul>

      <Button variant={ctaVariant} className="w-full">
        {ctaText}
      </Button>
    </div>
  )
}
