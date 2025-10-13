import { cn } from '@/lib/utils'
import { Button } from '@/components/atoms/Button/Button'
import { Check } from 'lucide-react'
import Link from 'next/link'

export interface PricingTierProps {
  title: string
  price?: string | number
  priceYearly?: string | number
  currency?: 'USD' | 'EUR' | 'GBP' | 'CUSTOM'
  period?: string
  features: string[]
  ctaText: string
  ctaHref?: string
  ctaVariant?: 'default' | 'outline'
  highlighted?: boolean
  badge?: string
  footnote?: string
  className?: string
  isYearly?: boolean
  saveBadge?: string
}

export function PricingTier({
  title,
  price,
  priceYearly,
  currency = 'CUSTOM',
  period,
  features,
  ctaText,
  ctaHref,
  ctaVariant = 'default',
  highlighted = false,
  badge,
  footnote,
  className,
  isYearly = false,
  saveBadge,
}: PricingTierProps) {
  const displayPrice = isYearly && priceYearly ? priceYearly : price
  const displayPeriod = isYearly && priceYearly ? '/year' : period

  const titleId = `pricing-${title.toLowerCase().replace(/\s+/g, '-')}`

  return (
    <section
      aria-labelledby={titleId}
      className={cn(
        'flex flex-col h-full',
        'bg-card/90 backdrop-blur supports-[backdrop-filter]:bg-card/75',
        'rounded-2xl p-7 md:p-8 border-2',
        'motion-safe:hover:translate-y-[-2px] motion-safe:hover:shadow-lg motion-safe:transition-all',
        highlighted
          ? 'border-primary shadow-[0_0_0_1px_var(--primary)] shadow-primary/20 hover:shadow-primary/30 ring-1 ring-primary/30'
          : 'border-border',
        className,
      )}
    >
      {badge && (
        <div className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] bg-primary/15 text-primary font-semibold w-fit mb-4 h-[22px]">
          {badge}
        </div>
      )}
      {!badge && <div className="h-[22px] mb-4" aria-hidden="true" />}

      <div>
        <h3 id={titleId} className="text-xl font-semibold tracking-tight text-foreground">
          {title}
        </h3>
        <div className="mt-3">
          <span className="text-4xl font-extrabold text-foreground">{displayPrice}</span>
          {displayPeriod && <span className="text-sm text-muted-foreground ml-2">{displayPeriod}</span>}
          {isYearly && saveBadge && (
            <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded-full text-[10px] bg-chart-3/15 text-chart-3 font-medium">
              {saveBadge}
            </span>
          )}
        </div>
      </div>

      <ul className="mt-5 space-y-2 text-sm" role="list" aria-label={`${title} features`}>
        {features.map((feature, index) => (
          <li key={index} className="flex items-start gap-2">
            <Check className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5" aria-hidden="true" />
            <span className="text-muted-foreground">{feature}</span>
          </li>
        ))}
      </ul>

      <div className="mt-auto pt-6">
        {ctaHref ? (
          <Button variant={ctaVariant} className="w-full" asChild>
            <Link href={ctaHref} aria-label={`${ctaText} for ${title} plan`}>
              {ctaText}
            </Link>
          </Button>
        ) : (
          <Button variant={ctaVariant} className="w-full" aria-label={`${ctaText} for ${title} plan`}>
            {ctaText}
          </Button>
        )}
        {footnote && <p className="text-[12px] text-muted-foreground/90 mt-2 text-center">{footnote}</p>}
      </div>
    </section>
  )
}
