import type { LucideIcon } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/atoms/Button/Button'
import { ArrowRight } from 'lucide-react'
import Link from 'next/link'

export interface AudienceCardProps {
  icon: LucideIcon
  category: string
  title: string
  description: string
  features: string[]
  href: string
  ctaText: string
  color: string
  className?: string
  imageSlot?: React.ReactNode
  badgeSlot?: React.ReactNode
  decisionLabel?: string
}

export function AudienceCard({
  icon: Icon,
  category,
  title,
  description,
  features,
  href,
  ctaText,
  color,
  className,
  imageSlot,
  badgeSlot,
  decisionLabel,
}: AudienceCardProps) {
  const colorClasses = {
    primary: {
      hoverBorder: 'hover:border-primary/50',
      gradient: 'from-primary/0 via-primary/0 to-primary/0 group-hover:from-primary/5 group-hover:via-primary/10',
      iconBg: 'from-primary to-primary',
      text: 'text-primary',
      button: 'bg-primary',
    },
    'chart-1': {
      hoverBorder: 'hover:border-chart-1/50',
      gradient: 'from-chart-1/0 via-chart-1/0 to-chart-1/0 group-hover:from-chart-1/5 group-hover:via-chart-1/10',
      iconBg: 'from-chart-1 to-chart-1',
      text: 'text-chart-1',
      button: 'bg-chart-1',
    },
    'chart-2': {
      hoverBorder: 'hover:border-chart-2/50',
      gradient: 'from-chart-2/0 via-chart-2/0 to-chart-2/0 group-hover:from-chart-2/5 group-hover:via-chart-2/10',
      iconBg: 'from-chart-2 to-chart-2',
      text: 'text-chart-2',
      button: 'bg-chart-2',
    },
    'chart-3': {
      hoverBorder: 'hover:border-chart-3/50',
      gradient: 'from-chart-3/0 via-chart-3/0 to-chart-3/0 group-hover:from-chart-3/5 group-hover:via-chart-3/10',
      iconBg: 'from-chart-3 to-chart-3',
      text: 'text-chart-3',
      button: 'bg-chart-3',
    },
    'chart-4': {
      hoverBorder: 'hover:border-chart-4/50',
      gradient: 'from-chart-4/0 via-chart-4/0 to-chart-4/0 group-hover:from-chart-4/5 group-hover:via-chart-4/10',
      iconBg: 'from-chart-4 to-chart-4',
      text: 'text-chart-4',
      button: 'bg-chart-4',
    },
    'chart-5': {
      hoverBorder: 'hover:border-chart-5/50',
      gradient: 'from-chart-5/0 via-chart-5/0 to-chart-5/0 group-hover:from-chart-5/5 group-hover:via-chart-5/10',
      iconBg: 'from-chart-5 to-chart-5',
      text: 'text-chart-5',
      button: 'bg-chart-5',
    },
  }

  const colors = colorClasses[color as keyof typeof colorClasses] || colorClasses.primary
  const descriptionId = `${title.toLowerCase().replace(/\s+/g, '-')}-description`

  return (
    <div className="flex flex-col">
      {/* Optional decision label above card */}
      {decisionLabel && (
        <div className={cn('mb-3 text-sm font-medium', colors.text)}>
          {decisionLabel}
        </div>
      )}

      <div
        className={cn(
          'group relative flex flex-col overflow-hidden border-border bg-card backdrop-blur-sm transition-all duration-300 hover:scale-[1.02] rounded-lg border',
          colors.hoverBorder,
          className,
        )}
      >
      <div
        className={cn(
          'absolute inset-0 -z-10 bg-gradient-to-br opacity-0 transition-all duration-500 group-hover:to-transparent group-hover:opacity-100',
          colors.gradient,
        )}
      />

      {/* Icons side-by-side at top */}
      <div className="mb-6 flex items-center gap-3">
        <div
          className={cn(
            'flex h-14 w-14 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br shadow-lg',
            colors.iconBg,
          )}
        >
          <Icon className="h-7 w-7 text-primary-foreground" aria-hidden="true" />
        </div>
        {imageSlot && (
          <div className="h-14 w-14 shrink-0 overflow-hidden rounded-lg bg-card ring-1 ring-border">
            {imageSlot}
          </div>
        )}
      </div>

      <div className={cn('mb-2 text-sm font-medium uppercase tracking-wider', colors.text)}>{category}</div>
      <h3 className="mb-3 text-2xl font-semibold text-card-foreground">{title}</h3>
      <p id={descriptionId} className="mb-6 text-sm leading-relaxed text-muted-foreground sm:text-base">
        {description}
      </p>

      <ul className="mb-8 space-y-3 text-sm text-muted-foreground sm:text-base">
        {features.map((feature, index) => (
          <li key={index} className="flex items-start gap-2">
            <span className={cn('mt-1', colors.text)} aria-hidden="true">
              →
            </span>
            <span>{feature}</span>
          </li>
        ))}
      </ul>

      {/* Optional badge slot */}
      {badgeSlot && <div className="mb-4">{badgeSlot}</div>}

      {/* Spacer to push button to bottom */}
      <div className="flex-1" />

      <Link href={href}>
        <Button className={cn('w-full', colors.button)} aria-describedby={descriptionId}>
          {ctaText}
          <ArrowRight className="ml-2 h-4 w-4" aria-hidden="true" />
        </Button>
      </Link>
      </div>
    </div>
  )
}
