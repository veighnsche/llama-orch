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
}: AudienceCardProps) {
  return (
    <div
      className={cn(
        'group relative overflow-hidden border-border bg-card p-8 backdrop-blur-sm transition-all duration-300 hover:scale-[1.02] rounded-lg border',
        `hover:border-${color}/50`,
        className
      )}
    >
      <div
        className={cn(
          'absolute inset-0 -z-10 bg-gradient-to-br opacity-0 transition-all duration-500',
          `from-${color}/0 via-${color}/0 to-${color}/0`,
          `group-hover:from-${color}/5 group-hover:via-${color}/10 group-hover:to-transparent group-hover:opacity-100`
        )}
      />

      <div
        className={cn(
          'mb-6 flex h-14 w-14 items-center justify-center rounded-xl bg-gradient-to-br shadow-lg',
          `from-${color} to-${color}`
        )}
      >
        <Icon className="h-7 w-7 text-primary-foreground" />
      </div>

      <div
        className={cn(
          'mb-2 text-sm font-medium uppercase tracking-wider',
          `text-${color}`
        )}
      >
        {category}
      </div>
      <h3 className="mb-3 text-2xl font-semibold text-card-foreground">
        {title}
      </h3>
      <p className="mb-6 leading-relaxed text-muted-foreground">
        {description}
      </p>

      <ul className="mb-8 space-y-3 text-sm text-muted-foreground">
        {features.map((feature, index) => (
          <li key={index} className="flex items-start gap-2">
            <span className={cn('mt-1', `text-${color}`)}>→</span>
            <span>{feature}</span>
          </li>
        ))}
      </ul>

      <Link href={href}>
        <Button className={cn('w-full', `bg-${color}`)}>
          {ctaText}
          <ArrowRight className="ml-2 h-4 w-4" />
        </Button>
      </Link>
    </div>
  )
}
