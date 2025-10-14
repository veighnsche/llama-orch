import { type ReactNode } from 'react'
import Link from 'next/link'
import { cn } from '@rbee/ui/utils'
import { Button } from '@rbee/ui/atoms/Button'

export type Feature = {
  icon: ReactNode
  title: string
  body: string
  badge?: string | ReactNode
}

export type Step = {
  title: string
  body: string
}

export type EarningRow = {
  model: string
  meta: string
  value: string
  note?: string
}

export type Earnings = {
  title?: string
  rows: EarningRow[]
  disclaimer?: string
  imageSrc?: string
}

export interface SolutionSectionProps {
  kicker?: string
  eyebrowIcon?: ReactNode
  title: string
  subtitle?: string
  features: Feature[]
  steps: Step[]
  earnings?: Earnings
  aside?: ReactNode
  illustration?: ReactNode
  ctaPrimary?: { label: string; href: string; ariaLabel?: string }
  ctaSecondary?: { label: string; href: string; ariaLabel?: string }
  ctaCaption?: string
  id?: string
  className?: string
}

export function SolutionSection({
  kicker,
  eyebrowIcon,
  title,
  subtitle,
  features,
  steps,
  earnings,
  aside,
  illustration,
  ctaPrimary,
  ctaSecondary,
  ctaCaption,
  id,
  className,
}: SolutionSectionProps) {
  return (
    <section
      id={id}
      aria-labelledby={id ? `${id}-h2` : undefined}
      className={cn(
        'relative border-b border-border bg-radial-glow px-6 py-24 lg:py-28',
        className,
      )}
    >
      {/* Decorative illustration */}
      {illustration}

      <div className="relative z-10 mx-auto max-w-7xl">
        {/* Header */}
        <div className="mb-12 text-center animate-in fade-in-50 slide-in-from-bottom-2 duration-500 motion-reduce:animate-none">
          {kicker && (
            <p className="mb-2 inline-flex items-center gap-2 text-sm font-medium text-primary/80">
              {eyebrowIcon}
              <span>{kicker}</span>
            </p>
          )}
          <h2 id={id ? `${id}-h2` : undefined} className="text-balance text-4xl font-extrabold tracking-tight lg:text-5xl">
            {title}
          </h2>
          {subtitle && (
            <p className="mx-auto mt-4 max-w-2xl text-pretty text-lg leading-snug text-foreground/85 lg:text-xl">
              {subtitle}
            </p>
          )}
        </div>

        {/* Feature Tiles */}
        <div className="animate-in fade-in-50 mb-12 mt-12 grid gap-6 [animation-delay:100ms] md:grid-cols-2 lg:grid-cols-4">
          {features.map((feature, idx) => (
            <div
              key={idx}
              className="flex h-full items-start gap-4 rounded-2xl border border-border bg-card/50 p-6"
            >
              <div className="flex-shrink-0 rounded-xl bg-primary/10 p-3 text-primary">{feature.icon}</div>
              <div className="flex-1">
                <div className="mb-1 flex items-start justify-between gap-2">
                  <h3 className="text-base font-semibold text-foreground">{feature.title}</h3>
                  {feature.badge && (
                    <span className="flex-shrink-0 rounded-full border border-primary/20 bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary">
                      {feature.badge}
                    </span>
                  )}
                </div>
                <p className="text-sm leading-relaxed text-muted-foreground">{feature.body}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Timeline + Aside */}
        <div className="grid gap-8 lg:grid-cols-[1.2fr_0.8fr] lg:gap-12">
          {/* Steps Card */}
          <div className="animate-in fade-in-50 rounded-2xl border border-border bg-card/40 p-6 [animation-delay:150ms] md:p-8">
            <h3 className="mb-6 text-2xl font-bold">How It Works</h3>
            <ol className="space-y-6" role="list">
              {steps.map((step, idx) => (
                <li
                  key={idx}
                  className="grid grid-cols-[auto_1fr] items-start gap-4"
                  aria-label={`Step ${idx + 1}: ${step.title}`}
                >
                  <div className="grid h-8 w-8 shrink-0 place-content-center rounded-full bg-primary/10 text-sm font-semibold text-primary">
                    {idx + 1}
                  </div>
                  <div>
                    <div className="mb-1 font-semibold text-foreground">{step.title}</div>
                    <div className="text-sm leading-relaxed text-muted-foreground">{step.body}</div>
                  </div>
                </li>
              ))}
            </ol>
          </div>

          {/* Aside (custom or earnings) */}
          {aside ? (
            aside
          ) : earnings ? (
            <div className="animate-in fade-in-50 slide-in-from-right-2 [animation-delay:200ms] lg:sticky lg:top-24 lg:self-start">
              <div className="rounded-2xl border border-border bg-card p-6">
                <div className="mb-4 text-sm font-semibold text-foreground">
                  {earnings.title || 'Compliance Metrics'}
                </div>
                <div className="mb-6 space-y-4">
                  {earnings.rows.map((row, idx) => (
                    <div key={idx} className="flex items-start justify-between gap-4">
                      <div className="flex-1">
                        <div className="text-sm text-muted-foreground">{row.model}</div>
                        <div className="text-xs text-muted-foreground/70">{row.meta}</div>
                      </div>
                      <div className="text-right">
                        <div className="font-semibold text-primary tabular-nums">{row.value}</div>
                        {row.note && <div className="text-xs text-muted-foreground/70 tabular-nums">{row.note}</div>}
                      </div>
                    </div>
                  ))}
                </div>

                {earnings.disclaimer && (
                  <div className="rounded-xl border border-primary/20 bg-primary/5 p-3 text-xs text-foreground/90">
                    {earnings.disclaimer}
                  </div>
                )}
              </div>
            </div>
          ) : null}
        </div>

        {/* CTA Bar */}
        {(ctaPrimary || ctaSecondary) && (
          <div className="mt-12 text-center">
            <div className="flex flex-col items-center justify-center gap-3 sm:flex-row">
              {ctaPrimary && (
                <Button
                  asChild
                  size="lg"
                  className="transition-transform active:scale-[0.98]"
                  aria-label={ctaPrimary.ariaLabel || ctaPrimary.label}
                >
                  <Link href={ctaPrimary.href}>{ctaPrimary.label}</Link>
                </Button>
              )}
              {ctaSecondary && (
                <Button
                  asChild
                  size="lg"
                  variant="outline"
                  className="transition-transform active:scale-[0.98]"
                  aria-label={ctaSecondary.ariaLabel || ctaSecondary.label}
                >
                  <Link href={ctaSecondary.href}>{ctaSecondary.label}</Link>
                </Button>
              )}
            </div>
            {ctaCaption && <p className="mt-4 text-xs text-muted-foreground">{ctaCaption}</p>}
          </div>
        )}
      </div>
    </section>
  )
}
