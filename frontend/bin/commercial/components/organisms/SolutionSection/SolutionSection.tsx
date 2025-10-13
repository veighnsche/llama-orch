import { type ReactNode } from 'react'
import Link from 'next/link'
import { cn } from '@/lib/utils'
import { Button } from '@/components/atoms/Button/Button'

export type Feature = {
  icon: ReactNode
  title: string
  body: string
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
  title: string
  subtitle?: string
  features: Feature[]
  steps: Step[]
  earnings?: Earnings
  ctaPrimary?: { label: string; href: string; ariaLabel?: string }
  ctaSecondary?: { label: string; href: string; ariaLabel?: string }
  id?: string
  className?: string
}

export function SolutionSection({
  kicker,
  title,
  subtitle,
  features,
  steps,
  earnings,
  ctaPrimary,
  ctaSecondary,
  id,
  className,
}: SolutionSectionProps) {
  return (
    <section
      id={id}
      className={cn(
        'border-b border-border bg-[radial-gradient(60rem_40rem_at_10%_-20%,theme(colors.primary/10),transparent)] px-6 py-20 lg:py-28',
        className,
      )}
    >
      <div className="mx-auto max-w-7xl">
        {/* Header */}
        <div className="mb-12 text-center animate-in fade-in slide-in-from-bottom-2 duration-500 motion-reduce:animate-none">
          {kicker && <p className="mb-2 text-sm font-medium text-primary/80">{kicker}</p>}
          <h2 className="text-balance text-4xl font-extrabold tracking-tight lg:text-5xl">{title}</h2>
          {subtitle && (
            <p className="mx-auto mt-4 max-w-2xl text-pretty text-lg leading-snug text-muted-foreground lg:text-xl">
              {subtitle}
            </p>
          )}
        </div>

        {/* Feature Tiles */}
        <div className="mb-12 mt-12 grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          {features.map((feature, idx) => (
            <div
              key={idx}
              className={cn(
                'animate-in fade-in slide-in-from-bottom-2 motion-reduce:animate-none',
                'rounded-2xl border border-border/60 bg-card/60 p-6 text-center backdrop-blur supports-[backdrop-filter]:bg-background/60',
                idx === 0 && 'delay-75',
                idx === 1 && 'delay-150',
                idx === 2 && 'delay-200',
                idx === 3 && 'delay-300',
              )}
            >
              <div className="mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-xl bg-primary/10 text-primary">
                {feature.icon}
              </div>
              <h3 className="text-base font-semibold">{feature.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{feature.body}</p>
            </div>
          ))}
        </div>

        {/* Timeline + Earnings */}
        <div className="rounded-2xl border border-border bg-gradient-to-b from-card to-background p-8 sm:p-10">
          <div className="grid gap-12 lg:grid-cols-[1.1fr_0.9fr]">
            {/* Timeline */}
            <div>
              <h3 className="mb-6 text-2xl font-bold">How It Works</h3>
              <div className="relative">
                {/* Vertical line */}
                <div className="absolute bottom-0 left-4 top-0 w-px bg-border" aria-hidden="true" />

                {steps.map((step, idx) => (
                  <div key={idx} className={cn('relative mb-6 space-y-1 pl-12 last:mb-0')}>
                    <div className="absolute left-0 top-0 flex h-8 w-8 items-center justify-center rounded-full bg-primary text-sm font-bold text-primary-foreground">
                      {idx + 1}
                    </div>
                    <div className="font-medium">{step.title}</div>
                    <div className="text-sm leading-relaxed text-muted-foreground">{step.body}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Earnings Card */}
            {earnings && (
              <div className="animate-in fade-in slide-in-from-bottom-2 delay-150 motion-reduce:animate-none lg:ml-auto">
                <div className="w-full max-w-md rounded-2xl border border-border bg-background p-6">
                  <div className="mb-4 text-sm font-medium text-muted-foreground">
                    {earnings.title || 'Example Earnings'}
                  </div>
                  <div className="mb-6 space-y-4">
                    {earnings.rows.map((row, idx) => (
                      <div key={idx} className="flex items-center justify-between">
                        <div>
                          <div className="text-sm font-medium">{row.model}</div>
                          <div className="text-xs text-muted-foreground">{row.meta}</div>
                        </div>
                        <div className="text-right">
                          <div className="tabular-nums text-lg font-bold text-primary">{row.value}</div>
                          {row.note && (
                            <div className="text-[11px] text-muted-foreground tabular-nums">{row.note}</div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>

                  {earnings.disclaimer && (
                    <div className="rounded-lg border border-primary/20 bg-primary/10 p-4">
                      <div className="text-xs text-primary">{earnings.disclaimer}</div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* CTA Bar */}
        {(ctaPrimary || ctaSecondary) && (
          <div className="mt-10 text-center animate-in fade-in slide-in-from-bottom-2 delay-200 motion-reduce:animate-none">
            {ctaPrimary && (
              <Button
                asChild
                className="transition-transform active:scale-[0.98]"
                aria-label={ctaPrimary.ariaLabel || ctaPrimary.label}
              >
                <Link href={ctaPrimary.href}>{ctaPrimary.label}</Link>
              </Button>
            )}
            {ctaSecondary && (
              <Button
                asChild
                variant="outline"
                className="ml-0 mt-3 transition-transform active:scale-[0.98] sm:ml-3 sm:mt-0"
                aria-label={ctaSecondary.ariaLabel || ctaSecondary.label}
              >
                <Link href={ctaSecondary.href}>{ctaSecondary.label}</Link>
              </Button>
            )}
          </div>
        )}
      </div>
    </section>
  )
}
