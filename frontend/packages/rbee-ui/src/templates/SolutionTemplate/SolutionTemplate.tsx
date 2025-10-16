import { Button, Card, CardContent, CardHeader, CardTitle } from '@rbee/ui/atoms'
import { BeeArchitecture, type BeeTopology, FeatureInfoCard } from '@rbee/ui/molecules'
import { StepListItem } from '@rbee/ui/molecules/StepListItem'
import { cn } from '@rbee/ui/utils'
import Link from 'next/link'
import type { ReactNode } from 'react'

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

export interface SolutionTemplateProps {
  /** Feature cards to display */
  features: Feature[]
  /** Optional steps for "How It Works" section */
  steps?: Step[]
  /** Optional earnings/metrics sidebar */
  earnings?: Earnings
  /** Optional custom aside content (overrides earnings) */
  aside?: ReactNode
  /** Optional BeeArchitecture topology diagram */
  topology?: BeeTopology
  /** Optional decorative illustration */
  illustration?: ReactNode
  /** Primary CTA button */
  ctaPrimary?: { label: string; href: string; ariaLabel?: string }
  /** Secondary CTA button */
  ctaSecondary?: { label: string; href: string; ariaLabel?: string }
  /** Caption text below CTAs */
  ctaCaption?: string
  /** Custom class name */
  className?: string
}

export function SolutionTemplate({
  features,
  steps,
  earnings,
  aside,
  topology,
  illustration,
  ctaPrimary,
  ctaSecondary,
  ctaCaption,
  className,
}: SolutionTemplateProps) {
  return (
    <div className={cn('relative', className)}>
      {/* Decorative illustration */}
      {illustration}

      <div className="relative z-10">
        {/* Feature Tiles */}
        <div className="mb-12 grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          {features.map((feature, idx) => (
            <FeatureInfoCard
              key={idx}
              icon={feature.icon}
              title={feature.title}
              body={feature.body}
              tag={typeof feature.badge === 'string' ? feature.badge : undefined}
              tone="neutral"
              size="sm"
              delay="[animation-delay:100ms]"
            />
          ))}
        </div>

        {/* Optional BeeArchitecture Diagram */}
        {topology && <BeeArchitecture topology={topology} />}

        {/* Timeline + Aside (only if steps or aside/earnings provided) */}
        {(steps || aside || earnings) && (
          <div className="mt-12 grid gap-8 lg:grid-cols-[1.2fr_0.8fr] lg:gap-12">
            {/* Steps Card */}
            {steps && (
              <Card className="animate-in fade-in-50 bg-card/40 [animation-delay:150ms]">
                <CardHeader>
                  <CardTitle className="text-2xl">How It Works</CardTitle>
                </CardHeader>
                <CardContent>
                  <ol className="space-y-6">
                    {steps.map((step, idx) => (
                      <StepListItem key={idx} number={idx + 1} title={step.title} body={step.body} />
                    ))}
                  </ol>
                </CardContent>
              </Card>
            )}

            {/* Aside (custom or earnings) */}
            {aside ? (
              aside
            ) : earnings ? (
              <div className="animate-in fade-in-50 slide-in-from-right-2 [animation-delay:200ms] lg:sticky lg:top-24 lg:self-start">
                <div className="rounded-2xl border bg-card p-6">
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
        )}

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
    </div>
  )
}
