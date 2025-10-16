import { Button, Card, CardContent, CardHeader, CardTitle } from '@rbee/ui/atoms'
import { FeatureInfoCard } from '@rbee/ui/molecules/FeatureInfoCard'
import { SectionContainer } from '@rbee/ui/molecules/SectionContainer'
import { StepListItem } from '@rbee/ui/molecules/StepListItem'
import { cn } from '@rbee/ui/utils'
import Link from 'next/link'
import type { ReactNode } from 'react'

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

export type SolutionFeature = {
  icon: ReactNode
  title: string
  body: string
  badge?: string | ReactNode
}

export type SolutionStep = {
  title: string
  body: string
}

export type EarningRow = {
  model: string
  meta: string
  value: string
  note?: string
}

export type EnterpriseSolutionTemplateProps = {
  id?: string
  kicker?: string
  eyebrowIcon?: ReactNode
  title: string
  subtitle?: string
  features: SolutionFeature[]
  steps: SolutionStep[]
  earnings?: {
    title?: string
    rows: EarningRow[]
    disclaimer?: string
  }
  illustration?: ReactNode
  ctaPrimary?: { label: string; href: string; ariaLabel?: string }
  ctaSecondary?: { label: string; href: string; ariaLabel?: string }
  ctaCaption?: string
  className?: string
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function EnterpriseSolutionTemplate({
  kicker,
  eyebrowIcon,
  title,
  subtitle,
  features,
  steps,
  earnings,
  illustration,
  ctaPrimary,
  ctaSecondary,
  ctaCaption,
  id,
  className,
}: EnterpriseSolutionTemplateProps) {
  return (
    <div className={cn('relative border-b border-border bg-radial-glow', className)}>
      {/* Decorative illustration */}
      {illustration}

      <SectionContainer
        title={title}
        description={subtitle}
        kicker={
          kicker ? (
            <span className="inline-flex items-center gap-2">
              {eyebrowIcon}
              <span>{kicker}</span>
            </span>
          ) : undefined
        }
        headingId={id ? `${id}-h2` : undefined}
        align="center"
        maxWidth="7xl"
        paddingY="2xl"
        bgVariant="default"
        className="relative z-10"
      >
        {/* Feature Tiles */}
        <div className="mb-12 mt-12 grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          {features.map((feature, idx) => (
            <FeatureInfoCard
              key={idx}
              icon={feature.icon}
              title={feature.title}
              body={feature.body}
              tag={typeof feature.badge === 'string' ? feature.badge : undefined}
              delay="[animation-delay:100ms]"
            />
          ))}
        </div>

        {/* Timeline + Aside */}
        <div className="grid gap-8 lg:grid-cols-[1.2fr_0.8fr] lg:gap-12">
          {/* Steps Card */}
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

          {/* Earnings aside */}
          {earnings && (
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
          )}
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
      </SectionContainer>
    </div>
  )
}
