import { FeatureCard } from '@rbee/ui/molecules'
import type * as React from 'react'

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

export type EnterpriseFeature = {
  icon: React.ReactNode
  title: string
  intro: string
  bullets: string[]
}

export type OutcomeStat = {
  value: string
  label: string
}

export type EnterpriseFeaturesTemplateProps = {
  eyebrow: string
  heading: string
  description: string
  features: EnterpriseFeature[]
  outcomes: {
    heading: string
    stats: OutcomeStat[]
    linkText: string
    linkHref: string
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function EnterpriseFeaturesTemplate({
  eyebrow,
  heading,
  description,
  features,
  outcomes,
}: EnterpriseFeaturesTemplateProps) {
  return (
    <section
      aria-labelledby="enterprise-features-h2"
      className="relative border-b border-border bg-background px-6 py-24 overflow-hidden"
    >
      {/* Decorative Gradient */}
      <div className="pointer-events-none absolute inset-0 bg-radial-glow" aria-hidden="true" />

      <div className="relative mx-auto max-w-7xl">
        {/* Header Block */}
        <div className="mb-12 text-center animate-in fade-in-50 slide-in-from-bottom-2">
          <p className="mb-2 text-sm font-semibold uppercase tracking-wide text-primary">{eyebrow}</p>
          <h2 id="enterprise-features-h2" className="mb-4 text-4xl font-bold text-foreground">
            {heading}
          </h2>
          <p className="mx-auto max-w-3xl text-balance text-xl text-muted-foreground">
            {description}
          </p>
        </div>

        {/* Feature Grid */}
        <div className="grid gap-8 md:grid-cols-2 animate-in fade-in-50" style={{ animationDelay: '120ms' }}>
          {features.map((feature, index) => (
            <FeatureCard key={index} {...feature} />
          ))}
        </div>

        {/* Outcomes Band */}
        <div
          className="mt-10 rounded-2xl border border-primary/20 bg-primary/5 p-6 md:p-8 animate-in fade-in-50"
          style={{ animationDelay: '200ms' }}
        >
          <h3 className="mb-6 text-lg font-semibold text-foreground">{outcomes.heading}</h3>
          <div className="grid gap-6 sm:grid-cols-3">
            {outcomes.stats.map((stat, idx) => (
              <div key={idx} className="text-center">
                <div className="mb-1 text-3xl font-bold text-foreground">{stat.value}</div>
                <div className="text-sm text-muted-foreground">{stat.label}</div>
              </div>
            ))}
          </div>
          <div className="mt-6 text-center">
            <a
              href={outcomes.linkHref}
              className="inline-flex items-center gap-1 text-sm text-primary hover:underline focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none rounded"
            >
              {outcomes.linkText} →
            </a>
          </div>
        </div>
      </div>
    </section>
  )
}
