import { Badge } from '@rbee/ui/atoms/Badge'
import { Button } from '@rbee/ui/atoms/Button'
import { ComplianceChip, StatsGrid } from '@rbee/ui/molecules'
import Link from 'next/link'
import type * as React from 'react'
import type { ReactNode } from 'react'

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

export type AuditEvent = {
  event: string
  user: string
  time: string
  displayTime: string
  status: string
}

export type StatItem = {
  value: string
  label: string
  helpText?: string
}

export type ComplianceChipData = {
  icon: React.ReactNode
  label: string
  ariaLabel: string
}

export type FilterButton = {
  label: string
  ariaLabel: string
  active?: boolean
}

export type FloatingBadge = {
  label: string
  value: string
  ariaLabel: string
  position: 'top-right' | 'bottom-left'
}

export type EnterpriseHeroProps = {
  badge: {
    icon: ReactNode
    text: string
  }
  heading: string
  description: string
  stats: StatItem[]
  primaryCta: {
    text: string
    ariaLabel?: string
  }
  secondaryCta: {
    text: string
    href: string
  }
  helperText: string
  complianceChips: ComplianceChipData[]
  auditConsole: {
    title: string
    badge: string
    filterButtons: FilterButton[]
    events: AuditEvent[]
    footer: {
      retention: string
      tamperProof: string
    }
  }
  floatingBadges: FloatingBadge[]
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function EnterpriseHero({
  badge,
  heading,
  description,
  stats,
  primaryCta,
  secondaryCta,
  helperText,
  complianceChips,
  auditConsole,
  floatingBadges,
}: EnterpriseHeroProps) {
  const BadgeIcon = badge.icon

  return (
    <section
      className="relative z-0 overflow-hidden bg-gradient-to-b from-background via-card to-background bg-[radial-gradient(60rem_40rem_at_20%_-10%,theme(colors.primary/8),transparent)] px-6 py-24 lg:py-32"
      aria-labelledby="enterprise-hero-h1"
    >
      <div className="relative z-10 mx-auto max-w-7xl">
        <div className="grid gap-12 lg:grid-cols-2 lg:gap-16">
          {/* Left: Messaging */}
          <div className="animate-in fade-in-50 slide-in-from-bottom-2 flex flex-col justify-center duration-500">
            {/* Eyebrow Badge */}
            <Badge
              variant="outline"
              className="mb-6 w-fit border-primary/20 bg-primary/10 px-4 py-2 text-sm text-primary"
            >
              <div className="flex h-4 w-4 items-center justify-center">{BadgeIcon}</div>
              <span>{badge.text}</span>
            </Badge>

            {/* H1 */}
            <h1
              id="enterprise-hero-h1"
              className="mb-6 text-balance text-5xl font-bold leading-tight text-foreground lg:text-6xl"
            >
              {heading}
            </h1>

            {/* Support Copy */}
            <p className="mb-8 text-pretty text-xl leading-relaxed text-foreground/85">{description}</p>

            {/* Proof Tiles */}
            <StatsGrid
              variant="tiles"
              columns={3}
              className="animate-in fade-in-50 mb-8 [animation-delay:120ms]"
              stats={stats}
            />

            {/* Primary CTAs */}
            <div className="mb-6 flex flex-wrap gap-4">
              <Button
                size="lg"
                className="bg-primary text-primary-foreground hover:bg-primary/90"
                aria-label={primaryCta.ariaLabel}
                aria-describedby="compliance-proof-bar"
              >
                {primaryCta.text}
              </Button>
              <Button
                size="lg"
                variant="outline"
                className="border-border bg-transparent text-foreground hover:bg-secondary"
                asChild
                aria-describedby="compliance-proof-bar"
              >
                <Link href={secondaryCta.href}>{secondaryCta.text}</Link>
              </Button>
            </div>

            {/* Helper text */}
            <p className="mb-6 text-xs text-muted-foreground">{helperText}</p>

            {/* Compliance Proof Bar */}
            <div id="compliance-proof-bar" className="flex flex-wrap items-center gap-3" aria-live="polite">
              {complianceChips.map((chip, idx) => (
                <ComplianceChip key={idx} icon={chip.icon} ariaLabel={chip.ariaLabel}>
                  {chip.label}
                </ComplianceChip>
              ))}
            </div>
          </div>

          {/* Right: Audit Console Visual */}
          <div className="animate-in fade-in-50 slide-in-from-right-2 flex items-center justify-center pb-12 [animation-delay:150ms]">
            <div className="relative w-full max-w-lg lg:sticky lg:top-24">
              {/* Immutable Audit Trail Console */}
              <div className="rounded-xl border bg-card p-6 shadow-2xl">
                {/* Header */}
                <div className="mb-4 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="h-5 w-5 text-primary" aria-hidden="true">
                      {BadgeIcon}
                    </div>
                    <span className="font-semibold text-foreground">{auditConsole.title}</span>
                  </div>
                  <Badge variant="secondary" className="bg-chart-3/20 text-xs text-chart-3">
                    {auditConsole.badge}
                  </Badge>
                </div>

                {/* Filter Strip */}
                <div className="mb-4 flex items-center gap-2 text-xs text-muted-foreground" role="presentation">
                  <div className="h-3 w-3" aria-hidden="true">
                    {BadgeIcon}
                  </div>
                  {auditConsole.filterButtons.map((filter, idx) => (
                    <button
                      key={idx}
                      type="button"
                      className={
                        filter.active
                          ? 'rounded-md bg-primary/10 px-2 py-1 text-primary'
                          : 'px-2 py-1 hover:text-foreground'
                      }
                      aria-label={filter.ariaLabel}
                    >
                      {filter.label}
                    </button>
                  ))}
                </div>

                {/* Audit Events List */}
                <ul className="space-y-3" aria-label="Recent audit events">
                  {auditConsole.events.map((log, i) => (
                    <li
                      key={i}
                      className="rounded-lg border bg-background p-3"
                      aria-label={`${log.event} by ${log.user} at ${log.displayTime} – ${log.status}`}
                    >
                      <div className="mb-1 flex items-center justify-between">
                        <span className="font-mono text-sm text-primary">{log.event}</span>
                        <Badge variant="secondary" className="bg-chart-3/20 px-2 py-0.5 text-xs text-chart-3">
                          {log.status}
                        </Badge>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        <div>{log.user}</div>
                        <time dateTime={log.time} className="text-muted-foreground/70">
                          {log.displayTime}
                        </time>
                      </div>
                    </li>
                  ))}
                </ul>

                {/* Footer */}
                <div className="mt-4 flex items-center justify-between border-t border-border pt-4 text-xs text-foreground/85">
                  <span>{auditConsole.footer.retention}</span>
                  <span className="flex items-center gap-1">
                    <div className="h-3 w-3" aria-hidden="true">
                      {BadgeIcon}
                    </div>
                    {auditConsole.footer.tamperProof}
                  </span>
                </div>
              </div>

              {/* Floating badges */}
              {floatingBadges.map((floatingBadge, idx) => (
                <div
                  key={idx}
                  className={`absolute rounded-xl border border-primary/20 bg-card px-4 py-2 shadow-md drop-shadow-md ${
                    floatingBadge.position === 'top-right' ? '-right-4 -top-4' : '-bottom-4 -left-4'
                  }`}
                  role="status"
                  aria-live="polite"
                  aria-label={floatingBadge.ariaLabel}
                >
                  <div className="text-xs text-muted-foreground">{floatingBadge.label}</div>
                  <div className="font-semibold text-primary">{floatingBadge.value}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
