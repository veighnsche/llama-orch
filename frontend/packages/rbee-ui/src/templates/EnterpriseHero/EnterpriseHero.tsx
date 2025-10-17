import { Badge } from '@rbee/ui/atoms/Badge'
import { HeroTemplate } from '@rbee/ui/templates/HeroTemplate'
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

  const asideContent = (
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
  )

  return (
    <HeroTemplate
      badge={{ variant: 'icon', text: badge.text, icon: BadgeIcon }}
      headline={{ variant: 'simple', content: heading }}
      subcopy={description}
      subcopyMaxWidth="medium"
      proofElements={{
        variant: 'stats-tiles',
        items: stats,
        columns: 3,
      }}
      ctas={{
        primary: {
          label: primaryCta.text,
          ariaLabel: primaryCta.ariaLabel,
        },
        secondary: {
          label: secondaryCta.text,
          href: secondaryCta.href,
          variant: 'outline',
        },
      }}
      helperText={helperText}
      trustElements={{
        variant: 'chips',
        items: complianceChips,
      }}
      aside={asideContent}
      asideAriaLabel="Immutable audit trail console showing recent compliance events"
      background={{
        variant: 'custom',
        className: 'bg-[radial-gradient(60rem_40rem_at_20%_-10%,theme(colors.primary/8),transparent)]',
      }}
      padding="spacious"
      headingId="enterprise-hero-h1"
    />
  )
}
