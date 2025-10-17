import { GlassCard } from '@rbee/ui/atoms'
import { Badge } from '@rbee/ui/atoms/Badge'
import { Card, CardAction, CardContent } from '@rbee/ui/atoms/Card'
import { AuditEventItem, IconCardHeader } from '@rbee/ui/molecules'
import { FilterButton } from '@rbee/ui/molecules/FilterButton'
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
  const asideContent = (
    <div className="animate-in fade-in-50 slide-in-from-right-2 flex items-center justify-center pb-12 [animation-delay:150ms]">
      <div className="relative w-full max-w-lg lg:sticky lg:top-24">
        {/* Immutable Audit Trail Console */}
        <Card className="shadow-2xl">
          <IconCardHeader
            icon={badge.icon}
            title={auditConsole.title}
            iconSize="md"
            iconTone="primary"
            titleClassName="font-semibold text-foreground"
            align="center"
          />
          <CardAction className="absolute right-6 top-6">
            <Badge variant="secondary" className="bg-chart-3/20 text-xs text-chart-3">
              {auditConsole.badge}
            </Badge>
          </CardAction>

          <CardContent>
            {/* Filter Strip */}
            <div className="mb-4 flex items-center gap-2" role="presentation">
              {auditConsole.filterButtons.map((filter, idx) => (
                <FilterButton key={idx} label={filter.label} active={filter.active} ariaLabel={filter.ariaLabel} />
              ))}
            </div>

            {/* Audit Events List */}
            <ul className="space-y-3" aria-label="Recent audit events">
              {auditConsole.events.map((log, i) => (
                <AuditEventItem
                  key={i}
                  event={log.event}
                  user={log.user}
                  time={log.time}
                  displayTime={log.displayTime}
                  status={log.status}
                />
              ))}
            </ul>

            {/* Footer */}
            <div className="mt-4 flex items-center justify-between border-t border-border pt-4 text-xs text-foreground/85">
              <span>{auditConsole.footer.retention}</span>
              <span className="flex items-center gap-1.5">
                <span className="h-3 w-3 shrink-0 flex items-center justify-center" aria-hidden="true">
                  {badge.icon}
                </span>
                {auditConsole.footer.tamperProof}
              </span>
            </div>
          </CardContent>
        </Card>

        {/* Floating badges */}
        {floatingBadges.map((floatingBadge, idx) => (
          <GlassCard
            key={idx}
            className={`absolute px-4 py-2 ${
              floatingBadge.position === 'top-right' ? 'right-4 -top-8' : 'left-4 -bottom-8'
            }`}
            role="status"
            aria-live="polite"
            aria-label={floatingBadge.ariaLabel}
          >
            <div className="text-xs text-muted-foreground">{floatingBadge.label}</div>
            <div className="font-semibold text-primary">{floatingBadge.value}</div>
          </GlassCard>
        ))}
      </div>
    </div>
  )

  return (
    <HeroTemplate
      badge={{ variant: 'icon', text: badge.text, icon: badge.icon }}
      headline={{ variant: 'simple', content: heading }}
      subcopy={description}
      subcopyMaxWidth="medium"
      proofElements={{
        variant: 'stats-pills',
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
