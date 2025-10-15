import { Badge } from '@rbee/ui/atoms/Badge'
import { Button } from '@rbee/ui/atoms/Button'
import { Card } from '@rbee/ui/atoms/Card'
import { Separator } from '@rbee/ui/atoms/Separator'
import { ComplianceChip, StatsGrid } from '@rbee/ui/molecules'
import { Building2, Check, FileCheck, Filter, Lock, Shield } from 'lucide-react'
import Link from 'next/link'

const auditEvents = [
  {
    event: 'auth.success',
    user: 'admin@company.eu',
    time: '2025-10-11T14:23:15Z',
    displayTime: '2025-10-11 14:23:15 UTC',
    status: 'success',
  },
  {
    event: 'data.access',
    user: 'analyst@company.eu',
    time: '2025-10-11T14:22:48Z',
    displayTime: '2025-10-11 14:22:48 UTC',
    status: 'success',
  },
  {
    event: 'task.submitted',
    user: 'dev@company.eu',
    time: '2025-10-11T14:21:33Z',
    displayTime: '2025-10-11 14:21:33 UTC',
    status: 'success',
  },
  {
    event: 'compliance.export',
    user: 'dpo@company.eu',
    time: '2025-10-11T14:20:12Z',
    displayTime: '2025-10-11 14:20:12 UTC',
    status: 'success',
  },
]

export function EnterpriseHero() {
  return (
    <section
      className="relative z-0 overflow-hidden border-b border-border bg-gradient-to-b from-background via-card to-background bg-[radial-gradient(60rem_40rem_at_20%_-10%,theme(colors.primary/8),transparent)] px-6 py-24 lg:py-32"
      aria-labelledby="enterprise-hero-h1"
      role="region"
    >
      {/* Decorative background illustration - removed, file doesn't exist */}

      <div className="relative z-10 mx-auto max-w-7xl">
        <div className="grid gap-12 lg:grid-cols-2 lg:gap-16">
          {/* Left: Messaging */}
          <div className="animate-in fade-in-50 slide-in-from-bottom-2 flex flex-col justify-center duration-500">
            {/* Eyebrow Badge */}
            <Badge
              variant="outline"
              className="mb-6 w-fit border-primary/20 bg-primary/10 px-4 py-2 text-sm text-primary"
            >
              <Shield className="h-4 w-4" />
              <span>EU-Native AI Infrastructure</span>
            </Badge>

            {/* H1 */}
            <h1
              id="enterprise-hero-h1"
              className="mb-6 text-balance text-5xl font-bold leading-tight text-foreground lg:text-6xl"
            >
              AI Infrastructure That Meets Your Compliance Requirements
            </h1>

            {/* Support Copy */}
            <p className="mb-8 text-pretty text-xl leading-relaxed text-foreground/85">
              GDPR-compliant by design. SOC2 ready. ISO 27001 aligned. Build AI on your terms with EU data residency,
              immutable audit trails, and enterprise-grade security.
            </p>

            {/* Proof Tiles */}
            <StatsGrid
              variant="tiles"
              columns={3}
              className="animate-in fade-in-50 mb-8 [animation-delay:120ms]"
              stats={[
                {
                  value: '100%',
                  label: 'GDPR Compliant',
                  helpText: 'Full compliance with EU General Data Protection Regulation',
                },
                {
                  value: '7 Years',
                  label: 'Audit Retention',
                  helpText: 'Immutable audit logs retained for 7 years per GDPR requirements',
                },
                {
                  value: 'Zero',
                  label: 'US Cloud Deps',
                  helpText: 'No dependencies on US cloud providers; EU-native infrastructure',
                },
              ]}
            />

            {/* Primary CTAs */}
            <div className="mb-6 flex flex-wrap gap-4">
              <Button
                size="lg"
                className="bg-primary text-primary-foreground hover:bg-primary/90"
                aria-label="Schedule a compliance demo"
                aria-describedby="compliance-proof-bar"
              >
                Schedule Demo
              </Button>
              <Button
                size="lg"
                variant="outline"
                className="border-border bg-transparent text-foreground hover:bg-secondary"
                asChild
                aria-describedby="compliance-proof-bar"
              >
                <Link href="#compliance">View Compliance Details</Link>
              </Button>
            </div>

            {/* Helper text */}
            <p className="mb-6 text-xs text-muted-foreground">
              EU data residency guaranteed. Audited event types updated quarterly.
            </p>

            {/* Compliance Proof Bar */}
            <div id="compliance-proof-bar" className="flex flex-wrap items-center gap-3" aria-live="polite">
              <ComplianceChip icon={<FileCheck className="h-3 w-3" />} ariaLabel="GDPR Compliant certification">
                GDPR Compliant
              </ComplianceChip>
              <ComplianceChip icon={<Shield className="h-3 w-3" />} ariaLabel="SOC2 Ready certification">
                SOC2 Ready
              </ComplianceChip>
              <ComplianceChip icon={<Lock className="h-3 w-3" />} ariaLabel="ISO 27001 Aligned certification">
                ISO 27001 Aligned
              </ComplianceChip>
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
                    <Lock className="h-5 w-5 text-primary" aria-hidden="true" />
                    <span className="font-semibold text-foreground">Immutable Audit Trail</span>
                  </div>
                  <Badge variant="secondary" className="bg-chart-3/20 text-xs text-chart-3">
                    Compliant
                  </Badge>
                </div>

                {/* Filter Strip (non-functional UI affordance) */}
                <div className="mb-4 flex items-center gap-2 text-xs text-muted-foreground" role="presentation">
                  <Filter className="h-3 w-3" aria-hidden="true" />
                  <button
                    type="button"
                    className="rounded-md bg-primary/10 px-2 py-1 text-primary"
                    aria-label="Filter: All events"
                  >
                    All
                  </button>
                  <button type="button" className="px-2 py-1 hover:text-foreground" aria-label="Filter: Auth events">
                    Auth
                  </button>
                  <button type="button" className="px-2 py-1 hover:text-foreground" aria-label="Filter: Data events">
                    Data
                  </button>
                  <button type="button" className="px-2 py-1 hover:text-foreground" aria-label="Filter: Export events">
                    Exports
                  </button>
                </div>

                {/* Audit Events List */}
                <ul className="space-y-3" role="list" aria-label="Recent audit events">
                  {auditEvents.map((log, i) => (
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
                  <span>Retention: 7 years (GDPR)</span>
                  <span className="flex items-center gap-1">
                    <Shield className="h-3 w-3" aria-hidden="true" />
                    Tamper-evident
                  </span>
                </div>
              </div>

              {/* Floating toast-like badges */}
              <div
                className="absolute -right-4 -top-4 rounded-xl border border-primary/20 bg-card px-4 py-2 shadow-md drop-shadow-md"
                role="status"
                aria-live="polite"
                aria-label="All data processed and stored within the EU"
              >
                <div className="text-xs text-muted-foreground">Data Residency</div>
                <div className="font-semibold text-primary">🇪🇺 EU Only</div>
              </div>

              <div
                className="absolute -bottom-4 -left-4 rounded-xl border border-primary/20 bg-card px-4 py-2 shadow-md drop-shadow-md"
                role="status"
                aria-live="polite"
                aria-label="32 distinct audit event types tracked"
              >
                <div className="text-xs text-muted-foreground">Audit Events</div>
                <div className="font-semibold text-primary">32 Types</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
