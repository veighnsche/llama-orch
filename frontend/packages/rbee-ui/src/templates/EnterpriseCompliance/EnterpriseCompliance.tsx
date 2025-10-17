import { Card, CardContent, HighlightCard } from '@rbee/ui/atoms'
import { AuditReadinessCTA, BulletListItem, IconCardHeader } from '@rbee/ui/molecules'
import Image from 'next/image'
import type { ReactNode } from 'react'

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

export type ComplianceBullet = {
  title: string
}

export type ComplianceBox = {
  heading: string
  items: string[]
  checkmarkColor?: 'primary' | 'chart-1' | 'chart-2' | 'chart-3' | 'chart-4' | 'chart-5' | 'white'
  disabledCheckmarks?: boolean
}

export type CompliancePillar = {
  icon: ReactNode
  title: string
  subtitle: string
  titleId: string
  bullets: ComplianceBullet[]
  box: ComplianceBox
}

export type EnterpriseComplianceProps = {
  id?: string
  backgroundImage: {
    src: string
    alt: string
  }
  pillars: CompliancePillar[]
  auditReadiness: {
    heading: string
    description: string
    note: string
    noteAriaLabel: string
    buttons: Array<{
      text: string
      href: string
      variant?: 'default' | 'outline'
      ariaDescribedby?: string
    }>
    footnote: string
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function EnterpriseCompliance({
  id,
  backgroundImage,
  pillars,
  auditReadiness,
}: EnterpriseComplianceProps) {
  return (
    <div id={id} className="relative">
      {/* Decorative background illustration */}
      <Image
        src={backgroundImage.src}
        width={1200}
        height={640}
        className="pointer-events-none absolute left-1/2 top-6 -z-10 hidden w-[50rem] -translate-x-1/2 opacity-15 blur-[0.5px] md:block"
        alt={backgroundImage.alt}
        aria-hidden="true"
      />

      <div className="relative z-10">
        {/* Three Pillars */}
        <div className="animate-in fade-in-50 mb-12 grid gap-8 [animation-delay:120ms] lg:grid-cols-3">
          {pillars.map((pillar, idx) => (
            <Card
              key={idx}
              className="h-full rounded-2xl border-border bg-card/60 transition-shadow hover:shadow-lg"
              aria-labelledby={pillar.titleId}
            >
              <IconCardHeader
                icon={pillar.icon}
                title={pillar.title}
                subtitle={pillar.subtitle}
                titleId={pillar.titleId}
              />
              <CardContent className="p-0 px-6 pb-6">
                <ul className="space-y-3">
                  {pillar.bullets.map((bullet, bulletIdx) => (
                    <BulletListItem key={bulletIdx} variant="check" showPlate={false} title={bullet.title} />
                  ))}
                </ul>
                <div className="mt-6">
                  <HighlightCard
                    heading={pillar.box.heading}
                    items={pillar.box.items}
                    color="chart-3"
                    checkmarkColor={pillar.box.checkmarkColor}
                    disabledCheckmarks={pillar.box.disabledCheckmarks}
                  />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Audit Readiness Band */}
        <AuditReadinessCTA
          heading={auditReadiness.heading}
          description={auditReadiness.description}
          note={auditReadiness.note}
          noteAriaLabel={auditReadiness.noteAriaLabel}
          buttons={auditReadiness.buttons}
          footnote={auditReadiness.footnote}
        />
      </div>
    </div>
  )
}
