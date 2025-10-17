import { Card, CardContent, HighlightCard } from '@rbee/ui/atoms'
import { BulletListItem, IconCardHeader } from '@rbee/ui/molecules'
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
  pillars: CompliancePillar[]
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function EnterpriseCompliance({ pillars }: EnterpriseComplianceProps) {
  return (
    <div className="animate-in fade-in-50 grid gap-8 [animation-delay:120ms] lg:grid-cols-3">
      {pillars.map((pillar, idx) => (
        <Card
          key={idx}
          className="h-full rounded-2xl border-border bg-card/60 transition-shadow hover:shadow-lg"
          aria-labelledby={pillar.titleId}
        >
          <IconCardHeader icon={pillar.icon} title={pillar.title} subtitle={pillar.subtitle} titleId={pillar.titleId} />
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
  )
}
