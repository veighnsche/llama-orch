import { Button, Card, CardContent } from '@rbee/ui/atoms'
import { BulletListItem, IconCardHeader } from '@rbee/ui/molecules'
import type { LucideIcon } from 'lucide-react'
import Image from 'next/image'
import Link from 'next/link'

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

export type ComplianceBullet = {
  title: string
}

export type ComplianceBox = {
  heading: string
  items: string[]
}

export type CompliancePillar = {
  icon: LucideIcon
  title: string
  subtitle: string
  titleId: string
  bullets: ComplianceBullet[]
  box: ComplianceBox
}

export type EnterpriseComplianceTemplateProps = {
  id?: string
  backgroundImage: {
    src: string
    alt: string
  }
  eyebrow: string
  heading: string
  description: string
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

export function EnterpriseComplianceTemplate({
  id,
  backgroundImage,
  eyebrow,
  heading,
  description,
  pillars,
  auditReadiness,
}: EnterpriseComplianceTemplateProps) {
  return (
    <section
      id={id}
      aria-labelledby="compliance-h2"
      className="relative border-b border-border bg-radial-glow px-6 py-24"
    >
      {/* Decorative background illustration */}
      <Image
        src={backgroundImage.src}
        width={1200}
        height={640}
        className="pointer-events-none absolute left-1/2 top-6 -z-10 hidden w-[50rem] -translate-x-1/2 opacity-15 blur-[0.5px] md:block"
        alt={backgroundImage.alt}
        aria-hidden="true"
      />

      <div className="relative z-10 mx-auto max-w-7xl">
        {/* Header */}
        <div className="animate-in fade-in-50 slide-in-from-bottom-2 mb-16 text-center duration-500">
          <p className="mb-2 text-sm font-medium text-primary/80">{eyebrow}</p>
          <h2 id="compliance-h2" className="mb-4 text-4xl font-bold text-foreground">
            {heading}
          </h2>
          <p className="mx-auto max-w-3xl text-balance text-xl text-foreground/85">
            {description}
          </p>
        </div>

        {/* Three Pillars */}
        <div className="animate-in fade-in-50 mb-12 grid gap-8 [animation-delay:120ms] lg:grid-cols-3">
          {pillars.map((pillar, idx) => (
            <Card
              key={idx}
              className="h-full rounded-2xl border-border bg-card/60 p-8 transition-shadow hover:shadow-lg"
              aria-labelledby={pillar.titleId}
            >
              <IconCardHeader
                icon={pillar.icon}
                title={pillar.title}
                subtitle={pillar.subtitle}
                titleId={pillar.titleId}
              />
              <CardContent className="p-0">
                <ul className="space-y-3">
                  {pillar.bullets.map((bullet, bulletIdx) => (
                    <BulletListItem key={bulletIdx} variant="check" title={bullet.title} />
                  ))}
                </ul>
                <div className="mt-6">
                  <div className="rounded-lg border border-chart-3/50 bg-chart-3/10 p-4">
                    <div className="mb-2 font-semibold text-chart-3">{pillar.box.heading}</div>
                    <div className="space-y-1 text-xs text-foreground/85">
                      {pillar.box.items.map((item, itemIdx) => (
                        <div key={itemIdx}>{item}</div>
                      ))}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Audit Readiness Band */}
        <div className="animate-in fade-in-50 rounded-2xl border border-primary/20 bg-primary/5 p-8 text-center [animation-delay:200ms]">
          <h3 className="mb-2 text-2xl font-semibold text-foreground">{auditReadiness.heading}</h3>
          <p className="mb-2 text-foreground/85">
            {auditReadiness.description}
          </p>
          <p
            id="compliance-pack-note"
            className="mb-6 text-sm text-muted-foreground"
            aria-label={auditReadiness.noteAriaLabel}
          >
            {auditReadiness.note}
          </p>
          <div className="flex flex-col items-center justify-center gap-3 sm:flex-row">
            {auditReadiness.buttons.map((button, idx) => (
              <Button
                key={idx}
                size="lg"
                variant={button.variant}
                asChild
                aria-describedby={button.ariaDescribedby}
                className="transition-transform active:scale-[0.98]"
              >
                <Link href={button.href}>{button.text}</Link>
              </Button>
            ))}
          </div>
          <p className="mt-6 text-xs text-muted-foreground">{auditReadiness.footnote}</p>
        </div>
      </div>
    </section>
  )
}
