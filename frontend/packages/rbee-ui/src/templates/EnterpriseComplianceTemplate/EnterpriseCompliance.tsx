import { Button, Card, CardContent } from '@rbee/ui/atoms'
import { BulletListItem, IconCardHeader } from '@rbee/ui/molecules'
import { Globe, Lock, Shield } from 'lucide-react'
import Image from 'next/image'
import Link from 'next/link'

export function EnterpriseCompliance() {
  return (
    <section
      id="compliance"
      aria-labelledby="compliance-h2"
      className="relative border-b border-border bg-radial-glow px-6 py-24"
    >
      {/* Decorative background illustration */}
      <Image
        src="/decor/compliance-ledger.webp"
        width={1200}
        height={640}
        className="pointer-events-none absolute left-1/2 top-6 -z-10 hidden w-[50rem] -translate-x-1/2 opacity-15 blur-[0.5px] md:block"
        alt="Abstract EU-blue ledger lines with checkpoint nodes; evokes immutable audit trails, GDPR alignment, SOC2 controls, ISO 27001 ISMS"
        aria-hidden="true"
      />

      <div className="relative z-10 mx-auto max-w-7xl">
        {/* Header */}
        <div className="animate-in fade-in-50 slide-in-from-bottom-2 mb-16 text-center duration-500">
          <p className="mb-2 text-sm font-medium text-primary/80">Security & Certifications</p>
          <h2 id="compliance-h2" className="mb-4 text-4xl font-bold text-foreground">
            Compliance by Design
          </h2>
          <p className="mx-auto max-w-3xl text-balance text-xl text-foreground/85">
            Built from the ground up to meet GDPR, SOC2, and ISO 27001 requirements—security is engineered in, not
            bolted on.
          </p>
        </div>

        {/* Three Pillars */}
        <div className="animate-in fade-in-50 mb-12 grid gap-8 [animation-delay:120ms] lg:grid-cols-3">
          {/* GDPR Pillar */}
          <Card
            className="h-full rounded-2xl border-border bg-card/60 p-8 transition-shadow hover:shadow-lg"
            aria-labelledby="compliance-gdpr"
          >
            <IconCardHeader
              icon={<Globe className="w-6 h-6" />}
              title="GDPR"
              subtitle="EU Regulation"
              titleId="compliance-gdpr"
            />
            <CardContent className="p-0">
              <ul className="space-y-3">
                <BulletListItem variant="check" title="7-year audit retention (Art. 30)" />
                <BulletListItem variant="check" title="Data access records (Art. 15)" />
                <BulletListItem variant="check" title="Erasure tracking (Art. 17)" />
                <BulletListItem variant="check" title="Consent management (Art. 7)" />
                <BulletListItem variant="check" title="Data residency controls (Art. 44)" />
                <BulletListItem variant="check" title="Breach notification (Art. 33)" />
              </ul>
              <div className="mt-6">
                <div className="rounded-lg border border-chart-3/50 bg-chart-3/10 p-4">
                  <div className="mb-2 font-semibold text-chart-3">Compliance Endpoints</div>
                  <div className="space-y-1 font-mono text-xs text-foreground/85">
                    <div>GET /v2/compliance/data-access</div>
                    <div>POST /v2/compliance/data-export</div>
                    <div>POST /v2/compliance/data-deletion</div>
                    <div>GET /v2/compliance/audit-trail</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* SOC2 Pillar */}
          <Card
            className="h-full rounded-2xl border-border bg-card/60 p-8 transition-shadow hover:shadow-lg"
            aria-labelledby="compliance-soc2"
          >
            <IconCardHeader
              icon={<Shield className="w-6 h-6" />}
              title="SOC2"
              subtitle="US Standard"
              titleId="compliance-soc2"
            />
            <CardContent className="p-0">
              <ul className="space-y-3">
                <BulletListItem variant="check" title="Auditor query API" />
                <BulletListItem variant="check" title="32 audit event types" />
                <BulletListItem variant="check" title="7-year retention (Type II)" />
                <BulletListItem variant="check" title="Tamper-evident hash chains" />
                <BulletListItem variant="check" title="Access control logging" />
                <BulletListItem variant="check" title="Encryption at rest" />
              </ul>
              <div className="mt-6">
                <div className="rounded-lg border border-chart-3/50 bg-chart-3/10 p-4">
                  <div className="mb-2 font-semibold text-chart-3">Trust Service Criteria</div>
                  <div className="space-y-1 text-xs text-foreground/85">
                    <div>✓ Security (CC1-CC9)</div>
                    <div>✓ Availability (A1.1-A1.3)</div>
                    <div>✓ Confidentiality (C1.1-C1.2)</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* ISO 27001 Pillar */}
          <Card
            className="h-full rounded-2xl border-border bg-card/60 p-8 transition-shadow hover:shadow-lg"
            aria-labelledby="compliance-iso27001"
          >
            <IconCardHeader
              icon={<Lock className="w-6 h-6" />}
              title="ISO 27001"
              subtitle="International Standard"
              titleId="compliance-iso27001"
            />
            <CardContent className="p-0">
              <ul className="space-y-3">
                <BulletListItem variant="check" title="Incident records (A.16)" />
                <BulletListItem variant="check" title="3-year minimum retention" />
                <BulletListItem variant="check" title="Access logging (A.9)" />
                <BulletListItem variant="check" title="Crypto controls (A.10)" />
                <BulletListItem variant="check" title="Ops security (A.12)" />
                <BulletListItem variant="check" title="Security policies (A.5)" />
              </ul>
              <div className="mt-6">
                <div className="rounded-lg border border-chart-3/50 bg-chart-3/10 p-4">
                  <div className="mb-2 font-semibold text-chart-3">ISMS Controls</div>
                  <div className="space-y-1 text-xs text-foreground/85">
                    <div>✓ 114 controls implemented</div>
                    <div>✓ Risk assessment framework</div>
                    <div>✓ Continuous monitoring</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Audit Readiness Band */}
        <div className="animate-in fade-in-50 rounded-2xl border border-primary/20 bg-primary/5 p-8 text-center [animation-delay:200ms]">
          <h3 className="mb-2 text-2xl font-semibold text-foreground">Ready for Your Compliance Audit</h3>
          <p className="mb-2 text-foreground/85">
            Download our compliance documentation package or schedule a call with our compliance team.
          </p>
          <p
            id="compliance-pack-note"
            className="mb-6 text-sm text-muted-foreground"
            aria-label="Compliance pack includes endpoints, retention policy, and audit-logging design"
          >
            Pack includes endpoints, retention policy, and audit-logging design.
          </p>
          <div className="flex flex-col items-center justify-center gap-3 sm:flex-row">
            <Button
              size="lg"
              asChild
              aria-describedby="compliance-pack-note"
              className="transition-transform active:scale-[0.98]"
            >
              <Link href="/compliance/download">Download Compliance Pack</Link>
            </Button>
            <Button
              size="lg"
              variant="outline"
              asChild
              aria-describedby="compliance-pack-note"
              className="transition-transform active:scale-[0.98]"
            >
              <Link href="/contact/compliance">Talk to Compliance Team</Link>
            </Button>
          </div>
          <p className="mt-6 text-xs text-muted-foreground">rbee (pronounced "are-bee")</p>
        </div>
      </div>
    </section>
  )
}
