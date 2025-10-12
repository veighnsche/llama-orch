import { Button } from '@/components/atoms/Button/Button'
import { Shield, Lock, FileCheck } from "lucide-react"
import Link from "next/link"

export function EnterpriseHero() {
  return (
    <section className="relative overflow-hidden border-b border-border bg-gradient-to-b from-background via-card to-background px-6 py-24 lg:py-32">
      <div className="mx-auto max-w-7xl">
        <div className="grid gap-12 lg:grid-cols-2 lg:gap-16">
          {/* Left: Messaging */}
          <div className="flex flex-col justify-center">
            <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/10 px-4 py-2 text-sm text-primary">
              <Shield className="h-4 w-4" />
              <span>EU-Native AI Infrastructure</span>
            </div>

            <h1 className="mb-6 text-balance text-5xl font-bold leading-tight text-foreground lg:text-6xl">
              AI Infrastructure That Meets Your Compliance Requirements
            </h1>

            <p className="mb-8 text-pretty text-xl leading-relaxed text-muted-foreground">
              GDPR-compliant by design. SOC2 ready. ISO 27001 aligned. Build AI infrastructure on your terms with
              complete data sovereignty, immutable audit trails, and enterprise-grade security.
            </p>

            {/* Key Stats */}
            <div className="mb-8 grid grid-cols-3 gap-4">
              <div className="rounded-lg border border-border bg-card/50 p-4">
                <div className="mb-1 text-2xl font-bold text-primary">100%</div>
                <div className="text-sm text-muted-foreground">GDPR Compliant</div>
              </div>
              <div className="rounded-lg border border-border bg-card/50 p-4">
                <div className="mb-1 text-2xl font-bold text-primary">7 Years</div>
                <div className="text-sm text-muted-foreground">Audit Retention</div>
              </div>
              <div className="rounded-lg border border-border bg-card/50 p-4">
                <div className="mb-1 text-2xl font-bold text-primary">Zero</div>
                <div className="text-sm text-muted-foreground">US Cloud Deps</div>
              </div>
            </div>

            {/* CTAs */}
            <div className="flex flex-wrap gap-4">
              <Button size="lg" className="bg-primary text-primary-foreground hover:bg-primary/90">
                Schedule Demo
              </Button>
              <Button
                size="lg"
                variant="outline"
                className="border-border text-foreground hover:bg-secondary bg-transparent"
                asChild
              >
                <Link href="#compliance">View Compliance Details</Link>
              </Button>
            </div>

            {/* Trust Indicators */}
            <div className="mt-8 flex flex-wrap items-center gap-6 text-sm text-muted-foreground">
              <div className="flex items-center gap-2">
                <FileCheck className="h-4 w-4 text-primary" />
                <span>GDPR Compliant</span>
              </div>
              <div className="flex items-center gap-2">
                <Shield className="h-4 w-4 text-primary" />
                <span>SOC2 Ready</span>
              </div>
              <div className="flex items-center gap-2">
                <Lock className="h-4 w-4 text-primary" />
                <span>ISO 27001 Aligned</span>
              </div>
            </div>
          </div>

          {/* Right: Visual */}
          <div className="flex items-center justify-center">
            <div className="relative w-full max-w-lg">
              {/* Audit Log Mockup */}
              <div className="rounded-lg border border-border bg-card p-6 shadow-2xl">
                <div className="mb-4 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Lock className="h-5 w-5 text-primary" />
                    <span className="font-semibold text-foreground">Immutable Audit Trail</span>
                  </div>
                  <div className="rounded-full bg-chart-3/20 px-3 py-1 text-xs text-chart-3">Compliant</div>
                </div>

                {/* Audit Events */}
                <div className="space-y-3">
                  {[
                    { event: "auth.success", user: "admin@company.eu", time: "2025-10-11 14:23:15", status: "success" },
                    {
                      event: "data.access",
                      user: "analyst@company.eu",
                      time: "2025-10-11 14:22:48",
                      status: "success",
                    },
                    { event: "task.submitted", user: "dev@company.eu", time: "2025-10-11 14:21:33", status: "success" },
                    {
                      event: "compliance.export",
                      user: "dpo@company.eu",
                      time: "2025-10-11 14:20:12",
                      status: "success",
                    },
                  ].map((log, i) => (
                    <div key={i} className="rounded-lg border border-border bg-background p-3">
                      <div className="mb-1 flex items-center justify-between">
                        <span className="font-mono text-sm text-primary">{log.event}</span>
                        <span className="rounded-md bg-chart-3/20 px-2 py-0.5 text-xs text-chart-3">{log.status}</span>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        <div>{log.user}</div>
                        <div className="text-muted-foreground/70">{log.time} UTC</div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Footer */}
                <div className="mt-4 flex items-center justify-between border-t border-border pt-4 text-xs text-muted-foreground/70">
                  <span>Retention: 7 years (GDPR)</span>
                  <span className="flex items-center gap-1">
                    <Shield className="h-3 w-3" />
                    Tamper-evident
                  </span>
                </div>
              </div>

              {/* Floating badges */}
              <div className="absolute -right-4 -top-4 rounded-lg border border-primary/20 bg-card px-4 py-2 shadow-lg">
                <div className="text-xs text-muted-foreground">Data Residency</div>
                <div className="font-semibold text-primary">🇪🇺 EU Only</div>
              </div>

              <div className="absolute -bottom-4 -left-4 rounded-lg border border-primary/20 bg-card px-4 py-2 shadow-lg">
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
