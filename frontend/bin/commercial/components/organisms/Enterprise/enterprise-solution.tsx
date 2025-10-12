import { Shield, Lock, FileCheck, Server } from "lucide-react"
import { cn } from "@/lib/utils"

export function EnterpriseSolution() {
  return (
    <section className="border-b border-border bg-background px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <div className="mb-16 text-center">
          <h2 className="mb-4 text-4xl font-bold text-foreground">EU-Native AI Infrastructure</h2>
          <p className="mx-auto max-w-3xl text-balance text-xl text-muted-foreground">
            rbee provides enterprise-grade AI infrastructure that meets your compliance requirements by design.
            Self-hosted, EU-resident, fully auditable.
          </p>
        </div>

        {/* Architecture Diagram */}
        <div className="mb-16 rounded-lg border border-border bg-card p-8">
          <div className="mb-6 text-center">
            <h3 className="mb-2 text-2xl font-semibold text-foreground">Defense-in-Depth Security Architecture</h3>
            <p className="text-muted-foreground">Five specialized security layers working together</p>
          </div>

          <div className="space-y-4">
            {[
              {
                icon: Shield,
                name: "Input Validation",
                description: "First line of defense - Rejects malicious input, prevents injection attacks",
                color: "text-chart-2",
              },
              {
                icon: Lock,
                name: "Authentication (auth-min)",
                description: "Timing-safe token validation, zero-trust principles",
                color: "text-primary",
              },
              {
                icon: FileCheck,
                name: "Secrets Management",
                description: "File-based credentials, memory zeroization, no environment leakage",
                color: "text-chart-3",
              },
              {
                icon: FileCheck,
                name: "Audit Logging",
                description: "Immutable audit trail, 7-year retention, tamper-evident",
                color: "text-chart-4",
              },
              {
                icon: Server,
                name: "Deadline Propagation",
                description: "Resource enforcement, prevents exhaustion attacks",
                color: "text-destructive",
              },
            ].map((layer, i) => (
              <div key={i} className="flex items-start gap-4 rounded-lg border border-border bg-background p-4">
                <div
                  className={cn(
                    'flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-card',
                    layer.color
                  )}
                >
                  <layer.icon className="h-5 w-5" />
                </div>
                <div>
                  <div className="mb-1 font-semibold text-foreground">{layer.name}</div>
                  <div className="text-sm leading-relaxed text-muted-foreground">{layer.description}</div>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-6 text-center text-sm text-muted-foreground/70">
            Each layer catches what others might miss - Defense-in-depth architecture
          </div>
        </div>

        {/* Key Benefits */}
        <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-lg border border-border bg-card/50 p-6">
            <div className="mb-3 text-3xl font-bold text-primary">100%</div>
            <div className="mb-2 font-semibold text-foreground">Data Sovereignty</div>
            <div className="text-sm leading-relaxed text-muted-foreground">
              Data never leaves your infrastructure. EU-only deployment. Complete control.
            </div>
          </div>

          <div className="rounded-lg border border-border bg-card/50 p-6">
            <div className="mb-3 text-3xl font-bold text-primary">7 Years</div>
            <div className="mb-2 font-semibold text-foreground">Audit Retention</div>
            <div className="text-sm leading-relaxed text-muted-foreground">
              GDPR-compliant audit logs. Immutable, tamper-evident, legally defensible.
            </div>
          </div>

          <div className="rounded-lg border border-border bg-card/50 p-6">
            <div className="mb-3 text-3xl font-bold text-primary">32</div>
            <div className="mb-2 font-semibold text-foreground">Audit Event Types</div>
            <div className="text-sm leading-relaxed text-muted-foreground">
              Complete visibility. Authentication, data access, compliance events.
            </div>
          </div>

          <div className="rounded-lg border border-border bg-card/50 p-6">
            <div className="mb-3 text-3xl font-bold text-primary">Zero</div>
            <div className="mb-2 font-semibold text-foreground">US Cloud Dependencies</div>
            <div className="text-sm leading-relaxed text-muted-foreground">
              Self-hosted or EU marketplace. No Schrems II concerns. Full compliance.
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
