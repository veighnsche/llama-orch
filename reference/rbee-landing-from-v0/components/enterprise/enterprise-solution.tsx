import { Shield, Lock, FileCheck, Server } from "lucide-react"

export function EnterpriseSolution() {
  return (
    <section className="border-b border-slate-800 bg-slate-950 px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <div className="mb-16 text-center">
          <h2 className="mb-4 text-4xl font-bold text-white">EU-Native AI Infrastructure</h2>
          <p className="mx-auto max-w-3xl text-balance text-xl text-slate-300">
            rbee provides enterprise-grade AI infrastructure that meets your compliance requirements by design.
            Self-hosted, EU-resident, fully auditable.
          </p>
        </div>

        {/* Architecture Diagram */}
        <div className="mb-16 rounded-lg border border-slate-800 bg-slate-900 p-8">
          <div className="mb-6 text-center">
            <h3 className="mb-2 text-2xl font-semibold text-white">Defense-in-Depth Security Architecture</h3>
            <p className="text-slate-400">Five specialized security layers working together</p>
          </div>

          <div className="space-y-4">
            {[
              {
                icon: Shield,
                name: "Input Validation",
                description: "First line of defense - Rejects malicious input, prevents injection attacks",
                color: "text-blue-400",
              },
              {
                icon: Lock,
                name: "Authentication (auth-min)",
                description: "Timing-safe token validation, zero-trust principles",
                color: "text-amber-400",
              },
              {
                icon: FileCheck,
                name: "Secrets Management",
                description: "File-based credentials, memory zeroization, no environment leakage",
                color: "text-green-400",
              },
              {
                icon: FileCheck,
                name: "Audit Logging",
                description: "Immutable audit trail, 7-year retention, tamper-evident",
                color: "text-purple-400",
              },
              {
                icon: Server,
                name: "Deadline Propagation",
                description: "Resource enforcement, prevents exhaustion attacks",
                color: "text-red-400",
              },
            ].map((layer, i) => (
              <div key={i} className="flex items-start gap-4 rounded-lg border border-slate-800 bg-slate-950 p-4">
                <div
                  className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-slate-900 ${layer.color}`}
                >
                  <layer.icon className="h-5 w-5" />
                </div>
                <div>
                  <div className="mb-1 font-semibold text-white">{layer.name}</div>
                  <div className="text-sm leading-relaxed text-slate-400">{layer.description}</div>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-6 text-center text-sm text-slate-500">
            Each layer catches what others might miss - Defense-in-depth architecture
          </div>
        </div>

        {/* Key Benefits */}
        <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-6">
            <div className="mb-3 text-3xl font-bold text-amber-400">100%</div>
            <div className="mb-2 font-semibold text-white">Data Sovereignty</div>
            <div className="text-sm leading-relaxed text-slate-400">
              Data never leaves your infrastructure. EU-only deployment. Complete control.
            </div>
          </div>

          <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-6">
            <div className="mb-3 text-3xl font-bold text-amber-400">7 Years</div>
            <div className="mb-2 font-semibold text-white">Audit Retention</div>
            <div className="text-sm leading-relaxed text-slate-400">
              GDPR-compliant audit logs. Immutable, tamper-evident, legally defensible.
            </div>
          </div>

          <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-6">
            <div className="mb-3 text-3xl font-bold text-amber-400">32</div>
            <div className="mb-2 font-semibold text-white">Audit Event Types</div>
            <div className="text-sm leading-relaxed text-slate-400">
              Complete visibility. Authentication, data access, compliance events.
            </div>
          </div>

          <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-6">
            <div className="mb-3 text-3xl font-bold text-amber-400">Zero</div>
            <div className="mb-2 font-semibold text-white">US Cloud Dependencies</div>
            <div className="text-sm leading-relaxed text-slate-400">
              Self-hosted or EU marketplace. No Schrems II concerns. Full compliance.
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
