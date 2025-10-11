import { Button } from "@/components/ui/button"
import { Shield, Lock, FileCheck } from "lucide-react"
import Link from "next/link"

export function EnterpriseHero() {
  return (
    <section className="relative overflow-hidden border-b border-slate-800 bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 px-6 py-24 lg:py-32">
      <div className="mx-auto max-w-7xl">
        <div className="grid gap-12 lg:grid-cols-2 lg:gap-16">
          {/* Left: Messaging */}
          <div className="flex flex-col justify-center">
            <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-amber-500/20 bg-amber-500/10 px-4 py-2 text-sm text-amber-400">
              <Shield className="h-4 w-4" />
              <span>EU-Native AI Infrastructure</span>
            </div>

            <h1 className="mb-6 text-balance text-5xl font-bold leading-tight text-white lg:text-6xl">
              AI Infrastructure That Meets Your Compliance Requirements
            </h1>

            <p className="mb-8 text-pretty text-xl leading-relaxed text-slate-300">
              GDPR-compliant by design. SOC2 ready. ISO 27001 aligned. Build AI infrastructure on your terms with
              complete data sovereignty, immutable audit trails, and enterprise-grade security.
            </p>

            {/* Key Stats */}
            <div className="mb-8 grid grid-cols-3 gap-4">
              <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-4">
                <div className="mb-1 text-2xl font-bold text-amber-400">100%</div>
                <div className="text-sm text-slate-400">GDPR Compliant</div>
              </div>
              <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-4">
                <div className="mb-1 text-2xl font-bold text-amber-400">7 Years</div>
                <div className="text-sm text-slate-400">Audit Retention</div>
              </div>
              <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-4">
                <div className="mb-1 text-2xl font-bold text-amber-400">Zero</div>
                <div className="text-sm text-slate-400">US Cloud Deps</div>
              </div>
            </div>

            {/* CTAs */}
            <div className="flex flex-wrap gap-4">
              <Button size="lg" className="bg-amber-500 text-slate-950 hover:bg-amber-400">
                Schedule Demo
              </Button>
              <Button
                size="lg"
                variant="outline"
                className="border-slate-700 text-white hover:bg-slate-800 bg-transparent"
                asChild
              >
                <Link href="#compliance">View Compliance Details</Link>
              </Button>
            </div>

            {/* Trust Indicators */}
            <div className="mt-8 flex flex-wrap items-center gap-6 text-sm text-slate-400">
              <div className="flex items-center gap-2">
                <FileCheck className="h-4 w-4 text-amber-400" />
                <span>GDPR Compliant</span>
              </div>
              <div className="flex items-center gap-2">
                <Shield className="h-4 w-4 text-amber-400" />
                <span>SOC2 Ready</span>
              </div>
              <div className="flex items-center gap-2">
                <Lock className="h-4 w-4 text-amber-400" />
                <span>ISO 27001 Aligned</span>
              </div>
            </div>
          </div>

          {/* Right: Visual */}
          <div className="flex items-center justify-center">
            <div className="relative w-full max-w-lg">
              {/* Audit Log Mockup */}
              <div className="rounded-lg border border-slate-700 bg-slate-900 p-6 shadow-2xl">
                <div className="mb-4 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Lock className="h-5 w-5 text-amber-400" />
                    <span className="font-semibold text-white">Immutable Audit Trail</span>
                  </div>
                  <div className="rounded-full bg-green-500/20 px-3 py-1 text-xs text-green-400">Compliant</div>
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
                    <div key={i} className="rounded border border-slate-800 bg-slate-950 p-3">
                      <div className="mb-1 flex items-center justify-between">
                        <span className="font-mono text-sm text-amber-400">{log.event}</span>
                        <span className="rounded bg-green-500/20 px-2 py-0.5 text-xs text-green-400">{log.status}</span>
                      </div>
                      <div className="text-xs text-slate-400">
                        <div>{log.user}</div>
                        <div className="text-slate-500">{log.time} UTC</div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Footer */}
                <div className="mt-4 flex items-center justify-between border-t border-slate-800 pt-4 text-xs text-slate-500">
                  <span>Retention: 7 years (GDPR)</span>
                  <span className="flex items-center gap-1">
                    <Shield className="h-3 w-3" />
                    Tamper-evident
                  </span>
                </div>
              </div>

              {/* Floating badges */}
              <div className="absolute -right-4 -top-4 rounded-lg border border-amber-500/20 bg-slate-900 px-4 py-2 shadow-lg">
                <div className="text-xs text-slate-400">Data Residency</div>
                <div className="font-semibold text-amber-400">🇪🇺 EU Only</div>
              </div>

              <div className="absolute -bottom-4 -left-4 rounded-lg border border-amber-500/20 bg-slate-900 px-4 py-2 shadow-lg">
                <div className="text-xs text-slate-400">Audit Events</div>
                <div className="font-semibold text-amber-400">32 Types</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
