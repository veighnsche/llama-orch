import { FileCheck, Shield, Lock, Globe } from "lucide-react"

export function EnterpriseCompliance() {
  return (
    <section id="compliance" className="border-b border-slate-800 bg-slate-950 px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <div className="mb-16 text-center">
          <h2 className="mb-4 text-4xl font-bold text-white">Compliance by Design</h2>
          <p className="mx-auto max-w-3xl text-balance text-xl text-slate-300">
            rbee is built from the ground up to meet GDPR, SOC2, and ISO 27001 requirements. Not bolted on as an
            afterthought.
          </p>
        </div>

        <div className="grid gap-8 lg:grid-cols-3">
          {/* GDPR */}
          <div className="rounded-lg border border-slate-800 bg-slate-900 p-8">
            <div className="mb-6 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-amber-500/10">
                <Globe className="h-6 w-6 text-amber-400" />
              </div>
              <div>
                <h3 className="text-2xl font-bold text-white">GDPR</h3>
                <p className="text-sm text-slate-400">EU Regulation</p>
              </div>
            </div>

            <div className="space-y-3">
              {[
                "7-year audit retention (Article 30)",
                "Data access records (Article 15)",
                "Right to erasure tracking (Article 17)",
                "Consent management (Article 7)",
                "Data residency controls (Article 44)",
                "Breach notification (Article 33)",
              ].map((item, i) => (
                <div key={i} className="flex items-start gap-2">
                  <FileCheck className="mt-0.5 h-4 w-4 shrink-0 text-green-400" />
                  <span className="text-sm leading-relaxed text-slate-300">{item}</span>
                </div>
              ))}
            </div>

            <div className="mt-6 rounded-lg border border-green-900/50 bg-green-950/20 p-4">
              <div className="mb-1 font-semibold text-green-400">Compliance Endpoints</div>
              <div className="space-y-1 text-xs text-slate-400">
                <div>GET /v2/compliance/data-access</div>
                <div>POST /v2/compliance/data-export</div>
                <div>POST /v2/compliance/data-deletion</div>
                <div>GET /v2/compliance/audit-trail</div>
              </div>
            </div>
          </div>

          {/* SOC2 */}
          <div className="rounded-lg border border-slate-800 bg-slate-900 p-8">
            <div className="mb-6 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-amber-500/10">
                <Shield className="h-6 w-6 text-amber-400" />
              </div>
              <div>
                <h3 className="text-2xl font-bold text-white">SOC2</h3>
                <p className="text-sm text-slate-400">US Standard</p>
              </div>
            </div>

            <div className="space-y-3">
              {[
                "Auditor access (query API)",
                "Security event logging (32 types)",
                "7-year retention (Type II)",
                "Tamper-evident storage (hash chains)",
                "Access control logging",
                "Encryption at rest",
              ].map((item, i) => (
                <div key={i} className="flex items-start gap-2">
                  <FileCheck className="mt-0.5 h-4 w-4 shrink-0 text-green-400" />
                  <span className="text-sm leading-relaxed text-slate-300">{item}</span>
                </div>
              ))}
            </div>

            <div className="mt-6 rounded-lg border border-green-900/50 bg-green-950/20 p-4">
              <div className="mb-1 font-semibold text-green-400">Trust Service Criteria</div>
              <div className="space-y-1 text-xs text-slate-400">
                <div>✓ Security (CC1-CC9)</div>
                <div>✓ Availability (A1.1-A1.3)</div>
                <div>✓ Confidentiality (C1.1-C1.2)</div>
              </div>
            </div>
          </div>

          {/* ISO 27001 */}
          <div className="rounded-lg border border-slate-800 bg-slate-900 p-8">
            <div className="mb-6 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-amber-500/10">
                <Lock className="h-6 w-6 text-amber-400" />
              </div>
              <div>
                <h3 className="text-2xl font-bold text-white">ISO 27001</h3>
                <p className="text-sm text-slate-400">International</p>
              </div>
            </div>

            <div className="space-y-3">
              {[
                "Security incident records (A.16)",
                "3-year retention (minimum)",
                "Access control logging (A.9)",
                "Cryptographic controls (A.10)",
                "Operations security (A.12)",
                "Information security policies (A.5)",
              ].map((item, i) => (
                <div key={i} className="flex items-start gap-2">
                  <FileCheck className="mt-0.5 h-4 w-4 shrink-0 text-green-400" />
                  <span className="text-sm leading-relaxed text-slate-300">{item}</span>
                </div>
              ))}
            </div>

            <div className="mt-6 rounded-lg border border-green-900/50 bg-green-950/20 p-4">
              <div className="mb-1 font-semibold text-green-400">ISMS Controls</div>
              <div className="space-y-1 text-xs text-slate-400">
                <div>✓ 114 controls implemented</div>
                <div>✓ Risk assessment framework</div>
                <div>✓ Continuous monitoring</div>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom CTA */}
        <div className="mt-12 rounded-lg border border-amber-500/20 bg-amber-500/5 p-8 text-center">
          <h3 className="mb-2 text-2xl font-semibold text-white">Ready for Your Compliance Audit</h3>
          <p className="mb-6 text-slate-300">
            Download our compliance documentation package or schedule a call with our compliance team.
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <button className="rounded-lg bg-amber-500 px-6 py-3 font-semibold text-slate-950 transition-colors hover:bg-amber-400">
              Download Compliance Pack
            </button>
            <button className="rounded-lg border border-slate-700 px-6 py-3 font-semibold text-white transition-colors hover:bg-slate-800">
              Talk to Compliance Team
            </button>
          </div>
        </div>
      </div>
    </section>
  )
}
