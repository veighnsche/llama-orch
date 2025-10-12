import { FileCheck, Shield, Lock, Globe } from 'lucide-react'

export function EnterpriseCompliance() {
  return (
    <section id="compliance" className="border-b border-border bg-background px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <div className="mb-16 text-center">
          <h2 className="mb-4 text-4xl font-bold text-foreground">Compliance by Design</h2>
          <p className="mx-auto max-w-3xl text-balance text-xl text-muted-foreground">
            rbee is built from the ground up to meet GDPR, SOC2, and ISO 27001 requirements. Not bolted on as an
            afterthought.
          </p>
        </div>

        <div className="grid gap-8 lg:grid-cols-3">
          {/* GDPR */}
          <div className="rounded-lg border border-border bg-card p-8">
            <div className="mb-6 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                <Globe className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h3 className="text-2xl font-bold text-foreground">GDPR</h3>
                <p className="text-sm text-muted-foreground">EU Regulation</p>
              </div>
            </div>

            <div className="space-y-3">
              {[
                '7-year audit retention (Article 30)',
                'Data access records (Article 15)',
                'Right to erasure tracking (Article 17)',
                'Consent management (Article 7)',
                'Data residency controls (Article 44)',
                'Breach notification (Article 33)',
              ].map((item, i) => (
                <div key={i} className="flex items-start gap-2">
                  <FileCheck className="mt-0.5 h-4 w-4 shrink-0 text-chart-3" />
                  <span className="text-sm leading-relaxed text-muted-foreground">{item}</span>
                </div>
              ))}
            </div>

            <div className="mt-6 rounded-lg border border-chart-3/50 bg-chart-3/10 p-4">
              <div className="mb-1 font-semibold text-chart-3">Compliance Endpoints</div>
              <div className="space-y-1 text-xs text-muted-foreground">
                <div>GET /v2/compliance/data-access</div>
                <div>POST /v2/compliance/data-export</div>
                <div>POST /v2/compliance/data-deletion</div>
                <div>GET /v2/compliance/audit-trail</div>
              </div>
            </div>
          </div>

          {/* SOC2 */}
          <div className="rounded-lg border border-border bg-card p-8">
            <div className="mb-6 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                <Shield className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h3 className="text-2xl font-bold text-foreground">SOC2</h3>
                <p className="text-sm text-muted-foreground">US Standard</p>
              </div>
            </div>

            <div className="space-y-3">
              {[
                'Auditor access (query API)',
                'Security event logging (32 types)',
                '7-year retention (Type II)',
                'Tamper-evident storage (hash chains)',
                'Access control logging',
                'Encryption at rest',
              ].map((item, i) => (
                <div key={i} className="flex items-start gap-2">
                  <FileCheck className="mt-0.5 h-4 w-4 shrink-0 text-chart-3" />
                  <span className="text-sm leading-relaxed text-muted-foreground">{item}</span>
                </div>
              ))}
            </div>

            <div className="mt-6 rounded-lg border border-chart-3/50 bg-chart-3/10 p-4">
              <div className="mb-1 font-semibold text-chart-3">Trust Service Criteria</div>
              <div className="space-y-1 text-xs text-muted-foreground">
                <div>✓ Security (CC1-CC9)</div>
                <div>✓ Availability (A1.1-A1.3)</div>
                <div>✓ Confidentiality (C1.1-C1.2)</div>
              </div>
            </div>
          </div>

          {/* ISO 27001 */}
          <div className="rounded-lg border border-border bg-card p-8">
            <div className="mb-6 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                <Lock className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h3 className="text-2xl font-bold text-foreground">ISO 27001</h3>
                <p className="text-sm text-muted-foreground">International</p>
              </div>
            </div>

            <div className="space-y-3">
              {[
                'Security incident records (A.16)',
                '3-year retention (minimum)',
                'Access control logging (A.9)',
                'Cryptographic controls (A.10)',
                'Operations security (A.12)',
                'Information security policies (A.5)',
              ].map((item, i) => (
                <div key={i} className="flex items-start gap-2">
                  <FileCheck className="mt-0.5 h-4 w-4 shrink-0 text-chart-3" />
                  <span className="text-sm leading-relaxed text-muted-foreground">{item}</span>
                </div>
              ))}
            </div>

            <div className="mt-6 rounded-lg border border-chart-3/50 bg-chart-3/10 p-4">
              <div className="mb-1 font-semibold text-chart-3">ISMS Controls</div>
              <div className="space-y-1 text-xs text-muted-foreground">
                <div>✓ 114 controls implemented</div>
                <div>✓ Risk assessment framework</div>
                <div>✓ Continuous monitoring</div>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom CTA */}
        <div className="mt-12 rounded-lg border border-primary/20 bg-primary/5 p-8 text-center">
          <h3 className="mb-2 text-2xl font-semibold text-foreground">Ready for Your Compliance Audit</h3>
          <p className="mb-6 text-muted-foreground">
            Download our compliance documentation package or schedule a call with our compliance team.
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <button className="rounded-lg bg-primary px-6 py-3 font-semibold text-primary-foreground transition-colors hover:bg-primary/90">
              Download Compliance Pack
            </button>
            <button className="rounded-lg border border-border px-6 py-3 font-semibold text-foreground transition-colors hover:bg-secondary">
              Talk to Compliance Team
            </button>
          </div>
        </div>
      </div>
    </section>
  )
}
