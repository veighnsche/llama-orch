import { Server, Shield, CheckCircle, Rocket } from 'lucide-react'

export function EnterpriseHowItWorks() {
  return (
    <section className="border-b border-border bg-background px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <div className="mb-16 text-center">
          <h2 className="mb-4 text-4xl font-bold text-foreground">Enterprise Deployment Process</h2>
          <p className="mx-auto max-w-3xl text-balance text-xl text-muted-foreground">
            From initial consultation to production deployment, we guide you through every step of the compliance
            journey.
          </p>
        </div>

        <div className="space-y-8">
          {/* Step 1 */}
          <div className="flex gap-6">
            <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-primary text-xl font-bold text-primary-foreground">
              1
            </div>
            <div className="flex-1">
              <div className="mb-4 flex items-center gap-3">
                <Shield className="h-6 w-6 text-primary" />
                <h3 className="text-2xl font-semibold text-foreground">Compliance Assessment</h3>
              </div>
              <p className="mb-4 leading-relaxed text-muted-foreground">
                We review your compliance requirements (GDPR, SOC2, ISO 27001, HIPAA, PCI-DSS) and create a tailored
                deployment plan. Identify data residency requirements, audit retention policies, and security controls.
              </p>
              <div className="rounded-lg border border-border bg-card p-4">
                <div className="mb-2 font-semibold text-foreground">Deliverables:</div>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  <li>• Compliance gap analysis</li>
                  <li>• Data flow mapping</li>
                  <li>• Risk assessment report</li>
                  <li>• Deployment architecture proposal</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Step 2 */}
          <div className="flex gap-6">
            <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-primary text-xl font-bold text-primary-foreground">
              2
            </div>
            <div className="flex-1">
              <div className="mb-4 flex items-center gap-3">
                <Server className="h-6 w-6 text-primary" />
                <h3 className="text-2xl font-semibold text-foreground">On-Premises Deployment</h3>
              </div>
              <p className="mb-4 leading-relaxed text-muted-foreground">
                Deploy rbee on your infrastructure (EU data centers, on-premises servers, or private cloud). Configure
                EU-only worker filtering, audit logging, and security controls. White-label option available.
              </p>
              <div className="rounded-lg border border-border bg-card p-4">
                <div className="mb-2 font-semibold text-foreground">Deployment Options:</div>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  <li>• EU data centers (Frankfurt, Amsterdam, Paris)</li>
                  <li>• On-premises (your servers)</li>
                  <li>• Private cloud (AWS EU, Azure EU, GCP EU)</li>
                  <li>• Hybrid (on-prem + marketplace)</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Step 3 */}
          <div className="flex gap-6">
            <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-primary text-xl font-bold text-primary-foreground">
              3
            </div>
            <div className="flex-1">
              <div className="mb-4 flex items-center gap-3">
                <CheckCircle className="h-6 w-6 text-primary" />
                <h3 className="text-2xl font-semibold text-foreground">Compliance Validation</h3>
              </div>
              <p className="mb-4 leading-relaxed text-muted-foreground">
                Validate compliance controls with your auditors. Provide audit trail access, compliance documentation,
                and security architecture review. Support for SOC2 Type II, ISO 27001, and GDPR audits.
              </p>
              <div className="rounded-lg border border-border bg-card p-4">
                <div className="mb-2 font-semibold text-foreground">Audit Support:</div>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  <li>• Compliance documentation package</li>
                  <li>• Auditor access to audit logs</li>
                  <li>• Security architecture review</li>
                  <li>• Penetration testing reports</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Step 4 */}
          <div className="flex gap-6">
            <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-primary text-xl font-bold text-primary-foreground">
              4
            </div>
            <div className="flex-1">
              <div className="mb-4 flex items-center gap-3">
                <Rocket className="h-6 w-6 text-primary" />
                <h3 className="text-2xl font-semibold text-foreground">Production Launch</h3>
              </div>
              <p className="mb-4 leading-relaxed text-muted-foreground">
                Go live with enterprise SLAs, 24/7 support, and dedicated account management. Continuous monitoring,
                health checks, and compliance reporting. Scale as your organization grows.
              </p>
              <div className="rounded-lg border border-border bg-card p-4">
                <div className="mb-2 font-semibold text-foreground">Enterprise Support:</div>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  <li>• 99.9% uptime SLA</li>
                  <li>• 24/7 support (1-hour response time)</li>
                  <li>• Dedicated account manager</li>
                  <li>• Quarterly compliance reviews</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Timeline */}
        <div className="mt-12 rounded-lg border border-primary/20 bg-primary/5 p-8 text-center">
          <h3 className="mb-2 text-2xl font-semibold text-foreground">Typical Deployment Timeline</h3>
          <p className="mb-6 text-muted-foreground">From initial consultation to production deployment</p>
          <div className="grid gap-4 md:grid-cols-4">
            <div>
              <div className="mb-2 text-3xl font-bold text-primary">Week 1-2</div>
              <div className="text-sm text-muted-foreground">Compliance Assessment</div>
            </div>
            <div>
              <div className="mb-2 text-3xl font-bold text-primary">Week 3-4</div>
              <div className="text-sm text-muted-foreground">Deployment & Configuration</div>
            </div>
            <div>
              <div className="mb-2 text-3xl font-bold text-primary">Week 5-6</div>
              <div className="text-sm text-muted-foreground">Compliance Validation</div>
            </div>
            <div>
              <div className="mb-2 text-3xl font-bold text-primary">Week 7</div>
              <div className="text-sm text-muted-foreground">Production Launch</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
