import { Building2, Heart, Scale, Shield } from "lucide-react"

export function EnterpriseUseCases() {
  return (
    <section className="border-b border-border bg-background px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <div className="mb-16 text-center">
          <h2 className="mb-4 text-4xl font-bold text-foreground">Built for Regulated Industries</h2>
          <p className="mx-auto max-w-3xl text-balance text-xl text-muted-foreground">
            Organizations in highly regulated industries trust rbee for AI infrastructure that meets their compliance
            requirements.
          </p>
        </div>

        <div className="grid gap-8 md:grid-cols-2">
          {/* Use Case 1: Financial Services */}
          <div className="rounded-lg border border-border bg-card p-8">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                <Building2 className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-foreground">Financial Services</h3>
                <p className="text-sm text-muted-foreground">Banks, Insurance, FinTech</p>
              </div>
            </div>

            <p className="mb-4 leading-relaxed text-muted-foreground">
              A European bank needed AI-powered code generation for internal tools but couldn't use external AI
              providers due to PCI-DSS and GDPR requirements.
            </p>

            <div className="mb-4 rounded-lg border border-border bg-background p-4">
              <div className="mb-2 font-semibold text-foreground">Challenge:</div>
              <ul className="space-y-1 text-sm text-muted-foreground">
                <li>• Cannot send code to external APIs (PCI-DSS)</li>
                <li>• Need complete audit trail (SOC2)</li>
                <li>• EU data residency required (GDPR)</li>
                <li>• 7-year audit retention mandatory</li>
              </ul>
            </div>

            <div className="rounded-lg border border-chart-3/50 bg-chart-3/10 p-4">
              <div className="mb-2 font-semibold text-chart-3">Solution with rbee:</div>
              <ul className="space-y-1 text-sm text-muted-foreground">
                <li>• On-premises deployment (EU data center)</li>
                <li>• Immutable audit logs (7-year retention)</li>
                <li>• Zero external dependencies</li>
                <li>• SOC2 Type II compliant</li>
              </ul>
            </div>
          </div>

          {/* Use Case 2: Healthcare */}
          <div className="rounded-lg border border-border bg-card p-8">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                <Heart className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-foreground">Healthcare</h3>
                <p className="text-sm text-muted-foreground">Hospitals, MedTech, Pharma</p>
              </div>
            </div>

            <p className="mb-4 leading-relaxed text-muted-foreground">
              A healthcare provider wanted AI-assisted development for patient management systems but faced strict HIPAA
              and GDPR compliance requirements.
            </p>

            <div className="mb-4 rounded-lg border border-border bg-background p-4">
              <div className="mb-2 font-semibold text-foreground">Challenge:</div>
              <ul className="space-y-1 text-sm text-muted-foreground">
                <li>• HIPAA compliance (PHI protection)</li>
                <li>• GDPR Article 9 (health data)</li>
                <li>• Cannot use US cloud providers</li>
                <li>• Need breach notification capability</li>
              </ul>
            </div>

            <div className="rounded-lg border border-chart-3/50 bg-chart-3/10 p-4">
              <div className="mb-2 font-semibold text-chart-3">Solution with rbee:</div>
              <ul className="space-y-1 text-sm text-muted-foreground">
                <li>• Self-hosted (hospital data center)</li>
                <li>• EU-only deployment</li>
                <li>• Complete audit trail (breach detection)</li>
                <li>• HIPAA-compliant architecture</li>
              </ul>
            </div>
          </div>

          {/* Use Case 3: Legal Services */}
          <div className="rounded-lg border border-border bg-card p-8">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                <Scale className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-foreground">Legal Services</h3>
                <p className="text-sm text-muted-foreground">Law Firms, Legal Tech</p>
              </div>
            </div>

            <p className="mb-4 leading-relaxed text-muted-foreground">
              A law firm needed AI-powered document analysis and code generation but couldn't risk client
              confidentiality by using external AI providers.
            </p>

            <div className="mb-4 rounded-lg border border-border bg-background p-4">
              <div className="mb-2 font-semibold text-foreground">Challenge:</div>
              <ul className="space-y-1 text-sm text-muted-foreground">
                <li>• Attorney-client privilege (confidentiality)</li>
                <li>• Cannot send documents to external APIs</li>
                <li>• Need complete audit trail (legal hold)</li>
                <li>• EU data residency (GDPR)</li>
              </ul>
            </div>

            <div className="rounded-lg border border-chart-3/50 bg-chart-3/10 p-4">
              <div className="mb-2 font-semibold text-chart-3">Solution with rbee:</div>
              <ul className="space-y-1 text-sm text-muted-foreground">
                <li>• On-premises deployment (law firm servers)</li>
                <li>• Zero external data transfer</li>
                <li>• Immutable audit logs (legal hold)</li>
                <li>• Complete client confidentiality</li>
              </ul>
            </div>
          </div>

          {/* Use Case 4: Government */}
          <div className="rounded-lg border border-border bg-card p-8">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                <Shield className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-foreground">Government</h3>
                <p className="text-sm text-muted-foreground">Public Sector, Defense</p>
              </div>
            </div>

            <p className="mb-4 leading-relaxed text-muted-foreground">
              A government agency needed AI-assisted development for citizen services but faced strict data sovereignty
              and security requirements.
            </p>

            <div className="mb-4 rounded-lg border border-border bg-background p-4">
              <div className="mb-2 font-semibold text-foreground">Challenge:</div>
              <ul className="space-y-1 text-sm text-muted-foreground">
                <li>• National security (data sovereignty)</li>
                <li>• Cannot use foreign cloud providers</li>
                <li>• Need complete audit trail (transparency)</li>
                <li>• ISO 27001 certification required</li>
              </ul>
            </div>

            <div className="rounded-lg border border-chart-3/50 bg-chart-3/10 p-4">
              <div className="mb-2 font-semibold text-chart-3">Solution with rbee:</div>
              <ul className="space-y-1 text-sm text-muted-foreground">
                <li>• On-premises deployment (government data center)</li>
                <li>• EU-only infrastructure</li>
                <li>• ISO 27001 aligned</li>
                <li>• Complete data sovereignty</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
