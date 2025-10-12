import { AlertTriangle, Globe, FileX, Scale } from "lucide-react"

export function EnterpriseProblem() {
  return (
    <section className="border-b border-border bg-gradient-to-b from-destructive/10 via-background to-background px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <div className="mb-16 text-center">
          <h2 className="mb-4 text-4xl font-bold text-foreground">The Compliance Challenge of Cloud AI</h2>
          <p className="mx-auto max-w-3xl text-balance text-xl text-muted-foreground">
            Using external AI providers creates compliance risks that can cost millions in fines and damage your
            reputation.
          </p>
        </div>

        <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-4">
          {/* Problem 1: Data Sovereignty */}
          <div className="group rounded-lg border border-destructive/50 bg-gradient-to-b from-destructive/20 to-background p-6 transition-all hover:border-destructive">
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-destructive/10">
              <Globe className="h-6 w-6 text-destructive" />
            </div>
            <h3 className="mb-3 text-xl font-semibold text-foreground">Data Sovereignty Violations</h3>
            <p className="text-pretty leading-relaxed text-muted-foreground">
              Your sensitive data crosses borders to US cloud providers. GDPR Article 44 violations. Schrems II
              compliance impossible. Data Protection Authorities watching.
            </p>
          </div>

          {/* Problem 2: No Audit Trail */}
          <div className="group rounded-lg border border-destructive/50 bg-gradient-to-b from-destructive/20 to-background p-6 transition-all hover:border-destructive">
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-destructive/10">
              <FileX className="h-6 w-6 text-destructive" />
            </div>
            <h3 className="mb-3 text-xl font-semibold text-foreground">Missing Audit Trails</h3>
            <p className="text-pretty leading-relaxed text-muted-foreground">
              No immutable logs. No proof of compliance. Cannot demonstrate GDPR Article 30 compliance. SOC2 audits
              fail. ISO 27001 certification impossible.
            </p>
          </div>

          {/* Problem 3: Regulatory Risk */}
          <div className="group rounded-lg border border-destructive/50 bg-gradient-to-b from-destructive/20 to-background p-6 transition-all hover:border-destructive">
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-destructive/10">
              <Scale className="h-6 w-6 text-destructive" />
            </div>
            <h3 className="mb-3 text-xl font-semibold text-foreground">Regulatory Fines</h3>
            <p className="text-pretty leading-relaxed text-muted-foreground">
              GDPR fines up to €20M or 4% of global revenue. Healthcare (HIPAA) violations: $50K per record. Financial
              services (PCI-DSS) breaches: reputation destroyed.
            </p>
          </div>

          {/* Problem 4: No Control */}
          <div className="group rounded-lg border border-destructive/50 bg-gradient-to-b from-destructive/20 to-background p-6 transition-all hover:border-destructive">
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-destructive/10">
              <AlertTriangle className="h-6 w-6 text-destructive" />
            </div>
            <h3 className="mb-3 text-xl font-semibold text-foreground">Zero Control</h3>
            <p className="text-pretty leading-relaxed text-muted-foreground">
              Provider changes terms. Data Processing Agreements worthless. Cannot guarantee data residency. Cannot
              prove compliance. Your DPO cannot sleep.
            </p>
          </div>
        </div>

        {/* Bottom emphasis */}
        <div className="mt-12 rounded-lg border border-destructive/50 bg-destructive/10 p-8 text-center">
          <p className="text-balance text-xl font-semibold text-destructive">
            "We cannot use external AI providers due to GDPR compliance requirements."
          </p>
          <p className="mt-2 text-muted-foreground">— Every EU CTO and Data Protection Officer</p>
        </div>
      </div>
    </section>
  )
}
