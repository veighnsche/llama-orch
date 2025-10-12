import { SectionContainer } from "@/components/primitives"

export function UseCasesIndustry() {
  return (
    <SectionContainer
      title="Industry-Specific Solutions"
      bgVariant="secondary"
      subtitle="rbee adapts to the unique compliance and security requirements of regulated industries."
    >
      <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          <div className="bg-card border border-border rounded-lg p-6 space-y-4">
            <h3 className="text-lg font-bold text-foreground">Financial Services</h3>
            <p className="text-muted-foreground text-sm leading-relaxed">
              GDPR compliance, audit trails, data residency controls. AI-powered code review and risk analysis without
              sending sensitive financial data to external APIs.
            </p>
          </div>

          <div className="bg-card border border-border rounded-lg p-6 space-y-4">
            <h3 className="text-lg font-bold text-foreground">Healthcare</h3>
            <p className="text-muted-foreground text-sm leading-relaxed">
              HIPAA-compliant infrastructure. Patient data never leaves your network. AI-assisted medical coding,
              documentation, and research without privacy concerns.
            </p>
          </div>

          <div className="bg-card border border-border rounded-lg p-6 space-y-4">
            <h3 className="text-lg font-bold text-foreground">Legal</h3>
            <p className="text-muted-foreground text-sm leading-relaxed">
              Attorney-client privilege maintained. Document analysis, contract review, and legal research with AI
              while keeping all client information confidential.
            </p>
          </div>

          <div className="bg-card border border-border rounded-lg p-6 space-y-4">
            <h3 className="text-lg font-bold text-foreground">Government</h3>
            <p className="text-muted-foreground text-sm leading-relaxed">
              Sovereign AI infrastructure. No foreign cloud dependencies. Complete audit trails for compliance with
              government security standards.
            </p>
          </div>

          <div className="bg-card border border-border rounded-lg p-6 space-y-4">
            <h3 className="text-lg font-bold text-foreground">Education</h3>
            <p className="text-muted-foreground text-sm leading-relaxed">
              Student data protection. AI-powered tutoring, grading assistance, and research tools without sending
              student information to external services.
            </p>
          </div>

          <div className="bg-card border border-border rounded-lg p-6 space-y-4">
            <h3 className="text-lg font-bold text-foreground">Manufacturing</h3>
            <p className="text-muted-foreground text-sm leading-relaxed">
              Protect trade secrets and proprietary designs. AI-assisted CAD, quality control, and process
              optimization without exposing intellectual property.
            </p>
          </div>
        </div>
      </SectionContainer>
  )
}
