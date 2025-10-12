import { Building2, Heart, Scale } from "lucide-react"

export function EnterpriseTestimonials() {
  return (
    <section className="border-b border-border bg-background px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <div className="mb-16 text-center">
          <h2 className="mb-4 text-4xl font-bold text-foreground">Trusted by Regulated Industries</h2>
          <p className="mx-auto max-w-3xl text-balance text-xl text-muted-foreground">
            Organizations in highly regulated industries trust rbee for compliance-first AI infrastructure.
          </p>
        </div>

        {/* Metrics */}
        <div className="mb-16 grid gap-8 md:grid-cols-4">
          <div className="text-center">
            <div className="mb-2 text-4xl font-bold text-primary">100%</div>
            <div className="text-sm text-muted-foreground">GDPR Compliant</div>
          </div>
          <div className="text-center">
            <div className="mb-2 text-4xl font-bold text-primary">7 Years</div>
            <div className="text-sm text-muted-foreground">Audit Retention</div>
          </div>
          <div className="text-center">
            <div className="mb-2 text-4xl font-bold text-primary">Zero</div>
            <div className="text-sm text-muted-foreground">Compliance Violations</div>
          </div>
          <div className="text-center">
            <div className="mb-2 text-4xl font-bold text-primary">24/7</div>
            <div className="text-sm text-muted-foreground">Enterprise Support</div>
          </div>
        </div>

        {/* Testimonials */}
        <div className="grid gap-8 md:grid-cols-3">
          {/* Testimonial 1 */}
          <div className="rounded-lg border border-border bg-card p-8">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                <Building2 className="h-6 w-6 text-primary" />
              </div>
              <div>
                <div className="font-semibold text-foreground">Dr. Klaus M.</div>
                <div className="text-sm text-muted-foreground">CTO, European Bank</div>
              </div>
            </div>
            <p className="leading-relaxed text-muted-foreground">
              "We couldn't use external AI providers due to PCI-DSS requirements. rbee gave us complete control with
              on-premises deployment and immutable audit logs. SOC2 audit passed with zero findings."
            </p>
          </div>

          {/* Testimonial 2 */}
          <div className="rounded-lg border border-border bg-card p-8">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                <Heart className="h-6 w-6 text-primary" />
              </div>
              <div>
                <div className="font-semibold text-foreground">Anna S.</div>
                <div className="text-sm text-muted-foreground">DPO, Healthcare Provider</div>
              </div>
            </div>
            <p className="leading-relaxed text-muted-foreground">
              "HIPAA and GDPR compliance were non-negotiable. rbee's EU-only deployment and 7-year audit retention gave
              us the confidence to use AI for patient management systems. Complete data sovereignty."
            </p>
          </div>

          {/* Testimonial 3 */}
          <div className="rounded-lg border border-border bg-card p-8">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                <Scale className="h-6 w-6 text-primary" />
              </div>
              <div>
                <div className="font-semibold text-foreground">Michael R.</div>
                <div className="text-sm text-muted-foreground">Managing Partner, Law Firm</div>
              </div>
            </div>
            <p className="leading-relaxed text-muted-foreground">
              "Attorney-client privilege meant we couldn't risk external AI providers. rbee's on-premises deployment and
              zero external data transfer gave us the security we needed. Client confidentiality protected."
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
