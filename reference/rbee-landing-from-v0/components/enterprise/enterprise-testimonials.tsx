import { Building2, Heart, Scale } from "lucide-react"

export function EnterpriseTestimonials() {
  return (
    <section className="border-b border-slate-800 bg-slate-950 px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <div className="mb-16 text-center">
          <h2 className="mb-4 text-4xl font-bold text-white">Trusted by Regulated Industries</h2>
          <p className="mx-auto max-w-3xl text-balance text-xl text-slate-300">
            Organizations in highly regulated industries trust rbee for compliance-first AI infrastructure.
          </p>
        </div>

        {/* Metrics */}
        <div className="mb-16 grid gap-8 md:grid-cols-4">
          <div className="text-center">
            <div className="mb-2 text-4xl font-bold text-amber-400">100%</div>
            <div className="text-sm text-slate-400">GDPR Compliant</div>
          </div>
          <div className="text-center">
            <div className="mb-2 text-4xl font-bold text-amber-400">7 Years</div>
            <div className="text-sm text-slate-400">Audit Retention</div>
          </div>
          <div className="text-center">
            <div className="mb-2 text-4xl font-bold text-amber-400">Zero</div>
            <div className="text-sm text-slate-400">Compliance Violations</div>
          </div>
          <div className="text-center">
            <div className="mb-2 text-4xl font-bold text-amber-400">24/7</div>
            <div className="text-sm text-slate-400">Enterprise Support</div>
          </div>
        </div>

        {/* Testimonials */}
        <div className="grid gap-8 md:grid-cols-3">
          {/* Testimonial 1 */}
          <div className="rounded-lg border border-slate-800 bg-slate-900 p-8">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-amber-500/10">
                <Building2 className="h-6 w-6 text-amber-400" />
              </div>
              <div>
                <div className="font-semibold text-white">Dr. Klaus M.</div>
                <div className="text-sm text-slate-400">CTO, European Bank</div>
              </div>
            </div>
            <p className="leading-relaxed text-slate-300">
              "We couldn't use external AI providers due to PCI-DSS requirements. rbee gave us complete control with
              on-premises deployment and immutable audit logs. SOC2 audit passed with zero findings."
            </p>
          </div>

          {/* Testimonial 2 */}
          <div className="rounded-lg border border-slate-800 bg-slate-900 p-8">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-amber-500/10">
                <Heart className="h-6 w-6 text-amber-400" />
              </div>
              <div>
                <div className="font-semibold text-white">Anna S.</div>
                <div className="text-sm text-slate-400">DPO, Healthcare Provider</div>
              </div>
            </div>
            <p className="leading-relaxed text-slate-300">
              "HIPAA and GDPR compliance were non-negotiable. rbee's EU-only deployment and 7-year audit retention gave
              us the confidence to use AI for patient management systems. Complete data sovereignty."
            </p>
          </div>

          {/* Testimonial 3 */}
          <div className="rounded-lg border border-slate-800 bg-slate-900 p-8">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-amber-500/10">
                <Scale className="h-6 w-6 text-amber-400" />
              </div>
              <div>
                <div className="font-semibold text-white">Michael R.</div>
                <div className="text-sm text-slate-400">Managing Partner, Law Firm</div>
              </div>
            </div>
            <p className="leading-relaxed text-slate-300">
              "Attorney-client privilege meant we couldn't risk external AI providers. rbee's on-premises deployment and
              zero external data transfer gave us the security we needed. Client confidentiality protected."
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
