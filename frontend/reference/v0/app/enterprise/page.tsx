import { EnterpriseHero } from "@/components/enterprise/enterprise-hero"
import { EnterpriseProblem } from "@/components/enterprise/enterprise-problem"
import { EnterpriseSolution } from "@/components/enterprise/enterprise-solution"
import { EnterpriseCompliance } from "@/components/enterprise/enterprise-compliance"
import { EnterpriseSecurity } from "@/components/enterprise/enterprise-security"
import { EnterpriseHowItWorks } from "@/components/enterprise/enterprise-how-it-works"
import { EnterpriseUseCases } from "@/components/enterprise/enterprise-use-cases"
import { EnterpriseComparison } from "@/components/enterprise/enterprise-comparison"
import { EnterpriseFeatures } from "@/components/enterprise/enterprise-features"
import { EnterpriseTestimonials } from "@/components/enterprise/enterprise-testimonials"
import { EnterpriseCTA } from "@/components/enterprise/enterprise-cta"
import { EmailCapture } from "@/components/email-capture"
import { Footer } from "@/components/footer"

export default function EnterprisePage() {
  return (
    <main className="min-h-screen bg-slate-950 pt-16">
      <EnterpriseHero />
      <EmailCapture />
      <EnterpriseProblem />
      <EnterpriseSolution />
      <EnterpriseCompliance />
      <EnterpriseSecurity />
      <EnterpriseHowItWorks />
      <EnterpriseUseCases />
      <EnterpriseComparison />
      <EnterpriseFeatures />
      <EnterpriseTestimonials />
      <EnterpriseCTA />
      <Footer />
    </main>
  )
}
