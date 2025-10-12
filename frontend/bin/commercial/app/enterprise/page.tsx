import { EnterpriseHero } from '@/components/organisms/Enterprise/enterprise-hero'
import { EnterpriseProblem } from '@/components/organisms/Enterprise/enterprise-problem'
import { EnterpriseSolution } from '@/components/organisms/Enterprise/enterprise-solution'
import { EnterpriseCompliance } from '@/components/organisms/Enterprise/enterprise-compliance'
import { EnterpriseSecurity } from '@/components/organisms/Enterprise/enterprise-security'
import { EnterpriseHowItWorks } from '@/components/organisms/Enterprise/enterprise-how-it-works'
import { EnterpriseUseCases } from '@/components/organisms/Enterprise/enterprise-use-cases'
import { EnterpriseComparison } from '@/components/organisms/Enterprise/enterprise-comparison'
import { EnterpriseFeatures } from '@/components/organisms/Enterprise/enterprise-features'
import { EnterpriseTestimonials } from '@/components/organisms/Enterprise/enterprise-testimonials'
import { EnterpriseCTA } from '@/components/organisms/Enterprise/enterprise-cta'
import { EmailCapture } from '@/components/organisms/EmailCapture/EmailCapture'
import { Footer } from '@/components/organisms/Footer/Footer'

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
