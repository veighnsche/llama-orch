import {
  EnterpriseHero,
  EnterpriseProblem,
  EnterpriseSolution,
  EnterpriseCompliance,
  EnterpriseSecurity,
  EnterpriseHowItWorks,
  EnterpriseUseCases,
  EnterpriseComparison,
  EnterpriseFeatures,
  EnterpriseTestimonials,
  EnterpriseCTA,
} from '@rbee/ui/organisms/Enterprise'
import { EmailCapture } from '@rbee/ui/organisms/EmailCapture'
import { Footer } from '@rbee/ui/organisms/Footer'

export default function EnterprisePage() {
  return (
    <main className="min-h-screen bg-slate-950">
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
    </main>
  )
}
