import { HeroSection } from '@/components/organisms/HeroSection/HeroSection'
import { WhatIsRbee } from '@/components/organisms/WhatIsRbee/WhatIsRbee'
import { AudienceSelector } from '@/components/organisms/AudienceSelector/AudienceSelector'
import { ProblemSection } from '@/components/organisms/ProblemSection/ProblemSection'
import { SolutionSection } from '@/components/organisms/SolutionSection/SolutionSection'
import { HowItWorksSection } from '@/components/organisms/HowItWorksSection/HowItWorksSection'
import { FeaturesSection } from '@/components/organisms/FeaturesSection/FeaturesSection'
import { UseCasesSection } from '@/components/organisms/UseCasesSection/UseCasesSection'
import { ComparisonSection } from '@/components/organisms/ComparisonSection/ComparisonSection'
import { PricingSection } from '@/components/organisms/PricingSection/PricingSection'
import { SocialProofSection } from '@/components/organisms/SocialProofSection/SocialProofSection'
import { TechnicalSection } from '@/components/organisms/TechnicalSection/TechnicalSection'
import { FAQSection } from '@/components/organisms/FaqSection/FaqSection'
import { CTASection } from '@/components/organisms/CtaSection/CtaSection'
import { EmailCapture } from '@/components/organisms/EmailCapture/EmailCapture'
import { Footer } from '@/components/organisms/Footer/Footer'

export default function Home() {
  return (
    <main className="min-h-screen pt-16">
      <HeroSection />
      <WhatIsRbee />
      <AudienceSelector />
      <EmailCapture />
      <ProblemSection />
      <SolutionSection />
      <HowItWorksSection />
      <FeaturesSection />
      <UseCasesSection />
      <ComparisonSection />
      <PricingSection />
      <SocialProofSection />
      <TechnicalSection />
      <FAQSection />
      <CTASection />
      <Footer />
    </main>
  )
}
