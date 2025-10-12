import { HeroSection } from "@/components/hero-section"
import { WhatIsRbee } from "@/components/what-is-rbee"
import { AudienceSelector } from "@/components/audience-selector"
import { ProblemSection } from "@/components/problem-section"
import { SolutionSection } from "@/components/solution-section"
import { HowItWorksSection } from "@/components/how-it-works-section"
import { FeaturesSection } from "@/components/features-section"
import { UseCasesSection } from "@/components/use-cases-section"
import { ComparisonSection } from "@/components/comparison-section"
import { PricingSection } from "@/components/pricing-section"
import { SocialProofSection } from "@/components/social-proof-section"
import { TechnicalSection } from "@/components/technical-section"
import { FAQSection } from "@/components/faq-section"
import { CTASection } from "@/components/cta-section"
import { EmailCapture } from "@/components/email-capture"
import { Footer } from "@/components/footer"

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
