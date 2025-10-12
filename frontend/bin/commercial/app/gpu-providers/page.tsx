import { ProvidersHero } from "@/components/providers/providers-hero"
import { ProvidersProblem } from "@/components/providers/providers-problem"
import { ProvidersSolution } from "@/components/providers/providers-solution"
import { ProvidersHowItWorks } from "@/components/providers/providers-how-it-works"
import { ProvidersFeatures } from "@/components/providers/providers-features"
import { ProvidersUseCases } from "@/components/providers/providers-use-cases"
import { ProvidersEarnings } from "@/components/providers/providers-earnings"
import { ProvidersMarketplace } from "@/components/providers/providers-marketplace"
import { ProvidersSecurity } from "@/components/providers/providers-security"
import { ProvidersTestimonials } from "@/components/providers/providers-testimonials"
import { ProvidersCTA } from "@/components/providers/providers-cta"
import { EmailCapture } from "@/components/email-capture"
import { Footer } from "@/components/footer"

export default function GPUProvidersPage() {
  return (
    <main className="min-h-screen bg-background pt-16">
      <ProvidersHero />
      <EmailCapture />
      <ProvidersProblem />
      <ProvidersSolution />
      <ProvidersHowItWorks />
      <ProvidersFeatures />
      <ProvidersUseCases />
      <ProvidersEarnings />
      <ProvidersMarketplace />
      <ProvidersSecurity />
      <ProvidersTestimonials />
      <ProvidersCTA />
      <Footer />
    </main>
  )
}
